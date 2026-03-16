import { uploadTriangles } from "../src/bvh";
import { createCubemap } from "../src/cubemap";
import { createLBVH, dispatchLBVH } from "../src/lbvh";
import { computeVertexAABB } from "../src/lighting";
import { lookAt, multiply, perspective } from "../src/math";
import { createGodRays } from "../src/godrays";
import { createPostProcess } from "../src/post";
import { createSplat } from "../src/splat";
import { createSphere, createStones } from "./scene";
import { createDirect } from "./direct";
import {
    createSky,
    sampleGradient,
    lightDirection,
    shadowFade,
    uploadSky,
    uploadSkyScene,
} from "../src/sky";

const PROBE_SIZE = 384;
const VIEWPORT_QUALITY = 2.5;
const FOV = 60;
const fovRad = (FOV * Math.PI) / 180;
const viewportH = Math.ceil(VIEWPORT_QUALITY * PROBE_SIZE * Math.tan(fovRad / 2));

const bar = document.querySelector(".loading-bar") as HTMLElement;
const overlay = document.querySelector(".loading-overlay") as HTMLElement;
function progress(p: number) {
    bar.style.width = `${p * 100}%`;
}
const yieldFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));

if (!navigator.gpu) {
    document.body.textContent = "WebGPU is not supported in this browser.";
    throw new Error("WebGPU not supported");
}
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    document.body.textContent = "No WebGPU adapter found.";
    throw new Error("No WebGPU adapter");
}

const canvas = document.querySelector("canvas")!;
const dpr = window.devicePixelRatio || 1;
const aspect = canvas.clientWidth / canvas.clientHeight;
let renderW = Math.ceil(viewportH * aspect);
const renderH = viewportH;
canvas.width = Math.floor(canvas.clientWidth * dpr);
canvas.height = Math.floor(canvas.clientHeight * dpr);

const device = await adapter.requestDevice();
const context = canvas.getContext("webgpu")!;
const format = navigator.gpu.getPreferredCanvasFormat();
canvas.style.imageRendering = "pixelated";
context.configure({ device, format, alphaMode: "premultiplied" });

progress(0.05);
await yieldFrame();

// Shell grass constants
const LAYERS = 8;
const SUBDIVISIONS = 16;
const AREA = 200;
const HEIGHT = 0.1;
const COLOR = 0x5a8a32;
const DENSITY = 80;
const ROOT_L = 0.85;
const TIP_L = 1.05;
const HUE_FREQ = 10.0;
const HUE_VAR = 0.02;

// Shell mesh: ground plane (y=0) + LAYERS shell layers
const vertsPerSide = SUBDIVISIONS + 1;
const totalLayers = LAYERS + 1;
const vertexCount = totalLayers * vertsPerSide * vertsPerSide;
const indexCount = totalLayers * SUBDIVISIONS * SUBDIVISIONS * 6;
const vertices = new Float32Array(vertexCount * 3);
const indices = new Uint16Array(indexCount);

let vi = 0;
let ii = 0;
for (let layer = 0; layer < totalLayers; layer++) {
    const y = layer / LAYERS;
    const baseVertex = layer * vertsPerSide * vertsPerSide;
    for (let gz = 0; gz < vertsPerSide; gz++) {
        for (let gx = 0; gx < vertsPerSide; gx++) {
            vertices[vi++] = gx / SUBDIVISIONS - 0.5;
            vertices[vi++] = y;
            vertices[vi++] = gz / SUBDIVISIONS - 0.5;
        }
    }
    for (let gz = 0; gz < SUBDIVISIONS; gz++) {
        for (let gx = 0; gx < SUBDIVISIONS; gx++) {
            const bl = baseVertex + gz * vertsPerSide + gx;
            const br = bl + 1;
            const tl = bl + vertsPerSide;
            const tr = tl + 1;
            indices[ii++] = bl;
            indices[ii++] = tl;
            indices[ii++] = br;
            indices[ii++] = br;
            indices[ii++] = tl;
            indices[ii++] = tr;
        }
    }
}

const vertexBuffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);

const indexBuffer = device.createBuffer({
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(indexBuffer, 0, indices);

// Stone geometry
const stones = createStones();
const stoneVertexBuffer = device.createBuffer({
    size: stones.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(stoneVertexBuffer, 0, stones.vertices);

const stoneIndexBuffer = device.createBuffer({
    size: stones.indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(stoneIndexBuffer, 0, stones.indices);
const stoneIndexCount = stones.indices.length;

// Sphere geometry (orb + wisps)
const sphere = createSphere(8);
const orbVertexBuffer = device.createBuffer({
    size: sphere.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(orbVertexBuffer, 0, sphere.vertices);
const orbIndexBuffer = device.createBuffer({
    size: sphere.indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(orbIndexBuffer, 0, sphere.indices);
const orbIndexCount = sphere.indices.length;

const orbBuffers = Array.from({ length: 4 }, () =>
    device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
);

// GPU LBVH from stone geometry
const { triBuffer, triAABBBuffer, count: triCount } = uploadTriangles(device, stones.vertices, stones.indices);
const lbvh = await createLBVH(device, triAABBBuffer, triCount);
const initEncoder = device.createCommandEncoder();
dispatchLBVH(lbvh, initEncoder, device);
device.queue.submit([initEncoder.finish()]);
await device.queue.onSubmittedWorkDone();
const nodeBuffer = lbvh.treeNodes;
const triIdBuffer = lbvh.sortedIds;

// Uniforms: viewProj(64) + sunDir(12)+shadowFade(4) + sunColor(12)+pad(4) + ambient(12)+pad(4) + cameraPos(12)+pad(4) = 128
const uniformBuffer = device.createBuffer({
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// Point lights: 64 * (position(12)+pad(4) + color(12)+radius(4)) = 64 * 32 = 2048
const lightBuffer = device.createBuffer({
    size: 64 * 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
const flickerQueue = new Float32Array(16);
for (let i = 0; i < 16; i++) flickerQueue[i] = 0.8 + Math.random() * 0.25;
let flickerIdx = 0;
let elapsed = 0;

const baseR = ((COLOR >> 16) & 0xff) / 255;
const baseG = ((COLOR >> 8) & 0xff) / 255;
const baseB = (COLOR & 0xff) / 255;

const sceneBindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: "uniform" },
        },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
});

const sky = createSky(device, "rgba8unorm");

const sceneBindGroup = device.createBindGroup({
    layout: sceneBindGroupLayout,
    entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: nodeBuffer } },
        { binding: 2, resource: { buffer: triBuffer } },
        { binding: 3, resource: { buffer: triIdBuffer } },
        { binding: 4, resource: { buffer: lightBuffer } },
        { binding: 5, resource: { buffer: sky.skyBuffer } },
    ],
});

progress(0.2);
await yieldFrame();

const stoneAABB = computeVertexAABB(stones.vertices, 6);
const grassAABB = { min: [-AREA / 2, 0, -AREA / 2] as [number, number, number], max: [AREA / 2, HEIGHT, AREA / 2] as [number, number, number] };
const orbsAABB = { min: [-3.5, 1.2, -3.5] as [number, number, number], max: [3.5, 2.8, 3.5] as [number, number, number] };

const sceneConfig = {
    device,
    vertexBuffer,
    indexBuffer,
    indexCount,
    stoneVertexBuffer,
    stoneIndexBuffer,
    stoneIndexCount,
    orbVertexBuffer,
    orbIndexBuffer,
    orbIndexCount,
    orbBuffers,
    meshAABBs: { grass: grassAABB, stones: stoneAABB, orbs: orbsAABB },
    area: AREA,
    height: HEIGHT,
    baseR,
    baseG,
    baseB,
    density: DENSITY,
    rootL: ROOT_L,
    tipL: TIP_L,
    hueFreq: HUE_FREQ,
    hueVar: HUE_VAR,
};

const bvhConfig = {
    nodeBuffer,
    triBuffer,
    triIdBuffer,
    lightBuffer,
    skyBuffer: sky.skyBuffer,
    sceneBuffer: sky.sceneBuffer,
};

const cubemap = createCubemap({ ...sceneConfig, ...bvhConfig });

const splat = createSplat({ ...sceneConfig, ...bvhConfig });

const godrays = createGodRays(device);
const post = createPostProcess(device, format);

progress(0.8);
await yieldFrame();

function projectSunToScreen(
    vp: Float32Array,
    dirX: number,
    dirY: number,
    dirZ: number,
): { u: number; v: number; visibility: number } {
    const sx = -dirX * 1000;
    const sy = -dirY * 1000;
    const sz = -dirZ * 1000;
    const cx = vp[0] * sx + vp[4] * sy + vp[8] * sz + vp[12];
    const cy = vp[1] * sx + vp[5] * sy + vp[9] * sz + vp[13];
    const cw = vp[3] * sx + vp[7] * sy + vp[11] * sz + vp[15];
    if (cw <= 0) return { u: 0, v: 0, visibility: 0 };
    const ndcX = cx / cw;
    const ndcY = cy / cw;
    const edge = Math.max(Math.abs(ndcX), Math.abs(ndcY));
    const t = Math.max(0, Math.min(1, (edge - 0.6) / 0.6));
    return {
        u: ndcX * 0.5 + 0.5,
        v: 1 - (ndcY * 0.5 + 0.5),
        visibility: Math.max(0, 1 - t * t * (3 - 2 * t)),
    };
}

let depthTexture = device.createTexture({
    size: [renderW, renderH],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

let colorTexture = device.createTexture({
    size: [renderW, renderH],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

let posterizeTexture = device.createTexture({
    size: [renderW, renderH],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

let colorView = colorTexture.createView();
let posterizeView = posterizeTexture.createView();
let depthView = depthTexture.createView();

const direct = createDirect({
    ...sceneConfig,
    sceneBindGroupLayout,
    sceneBindGroup,
    skyBuffer: sky.skyBuffer,
});

direct.resize(renderW, renderH, colorView, posterizeView);
post.resize(renderW, renderH);

let proj = perspective(FOV, renderW / renderH, 0.1, 200);

new ResizeObserver(() => {
    const newDpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(canvas.clientWidth * newDpr);
    canvas.height = Math.floor(canvas.clientHeight * newDpr);

    const newAspect = canvas.clientWidth / canvas.clientHeight;
    const w = Math.ceil(viewportH * newAspect);
    if (renderW === w) return;
    renderW = w;
    depthTexture.destroy();
    depthTexture = device.createTexture({
        size: [w, viewportH],
        format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    colorTexture.destroy();
    colorTexture = device.createTexture({
        size: [w, viewportH],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    posterizeTexture.destroy();
    posterizeTexture = device.createTexture({
        size: [w, viewportH],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    colorView = colorTexture.createView();
    posterizeView = posterizeTexture.createView();
    depthView = depthTexture.createView();
    direct.resize(w, viewportH, colorView, posterizeView);
    godrays.resize(w, viewportH);
    post.resize(w, viewportH);
    proj = perspective(FOV, w / viewportH, 0.1, 200);
}).observe(canvas);

let yaw = 0.927;
let pitch = 0;
let px = 20,
    py = 1.6,
    pz = 15;

let hour = 16.5;
const timeSpeed = 0.05;
let dragging = false;

const slider = document.getElementById("time-slider") as HTMLInputElement;
const clock = document.getElementById("time-clock")!;

function syncSlider() {
    const h = Math.floor(hour) % 24;
    const m = Math.floor((hour % 1) * 60);
    clock.textContent = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
    slider.value = String(Math.round(hour * 10));
}

slider.addEventListener("input", () => {
    hour = parseFloat(slider.value) / 10;
});
slider.addEventListener("mousedown", () => (dragging = true));
slider.addEventListener("mouseup", () => (dragging = false));
slider.addEventListener("touchstart", () => (dragging = true));
slider.addEventListener("touchend", () => (dragging = false));

const isTouchDevice = matchMedia("(pointer: coarse)").matches;

const keys: Record<string, boolean> = {};
if (!isTouchDevice) {
    canvas.addEventListener("click", () => canvas.requestPointerLock());
}
document.addEventListener("mousemove", (e) => {
    if (document.pointerLockElement !== canvas) return;
    const scale = 1.5 / canvas.clientHeight;
    const halfPi = Math.PI / 2 - 0.01;
    yaw -= e.movementX * scale;
    pitch = Math.max(-halfPi, Math.min(halfPi, pitch - e.movementY * scale));
});
const ui = document.querySelectorAll<HTMLElement>(".hud, .fps, .time-panel, .controls-hint");
document.addEventListener("keydown", (e) => {
    keys[e.code] = true;
    if (e.key === "f" || e.key === "F") {
        const hidden = ui[0].style.display === "none";
        ui.forEach((el) => (el.style.display = hidden ? "" : "none"));
    }
});
document.addEventListener("keyup", (e) => {
    keys[e.code] = false;
});

let mode: "none" | "cubemap" | "splat" = "splat";
const modeGroup = document.getElementById("mode-group")!;
modeGroup.addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest(".mode-btn") as HTMLElement | null;
    if (!btn) return;
    mode = btn.dataset.mode as typeof mode;
    modeGroup.querySelector(".active")?.classList.remove("active");
    btn.classList.add("active");
});

const controlsHint = document.getElementById("controls-hint")!;
let hintDismissed = false;
function dismissHint() {
    if (hintDismissed) return;
    hintDismissed = true;
    controlsHint.classList.add("hidden");
}
if (isTouchDevice) {
    dismissHint();
    const uiToggle = document.getElementById("ui-toggle");
    if (uiToggle) {
        let uiHidden = false;
        uiToggle.addEventListener("click", () => {
            uiHidden = !uiHidden;
            ui.forEach((el) => (el.style.display = uiHidden ? "none" : ""));
        });
    }
} else {
    document.addEventListener("keydown", (e) => {
        if (["KeyW", "KeyA", "KeyS", "KeyD"].includes(e.code)) dismissHint();
    }, { once: true });
    document.addEventListener("mousemove", (e) => {
        if (document.pointerLockElement === canvas) dismissHint();
    }, { once: true });
    setTimeout(dismissHint, 10000);
}

// Touch controls
let touchMX = 0;
let touchMZ = 0;
let joystickId: number | null = null;
let lookId: number | null = null;
let joystickOriginX = 0;
let joystickOriginY = 0;
let lookLastX = 0;
let lookLastY = 0;

let joystickBase: HTMLElement | null = null;
let joystickThumb: HTMLElement | null = null;

if (isTouchDevice) {
    const style = document.createElement("style");
    style.textContent = `
        .joystick-base {
            position: fixed;
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: 2px solid var(--border);
            background: var(--panel);
            pointer-events: none;
            z-index: 20;
            display: none;
            transform: translate(-50%, -50%);
        }
        .joystick-thumb {
            position: absolute;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--accent);
            opacity: 0.6;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
        }
    `;
    document.head.appendChild(style);

    joystickBase = document.createElement("div");
    joystickBase.className = "joystick-base";
    joystickThumb = document.createElement("div");
    joystickThumb.className = "joystick-thumb";
    joystickBase.appendChild(joystickThumb);
    document.body.appendChild(joystickBase);

    function isUIElement(target: EventTarget | null): boolean {
        if (!target || !(target instanceof HTMLElement)) return false;
        return !!target.closest(".time-panel, .hud, .fps");
    }

    canvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        for (let i = 0; i < e.changedTouches.length; i++) {
            const t = e.changedTouches[i];
            if (isUIElement(t.target)) continue;
            const midX = window.innerWidth / 2;
            if (t.clientX < midX && joystickId === null) {
                joystickId = t.identifier;
                joystickOriginX = t.clientX;
                joystickOriginY = t.clientY;
                joystickBase!.style.left = t.clientX + "px";
                joystickBase!.style.top = t.clientY + "px";
                joystickBase!.style.display = "block";
                joystickThumb!.style.transform = "translate(-50%, -50%)";
            } else if (lookId === null) {
                lookId = t.identifier;
                lookLastX = t.clientX;
                lookLastY = t.clientY;
            }
        }
    }, { passive: false });

    canvas.addEventListener("touchmove", (e) => {
        e.preventDefault();
        for (let i = 0; i < e.changedTouches.length; i++) {
            const t = e.changedTouches[i];
            if (t.identifier === joystickId) {
                let dx = t.clientX - joystickOriginX;
                let dy = t.clientY - joystickOriginY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const maxR = 60;
                if (dist > maxR) {
                    dx = (dx / dist) * maxR;
                    dy = (dy / dist) * maxR;
                }
                touchMX = dx / maxR;
                touchMZ = dy / maxR;
                joystickThumb!.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
            } else if (t.identifier === lookId) {
                const scale = 1.5 / canvas.clientHeight;
                const halfPi = Math.PI / 2 - 0.01;
                yaw -= (t.clientX - lookLastX) * scale;
                pitch = Math.max(-halfPi, Math.min(halfPi, pitch - (t.clientY - lookLastY) * scale));
                lookLastX = t.clientX;
                lookLastY = t.clientY;
            }
        }
    }, { passive: false });

    function handleTouchEnd(e: TouchEvent) {
        for (let i = 0; i < e.changedTouches.length; i++) {
            const t = e.changedTouches[i];
            if (t.identifier === joystickId) {
                joystickId = null;
                touchMX = 0;
                touchMZ = 0;
                joystickBase!.style.display = "none";
            } else if (t.identifier === lookId) {
                lookId = null;
            }
        }
    }
    canvas.addEventListener("touchend", handleTouchEnd);
    canvas.addEventListener("touchcancel", handleTouchEnd);
}


const fpsDisplay = document.getElementById("fps")!;
let fpsFrames = 0;
let fpsTime = performance.now();

let lastTime = performance.now();
let loaded = false;
const tmp4 = new Float32Array(4);
const tmp8 = new Float32Array(8);
const tmp16 = new Float32Array(16);

function frame() {
    const now = performance.now();
    const dt = (now - lastTime) / 1000;
    lastTime = now;

    fpsFrames++;
    if (now - fpsTime >= 500) {
        fpsDisplay.textContent = `${Math.round((fpsFrames * 1000) / (now - fpsTime))} fps`;
        fpsFrames = 0;
        fpsTime = now;
    }

    const sprinting = keys["ShiftLeft"] || keys["ShiftRight"];
    const speed = 2.5 * (sprinting ? 3 : 1);

    let mx = 0,
        mz = 0;
    if (keys["KeyW"]) mz -= 1;
    if (keys["KeyS"]) mz += 1;
    if (keys["KeyA"]) mx -= 1;
    if (keys["KeyD"]) mx += 1;
    mx += touchMX;
    mz += touchMZ;

    const len = Math.sqrt(mx * mx + mz * mz);
    if (len > 0) {
        const move = (speed * dt) / len;
        const cy = Math.cos(yaw),
            sy = Math.sin(yaw);
        px += (mz * sy + mx * cy) * move;
        pz += (mz * cy - mx * sy) * move;
    }

    if (!dragging) {
        hour += dt * timeSpeed;
        if (hour >= 24) hour -= 24;
    }
    syncSlider();
    const elevation = 55 * Math.sin(((hour - 6) / 12) * Math.PI);
    const azimuth = (hour / 24) * 360 - 25;
    const output = sampleGradient(elevation);
    const [ldx, ldy, ldz] = lightDirection(azimuth, elevation);
    const sf = shadowFade(elevation);

    const lx = -Math.sin(yaw) * Math.cos(pitch);
    const ly = Math.sin(pitch);
    const lz = -Math.cos(yaw) * Math.cos(pitch);
    const view = lookAt(px, py, pz, px + lx, py + ly, pz + lz, 0, 1, 0);
    const viewProj = multiply(proj, view);
    elapsed += dt;
    const patrolAngle = elapsed * 0.4;
    const bobY =
        2 +
        Math.sin(elapsed * 1.5) * 0.25 +
        Math.sin(elapsed * 2.3 + 1.7) * 0.1 +
        Math.sin(elapsed * 3.7 + 4.1) * 0.05;
    const lightX =
        Math.cos(patrolAngle) * 3 +
        Math.sin(elapsed * 0.7 + 0.3) * 0.12 +
        Math.sin(elapsed * 1.9 + 2.5) * 0.06;
    const lightZ =
        Math.sin(patrolAngle) * 3 +
        Math.sin(elapsed * 0.9 + 1.1) * 0.12 +
        Math.sin(elapsed * 2.1 + 3.8) * 0.06;

    flickerQueue[flickerIdx] = 0.8 + Math.random() * 0.25;
    flickerIdx = (flickerIdx + 1) & 15;
    let flickerSum = 0;
    for (let i = 0; i < 16; i++) flickerSum += flickerQueue[i];
    const flicker = flickerSum / 16;
    const heightDim = 0.85 + 0.15 * Math.max(0, Math.min(1, (bobY - 1.5) / 1.0));
    const intensity = 3 * flicker * heightDim;
    const flickerNorm = (flicker - 0.8) / 0.25;
    const lr = 0.3 + (flickerNorm > 0 ? flickerNorm * 0.2 : flickerNorm * 0.15);
    const lg = 0.75 + (flickerNorm > 0 ? flickerNorm * 0.15 : flickerNorm * 0.25);
    const lb = 0.82 + (flickerNorm > 0 ? flickerNorm * 0.13 : flickerNorm * 0.22);
    tmp4[0] = lightX; tmp4[1] = bobY; tmp4[2] = lightZ; tmp4[3] = 0;
    device.queue.writeBuffer(lightBuffer, 0, tmp4);
    tmp4[0] = lr * intensity; tmp4[1] = lg * intensity; tmp4[2] = lb * intensity; tmp4[3] = 15;
    device.queue.writeBuffer(lightBuffer, 16, tmp4);

    tmp8[0] = lightX; tmp8[1] = bobY; tmp8[2] = lightZ; tmp8[3] = 0.2;
    tmp8[4] = 0.75; tmp8[5] = 1.0; tmp8[6] = 1.0; tmp8[7] = 0;
    device.queue.writeBuffer(orbBuffers[0], 0, tmp8);
    const w1a = elapsed * 1.5;
    const w1b = Math.sin(elapsed * 2) * 0.1;
    tmp8[0] = lightX + Math.cos(w1a) * 0.3; tmp8[1] = bobY + w1b;
    tmp8[2] = lightZ + Math.sin(w1a) * 0.3; tmp8[3] = 0.12;
    tmp8[4] = 0.5; tmp8[5] = 0.9; tmp8[6] = 0.95; tmp8[7] = 0;
    device.queue.writeBuffer(orbBuffers[1], 0, tmp8);
    const w2a = elapsed * -1.1 + 2.1;
    const w2b = Math.sin(elapsed * 2 + 2.1) * 0.08;
    tmp8[0] = lightX + Math.cos(w2a) * 0.4; tmp8[1] = bobY + w2b;
    tmp8[2] = lightZ + Math.sin(w2a) * 0.4; tmp8[3] = 0.1;
    device.queue.writeBuffer(orbBuffers[2], 0, tmp8);
    const w3a = elapsed * 2.0 + 4.2;
    const w3b = Math.sin(elapsed * 2 + 4.2) * 0.06;
    tmp8[0] = lightX + Math.cos(w3a) * 0.22; tmp8[1] = bobY + w3b;
    tmp8[2] = lightZ + Math.sin(w3a) * 0.22; tmp8[3] = 0.08;
    device.queue.writeBuffer(orbBuffers[3], 0, tmp8);

    device.queue.writeBuffer(uniformBuffer, 0, viewProj);
    tmp4[0] = -ldx; tmp4[1] = -ldy; tmp4[2] = -ldz; tmp4[3] = sf;
    device.queue.writeBuffer(uniformBuffer, 64, tmp4);
    const sc = output.sunColor;
    const si = output.sunIntensity;
    tmp4[0] = sc.r * si; tmp4[1] = sc.g * si; tmp4[2] = sc.b * si; tmp4[3] = 0;
    device.queue.writeBuffer(uniformBuffer, 80, tmp4);
    const ac = output.ambientColor;
    const ai = output.ambientIntensity;
    tmp4[0] = ac.r * ai; tmp4[1] = ac.g * ai; tmp4[2] = ac.b * ai; tmp4[3] = 0;
    device.queue.writeBuffer(uniformBuffer, 96, tmp4);
    tmp4[0] = px; tmp4[1] = py; tmp4[2] = pz; tmp4[3] = 0;
    device.queue.writeBuffer(uniformBuffer, 112, tmp4);

    uploadSky(device, sky.skyBuffer, output, azimuth, elevation);
    const cy = Math.cos(yaw),
        sy = Math.sin(yaw);
    const cp = Math.cos(pitch),
        sp = Math.sin(pitch);
    tmp16[0] = cy; tmp16[1] = 0; tmp16[2] = -sy; tmp16[3] = 0;
    tmp16[4] = sp * sy; tmp16[5] = cp; tmp16[6] = sp * cy; tmp16[7] = 0;
    tmp16[8] = sy * cp; tmp16[9] = -sp; tmp16[10] = cy * cp; tmp16[11] = 0;
    tmp16[12] = px; tmp16[13] = py; tmp16[14] = pz; tmp16[15] = 1;
    uploadSkyScene(
        device,
        sky.sceneBuffer,
        tmp16,
        [ldx, ldy, ldz],
        [sc.r * si, sc.g * si, sc.b * si],
        output.zenith,
        renderW,
        renderH,
        FOV,
    );

    const encoder = device.createCommandEncoder();

    let postInput: GPUTextureView;

    const sun = projectSunToScreen(viewProj, ldx, ldy, ldz);

    if (mode === "splat") {
        splat.encode(
            encoder,
            {
                cameraPos: [px, py, pz],
                cameraFwd: [lx, ly, lz],
                sunDir: [-ldx, -ldy, -ldz],
                sunColor: [sc.r * si, sc.g * si, sc.b * si],
                ambient: [ac.r * ai, ac.g * ai, ac.b * ai],
                shadowFade: sf,
                pointLightCount: 1,
                time: performance.now() / 1000,
            },
            colorView,
            depthView,
            viewProj,
        );
        godrays.encode(encoder, colorView, depthView, posterizeView, sun);
        postInput = posterizeView;
    } else if (mode === "cubemap") {
        cubemap.encode(
            encoder,
            {
                cameraPos: [px, py, pz],
                cameraFwd: [lx, ly, lz],
                sunDir: [-ldx, -ldy, -ldz],
                sunColor: [sc.r * si, sc.g * si, sc.b * si],
                ambient: [ac.r * ai, ac.g * ai, ac.b * ai],
                shadowFade: sf,
            },
            colorView,
            depthView,
        );
        godrays.encode(encoder, colorView, depthView, posterizeView, sun);
        postInput = posterizeView;
    } else {
        direct.encode(
            encoder,
            (e, c, d) => sky.encode(e, c, d),
            colorView,
            depthView,
            posterizeView,
        );
        godrays.encode(encoder, colorView, depthView, posterizeView, sun);
        postInput = posterizeView;
    }

    post.encode(encoder, postInput, context.getCurrentTexture().createView());

    device.queue.submit([encoder.finish()]);
    if (loaded) requestAnimationFrame(frame);
}

// Warmup: submit one frame to force GPU shader compilation
lastTime = performance.now();
frame();
await device.queue.onSubmittedWorkDone();
progress(1);
await yieldFrame();

loaded = true;
lastTime = performance.now();
overlay.style.opacity = "0";
overlay.addEventListener("transitionend", () => overlay.remove());
requestAnimationFrame(frame);
