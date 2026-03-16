const EXPOSURE = 0.75;
const BLOOM_INTENSITY = 0.15;
const BLOOM_THRESHOLD = 0.0;
const BLOOM_KNEE = 0.15;
const BLOOM_RADIUS = 0.5;
const MIP_FORMAT: GPUTextureFormat = "rgba16float";
const MAX_MIPS = 5;

const vertexWGSL = /* wgsl */ `
struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    let positions = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    let p = positions[vi];
    var out: VsOut;
    out.pos = vec4f(p, 0, 1);
    out.uv = p * vec2f(0.5, -0.5) + 0.5;
    return out;
}`;

const thresholdShader = /* wgsl */ `
${vertexWGSL}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let color = textureSample(srcTex, srcSamp, in.uv).rgb;
    let brightness = max(max(color.r, color.g), color.b);
    let soft = clamp((brightness - ${BLOOM_THRESHOLD} + ${BLOOM_KNEE}) / (2.0 * ${BLOOM_KNEE} + 0.0001), 0.0, 1.0);
    return vec4f(color * soft * soft, 1.0);
}`;

const downsampleShader = /* wgsl */ `
${vertexWGSL}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;
@group(0) @binding(2) var<uniform> params: vec4f;

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let ts = params.xy;
    let uv = in.uv;
    var color = textureSample(srcTex, srcSamp, uv).rgb * 4.0;
    color += textureSample(srcTex, srcSamp, uv + vec2f(-ts.x, -ts.y)).rgb;
    color += textureSample(srcTex, srcSamp, uv + vec2f(ts.x, -ts.y)).rgb;
    color += textureSample(srcTex, srcSamp, uv + vec2f(-ts.x, ts.y)).rgb;
    color += textureSample(srcTex, srcSamp, uv + vec2f(ts.x, ts.y)).rgb;
    return vec4f(color / 8.0, 1.0);
}`;

const upsampleShader = /* wgsl */ `
${vertexWGSL}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;
@group(0) @binding(2) var<uniform> params: vec4f;

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let ts = params.xy * params.z;
    let uv = in.uv;
    var color = vec3f(0.0);
    color += textureSample(srcTex, srcSamp, uv + vec2f(-ts.x, -ts.y)).rgb;
    color += textureSample(srcTex, srcSamp, uv + vec2f(0.0, -ts.y)).rgb * 2.0;
    color += textureSample(srcTex, srcSamp, uv + vec2f(ts.x, -ts.y)).rgb;
    color += textureSample(srcTex, srcSamp, uv + vec2f(-ts.x, 0.0)).rgb * 2.0;
    color += textureSample(srcTex, srcSamp, uv + vec2f(ts.x, 0.0)).rgb * 2.0;
    color += textureSample(srcTex, srcSamp, uv + vec2f(-ts.x, ts.y)).rgb;
    color += textureSample(srcTex, srcSamp, uv + vec2f(0.0, ts.y)).rgb * 2.0;
    color += textureSample(srcTex, srcSamp, uv + vec2f(ts.x, ts.y)).rgb;
    let weight = params.w;
    return vec4f(color / 12.0 * weight, 1.0);
}`;

const compositeShader = /* wgsl */ `
${vertexWGSL}

@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;
@group(0) @binding(2) var srcSamp: sampler;

fn aces(x: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let scene = textureSample(sceneTex, srcSamp, in.uv).rgb;
    let bloom = textureSample(bloomTex, srcSamp, in.uv).rgb;
    let combined = (scene + bloom * ${BLOOM_INTENSITY}) * ${EXPOSURE};
    return vec4f(aces(combined), 1.0);
}`;

// FXAA detects edges by comparing luminance of the 4 diagonal neighbours
// against the centre pixel, then blends along the dominant edge orientation.
// Applied at render resolution before nearest-neighbour upscale to canvas.
const fxaaShader = /* wgsl */ `
${vertexWGSL}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;
@group(0) @binding(2) var<uniform> invSize: vec2f;

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let uv = in.uv;
    let ts = invSize;

    let m  = textureSample(srcTex, srcSamp, uv).rgb;
    let nw = textureSample(srcTex, srcSamp, uv + vec2f(-ts.x, -ts.y)).rgb;
    let ne = textureSample(srcTex, srcSamp, uv + vec2f( ts.x, -ts.y)).rgb;
    let sw = textureSample(srcTex, srcSamp, uv + vec2f(-ts.x,  ts.y)).rgb;
    let se = textureSample(srcTex, srcSamp, uv + vec2f( ts.x,  ts.y)).rgb;

    let lumaNW = luma(nw); let lumaNE = luma(ne);
    let lumaSW = luma(sw); let lumaSE = luma(se);
    let lumaM  = luma(m);

    let lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    let lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    let contrast = lumaMax - lumaMin;

    // Pass non-edge pixels through unchanged
    if (contrast < max(0.02, lumaMax * 0.05)) { return vec4f(m, 1.0); }

    // Determine dominant edge orientation and blend perpendicular neighbours
    let edgeH = abs(lumaNW + lumaNE - 2.0 * lumaM) + abs(lumaSW + lumaSE - 2.0 * lumaM);
    let edgeV = abs(lumaNW + lumaSW - 2.0 * lumaM) + abs(lumaNE + lumaSE - 2.0 * lumaM);
    let step = select(vec2f(ts.x, 0.0), vec2f(0.0, ts.y), edgeH >= edgeV);

    let c1 = textureSample(srcTex, srcSamp, uv + step).rgb;
    let c2 = textureSample(srcTex, srcSamp, uv - step).rgb;
    return vec4f(mix(m, (c1 + c2) * 0.5, clamp(contrast * 8.0, 0.0, 0.75)), 1.0);
}`;

interface Mip {
    texture: GPUTexture;
    view: GPUTextureView;
    width: number;
    height: number;
}

function fullscreenPass(
    encoder: GPUCommandEncoder,
    pipeline: GPURenderPipeline,
    bindGroup: GPUBindGroup,
    targetView: GPUTextureView,
    loadOp: "clear" | "load",
): void {
    const pass = encoder.beginRenderPass({
        colorAttachments: [
            {
                view: targetView,
                loadOp,
                storeOp: "store" as const,
                ...(loadOp === "clear" ? { clearValue: { r: 0, g: 0, b: 0, a: 0 } } : {}),
            },
        ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
}

export interface PostProcess {
    resize(width: number, height: number): void;
    encode(encoder: GPUCommandEncoder, inputView: GPUTextureView, outputView: GPUTextureView): void;
    destroy(): void;
}

export function createPostProcess(device: GPUDevice, outputFormat: GPUTextureFormat): PostProcess {
    const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    const nearestSampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });

    const thresholdModule = device.createShaderModule({ code: thresholdShader });
    const downsampleModule = device.createShaderModule({ code: downsampleShader });
    const upsampleModule = device.createShaderModule({ code: upsampleShader });
    const compositeModule = device.createShaderModule({ code: compositeShader });

    function makePipeline(
        module: GPUShaderModule,
        targetFormat: GPUTextureFormat,
        additive: boolean,
    ): GPURenderPipeline {
        return device.createRenderPipeline({
            layout: "auto",
            vertex: { module, entryPoint: "vs" },
            fragment: {
                module,
                entryPoint: "fs",
                targets: [
                    {
                        format: targetFormat,
                        blend: additive
                            ? {
                                  color: { srcFactor: "one", dstFactor: "one", operation: "add" },
                                  alpha: { srcFactor: "one", dstFactor: "zero", operation: "add" },
                              }
                            : undefined,
                    },
                ],
            },
            primitive: { topology: "triangle-list" },
        });
    }

    const thresholdPipeline = makePipeline(thresholdModule, MIP_FORMAT, false);
    const downsamplePipeline = makePipeline(downsampleModule, MIP_FORMAT, false);
    const upsamplePipeline = makePipeline(upsampleModule, MIP_FORMAT, true);
    const compositePipeline = makePipeline(compositeModule, outputFormat, false);

    const fxaaModule = device.createShaderModule({ code: fxaaShader });
    const fxaaPipeline = makePipeline(fxaaModule, outputFormat, false);

    // Intermediate render-resolution texture: composite writes here, FXAA reads it
    let fxaaTexture: GPUTexture | null = null;
    let fxaaView: GPUTextureView | null = null;

    // Uniform buffer holding (1/renderWidth, 1/renderHeight) for FXAA pixel size
    const fxaaInvSizeBuffer = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const fxaaInvSizeData = new Float32Array(2);

    let fxaaBindGroup: GPUBindGroup | null = null;

    let mips: Mip[] = [];
    let cachedWidth = 0;
    let cachedHeight = 0;

    const uniformPool: GPUBuffer[] = [];
    function getUniform(index: number): GPUBuffer {
        while (uniformPool.length <= index) {
            uniformPool.push(
                device.createBuffer({
                    size: 16,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                }),
            );
        }
        return uniformPool[index];
    }

    const paramData = new Float32Array(4);
    function uploadParams(idx: number, a: number, b: number, c: number, d: number): void {
        paramData[0] = a;
        paramData[1] = b;
        paramData[2] = c;
        paramData[3] = d;
        device.queue.writeBuffer(getUniform(idx), 0, paramData);
    }

    let downBindGroups: GPUBindGroup[] = [];
    let upBindGroups: GPUBindGroup[] = [];
    let cachedInputView: GPUTextureView | null = null;
    let thresholdBindGroup: GPUBindGroup | null = null;
    let compositeBindGroup: GPUBindGroup | null = null;

    function ensureMips(width: number, height: number): void {
        if (width === cachedWidth && height === cachedHeight) return;

        for (const m of mips) m.texture.destroy();
        mips = [];

        // Rebuild the FXAA intermediate texture at the new render resolution
        fxaaTexture?.destroy();
        fxaaTexture = device.createTexture({
            size: { width, height },
            format: outputFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        fxaaView = fxaaTexture.createView();
        fxaaBindGroup = null; // rebuilt on next encode

        // Upload render-space pixel size for FXAA neighbourhood sampling
        fxaaInvSizeData[0] = 1 / width;
        fxaaInvSizeData[1] = 1 / height;
        device.queue.writeBuffer(fxaaInvSizeBuffer, 0, fxaaInvSizeData);

        let w = Math.max(1, width >> 1);
        let h = Math.max(1, height >> 1);
        const count = Math.min(MAX_MIPS, Math.floor(Math.log2(Math.min(width, height))));

        for (let i = 0; i < count; i++) {
            const texture = device.createTexture({
                size: { width: w, height: h },
                format: MIP_FORMAT,
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            });
            mips.push({ texture, view: texture.createView(), width: w, height: h });
            w = Math.max(1, w >> 1);
            h = Math.max(1, h >> 1);
        }

        downBindGroups = [];
        for (let i = 1; i < mips.length; i++) {
            downBindGroups.push(
                device.createBindGroup({
                    layout: downsamplePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: mips[i - 1].view },
                        { binding: 1, resource: sampler },
                        { binding: 2, resource: { buffer: getUniform(i) } },
                    ],
                }),
            );
        }

        upBindGroups = [];
        for (let i = mips.length - 2; i >= 0; i--) {
            const uIdx = mips.length + (mips.length - 2 - i);
            upBindGroups.push(
                device.createBindGroup({
                    layout: upsamplePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: mips[i + 1].view },
                        { binding: 1, resource: sampler },
                        { binding: 2, resource: { buffer: getUniform(uIdx) } },
                    ],
                }),
            );
        }

        cachedWidth = width;
        cachedHeight = height;
        cachedInputView = null;
    }

    return {
        resize(width: number, height: number): void {
            ensureMips(width, height);
        },

        encode(
            encoder: GPUCommandEncoder,
            inputView: GPUTextureView,
            outputView: GPUTextureView,
        ): void {
            ensureMips(cachedWidth, cachedHeight);
            if (mips.length < 2) return;

            for (let i = 1; i < mips.length; i++) {
                uploadParams(i, 1 / mips[i - 1].width, 1 / mips[i - 1].height, 0, 0);
            }
            const mipWeight = 1 / mips.length;
            for (let i = mips.length - 2; i >= 0; i--) {
                const uIdx = mips.length + (mips.length - 2 - i);
                uploadParams(
                    uIdx,
                    1 / mips[i + 1].width,
                    1 / mips[i + 1].height,
                    BLOOM_RADIUS,
                    mipWeight,
                );
            }

            if (inputView !== cachedInputView) {
                thresholdBindGroup = device.createBindGroup({
                    layout: thresholdPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: inputView },
                        { binding: 1, resource: sampler },
                    ],
                });
                compositeBindGroup = device.createBindGroup({
                    layout: compositePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: inputView },
                        { binding: 1, resource: mips[0].view },
                        { binding: 2, resource: nearestSampler },
                    ],
                });
                cachedInputView = inputView;
            }

            // Build FXAA bind group once per resize (fxaaView changes on resize)
            if (!fxaaBindGroup) {
                fxaaBindGroup = device.createBindGroup({
                    layout: fxaaPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: fxaaView! },
                        { binding: 1, resource: sampler },
                        { binding: 2, resource: { buffer: fxaaInvSizeBuffer } },
                    ],
                });
            }

            fullscreenPass(encoder, thresholdPipeline, thresholdBindGroup!, mips[0].view, "clear");

            for (let i = 0; i < downBindGroups.length; i++) {
                fullscreenPass(
                    encoder,
                    downsamplePipeline,
                    downBindGroups[i],
                    mips[i + 1].view,
                    "clear",
                );
            }

            for (let i = 0; i < upBindGroups.length; i++) {
                const targetIdx = mips.length - 2 - i;
                fullscreenPass(
                    encoder,
                    upsamplePipeline,
                    upBindGroups[i],
                    mips[targetIdx].view,
                    "load",
                );
            }

            // Composite → intermediate fxaaTexture (render resolution)
            fullscreenPass(encoder, compositePipeline, compositeBindGroup!, fxaaView!, "clear");

            // FXAA → final outputView (canvas), smoothing edges before upscale
            fullscreenPass(encoder, fxaaPipeline, fxaaBindGroup, outputView, "clear");
        },

        destroy(): void {
            for (const m of mips) m.texture.destroy();
            mips = [];
            for (const buf of uniformPool) buf.destroy();
            uniformPool.length = 0;
            fxaaTexture?.destroy();
            fxaaTexture = null;
            fxaaView = null;
            fxaaInvSizeBuffer.destroy();
            cachedWidth = 0;
            cachedHeight = 0;
            cachedInputView = null;
        },
    };
}
