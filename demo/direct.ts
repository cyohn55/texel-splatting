import { BVH_WGSL } from "../src/bvh";
import {
    type SceneConfig,
    LIGHTING_WGSL,
    POINT_LIGHT_WGSL,
    STONE_COLOR_WGSL,
    UNIFORMS_WGSL,
} from "../src/lighting";
import { OKLAB_WGSL, PATH_WGSL } from "../src/oklab";
import { SKY_STRUCT_WGSL, HAZE_WGSL } from "../src/sky";

interface DirectConfig extends SceneConfig {
    sceneBindGroupLayout: GPUBindGroupLayout;
    sceneBindGroup: GPUBindGroup;
    skyBuffer: GPUBuffer;
}

export interface DirectEncoder {
    encode(
        encoder: GPUCommandEncoder,
        skyEncode: (
            encoder: GPUCommandEncoder,
            colorView: GPUTextureView,
            depthView: GPUTextureView,
        ) => void,
        colorView: GPUTextureView,
        depthView: GPUTextureView,
        posterizeView: GPUTextureView,
    ): void;
    resize(
        width: number,
        height: number,
        colorView: GPUTextureView,
        posterizeView: GPUTextureView,
    ): void;
    destroy(): void;
}

export function createDirect(config: DirectConfig): DirectEncoder {
    const { device } = config;

    const scenePipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [config.sceneBindGroupLayout],
    });

    const orbLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            },
        ],
    });

    const emissiveShader = device.createShaderModule({
        code: /* wgsl */ `
            ${UNIFORMS_WGSL}
            @group(0) @binding(0) var<uniform> u: Uniforms;

            struct OrbParams {
                position: vec3f,
                scale: f32,
                color: vec3f,
                _pad: f32,
            }
            @group(1) @binding(0) var<uniform> orb: OrbParams;

            struct VsOut {
                @builtin(position) pos: vec4f,
                @location(0) worldPos: vec3f,
                @location(1) color: vec3f,
            }

            @vertex fn vs(@location(0) position: vec3f) -> VsOut {
                let world = position * orb.scale + orb.position;
                var out: VsOut;
                out.pos = u.viewProj * vec4f(world, 1);
                out.worldPos = world;
                out.color = orb.color;
                return out;
            }

            struct FsOut {
                @location(0) color: vec4f,
                @location(1) dist: f32,
            }

            @fragment fn fs(in: VsOut) -> FsOut {
                var out: FsOut;
                out.color = vec4f(min(in.color, vec3f(1)), 1);
                out.dist = length(in.worldPos - u.cameraPos);
                return out;
            }
        `,
    });

    const emissivePipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [config.sceneBindGroupLayout, orbLayout],
        }),
        vertex: {
            module: emissiveShader,
            buffers: [
                {
                    arrayStride: 24,
                    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" as const }],
                },
            ],
        },
        fragment: {
            module: emissiveShader,
            targets: [{ format: "rgba8unorm" as const }, { format: "r32float" as const }],
        },
        primitive: { topology: "triangle-list" as const },
        depthStencil: {
            format: "depth32float" as const,
            depthWriteEnabled: true,
            depthCompare: "less" as const,
        },
    });

    const orbBindGroups = config.orbBuffers.map((buf) =>
        device.createBindGroup({
            layout: orbLayout,
            entries: [{ binding: 0, resource: { buffer: buf } }],
        }),
    );

    const grassShader = device.createShaderModule({
        code: /* wgsl */ `
            ${UNIFORMS_WGSL}
            @group(0) @binding(0) var<uniform> u: Uniforms;
            @group(0) @binding(1) var<storage, read> bvhNodes: array<BVHNode>;
            @group(0) @binding(2) var<storage, read> bvhTris: array<BVHTri>;
            @group(0) @binding(3) var<storage, read> bvhTriIds: array<u32>;

            ${POINT_LIGHT_WGSL}
            @group(0) @binding(4) var<uniform> pointLights: array<PointLight, 64>;

            ${SKY_STRUCT_WGSL}
            @group(0) @binding(5) var<uniform> sky: Sky;

            ${BVH_WGSL}

            struct VsOut {
                @builtin(position) pos: vec4f,
                @location(0) worldPos: vec3f,
                @location(1) localY: f32,
            }

            @vertex fn vs(@location(0) position: vec3f) -> VsOut {
                let world = vec3(
                    position.x * ${config.area}.0,
                    position.y * ${config.height},
                    position.z * ${config.area}.0,
                );
                var out: VsOut;
                out.pos = u.viewProj * vec4f(world, 1);
                out.worldPos = world;
                out.localY = position.y;
                return out;
            }

            ${OKLAB_WGSL}
            ${PATH_WGSL}
            ${LIGHTING_WGSL}

            struct FsOut {
                @location(0) color: vec4f,
                @location(1) dist: f32,
            }

            // Mip-mapped hash: blends between adjacent LOD levels based on how many
            // density cells fit within a single pixel (computed via screen-space
            // derivatives). When cells are sub-pixel, a coarser level is used so
            // each pixel maps to one stable cell, eliminating temporal shimmer.
            fn lodHash2(wp: vec2f, density: f32) -> f32 {
                let densityWp = wp * density;
                let fw = max(fwidth(densityWp.x), fwidth(densityWp.y));
                let lod = max(log2(fw), 0.0);
                let lod0 = floor(lod);
                let h0 = hash2(floor(densityWp / exp2(lod0)));
                let h1 = hash2(floor(densityWp / exp2(lod0 + 1.0)));
                return mix(h0, h1, fract(lod));
            }

            @fragment fn fs(in: VsOut) -> FsOut {
                let t = clamp(in.worldPos.y / ${config.height}, 0.0, 1.0);
                let wp = in.worldPos.xz;

                // Broad-patch height scale: ~5-unit zones of taller/shorter grass
                let patchHeightScale = mix(0.3, 1.0, hash2(floor(wp * 0.2)));
                let h = lodHash2(wp, ${config.density}.0) * patchHeightScale;
                if (h < t) { discard; }
                if (t > 0.0 && pathGrassDiscard(wp)) { discard; }

                // 5-shade green palette (OKLab deltas from base): ~2.5-unit colour patches
                let shadeIdx = min(i32(hash2(floor(wp * 0.4)) * 5.0), 4);
                let base = vec3f(${config.baseR}, ${config.baseG}, ${config.baseB});
                let oklab = toOKLab(base);

                var shadeDL: f32; var shadeDA: f32; var shadeDB: f32;
                switch shadeIdx {
                    case 0: { shadeDL = -0.24; shadeDA = -0.03; shadeDB = -0.07; } // dark forest green (near-black, cool)
                    case 1: { shadeDL = -0.10; shadeDA = -0.02; shadeDB = -0.02; } // rich mid-green
                    case 2: { shadeDL =  0.00; shadeDA =  0.00; shadeDB =  0.00; } // base green
                    case 3: { shadeDL =  0.10; shadeDA =  0.07; shadeDB =  0.13; } // warm yellow-green
                    default: { shadeDL =  0.22; shadeDA =  0.04; shadeDB =  0.18; } // bright lime / dry straw-green
                }

                let lBase = oklab.x + shadeDL;
                let l = mix(lBase * ${config.rootL}, lBase * ${config.tipL}, t);
                let a = oklab.y + shadeDA;
                let b = mix(oklab.z + shadeDB - 0.01, oklab.z + shadeDB + 0.01, t);
                var color = fromOKLab(vec3(l, a, b));
                if (t == 0.0) { color = pathGroundColor(wp, color); }

                let normal = vec3f(0.0, 1.0, 0.0);
                let lit = computeLighting(in.worldPos, normal, color, u.sunDir, u.shadowFade, u.sunColor, u.ambient, 0.0, 1u);
                var out: FsOut;
                out.color = vec4f(lit, 1);
                out.dist = length(in.worldPos - u.cameraPos);
                return out;
            }
        `,
    });

    const stoneShader = device.createShaderModule({
        code: /* wgsl */ `
            ${UNIFORMS_WGSL}
            @group(0) @binding(0) var<uniform> u: Uniforms;
            @group(0) @binding(1) var<storage, read> bvhNodes: array<BVHNode>;
            @group(0) @binding(2) var<storage, read> bvhTris: array<BVHTri>;
            @group(0) @binding(3) var<storage, read> bvhTriIds: array<u32>;

            ${POINT_LIGHT_WGSL}
            @group(0) @binding(4) var<uniform> pointLights: array<PointLight, 64>;

            ${SKY_STRUCT_WGSL}
            @group(0) @binding(5) var<uniform> sky: Sky;

            ${BVH_WGSL}
            ${OKLAB_WGSL}
            ${STONE_COLOR_WGSL}
            ${LIGHTING_WGSL}

            struct VsOut {
                @builtin(position) pos: vec4f,
                @location(0) normal: vec3f,
                @location(1) worldPos: vec3f,
            }

            @vertex fn vs(@location(0) position: vec3f, @location(1) normal: vec3f) -> VsOut {
                var out: VsOut;
                out.pos = u.viewProj * vec4f(position, 1);
                out.normal = normal;
                out.worldPos = position;
                return out;
            }

            struct FsOut {
                @location(0) color: vec4f,
                @location(1) dist: f32,
            }

            @fragment fn fs(in: VsOut) -> FsOut {
                let color = stoneColor(in.worldPos);
                let n = normalize(in.normal);
                let lit = computeLighting(in.worldPos, n, color, u.sunDir, u.shadowFade, u.sunColor, u.ambient, 0.0, 1u);
                var out: FsOut;
                out.color = vec4f(lit, 1);
                out.dist = length(in.worldPos - u.cameraPos);
                return out;
            }
        `,
    });

    const grassPipeline = device.createRenderPipeline({
        layout: scenePipelineLayout,
        vertex: {
            module: grassShader,
            buffers: [
                {
                    arrayStride: 12,
                    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" as const }],
                },
            ],
        },
        fragment: {
            module: grassShader,
            targets: [{ format: "rgba8unorm" as const }, { format: "r32float" as const }],
        },
        primitive: { topology: "triangle-list" as const },
        depthStencil: {
            format: "depth32float" as const,
            depthWriteEnabled: true,
            depthCompare: "less" as const,
        },
    });

    const stonePipeline = device.createRenderPipeline({
        layout: scenePipelineLayout,
        vertex: {
            module: stoneShader,
            buffers: [
                {
                    arrayStride: 24,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x3" as const },
                        { shaderLocation: 1, offset: 12, format: "float32x3" as const },
                    ],
                },
            ],
        },
        fragment: {
            module: stoneShader,
            targets: [{ format: "rgba8unorm" as const }, { format: "r32float" as const }],
        },
        primitive: { topology: "triangle-list" as const, cullMode: "back" as const },
        depthStencil: {
            format: "depth32float" as const,
            depthWriteEnabled: true,
            depthCompare: "less" as const,
        },
    });

    const posterizeShader = device.createShaderModule({
        code: /* wgsl */ `
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
            }

            @group(0) @binding(0) var srcTex: texture_2d<f32>;
            @group(0) @binding(1) var srcSamp: sampler;

            ${OKLAB_WGSL}

            @fragment fn fs(in: VsOut) -> @location(0) vec4f {
                let color = textureSample(srcTex, srcSamp, in.uv).rgb;
                return vec4f(posterize(color), 1.0);
            }
        `,
    });

    const posterizePipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: posterizeShader, entryPoint: "vs" },
        fragment: {
            module: posterizeShader,
            entryPoint: "fs",
            targets: [{ format: "rgba8unorm" as const }],
        },
        primitive: { topology: "triangle-list" as const },
    });

    const posterizeSampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });

    const hazeShader = device.createShaderModule({
        code: /* wgsl */ `
            @group(0) @binding(0) var colorTex: texture_2d<f32>;
            @group(0) @binding(1) var distTex: texture_2d<f32>;

            ${SKY_STRUCT_WGSL}
            @group(0) @binding(2) var<uniform> sky: Sky;

            ${HAZE_WGSL}

            @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
                let coord = vec2u(pos.xy);
                let color = textureLoad(colorTex, coord, 0).rgb;
                let dist = textureLoad(distTex, coord, 0).r;
                if (dist < 0.0) { return vec4f(color, 1.0); }
                return vec4f(applyHaze(color, dist), 1.0);
            }

            @vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
                let positions = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
                return vec4f(positions[vi], 0, 1);
            }
        `,
    });

    const hazePipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: hazeShader, entryPoint: "vs" },
        fragment: {
            module: hazeShader,
            entryPoint: "fs",
            targets: [{ format: "rgba8unorm" as const }],
        },
        primitive: { topology: "triangle-list" as const },
    });

    let distanceTexture: GPUTexture | null = null;
    let distanceView: GPUTextureView;
    let posterizeBindGroup: GPUBindGroup;
    let hazeBindGroup: GPUBindGroup;

    function rebuild(
        width: number,
        height: number,
        colorView: GPUTextureView,
        posterizeView: GPUTextureView,
    ) {
        distanceTexture?.destroy();
        distanceTexture = device.createTexture({
            size: [width, height],
            format: "r32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        distanceView = distanceTexture.createView();
        posterizeBindGroup = device.createBindGroup({
            layout: posterizePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: colorView },
                { binding: 1, resource: posterizeSampler },
            ],
        });
        hazeBindGroup = device.createBindGroup({
            layout: hazePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: posterizeView },
                { binding: 1, resource: distanceView },
                { binding: 2, resource: { buffer: config.skyBuffer } },
            ],
        });
    }

    return {
        encode(encoder, skyEncode, colorView, depthView, posterizeView) {
            skyEncode(encoder, colorView, depthView);

            const scenePass = encoder.beginRenderPass({
                colorAttachments: [
                    { view: colorView, loadOp: "load", storeOp: "store" },
                    {
                        view: distanceView,
                        loadOp: "clear",
                        storeOp: "store",
                        clearValue: { r: -1, g: 0, b: 0, a: 0 },
                    },
                ],
                depthStencilAttachment: {
                    view: depthView,
                    depthLoadOp: "load",
                    depthStoreOp: "store",
                },
            });
            scenePass.setPipeline(grassPipeline);
            scenePass.setBindGroup(0, config.sceneBindGroup);
            scenePass.setVertexBuffer(0, config.vertexBuffer);
            scenePass.setIndexBuffer(config.indexBuffer, "uint16");
            scenePass.drawIndexed(config.indexCount);
            scenePass.setPipeline(stonePipeline);
            scenePass.setBindGroup(0, config.sceneBindGroup);
            scenePass.setVertexBuffer(0, config.stoneVertexBuffer);
            scenePass.setIndexBuffer(config.stoneIndexBuffer, "uint16");
            scenePass.drawIndexed(config.stoneIndexCount);
            scenePass.setPipeline(emissivePipeline);
            scenePass.setBindGroup(0, config.sceneBindGroup);
            scenePass.setVertexBuffer(0, config.orbVertexBuffer);
            scenePass.setIndexBuffer(config.orbIndexBuffer, "uint16");
            for (let i = 0; i < 4; i++) {
                scenePass.setBindGroup(1, orbBindGroups[i]);
                scenePass.drawIndexed(config.orbIndexCount);
            }
            scenePass.end();

            const posterizePass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: posterizeView,
                        loadOp: "clear",
                        storeOp: "store",
                        clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    },
                ],
            });
            posterizePass.setPipeline(posterizePipeline);
            posterizePass.setBindGroup(0, posterizeBindGroup);
            posterizePass.draw(3);
            posterizePass.end();

            const hazePass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: colorView,
                        loadOp: "clear",
                        storeOp: "store",
                        clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    },
                ],
            });
            hazePass.setPipeline(hazePipeline);
            hazePass.setBindGroup(0, hazeBindGroup);
            hazePass.draw(3);
            hazePass.end();
        },

        resize(width, height, colorView, posterizeView) {
            rebuild(width, height, colorView, posterizeView);
        },

        destroy() {
            distanceTexture?.destroy();
        },
    };
}
