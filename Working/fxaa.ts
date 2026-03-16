/**
 * Single-pass FXAA (Fast Approximate Anti-Aliasing) post-process.
 *
 * Detects edges by comparing luminance of the 4 diagonal neighbours against
 * the centre pixel, determines the dominant edge orientation (horizontal vs
 * vertical), and blends the centre sample toward the perpendicular neighbour
 * pair. Low-contrast pixels are passed through unchanged.
 *
 * Usage:
 *   const fxaa = createFxaa(device);
 *   fxaa.resize(width, height);
 *   // per frame:
 *   fxaa.encode(encoder, sceneView, outputView);
 *   fxaa.destroy();
 */

export interface Fxaa {
    resize(width: number, height: number): void;
    encode(
        encoder: GPUCommandEncoder,
        inputView: GPUTextureView,
        outputView: GPUTextureView,
        outputFormat: GPUTextureFormat,
    ): void;
    destroy(): void;
}

const FXAA_WGSL = /* wgsl */ `
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

    // Determine dominant edge orientation
    let edgeH = abs(lumaNW + lumaNE - 2.0 * lumaM) + abs(lumaSW + lumaSE - 2.0 * lumaM);
    let edgeV = abs(lumaNW + lumaSW - 2.0 * lumaM) + abs(lumaNE + lumaSE - 2.0 * lumaM);
    let step = select(vec2f(ts.x, 0.0), vec2f(0.0, ts.y), edgeH >= edgeV);

    let c1 = textureSample(srcTex, srcSamp, uv + step).rgb;
    let c2 = textureSample(srcTex, srcSamp, uv - step).rgb;
    return vec4f(mix(m, (c1 + c2) * 0.5, clamp(contrast * 8.0, 0.0, 0.75)), 1.0);
}
`;

export function createFxaa(device: GPUDevice): Fxaa {
    const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    const shaderModule = device.createShaderModule({ code: FXAA_WGSL });

    const invSizeBuffer = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const invSizeData = new Float32Array(2);

    // Pipeline is created lazily on first encode call (needs output format)
    const pipelineCache = new Map<GPUTextureFormat, GPURenderPipeline>();

    function getPipeline(outputFormat: GPUTextureFormat): GPURenderPipeline {
        let pipeline = pipelineCache.get(outputFormat);
        if (!pipeline) {
            pipeline = device.createRenderPipeline({
                layout: "auto",
                vertex: { module: shaderModule, entryPoint: "vs" },
                fragment: {
                    module: shaderModule,
                    entryPoint: "fs",
                    targets: [{ format: outputFormat }],
                },
                primitive: { topology: "triangle-list" },
            });
            pipelineCache.set(outputFormat, pipeline);
        }
        return pipeline;
    }

    let cachedInputView: GPUTextureView | null = null;
    let bindGroup: GPUBindGroup | null = null;
    let cachedPipeline: GPURenderPipeline | null = null;

    return {
        resize(width: number, height: number): void {
            invSizeData[0] = 1 / width;
            invSizeData[1] = 1 / height;
            device.queue.writeBuffer(invSizeBuffer, 0, invSizeData);
            // Force bind group rebuild on next encode (invSize buffer content changed)
            bindGroup = null;
        },

        encode(
            encoder: GPUCommandEncoder,
            inputView: GPUTextureView,
            outputView: GPUTextureView,
            outputFormat: GPUTextureFormat,
        ): void {
            const pipeline = getPipeline(outputFormat);

            if (inputView !== cachedInputView || pipeline !== cachedPipeline || !bindGroup) {
                bindGroup = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: inputView },
                        { binding: 1, resource: sampler },
                        { binding: 2, resource: { buffer: invSizeBuffer } },
                    ],
                });
                cachedInputView = inputView;
                cachedPipeline = pipeline;
            }

            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: outputView,
                        loadOp: "clear",
                        storeOp: "store",
                        clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    },
                ],
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.draw(3);
            pass.end();
        },

        destroy(): void {
            invSizeBuffer.destroy();
            pipelineCache.clear();
        },
    };
}
