#include "rt_reflections.h"

#include "core/engine.h"
#include "core/math/math_funcs.h"
#include "drivers/gles3/rasterizer_storage_gles3.h"
#include "drivers/gles3/shared_gl43_context.h"

#include "modules/modules_enabled.gen.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/multimesh_instance.h"
#include "scene/main/scene_tree.h"

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif
#include "scene/main/viewport.h"
#include "scene/resources/environment.h"
#include "scene/resources/sky.h"
#include "scene/resources/world.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual_server.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>

#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT 0x00000020
#endif
#ifndef GL_TEXTURE_FETCH_BARRIER_BIT
#define GL_TEXTURE_FETCH_BARRIER_BIT 0x00000008
#endif
#ifndef GL_SHADER_STORAGE_BARRIER_BIT
#define GL_SHADER_STORAGE_BARRIER_BIT 0x00002000
#endif
#ifndef GL_WRITE_ONLY
#define GL_WRITE_ONLY 0x88B9
#endif
#ifndef GL_FRAMEBUFFER_SRGB
#define GL_FRAMEBUFFER_SRGB 0x8DB9
#endif
#ifndef GL_RGBA16F
#define GL_RGBA16F 0x881A
#endif
#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif

// =========================================================
// GLSL SHADERS
// =========================================================
static const char *GBUF_VERT_GLSL = R"GLSL(
#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUv;
layout (location = 3) in float aSurfIdx;

struct RTInstance {
    mat4 invModel;
    mat4 model;
    vec4 aabbMin;
    vec4 aabbMax;
    ivec4 flags;
};

layout (std430, binding = 4) readonly buffer InstanceBuf {
    RTInstance instances[];
};

uniform int uInstIdx;
uniform mat4 uViewProj;
uniform int uMatOffset;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 vUv;
flat out int vMatId;

void main() {
    RTInstance inst = instances[uInstIdx];
    vec4 wp = inst.model * vec4(aPos, 1.0);
    vWorldPos = wp.xyz;
    vWorldNormal = normalize(transpose(mat3(inst.invModel)) * aNormal);
    vUv = aUv;
    vMatId = uMatOffset + int(aSurfIdx + 0.5);
    gl_Position = uViewProj * wp;
}
)GLSL";

static const char *GBUF_FRAG_GLSL = R"GLSL(
#version 430 core
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 vUv;
flat in int vMatId;

layout (location = 0) out vec4 outNormalRough;
layout (location = 1) out vec4 outAlbedoMetal;
layout (location = 2) out vec4 outExtra;

uniform sampler2DArray uTexArray;
uniform int uSkipReflection;
uniform int uPass; // 0 = opaque + scissor, 1 = transparent receivers
uniform int uRespectCullMode;
uniform int uFlipFacing;
uniform int uAlphaConsideration;
uniform float uTransparentMinAlpha;

struct RTMaterial {
    vec4 albedoAlpha;
    vec4 emissionMetal;
    vec4 pbrParams;
    vec4 uvScaleOffset;
    ivec4 texLayers;
    ivec4 chanNormal;
    ivec4 extraTex;
};

layout (std430, binding = 5) readonly buffer MaterialBuf {
    RTMaterial materials[];
};

float sampleChan(int layer, int ch, vec2 uv) {
    vec4 t = texture(uTexArray, vec3(uv, float(layer)));
    if (ch == 0) return t.r;
    if (ch == 1) return t.g;
    if (ch == 2) return t.b;
    if (ch == 3) return t.a;
    return dot(t.rgb, vec3(0.299, 0.587, 0.114));
}

void main() {
    RTMaterial m = materials[vMatId];
    int cullMode = m.extraTex.w & 3;
    bool frontFace = (uFlipFacing != 0) ? !gl_FrontFacing : gl_FrontFacing;
    if (uRespectCullMode != 0) {
        if (cullMode == 0 && !frontFace) discard;
        if (cullMode == 1 && frontFace) discard;
    }

    int alphaMode = (uAlphaConsideration != 0) ? m.extraTex.z : 0;
    if (uPass == 0 && alphaMode == 2) discard;
    if (uPass == 1 && (alphaMode != 2 || uSkipReflection != 0)) discard;

    vec2 uv = vUv * m.uvScaleOffset.xy + m.uvScaleOffset.zw;
    vec4 alb4 = m.albedoAlpha;
    if (m.texLayers.x >= 0) alb4 *= texture(uTexArray, vec3(uv, float(m.texLayers.x)));

    if (uAlphaConsideration != 0) {
        if (alphaMode == 1 && alb4.a < m.pbrParams.y) discard;
        if (alphaMode == 2 && alb4.a < uTransparentMinAlpha) discard;
    }

    bool isUnshaded = (m.extraTex.w & 8) != 0;
    if (uSkipReflection != 0 || isUnshaded) {
        outNormalRough = vec4(0.0);
        outAlbedoMetal = vec4(0.0);
        outExtra = vec4(0.0);
        return;
    }

    float rough = m.pbrParams.x;
    if (m.texLayers.y >= 0) rough *= sampleChan(m.texLayers.y, m.chanNormal.x, uv);

    float metal = m.emissionMetal.w;
    if (m.texLayers.z >= 0) metal *= sampleChan(m.texLayers.z, m.chanNormal.y, uv);

    vec3 N = normalize(vWorldNormal);
    if (cullMode == 2 && !frontFace) N = -N;

    if (m.chanNormal.z >= 0) {
        vec3 nm = texture(uTexArray, vec3(uv, float(m.chanNormal.z))).xyz * 2.0 - 1.0;
        nm.xy *= intBitsToFloat(m.chanNormal.w);
        nm.z = sqrt(max(1.0 - dot(nm.xy, nm.xy), 0.0));
        vec3 dp1 = dFdx(vWorldPos);
        vec3 dp2 = dFdy(vWorldPos);
        vec2 duv1 = dFdx(uv);
        vec2 duv2 = dFdy(uv);
        vec3 dp2perp = cross(dp2, N);
        vec3 dp1perp = cross(N, dp1);
        vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
        vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
        float maxTB = max(dot(T, T), dot(B, B));
        if (maxTB > 1e-10) {
            float invmax = inversesqrt(maxTB);
            N = normalize(mat3(T * invmax, B * invmax, N) * nm);
        }
    }

    outNormalRough = vec4(N, rough);
    outAlbedoMetal = vec4(alb4.rgb, metal);
    outExtra = vec4(clamp(m.pbrParams.z, 0.0, 1.0), (alphaMode == 2) ? 1.0 : 0.0, 0.0, 1.0);
}
)GLSL";

// Computes true L2 Spherical Harmonics (9 coeffs) using 1 KB of shared memory
static const char *SH_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (std430, binding = 7) writeonly buffer SkySHBuf {
    vec4 skySH[9];
};

uniform sampler2D uPanoramaTex;
uniform int uUsePanorama;
uniform int uPanoramaToLinear;
uniform float uSkyEnergy;
uniform vec3 uSkyTop;
uniform vec3 uSkyHorizon;
uniform vec3 uSkyGroundHorizon;
uniform vec3 uSkyGround;
uniform float uSkyCurve;
uniform float uGroundCurve;
uniform vec3 uSkySunDir;
uniform vec3 uSkySunColor;
uniform float uSkySunAngleMin;
uniform float uSkySunAngleMax;
uniform float uSkySunCurve;
uniform float uSkySunEnergy;

shared vec4 shRed[64];

const float PI = 3.14159265359;

vec3 sampleLocalSky(vec3 d) {
    if (uUsePanorama == 1) {
        float u = atan(d.x, -d.z) / (2.0 * PI) + 0.5;
        float v = acos(clamp(d.y, -1.0, 1.0)) / PI;
        ivec2 psz = textureSize(uPanoramaTex, 0);
        float maxLod = max(float(textureQueryLevels(uPanoramaTex)) - 1.0, 0.0);
        float lod = clamp(log2(float(max(psz.x, 1))) - 5.0, 0.0, maxLod);
        vec3 c = textureLod(uPanoramaTex, vec2(u, clamp(v, 0.001, 0.999)), lod).rgb;
        if (uPanoramaToLinear == 1) {
            c = pow(max(c, vec3(0.0)), vec3(2.2));
        }
        return c * uSkyEnergy;
    }

    float h = clamp(d.y, -1.0, 1.0);
    vec3 col;
    if (h >= 0.0) {
        float c = clamp(1.0 - acos(h) / (PI * 0.5), 0.0, 1.0);
        col = mix(uSkyHorizon, uSkyTop, pow(max(c, 1e-4), max(uSkyCurve, 0.005)));
    } else {
        float c = clamp(1.0 - acos(-h) / (PI * 0.5), 0.0, 1.0);
        col = mix(uSkyGroundHorizon, uSkyGround, pow(max(c, 1e-4), max(uGroundCurve, 0.005)));
    }

    if (uSkySunEnergy > 0.001) {
        float angDeg = degrees(acos(clamp(dot(d, normalize(uSkySunDir)), -1.0, 1.0)));
        if (angDeg < uSkySunAngleMax) {
            float c2 = clamp((angDeg - uSkySunAngleMin) / max(uSkySunAngleMax - uSkySunAngleMin, 0.001), 0.0, 1.0);
            float sunAmt = (angDeg <= uSkySunAngleMin) ? 1.0 : pow(1.0 - c2, 1.0 / max(uSkySunCurve, 0.001));
            col = mix(col, max(col, uSkySunColor * uSkySunEnergy), sunAmt);
        }
    }
    return col;
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    vec3 acc0 = vec3(0.0), acc1 = vec3(0.0), acc2 = vec3(0.0);
    vec3 acc3 = vec3(0.0), acc4 = vec3(0.0), acc5 = vec3(0.0);
    vec3 acc6 = vec3(0.0), acc7 = vec3(0.0), acc8 = vec3(0.0);

    for (int s = 0; s < 4; s++) {
        uint idx = tid * 4u + uint(s);
        float y = 1.0 - (2.0 * (float(idx) + 0.5) / 256.0);
        float r = sqrt(max(0.0, 1.0 - y * y));
        float phi = float(idx) * 2.399963229728653;
        vec3 dir = vec3(cos(phi) * r, y, sin(phi) * r);
        vec3 L = sampleLocalSky(dir);

        acc0 += L * 0.28209479177;
        acc1 += L * (0.48860251190 * dir.y);
        acc2 += L * (0.48860251190 * dir.z);
        acc3 += L * (0.48860251190 * dir.x);
        acc4 += L * (1.09254843059 * dir.x * dir.y);
        acc5 += L * (1.09254843059 * dir.y * dir.z);
        acc6 += L * (0.31539156525 * (3.0 * dir.z * dir.z - 1.0));
        acc7 += L * (1.09254843059 * dir.x * dir.z);
        acc8 += L * (0.54627421530 * (dir.x * dir.x - dir.y * dir.y));
    }

    const float w = (4.0 * PI) / 256.0;
    const float factors[9] = float[9](
        w * 1.0000000 * 0.28209479177,
        w * 0.6666667 * 0.48860251190,
        w * 0.6666667 * 0.48860251190,
        w * 0.6666667 * 0.48860251190,
        w * 0.2500000 * 1.09254843059,
        w * 0.2500000 * 1.09254843059,
        w * 0.2500000 * 0.31539156525,
        w * 0.2500000 * 1.09254843059,
        w * 0.2500000 * 0.54627421530
    );

    vec3 coeffs[9] = vec3[9](acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    for (int k = 0; k < 9; k++) {
        shRed[tid] = vec4(coeffs[k], 0.0);
        barrier();
        for (uint stride = 32u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                shRed[tid] += shRed[tid + stride];
            }
            barrier();
        }
        if (tid == 0u) {
            skySH[k] = vec4(shRed[0].rgb * factors[k], 0.0);
        }
        barrier();
    }
}
)GLSL";

static const char *RT_COMPUTE_HEADER_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutReflection;
layout (rgba16f, binding = 1) writeonly uniform image2D uOutCompositeMeta;

uniform sampler2D uGNormalRough;
uniform sampler2D uGAlbedoMetal;
uniform sampler2D uGExtra;
uniform sampler2D uGDepth;
uniform sampler2DArray uTexArray;
uniform sampler2D uPanoramaTex;

uniform mat4 uInvViewProj;
uniform vec3 uCamPos;
uniform int uUsePanorama;
uniform int uPanoramaToLinear;
uniform float uSkyEnergy;
uniform float uSkyReflectionEnergy;
uniform mat3 uSkyInvBasis;
uniform vec3 uSkyTop;
uniform vec3 uSkyHorizon;
uniform vec3 uSkyGroundHorizon;
uniform vec3 uSkyGround;
uniform float uSkyCurve;
uniform float uGroundCurve;
uniform float uSkyContrib;
uniform vec3 uAmbientColor;
uniform float uAmbientEnergy;
uniform float uAmbientRadianceBoost;
uniform float uAmbientBounceApprox;
uniform int uAmbientAOSamples;
uniform float uAmbientAORadius;
uniform float uAmbientAOIntensity;
uniform vec3 uSkySunDir;
uniform vec3 uSkySunColor;
uniform float uSkySunAngleMin;
uniform float uSkySunAngleMax;
uniform float uSkySunCurve;
uniform float uSkySunEnergy;
uniform int uHasTLAS;
uniform int uNumLights;
uniform int uNumDirLights;
uniform int uHasAlphaMaterials;
uniform int uMaterialCount;
uniform int uMaxBounces;
uniform float uMaxRoughness;
uniform float uRoughnessFadeRange;
uniform float uLodBias;
uniform float uDistLodStrength;
uniform float uShadowMaxDist;
uniform int uShadowBounces;
uniform int uPbrTexBounces;
uniform int uBounceLocalLights;
uniform int uNormalMapsInBounces;
uniform int uRespectCullMode;
uniform int uEnvFallback;
uniform float uMaxRayDistance;
uniform float uBounceDistFalloff;
uniform float uBounceThroughputCutoff;
uniform float uRayBias;
uniform float uShadowBias;
uniform int uSamplesPerPixel;
uniform float uGlossySpread;
uniform int uAllowStochastic;
uniform int uTemporalAccum;
uniform int uCheckerboard;
uniform int uFrameIdx;
uniform int uAlphaConsideration;
uniform int uAlphaRayTransparency;
uniform int uAlphaMaxSteps;
uniform int uAlphaShadowTransmission;
)GLSL";

static const char *RT_COMPUTE_BODY_GLSL = R"GLSL(
struct RTMaterial {
    vec4 albedoAlpha;
    vec4 emissionMetal;
    vec4 pbrParams;
    vec4 uvScaleOffset;
    ivec4 texLayers;
    ivec4 chanNormal;
    ivec4 extraTex;
};

struct RTLight {
    vec4 posRange;
    vec4 dirType;
    vec4 colorAtten;
    vec4 spotParams;
};

layout (std430, binding = 5) readonly buffer MaterialBuf { RTMaterial materials[]; };
layout (std430, binding = 6) readonly buffer LightBuf    { RTLight lights[]; };
layout (std430, binding = 7) readonly buffer SkySHBuf    { vec4 skySH[9]; };

const float PI = 3.14159265359;

float ignNoise(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

vec2 r2Sample(int idx, vec2 jitter) {
    return fract(jitter + float(idx) * vec2(0.7548776662, 0.5698402910));
}

vec3 sampleGGX_VNDF(vec2 xi, vec3 V, vec3 N, float alpha) {
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);
    vec3 Vlocal = vec3(dot(V, T), dot(V, B), max(dot(V, N), 0.001));
    vec3 Vh = normalize(vec3(alpha * Vlocal.x, alpha * Vlocal.y, Vlocal.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 1e-7 ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);

    float r = sqrt(clamp(xi.x, 0.0, 0.92));
    float phi = 2.0 * PI * xi.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;

    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    vec3 Ne = normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0001, Nh.z)));
    return normalize(T * Ne.x + B * Ne.y + N * Ne.z);
}

vec3 sampleCosineHemisphere(vec2 xi, vec3 N) {
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt(max(0.0, 1.0 - xi.y));
    float sinTheta = sqrt(max(0.0, xi.y));
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);
    return normalize(T * (cos(phi) * sinTheta) + B * (sin(phi) * sinTheta) + N * cosTheta);
}

// cullMode: 0 = double-sided, 1 = reflection/AO ray culling, 2 = shadow ray culling (toward light)
bool intersectTriangle(int triIdx, vec3 ro, vec3 rd, float maxT, int matOffset, int cullMode, out float outT, out float outU, out float outV) {
    float det;
    if (!rayTriangleGeomIdx(triIdx, ro, rd, maxT, det, outT, outU, outV)) return false;

    if (cullMode != 0) {
        int mId = clamp(matOffset + floatBitsToInt(triangles[triIdx].p0.w), 0, uMaterialCount - 1);
        int cm = materials[mId].extraTex.w & 3;
        if (cullMode == 1) {
            if (cm == 0 && det >= 0.0) return false;
            if (cm == 1 && det <= 0.0) return false;
        } else {
            if (cm == 0 && det <= 0.0) return false;
            if (cm == 1 && det >= 0.0) return false;
        }
    }
    return true;
}

struct HitInfo {
    bool hit;
    float t;
    vec3 pos;
    vec3 normal;
    vec2 uv;
    int matId;
    int instIdx;
    int triIdx;
};

void traverseBLAS(int rootNode, int matOffset, bool applyCull, vec3 ro, vec3 rd, vec3 invRd, inout float bestT, out int bestTri, out float bestU, out float bestV) {
    bestTri = -1;
    bestU = 0.0;
    bestV = 0.0;

    if (intersectAABB(blasNodes[rootNode].aabbMin.xyz, blasNodes[rootNode].aabbMax.xyz, ro, invRd, bestT) >= bestT)
        return;

    int stackId[18];
    float stackT[18];
    int sp = 0;
    int nodeIdx = rootNode;
    int cullMode = applyCull ? 1 : 0;

    while (true) {
        BVHNode node = blasNodes[nodeIdx];
        int leftOrFirst = floatBitsToInt(node.aabbMin.w);
        int count = floatBitsToInt(node.aabbMax.w);

        if (count > 0) {
            for (int i = 0; i < count; i++) {
                float t, u, v;
                int triIdx = leftOrFirst + i;
                if (intersectTriangle(triIdx, ro, rd, bestT, matOffset, cullMode, t, u, v)) {
                    bestT = t;
                    bestTri = triIdx;
                    bestU = u;
                    bestV = v;
                }
            }
        } else {
            int left = nodeIdx + 1;
            int right = leftOrFirst;
            float tL = intersectAABB(blasNodes[left].aabbMin.xyz, blasNodes[left].aabbMax.xyz, ro, invRd, bestT);
            float tR = intersectAABB(blasNodes[right].aabbMin.xyz, blasNodes[right].aabbMax.xyz, ro, invRd, bestT);
            if (tL < bestT && tR < bestT) {
                if (tL < tR) {
                    if (sp < 18) { stackId[sp] = right; stackT[sp] = tR; sp++; }
                    nodeIdx = left;
                    continue;
                } else {
                    if (sp < 18) { stackId[sp] = left; stackT[sp] = tL; sp++; }
                    nodeIdx = right;
                    continue;
                }
            } else if (tL < bestT) {
                nodeIdx = left;
                continue;
            } else if (tR < bestT) {
                nodeIdx = right;
                continue;
            }
        }

        bool popped = false;
        while (sp > 0) {
            sp--;
            if (stackT[sp] < bestT) {
                nodeIdx = stackId[sp];
                popped = true;
                break;
            }
        }
        if (!popped) break;
    }
}

bool traverseBLASShadow(int rootNode, int matOffset, int shadowCullMode, vec3 ro, vec3 rd, vec3 invRd, float maxT, inout float vis) {
    if (intersectAABB(blasNodes[rootNode].aabbMin.xyz, blasNodes[rootNode].aabbMax.xyz, ro, invRd, maxT) >= maxT)
        return true;

    int stackId[18];
    int sp = 0;
    int nodeIdx = rootNode;

    bool checkAlpha = (uHasAlphaMaterials != 0 && uAlphaConsideration != 0 && uAlphaShadowTransmission != 0);

    while (true) {
        BVHNode node = blasNodes[nodeIdx];
        int leftOrFirst = floatBitsToInt(node.aabbMin.w);
        int count = floatBitsToInt(node.aabbMax.w);

        if (count > 0) {
            for (int i = 0; i < count; i++) {
                float t, u, v;
                int triIdx = leftOrFirst + i;
                if (intersectTriangle(triIdx, ro, rd, maxT, matOffset, shadowCullMode, t, u, v)) {
                    if (!checkAlpha) {
                        vis = 0.0;
                        return false;
                    }
                    int mId = clamp(matOffset + floatBitsToInt(triangles[triIdx].p0.w), 0, uMaterialCount - 1);
                    RTMaterial hm = materials[mId];
                    int alphaMode = hm.extraTex.z;
                    if (alphaMode == 0) {
                        vis = 0.0;
                        return false;
                    }
                    RTTriangle tri = triangles[triIdx];
                    float w = 1.0 - u - v;
                    vec2 uv = (w * tri.uv01.xy + u * tri.uv01.zw + v * tri.uv2pad.xy) * hm.uvScaleOffset.xy + hm.uvScaleOffset.zw;
                    float texA = (hm.texLayers.x >= 0) ? textureLod(uTexArray, vec3(uv, float(hm.texLayers.x)), 2.0).a : 1.0;
                    float a = clamp(hm.albedoAlpha.a * texA, 0.0, 1.0);
                    if (alphaMode == 1) {
                        if (a >= hm.pbrParams.y) {
                            vis = 0.0;
                            return false;
                        }
                    } else {
                        vis *= (1.0 - a);
                        if (vis < 0.02) {
                            vis = 0.0;
                            return false;
                        }
                    }
                }
            }
        } else {
            int left = nodeIdx + 1;
            int right = leftOrFirst;
            bool hitL = intersectAABB(blasNodes[left].aabbMin.xyz, blasNodes[left].aabbMax.xyz, ro, invRd, maxT) < maxT;
            bool hitR = intersectAABB(blasNodes[right].aabbMin.xyz, blasNodes[right].aabbMax.xyz, ro, invRd, maxT) < maxT;
            if (hitL && hitR) {
                if (sp < 18) stackId[sp++] = right;
                nodeIdx = left;
                continue;
            } else if (hitL) {
                nodeIdx = left;
                continue;
            } else if (hitR) {
                nodeIdx = right;
                continue;
            }
        }

        if (sp == 0) break;
        nodeIdx = stackId[--sp];
    }
    return true;
}

HitInfo traceScene(vec3 ro, vec3 rd, float maxT, bool applyCull) {
    HitInfo res;
    res.hit = false;
    res.t = maxT;
    res.matId = 0;
    res.instIdx = -1;
    res.triIdx = -1;
    res.uv = vec2(0.0);

    if (uHasTLAS == 0) return res;

    vec3 invRd = safeInvDir(rd);
    if (intersectAABB(tlasNodes[0].aabbMin.xyz, tlasNodes[0].aabbMax.xyz, ro, invRd, res.t) >= res.t) return res;

    int stackId[18];
    float stackT[18];
    int sp = 0;
    int nodeIdx = 0;

    int winInst = -1;
    int winTri = -1;
    float winU = 0.0, winV = 0.0;

    while (true) {
        BVHNode node = tlasNodes[nodeIdx];
        int leftOrFirst = floatBitsToInt(node.aabbMin.w);
        int count = floatBitsToInt(node.aabbMax.w);

        if (count > 0) {
            for (int i = 0; i < count; i++) {
                int instIdx = tlasIndices[leftOrFirst + i];
                RTInstance inst = instances[instIdx];
                if ((inst.flags.x & 1) == 0)
                    continue;
                if (intersectAABB(inst.aabbMin.xyz, inst.aabbMax.xyz, ro, invRd, res.t) >= res.t)
                    continue;

                vec3 lro = (inst.invModel * vec4(ro, 1.0)).xyz;
                vec3 lrd = (inst.invModel * vec4(rd, 0.0)).xyz;
                vec3 invLrd = safeInvDir(lrd);
                int blasRoot = floatBitsToInt(inst.aabbMin.w);
                int matOffset = floatBitsToInt(inst.aabbMax.w);

                int tri; float u, v;
                float prevT = res.t;
                traverseBLAS(blasRoot, matOffset, applyCull, lro, lrd, invLrd, res.t, tri, u, v);
                if (tri >= 0 && res.t < prevT) {
                    winInst = instIdx;
                    winTri = tri;
                    winU = u;
                    winV = v;
                }
            }
        } else {
            int left = nodeIdx + 1;
            int right = leftOrFirst;
            float tL = intersectAABB(tlasNodes[left].aabbMin.xyz, tlasNodes[left].aabbMax.xyz, ro, invRd, res.t);
            float tR = intersectAABB(tlasNodes[right].aabbMin.xyz, tlasNodes[right].aabbMax.xyz, ro, invRd, res.t);
            if (tL < res.t && tR < res.t) {
                if (tL < tR) {
                    if (sp < 18) { stackId[sp] = right; stackT[sp] = tR; sp++; }
                    nodeIdx = left;
                    continue;
                } else {
                    if (sp < 18) { stackId[sp] = left; stackT[sp] = tL; sp++; }
                    nodeIdx = right;
                    continue;
                }
            } else if (tL < res.t) {
                nodeIdx = left;
                continue;
            } else if (tR < res.t) {
                nodeIdx = right;
                continue;
            }
        }

        bool popped = false;
        while (sp > 0) {
            sp--;
            if (stackT[sp] < res.t) {
                nodeIdx = stackId[sp];
                popped = true;
                break;
            }
        }
        if (!popped) break;
    }

    if (winTri >= 0) {
        res.hit = true;
        res.pos = ro + res.t * rd;
        res.instIdx = winInst;
        res.triIdx = winTri;
        RTTriangle tri = triangles[winTri];
        float w = 1.0 - winU - winV;
        vec3 localN = normalize(w * tri.n0.xyz + winU * tri.n1.xyz + winV * tri.n2.xyz);
        RTInstance inst = instances[winInst];
        vec3 worldN = normalize(transpose(mat3(inst.invModel)) * localN);
        if (dot(worldN, rd) > 0.0) worldN = -worldN;
        res.normal = worldN;
        res.uv = w * tri.uv01.xy + winU * tri.uv01.zw + winV * tri.uv2pad.xy;
        res.matId = clamp(floatBitsToInt(inst.aabbMax.w) + floatBitsToInt(tri.p0.w), 0, uMaterialCount - 1);
    }
    return res;
}

float traceShadow(vec3 ro, vec3 lightDir, float maxDist, bool isAO) {
    if (uHasTLAS == 0) return 1.0;
    vec3 invRd = safeInvDir(lightDir);
    if (intersectAABB(tlasNodes[0].aabbMin.xyz, tlasNodes[0].aabbMax.xyz, ro, invRd, maxDist) >= maxDist)
        return 1.0;

    int stackId[18];
    int sp = 0;
    int nodeIdx = 0;
    float vis = 1.0;

    while (true) {
        BVHNode node = tlasNodes[nodeIdx];
        int leftOrFirst = floatBitsToInt(node.aabbMin.w);
        int count = floatBitsToInt(node.aabbMax.w);

        if (count > 0) {
            for (int i = 0; i < count; i++) {
                RTInstance inst = instances[tlasIndices[leftOrFirst + i]];
                if ((inst.flags.x & 2) == 0)
                    continue;
                if (intersectAABB(inst.aabbMin.xyz, inst.aabbMax.xyz, ro, invRd, maxDist) >= maxDist)
                    continue;
                int shadowCull = (uRespectCullMode != 0 && (inst.flags.x & 4) == 0) ? (isAO ? 1 : 2) : 0;
                vec3 lro = (inst.invModel * vec4(ro, 1.0)).xyz;
                vec3 lrd = (inst.invModel * vec4(lightDir, 0.0)).xyz;
                if (!traverseBLASShadow(floatBitsToInt(inst.aabbMin.w), floatBitsToInt(inst.aabbMax.w), shadowCull, lro, lrd, safeInvDir(lrd), maxDist, vis))
                    return 0.0;
            }
        } else {
            int left = nodeIdx + 1;
            int right = leftOrFirst;
            bool hitL = intersectAABB(tlasNodes[left].aabbMin.xyz, tlasNodes[left].aabbMax.xyz, ro, invRd, maxDist) < maxDist;
            bool hitR = intersectAABB(tlasNodes[right].aabbMin.xyz, tlasNodes[right].aabbMax.xyz, ro, invRd, maxDist) < maxDist;
            if (hitL && hitR) {
                if (sp < 18) stackId[sp++] = right;
                nodeIdx = left;
                continue;
            } else if (hitL) {
                nodeIdx = left;
                continue;
            } else if (hitR) {
                nodeIdx = right;
                continue;
            }
        }

        if (sp == 0) break;
        nodeIdx = stackId[--sp];
    }
    return vis;
}

vec3 evalSkySH(vec3 worldN) {
    vec3 n = normalize(uSkyInvBasis * worldN);
    vec3 irr = skySH[0].rgb
             + skySH[1].rgb * n.y
             + skySH[2].rgb * n.z
             + skySH[3].rgb * n.x
             + skySH[4].rgb * (n.x * n.y)
             + skySH[5].rgb * (n.y * n.z)
             + skySH[6].rgb * (3.0 * n.z * n.z - 1.0)
             + skySH[7].rgb * (n.x * n.z)
             + skySH[8].rgb * (n.x * n.x - n.y * n.y);
    return max(irr, vec3(0.0));
}

vec2 dirToPanoramaUV(vec3 dir) {
    vec3 d = normalize(uSkyInvBasis * dir);
    float u = atan(d.x, -d.z) / (2.0 * PI) + 0.5;
    float v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2(u, clamp(v, 0.001, 0.999));
}

vec3 samplePanorama(vec3 dir, float lod) {
    vec3 c = textureLod(uPanoramaTex, dirToPanoramaUV(dir), lod).rgb;
    if (uPanoramaToLinear == 1) {
        c = pow(max(c, vec3(0.0)), vec3(2.2));
    }
    return c * uSkyEnergy;
}

vec3 sampleProceduralSky(vec3 dir, float rough) {
    vec3 d = normalize(uSkyInvBasis * dir);
    float h = clamp(d.y, -1.0, 1.0);
    vec3 col;
    if (h >= 0.0) {
        float c = clamp(1.0 - acos(h) / (PI * 0.5), 0.0, 1.0);
        col = mix(uSkyHorizon, uSkyTop, pow(max(c, 1e-4), max(uSkyCurve, 0.005)));
    } else {
        float c = clamp(1.0 - acos(-h) / (PI * 0.5), 0.0, 1.0);
        col = mix(uSkyGroundHorizon, uSkyGround, pow(max(c, 1e-4), max(uGroundCurve, 0.005)));
    }

    if (uSkySunEnergy > 0.001) {
        float angDeg = degrees(acos(clamp(dot(d, normalize(uSkySunDir)), -1.0, 1.0)));
        float maxAng = uSkySunAngleMax + rough * 35.0;
        if (angDeg < maxAng) {
            float c2 = clamp((angDeg - uSkySunAngleMin) / max(maxAng - uSkySunAngleMin, 0.001), 0.0, 1.0);
            float sunAmt = (angDeg <= uSkySunAngleMin) ? 1.0 : pow(1.0 - c2, 1.0 / max(uSkySunCurve, 0.001));
            sunAmt *= (1.0 - rough * 0.75);
            col = mix(col, max(col, uSkySunColor * uSkySunEnergy), sunAmt);
        }
    }
    return col;
}

vec3 sampleSkyRough(vec3 dir, float rough) {
    vec3 col;
    if (uUsePanorama == 1) {
        float maxLod = max(float(textureQueryLevels(uPanoramaTex)) - 1.0, 0.0);
        float lod = rough * maxLod;
        col = samplePanorama(dir, lod);
        if (maxLod < 0.5 || rough > 0.45) {
            float shMix = (maxLod < 0.5) ? clamp(rough * rough, 0.0, 1.0) : smoothstep(0.45, 1.0, rough);
            col = mix(col, evalSkySH(dir), shMix);
        }
    } else {
        col = mix(sampleProceduralSky(dir, rough), evalSkySH(dir), smoothstep(0.35, 1.0, rough));
    }

    col *= uSkyReflectionEnergy;

    for (int i = 0; i < uNumDirLights; i++) {
        float sunAmt = max(dot(dir, normalize(lights[i].dirType.xyz)), 0.0);
        float haloExp = mix(12.0, 2.0, clamp(rough, 0.0, 1.0));
        vec3 sunCol = lights[i].colorAtten.rgb / PI;
        col += sunCol * pow(sunAmt, haloExp) * 0.12 * (1.0 - rough * 0.7);
        col += sunCol * smoothstep(mix(0.9985, 0.85, rough * rough), mix(0.9996, 0.995, rough * rough), sunAmt) * mix(3.0, 0.25, rough);
    }
    return col;
}

float sampleChanLod(int layer, int ch, vec2 uv, float lod) {
    vec4 t = textureLod(uTexArray, vec3(uv, float(layer)), lod);
    if (ch == 0) return t.r;
    if (ch == 1) return t.g;
    if (ch == 2) return t.b;
    if (ch == 3) return t.a;
    return dot(t.rgb, vec3(0.299, 0.587, 0.114));
}

void evalMaterialAtHit(RTMaterial mat, HitInfo hit, float texLod, int bounce, out vec3 outNormal, out vec3 albedo, out float roughness, out float metallic, out vec3 emission, out float matAO) {
    vec2 uv = hit.uv * mat.uvScaleOffset.xy + mat.uvScaleOffset.zw;
    albedo = mat.albedoAlpha.rgb;
    if (mat.texLayers.x >= 0) albedo *= textureLod(uTexArray, vec3(uv, float(mat.texLayers.x)), texLod).rgb;

    bool samplePbrTex = (bounce < uPbrTexBounces) && (texLod < 5.0);
    roughness = mat.pbrParams.x;
    if (samplePbrTex && mat.texLayers.y >= 0) {
        roughness *= sampleChanLod(mat.texLayers.y, mat.chanNormal.x, uv, texLod);
    }

    metallic = mat.emissionMetal.w;
    if (samplePbrTex && mat.texLayers.z >= 0) {
        metallic *= sampleChanLod(mat.texLayers.z, mat.chanNormal.y, uv, texLod);
    }

    emission = mat.emissionMetal.rgb;
    if (mat.texLayers.w >= 0) emission *= textureLod(uTexArray, vec3(uv, float(mat.texLayers.w)), texLod).rgb;

    matAO = 1.0;
    if (samplePbrTex && mat.extraTex.x >= 0) {
        matAO = sampleChanLod(mat.extraTex.x, mat.extraTex.y, uv, texLod);
    }

    outNormal = hit.normal;
    if (uNormalMapsInBounces != 0 && samplePbrTex && mat.chanNormal.z >= 0 && hit.triIdx >= 0 && hit.instIdx >= 0) {
        RTTriangle tri = triangles[hit.triIdx];
        RTInstance inst = instances[hit.instIdx];
        vec3 wp0 = (inst.model * vec4(tri.p0.xyz, 1.0)).xyz;
        vec3 wp1 = (inst.model * vec4(tri.p1.xyz, 1.0)).xyz;
        vec3 wp2 = (inst.model * vec4(tri.p2.xyz, 1.0)).xyz;
        vec3 dp1 = wp1 - wp0;
        vec3 dp2 = wp2 - wp0;
        vec2 duv1 = (tri.uv01.zw - tri.uv01.xy) * mat.uvScaleOffset.xy;
        vec2 duv2 = (tri.uv2pad.xy - tri.uv01.xy) * mat.uvScaleOffset.xy;
        vec3 dp2perp = cross(dp2, outNormal);
        vec3 dp1perp = cross(outNormal, dp1);
        vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
        vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
        float maxTB = max(dot(T, T), dot(B, B));
        if (maxTB > 1e-10) {
            float invmax = inversesqrt(maxTB);
            vec3 nm = textureLod(uTexArray, vec3(uv, float(mat.chanNormal.z)), texLod).xyz * 2.0 - 1.0;
            nm.xy *= intBitsToFloat(mat.chanNormal.w);
            nm.z = sqrt(max(1.0 - dot(nm.xy, nm.xy), 0.0));
            outNormal = normalize(mat3(T * invmax, B * invmax, outNormal) * nm);
        }
    }
}

vec3 shadeHit(RTMaterial mat, vec3 hitPos, vec3 N, vec3 albedo, float metallic, vec3 emission, float matAO, float pathDist, int bounce, ivec2 px, int sppIdx) {
    int mflags = mat.extraTex.w;
    if ((mflags & 8) != 0) {
        return albedo + emission;
    }

    bool disableAmbient = (mflags & 4) != 0;
    vec3 col = emission;
    float aoLightScale = mix(1.0, matAO, mat.pbrParams.w);

    int aoCount = (!disableAmbient && bounce == 0 && uAmbientAOSamples > 0 && uAmbientAOIntensity > 0.001 && uAmbientAORadius > 0.01) ? uAmbientAOSamples : 0;
    vec3 baseAmbient = vec3(0.0);

    if (!disableAmbient) {
        vec3 skyIrradiance = evalSkySH(N);
        vec3 bAmb = mix(uAmbientColor, skyIrradiance * uAmbientRadianceBoost, uSkyContrib) * uAmbientEnergy;
        vec3 bounceFill = vec3(0.0);
        if (uAmbientBounceApprox > 0.001) {
            for (int i = 0; i < uNumDirLights; i++) {
                vec3 L = normalize(lights[i].dirType.xyz);
                float wrap = clamp(0.45 - 0.35 * dot(N, L), 0.0, 1.0);
                bounceFill += (lights[i].colorAtten.rgb / PI) * skyIrradiance * wrap * 0.15 * uAmbientBounceApprox;
            }
        }
        baseAmbient = (bAmb + bounceFill) * matAO * albedo * (1.0 - metallic * 0.75);
        if (aoCount == 0) {
            col += baseAmbient;
        } else {
            vec2 aoJitter = (uTemporalAccum != 0)
                ? vec2(fract(ignNoise(vec2(px) + 7.1) + float(uFrameIdx & 255) * 0.7548776662),
                       fract(ignNoise(vec2(px.yx) + 53.3) + float(uFrameIdx & 255) * 0.5698402910))
                : vec2(ignNoise(vec2(px)), ignNoise(vec2(px.yx) + vec2(17.3, 31.7)));
            vec3 aoOrigin = hitPos + N * max(uShadowBias, 0.002);
            float aoOcc = 0.0;
            for (int a = 0; a < aoCount; a++) {
                vec2 xi = r2Sample(a + sppIdx * 4, aoJitter);
                vec3 aoDir = sampleCosineHemisphere(xi, N);
                aoOcc += 1.0 - traceShadow(aoOrigin, aoDir, uAmbientAORadius, true);
            }
            float rtAO = clamp(1.0 - (aoOcc / float(aoCount)) * uAmbientAOIntensity, 0.0, 1.0);
            col += baseAmbient * rtAO;
        }
    }

    float diffuseScale = 1.0 - metallic;
    if (diffuseScale > 0.001) {
        bool traceShadows = (bounce < uShadowBounces) && (pathDist < uShadowMaxDist);
        int lightLimit = (bounce >= 2 || (bounce > 0 && uBounceLocalLights == 0) || (bounce == 1 && pathDist > uShadowMaxDist)) ? uNumDirLights : uNumLights;
        vec3 shadowOrigin = hitPos + N * max(uShadowBias, 0.002);

        for (int li = 0; li < lightLimit; li++) {
            RTLight light = lights[li];
            int ltype = int(light.dirType.w + 0.5);
            vec3 qDir = N;
            float qMaxDist = 200.0;
            float atten = 1.0;

            if (ltype == 0) {
                qDir = normalize(light.dirType.xyz);
            } else {
                vec3 toLight = light.posRange.xyz - hitPos;
                float dist = length(toLight);
                float range = max(light.posRange.w, 0.001);
                if (dist >= range || dist < 1e-5) continue;
                qDir = toLight / dist;
                qMaxDist = dist - max(uShadowBias, 0.002);
                float normDist = 1.0 - dist / range;
                atten = pow(normDist, max(light.colorAtten.w, 0.01));
                if (ltype == 2) {
                    float scos = dot(-qDir, normalize(light.dirType.xyz));
                    float spotMin = light.spotParams.x;
                    if (scos <= spotMin) continue;
                    float spotRatio = clamp((scos - spotMin) / max(1.0 - spotMin, 1e-4), 0.0, 1.0);
                    atten *= max(0.0, 1.0 - pow(1.0 - spotRatio, max(light.spotParams.y, 0.001)));
                }
            }

            float NdotL = max(dot(N, qDir), 0.0);
            if (NdotL <= 0.0 || atten <= 0.001) continue;

            vec3 lightContrib = (diffuseScale * albedo / PI) * light.colorAtten.rgb * atten * NdotL * aoLightScale;
            float lum = dot(lightContrib, vec3(0.2126, 0.7152, 0.0722));
            bool doTrace = (traceShadows && light.spotParams.z > 0.5 && lum >= 0.015);
            float vis = doTrace ? traceShadow(shadowOrigin, qDir, qMaxDist, false) : 1.0;
            col += lightContrib * vis;
        }
    }

    return col;
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 sz = imageSize(uOutReflection);
    if (px.x >= sz.x || px.y >= sz.y) return;

    float depth = texelFetch(uGDepth, px, 0).r;
    vec4 normRough = texelFetch(uGNormalRough, px, 0);
    vec4 albMetal  = texelFetch(uGAlbedoMetal, px, 0);
    vec4 extra     = texelFetch(uGExtra, px, 0);

    float roughness = normRough.w;
    float metallic  = albMetal.w;
    bool validPixel = (depth < 0.99999) && (dot(normRough.xyz, normRough.xyz) >= 0.01);
    bool reflectivePixel = validPixel && !(roughness > uMaxRoughness && metallic < 0.1);

    vec3 worldPos = vec3(0.0);
    vec3 N = vec3(0.0, 1.0, 0.0);
    vec3 V = vec3(0.0, 0.0, 1.0);
    float dist = -1.0;
    vec4 metaOut = vec4(0.0, 0.0, 0.0, -1.0);

    if (depth < 0.99999) {
        vec2 uv = (vec2(px) + 0.5) / vec2(sz);
        vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
        vec4 wp = uInvViewProj * ndc;
        worldPos = wp.xyz / wp.w;

        vec3 toCam = uCamPos - worldPos;
        dist = length(toCam);

        if (extra.g < 0.5) {
            float base_weight = 0.0;
            if (reflectivePixel) {
                N = normalize(normRough.xyz);
                V = toCam / max(dist, 1e-5);
                float ndotv = clamp(dot(N, V), 0.0, 1.0);
                vec3 F0 = mix(vec3(0.08 * extra.r), albMetal.rgb, metallic);
                vec3 F90 = mix(max(vec3(1.0 - roughness), F0), vec3(1.0), metallic);
                vec3 F = F0 + (F90 - F0) * pow(1.0 - ndotv, 5.0);
                float rough_fade = 1.0;
                if (metallic < 0.1) {
                    float fade_start = max(0.0, uMaxRoughness - uRoughnessFadeRange);
                    rough_fade = (uRoughnessFadeRange > 0.001)
                        ? (1.0 - smoothstep(fade_start, uMaxRoughness, roughness))
                        : ((roughness <= uMaxRoughness) ? 1.0 : 0.0);
                }
                float fresnel_max = max(max(F.r, F.g), F.b);
                base_weight = clamp(fresnel_max * rough_fade, 0.0, 1.0);
                if (base_weight <= 0.0005) {
                    base_weight = 0.0;
                }
            }
            metaOut = vec4(base_weight, roughness, 1.0, dist);
        }
    }
    imageStore(uOutCompositeMeta, px, metaOut);

    if (!reflectivePixel) {
        imageStore(uOutReflection, px, vec4(0.0, 0.0, 0.0, -1.0));
        return;
    }

    if (uCheckerboard != 0 && (((px.x + px.y) & 1) != (uFrameIdx & 1))) {
        imageStore(uOutReflection, px, vec4(0.0, 0.0, 0.0, -2.0));
        return;
    }

    bool useStochastic = (uAllowStochastic != 0) && (roughness > 0.025) && (uGlossySpread > 0.001);
    int spp = useStochastic ? max(uSamplesPerPixel, 1) : 1;
    float ggxAlpha = clamp(roughness * roughness * uGlossySpread, 0.001, 1.0);
    float jt0 = ignNoise(vec2(px));
    float jt1 = ignNoise(vec2(px.yx) + vec2(19.3, 41.7));
    vec2 baseJitter = (uTemporalAccum != 0)
        ? vec2(fract(jt0 + float(uFrameIdx & 255) * 0.61803398875),
               fract(jt1 + float(uFrameIdx & 255) * 0.7548776662))
        : vec2(jt0, jt1);

    vec3 totalAccum = vec3(0.0);
    vec3 baseThroughput = mix(vec3(1.0), albMetal.rgb, metallic);
    bool applyCull = (uRespectCullMode != 0);
    int maxAlphaSteps = (uAlphaConsideration != 0 && uAlphaRayTransparency != 0) ? max(1, uAlphaMaxSteps) : 1;

    vec3 rayOrigin = worldPos + N * max(uRayBias, 0.001);
    vec3 rayDir;
    if (useStochastic) {
        vec2 xi = r2Sample(0, baseJitter);
        vec3 H = sampleGGX_VNDF(xi, V, N, ggxAlpha);
        rayDir = reflect(-V, H);
        if (dot(rayDir, N) <= 0.001) rayDir = normalize(reflect(-V, N));
    } else {
        rayDir = normalize(reflect(-V, N));
    }

    vec3 throughput = baseThroughput;
    vec3 sampleAccum = vec3(0.0);
    float accumDist = dist;
    float bounceRange = uMaxRayDistance;
    float remT = bounceRange;
    int s = 0;
    int bounce = 0;
    int astep = 0;
    int maxTotalIters = spp * uMaxBounces * maxAlphaSteps;

    for (int iter = 0; iter < maxTotalIters; iter++) {
        HitInfo h = traceScene(rayOrigin, rayDir, remT, applyCull);
        bool startNextSample = false;

        if (!h.hit) {
            sampleAccum += throughput * sampleSkyRough(rayDir, bounce == 0 ? roughness * uGlossySpread : 0.25);
            startNextSample = true;
        } else {
            RTMaterial hm = materials[h.matId];
            int alphaMode = (uAlphaConsideration != 0) ? hm.extraTex.z : 0;
            bool treatAsSolid = (alphaMode == 0) || (astep + 1 >= maxAlphaSteps);

            if (!treatAsSolid) {
                vec2 huv = h.uv * hm.uvScaleOffset.xy + hm.uvScaleOffset.zw;
                float distLod = log2(max(1.0, (accumDist + h.t) * 0.35)) * uDistLodStrength;
                float hLod = clamp(distLod + uLodBias + float(bounce) * 1.25, 0.0, 8.0);
                vec4 albSample = hm.albedoAlpha;
                if (hm.texLayers.x >= 0) albSample *= textureLod(uTexArray, vec3(huv, float(hm.texLayers.x)), hLod);
                float hitAlpha = clamp(albSample.a, 0.0, 1.0);

                if ((alphaMode == 1 && hitAlpha >= hm.pbrParams.y) || (alphaMode == 2 && hitAlpha >= 0.99)) {
                    treatAsSolid = true;
                } else {
                    if (alphaMode == 2 && hitAlpha > 0.01) {
                        float tMetal = hm.emissionMetal.w;
                        float tRough = hm.pbrParams.x;
                        vec3 tEmis = hm.emissionMetal.rgb;
                        vec3 tAmb = evalSkySH(h.normal) * uAmbientRadianceBoost * uAmbientEnergy * uSkyContrib;
                        float tNdotV = max(dot(h.normal, -rayDir), 0.0);
                        vec3 tF0 = mix(vec3(0.08 * hm.pbrParams.z), albSample.rgb, tMetal);
                        vec3 tF = tF0 + (1.0 - tF0) * pow(1.0 - tNdotV, 5.0);
                        vec3 glassSpec = sampleSkyRough(reflect(rayDir, h.normal), tRough) * tF;
                        vec3 tBody = (tAmb * albSample.rgb * (1.0 - tMetal) + tEmis) * hitAlpha;

                        sampleAccum += throughput * (tBody + glassSpec * (1.0 - tRough * 0.7));
                        throughput *= mix(vec3(1.0), max(albSample.rgb, vec3(0.25)), hitAlpha * 0.5) * (1.0 - hitAlpha);
                    }

                    if (max(max(throughput.r, throughput.g), throughput.b) < uBounceThroughputCutoff) {
                        startNextSample = true;
                    } else {
                        float adv = max(h.t, 0.002);
                        accumDist += adv;
                        remT -= adv;
                        if (remT <= 0.01) {
                            sampleAccum += throughput * sampleSkyRough(rayDir, bounce == 0 ? roughness * uGlossySpread : 0.25);
                            startNextSample = true;
                        } else {
                            rayOrigin = h.pos + rayDir * max(uRayBias, 0.002);
                            astep++;
                        }
                    }
                }
            }

            if (treatAsSolid) {
                accumDist += h.t;
                float distLod = log2(max(1.0, accumDist * 0.35)) * uDistLodStrength;
                float texLod = clamp(distLod + uLodBias + (bounce == 0 ? roughness * 2.5 : 0.75) + float(bounce) * 1.25, 0.0, 8.0);

                vec3 hAlb, hEmis, hN;
                float hRough, hMetal, hAO;
                evalMaterialAtHit(hm, h, texLod, bounce, hN, hAlb, hRough, hMetal, hEmis, hAO);
                vec3 direct = shadeHit(hm, h.pos, hN, hAlb, hMetal, hEmis, hAO, accumDist, bounce, px, s);

                bool hReflective = !(hRough > uMaxRoughness && hMetal < 0.1) && ((hm.extraTex.w & 8) == 0);
                if (!hReflective) {
                    sampleAccum += throughput * direct;
                    startNextSample = true;
                } else {
                    vec3 hF0 = mix(vec3(0.08 * hm.pbrParams.z), hAlb, hMetal);
                    vec3 hF90 = mix(max(vec3(1.0 - hRough), hF0), vec3(1.0), hMetal);
                    float hNdotV = max(dot(hN, -rayDir), 0.0);
                    float f1 = 1.0 - clamp(hNdotV, 0.0, 1.0);
                    float f2 = f1 * f1;
                    vec3 hF = hF0 + (hF90 - hF0) * (f2 * f2 * f1);

                    if (bounce + 1 >= uMaxBounces) {
                        if (uEnvFallback != 0) {
                            vec3 envFallback = sampleSkyRough(reflect(rayDir, hN), hRough) * (1.0 - hRough * 0.65);
                            sampleAccum += throughput * (direct * (vec3(1.0) - hF) + envFallback * hF);
                        } else {
                            sampleAccum += throughput * direct;
                        }
                        startNextSample = true;
                    } else {
                        sampleAccum += throughput * direct * (vec3(1.0) - hF);
                        throughput *= hF;
                        if (max(max(throughput.r, throughput.g), throughput.b) < uBounceThroughputCutoff * float(bounce + 1)) {
                            startNextSample = true;
                        } else {
                            bounceRange *= uBounceDistFalloff;
                            remT = bounceRange;
                            rayOrigin = h.pos + hN * max(uRayBias, accumDist * 0.001);
                            rayDir = normalize(reflect(rayDir, hN));
                            bounce++;
                            astep = 0;
                        }
                    }
                }
            }
        }

        if (startNextSample) {
            totalAccum += sampleAccum;
            s++;
            if (s >= spp) break;
            sampleAccum = vec3(0.0);
            throughput = baseThroughput;
            accumDist = dist;
            bounceRange = uMaxRayDistance;
            remT = bounceRange;
            bounce = 0;
            astep = 0;
            rayOrigin = worldPos + N * max(uRayBias, 0.001);
            if (useStochastic) {
                vec2 xi = r2Sample(s, baseJitter);
                vec3 H = sampleGGX_VNDF(xi, V, N, ggxAlpha);
                rayDir = reflect(-V, H);
                if (dot(rayDir, N) <= 0.001) rayDir = normalize(reflect(-V, N));
            } else {
                rayDir = normalize(reflect(-V, N));
            }
        }
    }

    imageStore(uOutReflection, px, vec4(totalAccum / float(spp), dist));
}
)GLSL";

static const char *RECONSTRUCT_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutWork;
layout (rgba16f, binding = 1) writeonly uniform image2D uOutFinal;

uniform sampler2D uRawTex;
uniform sampler2D uHistoryTex;
uniform sampler2D uGNormalRough;
uniform sampler2D uGDepth;

uniform mat4 uInvViewProj;
uniform mat4 uPrevViewProj;
uniform vec3 uCamPos;
uniform vec3 uPrevCamPos;
uniform int uCheckerboard;
uniform int uTemporalAccum;
uniform int uHistoryValid;
uniform float uTemporalBlend;
uniform int uWriteFinal;

float hdrLum(vec3 c) {
    return dot(max(c, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
}

vec3 compressHDR(vec3 c) {
    c = max(c, vec3(0.0));
    return c / (1.0 + hdrLum(c));
}

vec3 uncompressHDR(vec3 c) {
    c = max(c, vec3(0.0));
    return c / max(1.0 - hdrLum(c), 1e-4);
}

vec3 rgbToYCoCg(vec3 c) {
    return vec3(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
         0.5  * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b
    );
}

vec3 ycocgToRgb(vec3 c) {
    float tmp = c.x - c.z;
    return max(vec3(tmp + c.y, c.x + c.z, tmp - c.y), vec3(0.0));
}

vec3 clipAABB(vec3 aabbMin, vec3 aabbMax, vec3 hist) {
    vec3 pClip = 0.5 * (aabbMax + aabbMin);
    vec3 eClip = 0.5 * (aabbMax - aabbMin) + 1e-5;
    vec3 vClip = hist - pClip;
    vec3 aUnit = abs(vClip / eClip);
    float maUnit = max(aUnit.x, max(aUnit.y, aUnit.z));
    if (maUnit > 1.0) {
        return pClip + vClip / maUnit;
    }
    return hist;
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 sz = imageSize(uOutWork);
    if (px.x >= sz.x || px.y >= sz.y) return;

    vec4 cur = texelFetch(uRawTex, px, 0);
    vec4 centerNR = texelFetch(uGNormalRough, px, 0);
    float centerDepth = texelFetch(uGDepth, px, 0).r;

    if (centerDepth >= 0.99999 || dot(centerNR.xyz, centerNR.xyz) < 0.01) {
        vec4 skipVal = vec4(0.0, 0.0, 0.0, -1.0);
        imageStore(uOutWork, px, skipVal);
        if (uWriteFinal != 0) imageStore(uOutFinal, px, skipVal);
        return;
    }

    vec3 centerN = normalize(centerNR.xyz);
    float rough = centerNR.w;
    vec2 uv = (vec2(px) + 0.5) / vec2(sz);
    vec4 wp = uInvViewProj * vec4(uv * 2.0 - 1.0, centerDepth * 2.0 - 1.0, 1.0);
    vec3 worldPos = wp.xyz / wp.w;
    float centerDist = length(uCamPos - worldPos);

    if (uCheckerboard != 0 && cur.a < -1.5) {
        vec3 sumC = vec3(0.0);
        float wSum = 0.0;
        const ivec2 offs[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
        for (int i = 0; i < 4; i++) {
            ivec2 np = clamp(px + offs[i], ivec2(0), sz - 1);
            vec4 nc = texelFetch(uRawTex, np, 0);
            if (nc.a >= 0.0) {
                vec4 nnr = texelFetch(uGNormalRough, np, 0);
                if (dot(nnr.xyz, nnr.xyz) > 0.01) {
                    float relD = abs(centerDist - nc.a) / max(centerDist, 0.25);
                    float nw = pow(max(0.0, dot(centerN, normalize(nnr.xyz))), 4.0) * exp(-relD * relD * 100.0);
                    sumC += compressHDR(nc.rgb) * nw;
                    wSum += nw;
                }
            }
        }
        if (wSum > 1e-5) {
            cur = vec4(uncompressHDR(sumC / wSum), centerDist);
        } else {
            cur = vec4(0.0, 0.0, 0.0, -1.0);
        }
    }

    if (cur.a >= 0.0) {
        // Compute 3x3 neighborhood statistics in compressed YCoCg space
        vec3 m1 = vec3(0.0);
        vec3 m2 = vec3(0.0);
        vec3 nMin = vec3(1e10);
        vec3 nMax = vec3(-1e10);
        float cnt = 0.0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                ivec2 np = clamp(px + ivec2(dx, dy), ivec2(0), sz - 1);
                vec4 nc = texelFetch(uRawTex, np, 0);
                if (nc.a >= 0.0) {
                    float relD = abs(centerDist - nc.a) / max(centerDist, 0.25);
                    if (relD < 0.25 || (dx == 0 && dy == 0)) {
                        vec3 yc = rgbToYCoCg(compressHDR(nc.rgb));
                        m1 += yc;
                        m2 += yc * yc;
                        nMin = min(nMin, yc);
                        nMax = max(nMax, yc);
                        cnt += 1.0;
                    }
                }
            }
        }

        if (cnt > 0.5) {
            vec3 mean = m1 / cnt;
            vec3 stddev = sqrt(max(m2 / cnt - mean * mean, vec3(0.0)));
            vec3 curYC = rgbToYCoCg(compressHDR(cur.rgb));

            // Suppress extreme 1-spp fireflies on glossy surfaces
            if (rough > 0.04 && cnt >= 3.0) {
                float maxAllowedY = mean.x + max(stddev.x * 2.25, 0.05);
                if (curYC.x > maxAllowedY && curYC.x > 1e-4) {
                    curYC *= maxAllowedY / curYC.x;
                    cur.rgb = uncompressHDR(ycocgToRgb(curYC));
                }
            }

            if (uTemporalAccum != 0 && uHistoryValid != 0) {
                vec4 prevClip = uPrevViewProj * vec4(worldPos, 1.0);
                if (prevClip.w > 0.001) {
                    vec2 prevUv = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
                    if (prevUv.x >= 0.0 && prevUv.x <= 1.0 && prevUv.y >= 0.0 && prevUv.y <= 1.0) {
                        vec4 hist = textureLod(uHistoryTex, prevUv, 0.0);
                        float prevDist = length(uPrevCamPos - worldPos);
                        if (hist.a >= 0.0 && abs(prevDist - hist.a) < max(0.10, prevDist * 0.06)) {
                            float gamma = mix(1.0, 2.25, smoothstep(0.03, 0.30, rough));
                            vec3 boxMin = max(nMin - vec3(0.01), mean - stddev * gamma - vec3(0.015));
                            vec3 boxMax = min(nMax + vec3(0.01), mean + stddev * gamma + vec3(0.015));

                            vec3 histYC = rgbToYCoCg(compressHDR(hist.rgb));
                            vec3 clippedYC = clipAABB(boxMin, boxMax, histYC);
                            vec3 outYC = mix(clippedYC, curYC, clamp(uTemporalBlend, 0.02, 1.0));
                            cur.rgb = uncompressHDR(ycocgToRgb(outYC));
                        }
                    }
                }
            }
        }
    }

    imageStore(uOutWork, px, cur);
    if (uWriteFinal != 0) {
        imageStore(uOutFinal, px, cur);
    }
}
)GLSL";

static const char *DENOISE_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutTex;

uniform sampler2D uSrcTex;
uniform sampler2D uGNormalRough;
uniform sampler2D uGDepth;
uniform mat4 uInvViewProj;
uniform vec3 uCamPos;
uniform int uStepSize;

float hdrLum(vec3 c) {
    return dot(max(c, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
}

vec3 compressHDR(vec3 c) {
    c = max(c, vec3(0.0));
    return c / (1.0 + hdrLum(c));
}

vec3 uncompressHDR(vec3 c) {
    c = max(c, vec3(0.0));
    return c / max(1.0 - hdrLum(c), 1e-4);
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 sz = imageSize(uOutTex);
    if (px.x >= sz.x || px.y >= sz.y) return;

    vec4 center = texelFetch(uSrcTex, px, 0);
    if (center.a < 0.0) {
        imageStore(uOutTex, px, center);
        return;
    }

    vec4 cNR = texelFetch(uGNormalRough, px, 0);
    float cDepth = texelFetch(uGDepth, px, 0).r;
    if (cDepth >= 0.99999 || dot(cNR.xyz, cNR.xyz) < 0.01) {
        imageStore(uOutTex, px, center);
        return;
    }

    float rough = cNR.w;
    float roughScale = smoothstep(0.02, 0.12, rough);
    if (roughScale <= 0.01) {
        imageStore(uOutTex, px, center);
        return;
    }

    vec3 cN = normalize(cNR.xyz);
    vec2 cuv = (vec2(px) + 0.5) / vec2(sz);
    vec4 cwp = uInvViewProj * vec4(cuv * 2.0 - 1.0, cDepth * 2.0 - 1.0, 1.0);
    vec3 cPos = cwp.xyz / cwp.w;

    vec3 cComp = compressHDR(center.rgb);
    float cLum = hdrLum(cComp);

    // Estimate local compressed-luma variance in 3x3 so stochastic noise blurs cleanly
    float lSum = 0.0;
    float lSqSum = 0.0;
    float lCnt = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 np = clamp(px + ivec2(dx, dy), ivec2(0), sz - 1);
            vec4 nc = texelFetch(uSrcTex, np, 0);
            if (nc.a >= 0.0) {
                float nl = hdrLum(compressHDR(nc.rgb));
                lSum += nl;
                lSqSum += nl * nl;
                lCnt += 1.0;
            }
        }
    }
    float lMean = (lCnt > 0.0) ? (lSum / lCnt) : cLum;
    float lVar = (lCnt > 0.0) ? max(0.0, lSqSum / lCnt - lMean * lMean) : 0.0;

    float sigmaL = max(0.12, sqrt(lVar) * 2.5) * mix(0.5, 2.0, roughScale);
    float sigmaN = mix(0.08, 0.40, rough);
    float sigmaPlane = max(0.015, center.a * 0.012);
    float invTwoSigmaL2 = 0.5 / (sigmaL * sigmaL);
    float invTwoSigmaPlane2 = 0.5 / (sigmaPlane * sigmaPlane);

    const float k1d[5] = float[5](0.0625, 0.25, 0.375, 0.25, 0.0625);
    float centerW = k1d[2] * k1d[2];
    vec3 sumC = cComp * centerW;
    float wSum = centerW;

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int ad = abs(dx) + abs(dy);
            if (ad == 0 || ad > 2) continue;
            ivec2 np = clamp(px + ivec2(dx, dy) * uStepSize, ivec2(0), sz - 1);
            vec4 nc = texelFetch(uSrcTex, np, 0);
            if (nc.a < 0.0) continue;

            vec4 nNR = texelFetch(uGNormalRough, np, 0);
            float nDepth = texelFetch(uGDepth, np, 0).r;
            if (nDepth >= 0.99999 || dot(nNR.xyz, nNR.xyz) < 0.01) continue;

            vec2 nuv = (vec2(np) + 0.5) / vec2(sz);
            vec4 nwp = uInvViewProj * vec4(nuv * 2.0 - 1.0, nDepth * 2.0 - 1.0, 1.0);
            vec3 nPos = nwp.xyz / nwp.w;

            float kw = k1d[dx + 2] * k1d[dy + 2] * roughScale;
            float nd = max(0.0, dot(cN, normalize(nNR.xyz)));
            float planeDist = abs(dot(nPos - cPos, cN));
            float relD = abs(center.a - nc.a) / max(center.a, 0.25);

            vec3 nComp = compressHDR(nc.rgb);
            float dl = cLum - hdrLum(nComp);

            float exponent = -((1.0 - nd) / sigmaN)
                             - (planeDist * planeDist) * invTwoSigmaPlane2
                             - relD * relD * 16.0
                             - abs(rough - nNR.w) * 10.0
                             - (dl * dl) * invTwoSigmaL2;
            float w = kw * exp(exponent);
            sumC += nComp * w;
            wSum += w;
        }
    }

    imageStore(uOutTex, px, vec4(uncompressHDR(sumC / wSum), center.a));
}
)GLSL";

static const char *COMPOSITE_FRAG_GLSL = R"GLSL(
#version 430 core
in vec2 vUv;
layout (location = 0) out vec4 outColor;

uniform sampler2D uReflTex;
uniform sampler2D uMetaTex;
uniform sampler2D uDepthTex;
uniform float uIntensity;
uniform float uBlurStrength;
uniform float uBlurLodScale;
uniform int uBicubicBlur;
uniform int uBilateralUpsample;
uniform mat4 uInvProj;

float linearizeDist(float rawD, vec2 uv) {
    vec4 vp = uInvProj * vec4(uv * 2.0 - 1.0, min(rawD, 0.99999) * 2.0 - 1.0, 1.0);
    return length(vp.xyz / vp.w);
}

vec4 reflBicubicMip(sampler2D tex, vec2 uv, int mip, vec2 mipSize) {
    vec2 invSize = 1.0 / mipSize;
    vec2 texCoords = uv * mipSize - 0.5;
    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    vec2 fxy2 = fxy * fxy;
    vec2 fxy3 = fxy2 * fxy;

    vec2 w0 = (1.0 / 6.0) * (1.0 - 3.0 * fxy + 3.0 * fxy2 - fxy3);
    vec2 w1 = (1.0 / 6.0) * (4.0 - 6.0 * fxy2 + 3.0 * fxy3);
    vec2 w2 = (1.0 / 6.0) * (1.0 + 3.0 * fxy + 3.0 * fxy2 - fxy3);
    vec2 w3 = (1.0 / 6.0) * fxy3;

    vec2 g0 = w0 + w1;
    vec2 g1 = w2 + w3;
    vec2 h0 = (w1 / g0) - 0.5 + texCoords;
    vec2 h1 = (w3 / g1) + 1.5 + texCoords;

    vec2 uv0 = h0 * invSize;
    vec2 uv1 = h1 * invSize;

    float flMip = float(mip);
    vec4 tex00 = textureLod(tex, vec2(uv0.x, uv0.y), flMip);
    vec4 tex10 = textureLod(tex, vec2(uv1.x, uv0.y), flMip);
    vec4 tex01 = textureLod(tex, vec2(uv0.x, uv1.y), flMip);
    vec4 tex11 = textureLod(tex, vec2(uv1.x, uv1.y), flMip);

    tex00 = mix(tex10, tex00, g0.x);
    tex01 = mix(tex11, tex01, g0.x);
    return mix(tex01, tex00, g0.y);
}

vec4 textureReflBicubic(sampler2D tex, vec2 uv, float lod) {
    vec2 baseSize = vec2(textureSize(tex, 0));
    float maxLod = log2(max(max(baseSize.x, baseSize.y), 1.0));
    lod = clamp(lod, 0.0, maxLod);

    int mip0 = int(floor(lod));
    int mip1 = min(mip0 + 1, int(maxLod));
    float f = fract(lod);

    vec4 c0 = reflBicubicMip(tex, uv, mip0, max(vec2(1.0), baseSize * exp2(-float(mip0))));
    if (f < 0.001 || mip0 == mip1) {
        return c0;
    }
    vec4 c1 = reflBicubicMip(tex, uv, mip1, max(vec2(1.0), baseSize * exp2(-float(mip1))));
    return mix(c0, c1, f);
}

void main() {
    float fullD = textureLod(uDepthTex, vUv, 0.0).r;
    if (fullD >= 0.99999) discard;

    float wSum = 0.0;
    vec3 refl = vec3(0.0);
    float weightVal = 0.0;
    float rough = 0.0;

    if (uBilateralUpsample != 0) {
        float fullDist = linearizeDist(fullD, vUv);
        vec2 texSize = vec2(textureSize(uReflTex, 0));
        vec2 fCoord = vUv * texSize - 0.5;
        ivec2 base = ivec2(floor(fCoord));
        vec2 f = clamp(fCoord - vec2(base), vec2(0.0), vec2(1.0));
        ivec2 maxIdx = ivec2(texSize) - 1;

        ivec2 c00 = clamp(base + ivec2(0, 0), ivec2(0), maxIdx);
        ivec2 c10 = clamp(base + ivec2(1, 0), ivec2(0), maxIdx);
        ivec2 c01 = clamp(base + ivec2(0, 1), ivec2(0), maxIdx);
        ivec2 c11 = clamp(base + ivec2(1, 1), ivec2(0), maxIdx);

        vec4 s00 = texelFetch(uReflTex, c00, 0);
        vec4 s10 = texelFetch(uReflTex, c10, 0);
        vec4 s01 = texelFetch(uReflTex, c01, 0);
        vec4 s11 = texelFetch(uReflTex, c11, 0);

        vec4 m00 = texelFetch(uMetaTex, c00, 0);
        vec4 m10 = texelFetch(uMetaTex, c10, 0);
        vec4 m01 = texelFetch(uMetaTex, c01, 0);
        vec4 m11 = texelFetch(uMetaTex, c11, 0);

        float b00 = (1.0 - f.x) * (1.0 - f.y);
        float b10 =        f.x  * (1.0 - f.y);
        float b01 = (1.0 - f.x) *        f.y;
        float b11 =        f.x  *        f.y;

        float invD = 1.0 / max(fullDist, 0.25);
        float rd00 = abs(fullDist - m00.w) * invD;
        float rd10 = abs(fullDist - m10.w) * invD;
        float rd01 = abs(fullDist - m01.w) * invD;
        float rd11 = abs(fullDist - m11.w) * invD;

        float gw00 = (m00.z > 0.5 && m00.w >= 0.0) ? (b00 * exp(-rd00 * rd00 * 400.0)) : 0.0;
        float gw10 = (m10.z > 0.5 && m10.w >= 0.0) ? (b10 * exp(-rd10 * rd10 * 400.0)) : 0.0;
        float gw01 = (m01.z > 0.5 && m01.w >= 0.0) ? (b01 * exp(-rd01 * rd01 * 400.0)) : 0.0;
        float gw11 = (m11.z > 0.5 && m11.w >= 0.0) ? (b11 * exp(-rd11 * rd11 * 400.0)) : 0.0;

        float rw00 = (s00.a >= 0.0 && m00.x > 0.0005) ? gw00 : 0.0;
        float rw10 = (s10.a >= 0.0 && m10.x > 0.0005) ? gw10 : 0.0;
        float rw01 = (s01.a >= 0.0 && m01.x > 0.0005) ? gw01 : 0.0;
        float rw11 = (s11.a >= 0.0 && m11.x > 0.0005) ? gw11 : 0.0;

        float gwSum = gw00 + gw10 + gw01 + gw11;
        float rwSum = rw00 + rw10 + rw01 + rw11;
        if (gwSum > 1e-4 && rwSum > 1e-5) {
            wSum = rwSum;
            refl = (s00.rgb * rw00 + s10.rgb * rw10 + s01.rgb * rw01 + s11.rgb * rw11) / rwSum;
            weightVal = (m00.x * gw00 + m10.x * gw10 + m01.x * gw01 + m11.x * gw11) / gwSum;
            rough = (m00.y * rw00 + m10.y * rw10 + m01.y * rw01 + m11.y * rw11) / rwSum;
        }
    } else {
        vec4 sFast = textureLod(uReflTex, vUv, 0.0);
        vec4 mFast = textureLod(uMetaTex, vUv, 0.0);
        if (sFast.a >= 0.0 && mFast.z > 0.5 && mFast.x > 0.0005) {
            wSum = 1.0;
            refl = sFast.rgb;
            weightVal = mFast.x;
            rough = mFast.y;
        }
    }

    float finalWeight = (wSum > 1e-5) ? clamp(weightVal * min(uIntensity, 1.0), 0.0, 1.0) : 0.0;
    if (finalWeight <= 0.0005) discard;

    if (rough > 0.02 && uBlurStrength > 0.001 && uBlurLodScale > 0.001) {
        float lod = rough * uBlurLodScale * uBlurStrength;
        vec4 blurred4 = (uBicubicBlur != 0)
            ? textureReflBicubic(uReflTex, vUv, lod)
            : textureLod(uReflTex, vUv, lod);
        float blurValid = smoothstep(-0.25, 0.25, blurred4.a);
        refl = mix(refl, blurred4.rgb, clamp(lod, 0.0, 1.0) * blurValid);
    }

    outColor = vec4(max(refl, vec3(0.0)) * max(uIntensity, 1.0), finalWeight);
}
)GLSL";

static const char *GODOT_NEXT_PASS_SHADER_TEMPLATE = R"SHADER(
shader_type spatial;
render_mode unshaded, blend_mix, depth_draw_never, %s;

uniform sampler2D rt_refl_tex;
uniform float rt_roughness = 0.15;
uniform float rt_metallic = 0.0;
uniform float rt_specular = 0.5;
uniform vec3 rt_albedo = vec3(1.0);
uniform float rt_albedo_alpha = 1.0;
uniform float rt_intensity = 1.0;
uniform float rt_max_roughness = 0.65;
uniform float rt_roughness_fade_range = 0.15;
uniform float rt_blur_strength = 1.0;
uniform float rt_blur_lod_scale = 5.0;
uniform bool rt_bicubic_blur = true;
uniform bool rt_bilateral_upsample = true;

uniform vec2 rt_uv1_scale = vec2(1.0);
uniform vec2 rt_uv1_offset = vec2(0.0);

uniform bool rt_has_albedo_tex = false;
uniform sampler2D rt_albedo_tex : hint_albedo;

uniform bool rt_has_rough_tex = false;
uniform sampler2D rt_rough_tex;
uniform int rt_rough_channel = 0;

uniform bool rt_has_metal_tex = false;
uniform sampler2D rt_metal_tex;
uniform int rt_metal_channel = 0;

uniform bool rt_has_normal_tex = false;
uniform sampler2D rt_normal_tex;
uniform float rt_normal_scale = 1.0;

uniform bool rt_alpha_consideration = true;
uniform int rt_alpha_mode = 0; // 0=opaque, 1=scissor, 2=transparent
uniform float rt_alpha_scissor = 0.5;
uniform float rt_transparent_fresnel_boost = 0.75;

float sample_channel(sampler2D tex, vec2 uv, int ch) {
    vec4 t = texture(tex, uv);
    if (ch == 0) return t.r;
    if (ch == 1) return t.g;
    if (ch == 2) return t.b;
    if (ch == 3) return t.a;
    return dot(t.rgb, vec3(0.299, 0.587, 0.114));
}

vec4 refl_bicubic_mip(sampler2D tex, vec2 uv, int mip, vec2 mip_size) {
    vec2 inv_size = 1.0 / mip_size;
    vec2 tex_coords = uv * mip_size - 0.5;
    vec2 fxy = fract(tex_coords);
    tex_coords -= fxy;

    vec2 fxy2 = fxy * fxy;
    vec2 fxy3 = fxy2 * fxy;

    vec2 w0 = (1.0 / 6.0) * (1.0 - 3.0 * fxy + 3.0 * fxy2 - fxy3);
    vec2 w1 = (1.0 / 6.0) * (4.0 - 6.0 * fxy2 + 3.0 * fxy3);
    vec2 w2 = (1.0 / 6.0) * (1.0 + 3.0 * fxy + 3.0 * fxy2 - fxy3);
    vec2 w3 = (1.0 / 6.0) * fxy3;

    vec2 g0 = w0 + w1;
    vec2 g1 = w2 + w3;
    vec2 h0 = (w1 / g0) - 0.5 + tex_coords;
    vec2 h1 = (w3 / g1) + 1.5 + tex_coords;

    vec2 uv0 = h0 * inv_size;
    vec2 uv1 = h1 * inv_size;

    float fl_mip = float(mip);
    vec4 tex00 = textureLod(tex, vec2(uv0.x, uv0.y), fl_mip);
    vec4 tex10 = textureLod(tex, vec2(uv1.x, uv0.y), fl_mip);
    vec4 tex01 = textureLod(tex, vec2(uv0.x, uv1.y), fl_mip);
    vec4 tex11 = textureLod(tex, vec2(uv1.x, uv1.y), fl_mip);

    tex00 = mix(tex10, tex00, g0.x);
    tex01 = mix(tex11, tex01, g0.x);
    return mix(tex01, tex00, g0.y);
}

vec4 texture_refl_bicubic(sampler2D tex, vec2 uv, float lod) {
    vec2 base_size = vec2(textureSize(tex, 0));
    float max_lod = log2(max(max(base_size.x, base_size.y), 1.0));
    lod = clamp(lod, 0.0, max_lod);

    int mip0 = int(floor(lod));
    int mip1 = min(mip0 + 1, int(max_lod));
    float f = fract(lod);

    vec4 c0 = refl_bicubic_mip(tex, uv, mip0, max(vec2(1.0), base_size * exp2(-float(mip0))));
    if (f < 0.001 || mip0 == mip1) {
        return c0;
    }
    vec4 c1 = refl_bicubic_mip(tex, uv, mip1, max(vec2(1.0), base_size * exp2(-float(mip1))));
    return mix(c0, c1, f);
}

void fragment() {
    vec2 uv = UV * rt_uv1_scale + rt_uv1_offset;

    vec4 alb4 = rt_has_albedo_tex ? texture(rt_albedo_tex, uv) : vec4(1.0);
    vec3 albedo = rt_albedo * alb4.rgb;
    float surf_alpha = clamp(rt_albedo_alpha * alb4.a, 0.0, 1.0);

    if (rt_alpha_consideration && rt_alpha_mode == 1 && surf_alpha < rt_alpha_scissor) {
        discard;
    }

    float rough = rt_roughness * (rt_has_rough_tex ? sample_channel(rt_rough_tex, uv, rt_rough_channel) : 1.0);
    float metal = rt_metallic * (rt_has_metal_tex ? sample_channel(rt_metal_tex, uv, rt_metal_channel) : 1.0);

    vec2 tex_size = vec2(textureSize(rt_refl_tex, 0));
    vec2 f_coord = SCREEN_UV * tex_size - 0.5;
    ivec2 base = ivec2(floor(f_coord));
    vec2 f = f_coord - vec2(base);
    ivec2 max_idx = ivec2(tex_size) - 1;

    float full_dist = length(VERTEX);
    float w_sum = 0.0;
    vec3 refl = vec3(0.0);

    if (rt_bilateral_upsample) {
        vec4 s00 = texelFetch(rt_refl_tex, clamp(base + ivec2(0, 0), ivec2(0), max_idx), 0);
        vec4 s10 = texelFetch(rt_refl_tex, clamp(base + ivec2(1, 0), ivec2(0), max_idx), 0);
        vec4 s01 = texelFetch(rt_refl_tex, clamp(base + ivec2(0, 1), ivec2(0), max_idx), 0);
        vec4 s11 = texelFetch(rt_refl_tex, clamp(base + ivec2(1, 1), ivec2(0), max_idx), 0);

        float inv_d = 1.0 / max(full_dist, 0.25);
        float rd00 = abs(full_dist - s00.a) * inv_d;
        float rd10 = abs(full_dist - s10.a) * inv_d;
        float rd01 = abs(full_dist - s01.a) * inv_d;
        float rd11 = abs(full_dist - s11.a) * inv_d;

        float w00 = (s00.a >= 0.0) ? ((1.0 - f.x) * (1.0 - f.y) * exp(-rd00 * rd00 * 400.0)) : 0.0;
        float w10 = (s10.a >= 0.0) ? (      f.x  * (1.0 - f.y) * exp(-rd10 * rd10 * 400.0)) : 0.0;
        float w01 = (s01.a >= 0.0) ? ((1.0 - f.x) *       f.y  * exp(-rd01 * rd01 * 400.0)) : 0.0;
        float w11 = (s11.a >= 0.0) ? (      f.x  *       f.y  * exp(-rd11 * rd11 * 400.0)) : 0.0;

        w_sum = w00 + w10 + w01 + w11;
        refl = (w_sum > 1e-4) ? (s00.rgb * w00 + s10.rgb * w10 + s01.rgb * w01 + s11.rgb * w11) / w_sum : vec3(0.0);
    } else {
        vec4 s_fast = textureLod(rt_refl_tex, SCREEN_UV, 0.0);
        w_sum = (s_fast.a >= 0.0) ? 1.0 : 0.0;
        refl = (s_fast.a >= 0.0) ? s_fast.rgb : vec3(0.0);
    }

    if (w_sum > 1e-5 && rough > 0.02 && rt_blur_strength > 0.001 && rt_blur_lod_scale > 0.001) {
        float lod = rough * rt_blur_lod_scale * rt_blur_strength;
        vec4 blurred4 = rt_bicubic_blur
            ? texture_refl_bicubic(rt_refl_tex, SCREEN_UV, lod)
            : textureLod(rt_refl_tex, SCREEN_UV, lod);
        float blur_valid = smoothstep(-0.25, 0.25, blurred4.a);
        refl = mix(refl, blurred4.rgb, clamp(lod, 0.0, 1.0) * blur_valid);
    }

    vec3 N = normalize(NORMAL);
    if (rt_has_normal_tex && dot(TANGENT, TANGENT) > 1e-6 && dot(BINORMAL, BINORMAL) > 1e-6) {
        vec3 nm = texture(rt_normal_tex, uv).xyz * 2.0 - 1.0;
        nm.xy *= rt_normal_scale;
        nm.z = sqrt(max(1.0 - dot(nm.xy, nm.xy), 0.0));
        N = normalize(TANGENT * nm.x + BINORMAL * nm.y + N * nm.z);
    }

    vec3 V = normalize(VIEW);
    float ndotv = clamp(dot(N, V), 0.0, 1.0);
    vec3 F0 = mix(vec3(0.08 * rt_specular), albedo, metal);
    vec3 F90 = mix(max(vec3(1.0 - rough), F0), vec3(1.0), metal);
    vec3 F = F0 + (F90 - F0) * pow(1.0 - ndotv, 5.0);
    float rough_fade = 1.0;
    if (metal < 0.1) {
        float fade_start = max(0.0, rt_max_roughness - rt_roughness_fade_range);
        rough_fade = (rt_roughness_fade_range > 0.001)
            ? (1.0 - smoothstep(fade_start, rt_max_roughness, rough))
            : ((rough <= rt_max_roughness) ? 1.0 : 0.0);
    }

    float fresnel_max = max(max(F.r, F.g), F.b);
    float alpha_factor = 1.0;
    if (rt_alpha_consideration && rt_alpha_mode == 2) {
        float glass_weight = max(surf_alpha, fresnel_max);
        alpha_factor = mix(surf_alpha, glass_weight, rt_transparent_fresnel_boost);
    }

    float weight = fresnel_max * min(rt_intensity, 1.0) * rough_fade * alpha_factor;
    bool valid = (w_sum > 1e-5) && (rough_fade > 0.001) && (weight > 0.0005);
    ALBEDO = refl * max(rt_intensity, 1.0);
    ALPHA = valid ? clamp(weight, 0.0, 1.0) : 0.0;
}
)SHADER";

RTReflections::RTReflections() {
    refl_texture.instance();
    composite_meta_texture.instance();
}

RTReflections::~RTReflections() {
    clear_attached_passes();
    _unregister_scene_hook();
    cleanup_gl_resources();
}

void RTReflections::cleanup_gl_resources() {
    if (!gl_ready || gl_thread_id != Thread::get_caller_id()) {
        return;
    }

    SharedGL43Context::ScopedScope gl43;
    if (!gl43.is_active()) {
        return;
    }

    raytracer.cleanup_gl();

    if (gbuf_fbo) { glDeleteFramebuffers(1, &gbuf_fbo); gbuf_fbo = 0; }
    if (gbuf_normal_rough) { glDeleteTextures(1, &gbuf_normal_rough); gbuf_normal_rough = 0; }
    if (gbuf_albedo_metal) { glDeleteTextures(1, &gbuf_albedo_metal); gbuf_albedo_metal = 0; }
    if (gbuf_extra) { glDeleteTextures(1, &gbuf_extra); gbuf_extra = 0; }
    if (gbuf_depth) { glDeleteTextures(1, &gbuf_depth); gbuf_depth = 0; }
    if (gbuf_program) { glDeleteProgram(gbuf_program); gbuf_program = 0; }

    if (rt_program) { glDeleteProgram(rt_program); rt_program = 0; }
    if (sh_program) { glDeleteProgram(sh_program); sh_program = 0; }
    if (reconstruct_program) { glDeleteProgram(reconstruct_program); reconstruct_program = 0; }
    if (denoise_program) { glDeleteProgram(denoise_program); denoise_program = 0; }
    if (composite_program) { glDeleteProgram(composite_program); composite_program = 0; }

    if (rt_work_a) { glDeleteTextures(1, &rt_work_a); rt_work_a = 0; }
    if (rt_work_b) { glDeleteTextures(1, &rt_work_b); rt_work_b = 0; }
    if (rt_history_tex[0]) { glDeleteTextures(1, &rt_history_tex[0]); rt_history_tex[0] = 0; }
    if (rt_history_tex[1]) { glDeleteTextures(1, &rt_history_tex[1]); rt_history_tex[1] = 0; }

    tex_pool.cleanup_gl();
    if (dummy_white_tex) { glDeleteTextures(1, &dummy_white_tex); dummy_white_tex = 0; }
    if (pano_sampler_mip) { glDeleteSamplers(1, &pano_sampler_mip); pano_sampler_mip = 0; }
    if (pano_sampler_nomip) { glDeleteSamplers(1, &pano_sampler_nomip); pano_sampler_nomip = 0; }

    if (ssbo_materials) { glDeleteBuffers(1, &ssbo_materials); ssbo_materials = 0; }
    if (ssbo_lights) { glDeleteBuffers(1, &ssbo_lights); ssbo_lights = 0; }
    if (ssbo_sky_sh) { glDeleteBuffers(1, &ssbo_sky_sh); ssbo_sky_sh = 0; }

    gl_ready = false;
    gl_attempted = false;
}

void RTReflections::_bind_methods() {
    // Core
    ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &RTReflections::set_enabled);
    ClassDB::bind_method(D_METHOD("is_enabled"), &RTReflections::is_enabled);
    ClassDB::bind_method(D_METHOD("set_editor_preview", "preview"), &RTReflections::set_editor_preview);
    ClassDB::bind_method(D_METHOD("get_editor_preview"), &RTReflections::get_editor_preview);
    ClassDB::bind_method(D_METHOD("set_auto_composite", "auto_composite"), &RTReflections::set_auto_composite);
    ClassDB::bind_method(D_METHOD("get_auto_composite"), &RTReflections::get_auto_composite);
    ClassDB::bind_method(D_METHOD("set_auto_attach_materials", "auto_attach"), &RTReflections::set_auto_attach_materials);
    ClassDB::bind_method(D_METHOD("get_auto_attach_materials"), &RTReflections::get_auto_attach_materials);
    ClassDB::bind_method(D_METHOD("set_reflection_intensity", "intensity"), &RTReflections::set_reflection_intensity);
    ClassDB::bind_method(D_METHOD("get_reflection_intensity"), &RTReflections::get_reflection_intensity);

    // Resolution & Filtering
    ClassDB::bind_method(D_METHOD("set_reflection_scale", "scale"), &RTReflections::set_reflection_scale);
    ClassDB::bind_method(D_METHOD("get_reflection_scale"), &RTReflections::get_reflection_scale);
    ClassDB::bind_method(D_METHOD("set_reflection_filtering", "filtering"), &RTReflections::set_reflection_filtering);
    ClassDB::bind_method(D_METHOD("get_reflection_filtering"), &RTReflections::get_reflection_filtering);
    ClassDB::bind_method(D_METHOD("set_bilateral_upsample", "enable"), &RTReflections::set_bilateral_upsample);
    ClassDB::bind_method(D_METHOD("get_bilateral_upsample"), &RTReflections::get_bilateral_upsample);
    ClassDB::bind_method(D_METHOD("set_temporal_accumulation", "enable"), &RTReflections::set_temporal_accumulation);
    ClassDB::bind_method(D_METHOD("get_temporal_accumulation"), &RTReflections::get_temporal_accumulation);
    ClassDB::bind_method(D_METHOD("set_temporal_blend", "blend"), &RTReflections::set_temporal_blend);
    ClassDB::bind_method(D_METHOD("get_temporal_blend"), &RTReflections::get_temporal_blend);
    ClassDB::bind_method(D_METHOD("set_checkerboard_tracing", "enable"), &RTReflections::set_checkerboard_tracing);
    ClassDB::bind_method(D_METHOD("get_checkerboard_tracing"), &RTReflections::get_checkerboard_tracing);
    ClassDB::bind_method(D_METHOD("set_spatial_denoise_passes", "passes"), &RTReflections::set_spatial_denoise_passes);
    ClassDB::bind_method(D_METHOD("get_spatial_denoise_passes"), &RTReflections::get_spatial_denoise_passes);
    ClassDB::bind_method(D_METHOD("set_max_roughness", "roughness"), &RTReflections::set_max_roughness);
    ClassDB::bind_method(D_METHOD("get_max_roughness"), &RTReflections::get_max_roughness);
    ClassDB::bind_method(D_METHOD("set_roughness_fade_range", "range"), &RTReflections::set_roughness_fade_range);
    ClassDB::bind_method(D_METHOD("get_roughness_fade_range"), &RTReflections::get_roughness_fade_range);
    ClassDB::bind_method(D_METHOD("set_blur_strength", "strength"), &RTReflections::set_blur_strength);
    ClassDB::bind_method(D_METHOD("get_blur_strength"), &RTReflections::get_blur_strength);
    ClassDB::bind_method(D_METHOD("set_blur_lod_scale", "scale"), &RTReflections::set_blur_lod_scale);
    ClassDB::bind_method(D_METHOD("get_blur_lod_scale"), &RTReflections::get_blur_lod_scale);

    // Tracing & Bounces
    ClassDB::bind_method(D_METHOD("set_ray_samples_per_pixel", "samples"), &RTReflections::set_ray_samples_per_pixel);
    ClassDB::bind_method(D_METHOD("get_ray_samples_per_pixel"), &RTReflections::get_ray_samples_per_pixel);
    ClassDB::bind_method(D_METHOD("set_glossy_roughness_spread", "spread"), &RTReflections::set_glossy_roughness_spread);
    ClassDB::bind_method(D_METHOD("get_glossy_roughness_spread"), &RTReflections::get_glossy_roughness_spread);
    ClassDB::bind_method(D_METHOD("set_max_bounces", "bounces"), &RTReflections::set_max_bounces);
    ClassDB::bind_method(D_METHOD("get_max_bounces"), &RTReflections::get_max_bounces);
    ClassDB::bind_method(D_METHOD("set_max_ray_distance", "distance"), &RTReflections::set_max_ray_distance);
    ClassDB::bind_method(D_METHOD("get_max_ray_distance"), &RTReflections::get_max_ray_distance);
    ClassDB::bind_method(D_METHOD("set_bounce_distance_falloff", "falloff"), &RTReflections::set_bounce_distance_falloff);
    ClassDB::bind_method(D_METHOD("get_bounce_distance_falloff"), &RTReflections::get_bounce_distance_falloff);
    ClassDB::bind_method(D_METHOD("set_bounce_throughput_threshold", "threshold"), &RTReflections::set_bounce_throughput_threshold);
    ClassDB::bind_method(D_METHOD("get_bounce_throughput_threshold"), &RTReflections::get_bounce_throughput_threshold);
    ClassDB::bind_method(D_METHOD("set_bounce_local_lights", "enable"), &RTReflections::set_bounce_local_lights);
    ClassDB::bind_method(D_METHOD("get_bounce_local_lights"), &RTReflections::get_bounce_local_lights);
    ClassDB::bind_method(D_METHOD("set_pbr_texture_bounces", "bounces"), &RTReflections::set_pbr_texture_bounces);
    ClassDB::bind_method(D_METHOD("get_pbr_texture_bounces"), &RTReflections::get_pbr_texture_bounces);
    ClassDB::bind_method(D_METHOD("set_normal_maps_in_bounces", "enable"), &RTReflections::set_normal_maps_in_bounces);
    ClassDB::bind_method(D_METHOD("get_normal_maps_in_bounces"), &RTReflections::get_normal_maps_in_bounces);
    ClassDB::bind_method(D_METHOD("set_respect_material_cull_mode", "enable"), &RTReflections::set_respect_material_cull_mode);
    ClassDB::bind_method(D_METHOD("get_respect_material_cull_mode"), &RTReflections::get_respect_material_cull_mode);
    ClassDB::bind_method(D_METHOD("set_skinned_meshes_in_reflections", "enable"), &RTReflections::set_skinned_meshes_in_reflections);
    ClassDB::bind_method(D_METHOD("get_skinned_meshes_in_reflections"), &RTReflections::get_skinned_meshes_in_reflections);
    ClassDB::bind_method(D_METHOD("set_final_bounce_env_fallback", "enable"), &RTReflections::set_final_bounce_env_fallback);
    ClassDB::bind_method(D_METHOD("get_final_bounce_env_fallback"), &RTReflections::get_final_bounce_env_fallback);
    ClassDB::bind_method(D_METHOD("set_ray_origin_bias", "bias"), &RTReflections::set_ray_origin_bias);
    ClassDB::bind_method(D_METHOD("get_ray_origin_bias"), &RTReflections::get_ray_origin_bias);

    // Custom Sky
    ClassDB::bind_method(D_METHOD("set_custom_sky_mode", "mode"), &RTReflections::set_custom_sky_mode);
    ClassDB::bind_method(D_METHOD("get_custom_sky_mode"), &RTReflections::get_custom_sky_mode);
    ClassDB::bind_method(D_METHOD("set_custom_sky", "sky"), &RTReflections::set_custom_sky);
    ClassDB::bind_method(D_METHOD("get_custom_sky"), &RTReflections::get_custom_sky);
    ClassDB::bind_method(D_METHOD("set_custom_sky_color", "color"), &RTReflections::set_custom_sky_color);
    ClassDB::bind_method(D_METHOD("get_custom_sky_color"), &RTReflections::get_custom_sky_color);
    ClassDB::bind_method(D_METHOD("set_custom_sky_energy", "energy"), &RTReflections::set_custom_sky_energy);
    ClassDB::bind_method(D_METHOD("get_custom_sky_energy"), &RTReflections::get_custom_sky_energy);
    ClassDB::bind_method(D_METHOD("set_custom_sky_rotation", "rotation_degrees"), &RTReflections::set_custom_sky_rotation);
    ClassDB::bind_method(D_METHOD("get_custom_sky_rotation"), &RTReflections::get_custom_sky_rotation);
    ClassDB::bind_method(D_METHOD("set_sky_reflection_energy", "energy"), &RTReflections::set_sky_reflection_energy);
    ClassDB::bind_method(D_METHOD("get_sky_reflection_energy"), &RTReflections::get_sky_reflection_energy);

    // Ambient Light & Radiance
    ClassDB::bind_method(D_METHOD("set_ambient_light_mode", "mode"), &RTReflections::set_ambient_light_mode);
    ClassDB::bind_method(D_METHOD("get_ambient_light_mode"), &RTReflections::get_ambient_light_mode);
    ClassDB::bind_method(D_METHOD("set_ambient_light_color", "color"), &RTReflections::set_ambient_light_color);
    ClassDB::bind_method(D_METHOD("get_ambient_light_color"), &RTReflections::get_ambient_light_color);
    ClassDB::bind_method(D_METHOD("set_ambient_light_energy", "energy"), &RTReflections::set_ambient_light_energy);
    ClassDB::bind_method(D_METHOD("get_ambient_light_energy"), &RTReflections::get_ambient_light_energy);
    ClassDB::bind_method(D_METHOD("set_ambient_light_sky_contribution", "contribution"), &RTReflections::set_ambient_light_sky_contribution);
    ClassDB::bind_method(D_METHOD("get_ambient_light_sky_contribution"), &RTReflections::get_ambient_light_sky_contribution);
    ClassDB::bind_method(D_METHOD("set_ambient_radiance_boost", "boost"), &RTReflections::set_ambient_radiance_boost);
    ClassDB::bind_method(D_METHOD("get_ambient_radiance_boost"), &RTReflections::get_ambient_radiance_boost);
    ClassDB::bind_method(D_METHOD("set_ambient_bounce_approximation", "strength"), &RTReflections::set_ambient_bounce_approximation);
    ClassDB::bind_method(D_METHOD("get_ambient_bounce_approximation"), &RTReflections::get_ambient_bounce_approximation);
    ClassDB::bind_method(D_METHOD("set_ambient_occlusion_samples", "samples"), &RTReflections::set_ambient_occlusion_samples);
    ClassDB::bind_method(D_METHOD("get_ambient_occlusion_samples"), &RTReflections::get_ambient_occlusion_samples);
    ClassDB::bind_method(D_METHOD("set_ambient_occlusion_radius", "radius"), &RTReflections::set_ambient_occlusion_radius);
    ClassDB::bind_method(D_METHOD("get_ambient_occlusion_radius"), &RTReflections::get_ambient_occlusion_radius);
    ClassDB::bind_method(D_METHOD("set_ambient_occlusion_intensity", "intensity"), &RTReflections::set_ambient_occlusion_intensity);
    ClassDB::bind_method(D_METHOD("get_ambient_occlusion_intensity"), &RTReflections::get_ambient_occlusion_intensity);

    // Material Alpha
    ClassDB::bind_method(D_METHOD("set_alpha_consideration", "enable"), &RTReflections::set_alpha_consideration);
    ClassDB::bind_method(D_METHOD("get_alpha_consideration"), &RTReflections::get_alpha_consideration);
    ClassDB::bind_method(D_METHOD("set_alpha_ray_transparency", "enable"), &RTReflections::set_alpha_ray_transparency);
    ClassDB::bind_method(D_METHOD("get_alpha_ray_transparency"), &RTReflections::get_alpha_ray_transparency);
    ClassDB::bind_method(D_METHOD("set_alpha_max_steps", "steps"), &RTReflections::set_alpha_max_steps);
    ClassDB::bind_method(D_METHOD("get_alpha_max_steps"), &RTReflections::get_alpha_max_steps);
    ClassDB::bind_method(D_METHOD("set_alpha_shadow_transmission", "enable"), &RTReflections::set_alpha_shadow_transmission);
    ClassDB::bind_method(D_METHOD("get_alpha_shadow_transmission"), &RTReflections::get_alpha_shadow_transmission);
    ClassDB::bind_method(D_METHOD("set_transparent_receivers", "enable"), &RTReflections::set_transparent_receivers);
    ClassDB::bind_method(D_METHOD("get_transparent_receivers"), &RTReflections::get_transparent_receivers);
    ClassDB::bind_method(D_METHOD("set_transparent_receiver_min_alpha", "alpha"), &RTReflections::set_transparent_receiver_min_alpha);
    ClassDB::bind_method(D_METHOD("get_transparent_receiver_min_alpha"), &RTReflections::get_transparent_receiver_min_alpha);
    ClassDB::bind_method(D_METHOD("set_transparent_fresnel_boost", "boost"), &RTReflections::set_transparent_fresnel_boost);
    ClassDB::bind_method(D_METHOD("get_transparent_fresnel_boost"), &RTReflections::get_transparent_fresnel_boost);

    // Mesh Simplification
    ClassDB::bind_method(D_METHOD("set_mesh_simplification_enabled", "enable"), &RTReflections::set_mesh_simplification_enabled);
    ClassDB::bind_method(D_METHOD("is_mesh_simplification_enabled"), &RTReflections::is_mesh_simplification_enabled);
    ClassDB::bind_method(D_METHOD("set_mesh_simplification_mode", "mode"), &RTReflections::set_mesh_simplification_mode);
    ClassDB::bind_method(D_METHOD("get_mesh_simplification_mode"), &RTReflections::get_mesh_simplification_mode);
    ClassDB::bind_method(D_METHOD("set_mesh_simplification_ratio", "ratio"), &RTReflections::set_mesh_simplification_ratio);
    ClassDB::bind_method(D_METHOD("get_mesh_simplification_ratio"), &RTReflections::get_mesh_simplification_ratio);
    ClassDB::bind_method(D_METHOD("set_mesh_simplification_min_triangles", "min_triangles"), &RTReflections::set_mesh_simplification_min_triangles);
    ClassDB::bind_method(D_METHOD("get_mesh_simplification_min_triangles"), &RTReflections::get_mesh_simplification_min_triangles);
    ClassDB::bind_method(D_METHOD("set_mesh_simplification_max_error", "max_error"), &RTReflections::set_mesh_simplification_max_error);
    ClassDB::bind_method(D_METHOD("get_mesh_simplification_max_error"), &RTReflections::get_mesh_simplification_max_error);
    ClassDB::bind_method(D_METHOD("set_mesh_simplification_preserve_seams", "preserve"), &RTReflections::set_mesh_simplification_preserve_seams);
    ClassDB::bind_method(D_METHOD("get_mesh_simplification_preserve_seams"), &RTReflections::get_mesh_simplification_preserve_seams);
    ClassDB::bind_method(D_METHOD("set_mesh_simplification_skinned", "enable"), &RTReflections::set_mesh_simplification_skinned);
    ClassDB::bind_method(D_METHOD("get_mesh_simplification_skinned"), &RTReflections::get_mesh_simplification_skinned);
    ClassDB::bind_method(D_METHOD("set_mesh_simplification_gbuffer", "enable"), &RTReflections::set_mesh_simplification_gbuffer);
    ClassDB::bind_method(D_METHOD("get_mesh_simplification_gbuffer"), &RTReflections::get_mesh_simplification_gbuffer);

    // Shadows
    ClassDB::bind_method(D_METHOD("set_trace_shadows", "enable"), &RTReflections::set_trace_shadows);
    ClassDB::bind_method(D_METHOD("get_trace_shadows"), &RTReflections::get_trace_shadows);
    ClassDB::bind_method(D_METHOD("set_shadow_bounces", "bounces"), &RTReflections::set_shadow_bounces);
    ClassDB::bind_method(D_METHOD("get_shadow_bounces"), &RTReflections::get_shadow_bounces);
    ClassDB::bind_method(D_METHOD("set_shadow_max_distance", "distance"), &RTReflections::set_shadow_max_distance);
    ClassDB::bind_method(D_METHOD("get_shadow_max_distance"), &RTReflections::get_shadow_max_distance);
    ClassDB::bind_method(D_METHOD("set_shadow_ray_bias", "bias"), &RTReflections::set_shadow_ray_bias);
    ClassDB::bind_method(D_METHOD("get_shadow_ray_bias"), &RTReflections::get_shadow_ray_bias);

    // Scalability & Texture Pool
    ClassDB::bind_method(D_METHOD("set_update_interval", "interval"), &RTReflections::set_update_interval);
    ClassDB::bind_method(D_METHOD("get_update_interval"), &RTReflections::get_update_interval);
    ClassDB::bind_method(D_METHOD("set_blas_builds_per_frame", "budget"), &RTReflections::set_blas_builds_per_frame);
    ClassDB::bind_method(D_METHOD("get_blas_builds_per_frame"), &RTReflections::get_blas_builds_per_frame);
    ClassDB::bind_method(D_METHOD("set_max_tlas_instances", "max_instances"), &RTReflections::set_max_tlas_instances);
    ClassDB::bind_method(D_METHOD("get_max_tlas_instances"), &RTReflections::get_max_tlas_instances);
    ClassDB::bind_method(D_METHOD("set_frustum_cull_gbuffer", "enable"), &RTReflections::set_frustum_cull_gbuffer);
    ClassDB::bind_method(D_METHOD("get_frustum_cull_gbuffer"), &RTReflections::get_frustum_cull_gbuffer);
    ClassDB::bind_method(D_METHOD("set_texture_array_resolution", "resolution"), &RTReflections::set_texture_array_resolution);
    ClassDB::bind_method(D_METHOD("get_texture_array_resolution"), &RTReflections::get_texture_array_resolution);
    ClassDB::bind_method(D_METHOD("set_max_texture_layers", "layers"), &RTReflections::set_max_texture_layers);
    ClassDB::bind_method(D_METHOD("get_max_texture_layers"), &RTReflections::get_max_texture_layers);

    // LOD & Culling
    ClassDB::bind_method(D_METHOD("set_texture_lod_bias", "bias"), &RTReflections::set_texture_lod_bias);
    ClassDB::bind_method(D_METHOD("get_texture_lod_bias"), &RTReflections::get_texture_lod_bias);
    ClassDB::bind_method(D_METHOD("set_distance_lod_strength", "strength"), &RTReflections::set_distance_lod_strength);
    ClassDB::bind_method(D_METHOD("get_distance_lod_strength"), &RTReflections::get_distance_lod_strength);
    ClassDB::bind_method(D_METHOD("set_mesh_cull_distance", "distance"), &RTReflections::set_mesh_cull_distance);
    ClassDB::bind_method(D_METHOD("get_mesh_cull_distance"), &RTReflections::get_mesh_cull_distance);
    ClassDB::bind_method(D_METHOD("set_mesh_cull_min_angle", "angle"), &RTReflections::set_mesh_cull_min_angle);
    ClassDB::bind_method(D_METHOD("get_mesh_cull_min_angle"), &RTReflections::get_mesh_cull_min_angle);
    ClassDB::bind_method(D_METHOD("set_light_cull_distance", "distance"), &RTReflections::set_light_cull_distance);
    ClassDB::bind_method(D_METHOD("get_light_cull_distance"), &RTReflections::get_light_cull_distance);
    ClassDB::bind_method(D_METHOD("set_reflection_cull_mask", "mask"), &RTReflections::set_reflection_cull_mask);
    ClassDB::bind_method(D_METHOD("get_reflection_cull_mask"), &RTReflections::get_reflection_cull_mask);
    ClassDB::bind_method(D_METHOD("set_reflection_viewing_cull_mask", "mask"), &RTReflections::set_reflection_viewing_cull_mask);
    ClassDB::bind_method(D_METHOD("get_reflection_viewing_cull_mask"), &RTReflections::get_reflection_viewing_cull_mask);

    ClassDB::bind_method(D_METHOD("apply_quality_preset", "preset"), &RTReflections::apply_quality_preset);
    ClassDB::bind_method(D_METHOD("get_reflection_texture"), &RTReflections::get_reflection_texture);
    ClassDB::bind_method(D_METHOD("mark_geometry_dirty"), &RTReflections::mark_geometry_dirty);
    ClassDB::bind_method(D_METHOD("_on_frame_pre_draw"), &RTReflections::_on_frame_pre_draw);

    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_preview"), "set_editor_preview", "get_editor_preview");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_composite"), "set_auto_composite", "get_auto_composite");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_attach_materials"), "set_auto_attach_materials", "get_auto_attach_materials");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "reflection_intensity", PROPERTY_HINT_RANGE, "0.0,4.0,0.05"), "set_reflection_intensity", "get_reflection_intensity");

    ADD_GROUP("Resolution & Filtering", "");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "reflection_scale", PROPERTY_HINT_RANGE, "0.15,1.0,0.05"), "set_reflection_scale", "get_reflection_scale");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "reflection_filtering", PROPERTY_HINT_ENUM, "Bilinear (Fast),Bicubic (High)"), "set_reflection_filtering", "get_reflection_filtering");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bilateral_upsample"), "set_bilateral_upsample", "get_bilateral_upsample");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "temporal_accumulation"), "set_temporal_accumulation", "get_temporal_accumulation");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "temporal_blend", PROPERTY_HINT_RANGE, "0.02,1.0,0.02"), "set_temporal_blend", "get_temporal_blend");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "checkerboard_tracing"), "set_checkerboard_tracing", "get_checkerboard_tracing");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "spatial_denoise_passes", PROPERTY_HINT_RANGE, "0,5,1"), "set_spatial_denoise_passes", "get_spatial_denoise_passes");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_roughness", PROPERTY_HINT_RANGE, "0.05,1.0,0.05"), "set_max_roughness", "get_max_roughness");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "roughness_fade_range", PROPERTY_HINT_RANGE, "0.0,0.5,0.01"), "set_roughness_fade_range", "get_roughness_fade_range");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "blur_strength", PROPERTY_HINT_RANGE, "0.0,4.0,0.05"), "set_blur_strength", "get_blur_strength");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "blur_lod_scale", PROPERTY_HINT_RANGE, "0.0,8.0,0.25"), "set_blur_lod_scale", "get_blur_lod_scale");

    ADD_GROUP("Tracing & Bounces", "");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "ray_samples_per_pixel", PROPERTY_HINT_RANGE, "1,8,1"), "set_ray_samples_per_pixel", "get_ray_samples_per_pixel");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "glossy_roughness_spread", PROPERTY_HINT_RANGE, "0.0,1.0,0.05"), "set_glossy_roughness_spread", "get_glossy_roughness_spread");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_bounces", PROPERTY_HINT_RANGE, "1,4,1"), "set_max_bounces", "get_max_bounces");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_ray_distance", PROPERTY_HINT_RANGE, "10.0,2000.0,5.0"), "set_max_ray_distance", "get_max_ray_distance");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce_distance_falloff", PROPERTY_HINT_RANGE, "0.1,1.0,0.05"), "set_bounce_distance_falloff", "get_bounce_distance_falloff");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce_throughput_threshold", PROPERTY_HINT_RANGE, "0.0,0.25,0.005"), "set_bounce_throughput_threshold", "get_bounce_throughput_threshold");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bounce_local_lights"), "set_bounce_local_lights", "get_bounce_local_lights");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "pbr_texture_bounces", PROPERTY_HINT_RANGE, "0,4,1"), "set_pbr_texture_bounces", "get_pbr_texture_bounces");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "normal_maps_in_bounces"), "set_normal_maps_in_bounces", "get_normal_maps_in_bounces");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "respect_material_cull_mode"), "set_respect_material_cull_mode", "get_respect_material_cull_mode");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "skinned_meshes_in_reflections"), "set_skinned_meshes_in_reflections", "get_skinned_meshes_in_reflections");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "final_bounce_env_fallback"), "set_final_bounce_env_fallback", "get_final_bounce_env_fallback");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "ray_origin_bias", PROPERTY_HINT_RANGE, "0.0005,0.05,0.0005"), "set_ray_origin_bias", "get_ray_origin_bias");

    ADD_GROUP("Custom Reflection Sky", "");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "custom_sky_mode", PROPERTY_HINT_ENUM, "Environment,Custom Sky,Color"), "set_custom_sky_mode", "get_custom_sky_mode");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "custom_sky", PROPERTY_HINT_RESOURCE_TYPE, "Sky"), "set_custom_sky", "get_custom_sky");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR, "custom_sky_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_custom_sky_color", "get_custom_sky_color");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "custom_sky_energy", PROPERTY_HINT_RANGE, "0.0,16.0,0.05"), "set_custom_sky_energy", "get_custom_sky_energy");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "custom_sky_rotation"), "set_custom_sky_rotation", "get_custom_sky_rotation");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "sky_reflection_energy", PROPERTY_HINT_RANGE, "0.0,8.0,0.05"), "set_sky_reflection_energy", "get_sky_reflection_energy");

    ADD_GROUP("Ambient Light & Radiance", "");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "ambient_light_mode", PROPERTY_HINT_ENUM, "Environment,Custom,Disabled"), "set_ambient_light_mode", "get_ambient_light_mode");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ambient_light_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_ambient_light_color", "get_ambient_light_color");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "ambient_light_energy", PROPERTY_HINT_RANGE, "0.0,16.0,0.05"), "set_ambient_light_energy", "get_ambient_light_energy");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "ambient_light_sky_contribution", PROPERTY_HINT_RANGE, "0.0,1.0,0.02"), "set_ambient_light_sky_contribution", "get_ambient_light_sky_contribution");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "ambient_radiance_boost", PROPERTY_HINT_RANGE, "0.0,4.0,0.05"), "set_ambient_radiance_boost", "get_ambient_radiance_boost");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "ambient_bounce_approximation", PROPERTY_HINT_RANGE, "0.0,1.0,0.05"), "set_ambient_bounce_approximation", "get_ambient_bounce_approximation");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "ambient_occlusion_samples", PROPERTY_HINT_RANGE, "0,4,1"), "set_ambient_occlusion_samples", "get_ambient_occlusion_samples");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "ambient_occlusion_radius", PROPERTY_HINT_RANGE, "0.1,10.0,0.1"), "set_ambient_occlusion_radius", "get_ambient_occlusion_radius");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "ambient_occlusion_intensity", PROPERTY_HINT_RANGE, "0.0,2.0,0.05"), "set_ambient_occlusion_intensity", "get_ambient_occlusion_intensity");

    ADD_GROUP("Material Alpha", "");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alpha_consideration"), "set_alpha_consideration", "get_alpha_consideration");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alpha_ray_transparency"), "set_alpha_ray_transparency", "get_alpha_ray_transparency");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "alpha_max_steps", PROPERTY_HINT_RANGE, "1,16,1"), "set_alpha_max_steps", "get_alpha_max_steps");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alpha_shadow_transmission"), "set_alpha_shadow_transmission", "get_alpha_shadow_transmission");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "transparent_receivers"), "set_transparent_receivers", "get_transparent_receivers");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "transparent_receiver_min_alpha", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_transparent_receiver_min_alpha", "get_transparent_receiver_min_alpha");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "transparent_fresnel_boost", PROPERTY_HINT_RANGE, "0.0,1.0,0.05"), "set_transparent_fresnel_boost", "get_transparent_fresnel_boost");

    ADD_GROUP("Mesh Simplification", "");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mesh_simplification_enabled"), "set_mesh_simplification_enabled", "is_mesh_simplification_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "mesh_simplification_mode", PROPERTY_HINT_ENUM, "QEM Edge Collapse (High Quality),Vertex Clustering (Fast)"), "set_mesh_simplification_mode", "get_mesh_simplification_mode");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "mesh_simplification_ratio", PROPERTY_HINT_RANGE, "0.05,1.0,0.05"), "set_mesh_simplification_ratio", "get_mesh_simplification_ratio");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "mesh_simplification_min_triangles", PROPERTY_HINT_RANGE, "0,10000,16"), "set_mesh_simplification_min_triangles", "get_mesh_simplification_min_triangles");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "mesh_simplification_max_error", PROPERTY_HINT_RANGE, "0.0,1.0,0.005"), "set_mesh_simplification_max_error", "get_mesh_simplification_max_error");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mesh_simplification_preserve_seams"), "set_mesh_simplification_preserve_seams", "get_mesh_simplification_preserve_seams");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mesh_simplification_skinned"), "set_mesh_simplification_skinned", "get_mesh_simplification_skinned");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mesh_simplification_gbuffer"), "set_mesh_simplification_gbuffer", "get_mesh_simplification_gbuffer");

    ADD_GROUP("Shadows", "");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "trace_shadows"), "set_trace_shadows", "get_trace_shadows");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_bounces", PROPERTY_HINT_RANGE, "0,4,1"), "set_shadow_bounces", "get_shadow_bounces");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "shadow_max_distance", PROPERTY_HINT_RANGE, "0.0,500.0,1.0"), "set_shadow_max_distance", "get_shadow_max_distance");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "shadow_ray_bias", PROPERTY_HINT_RANGE, "0.0005,0.05,0.0005"), "set_shadow_ray_bias", "get_shadow_ray_bias");

    ADD_GROUP("Scalability & Textures", "");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "update_interval", PROPERTY_HINT_RANGE, "1,10,1"), "set_update_interval", "get_update_interval");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "blas_builds_per_frame", PROPERTY_HINT_RANGE, "0,64,1"), "set_blas_builds_per_frame", "get_blas_builds_per_frame");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tlas_instances", PROPERTY_HINT_RANGE, "0,8192,16"), "set_max_tlas_instances", "get_max_tlas_instances");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "frustum_cull_gbuffer"), "set_frustum_cull_gbuffer", "get_frustum_cull_gbuffer");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_array_resolution", PROPERTY_HINT_ENUM, "128px:128,256px:256,512px:512,1024px:1024"), "set_texture_array_resolution", "get_texture_array_resolution");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_texture_layers", PROPERTY_HINT_RANGE, "8,256,8"), "set_max_texture_layers", "get_max_texture_layers");

    ADD_GROUP("LOD & Culling", "");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "reflection_cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_reflection_cull_mask", "get_reflection_cull_mask");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "reflection_viewing_cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_reflection_viewing_cull_mask", "get_reflection_viewing_cull_mask");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "texture_lod_bias", PROPERTY_HINT_RANGE, "-2.0,6.0,0.1"), "set_texture_lod_bias", "get_texture_lod_bias");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "distance_lod_strength", PROPERTY_HINT_RANGE, "0.0,4.0,0.1"), "set_distance_lod_strength", "get_distance_lod_strength");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "mesh_cull_distance", PROPERTY_HINT_RANGE, "5.0,500.0,5.0"), "set_mesh_cull_distance", "get_mesh_cull_distance");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "mesh_cull_min_angle", PROPERTY_HINT_RANGE, "0.0,0.1,0.002"), "set_mesh_cull_min_angle", "get_mesh_cull_min_angle");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "light_cull_distance", PROPERTY_HINT_RANGE, "10.0,500.0,5.0"), "set_light_cull_distance", "get_light_cull_distance");

    BIND_ENUM_CONSTANT(FILTER_BILINEAR);
    BIND_ENUM_CONSTANT(FILTER_BICUBIC);
    BIND_ENUM_CONSTANT(PRESET_LOW);
    BIND_ENUM_CONSTANT(PRESET_MEDIUM);
    BIND_ENUM_CONSTANT(PRESET_HIGH);
    BIND_ENUM_CONSTANT(PRESET_ULTRA);
    BIND_ENUM_CONSTANT(SKY_MODE_ENVIRONMENT);
    BIND_ENUM_CONSTANT(SKY_MODE_CUSTOM_SKY);
    BIND_ENUM_CONSTANT(SKY_MODE_COLOR);
    BIND_ENUM_CONSTANT(AMBIENT_MODE_ENVIRONMENT);
    BIND_ENUM_CONSTANT(AMBIENT_MODE_CUSTOM);
    BIND_ENUM_CONSTANT(AMBIENT_MODE_DISABLED);
    BIND_ENUM_CONSTANT(SIMPLIFY_QEM);
    BIND_ENUM_CONSTANT(SIMPLIFY_VERTEX_CLUSTERING);
    BIND_ENUM_CONSTANT(TEX_RES_128);
    BIND_ENUM_CONSTANT(TEX_RES_256);
    BIND_ENUM_CONSTANT(TEX_RES_512);
    BIND_ENUM_CONSTANT(TEX_RES_1024);
}

void RTReflections::_validate_property(PropertyInfo &property) const {
    if (property.name == "custom_sky" && custom_sky_mode != SKY_MODE_CUSTOM_SKY) {
        property.usage = PROPERTY_USAGE_NOEDITOR;
    } else if (property.name == "custom_sky_color" && custom_sky_mode != SKY_MODE_COLOR) {
        property.usage = PROPERTY_USAGE_NOEDITOR;
    } else if ((property.name == "custom_sky_energy" || property.name == "custom_sky_rotation") && custom_sky_mode == SKY_MODE_ENVIRONMENT) {
        property.usage = PROPERTY_USAGE_NOEDITOR;
    } else if ((property.name == "ambient_light_color" || property.name == "ambient_light_energy" || property.name == "ambient_light_sky_contribution") && ambient_light_mode != AMBIENT_MODE_CUSTOM) {
        property.usage = PROPERTY_USAGE_NOEDITOR;
    } else if (property.name == "temporal_blend" && !temporal_accumulation) {
        property.usage = PROPERTY_USAGE_NOEDITOR;
    } else if (!alpha_consideration &&
               (property.name == "alpha_ray_transparency" ||
                property.name == "alpha_max_steps" ||
                property.name == "alpha_shadow_transmission" ||
                property.name == "transparent_receivers" ||
                property.name == "transparent_receiver_min_alpha" ||
                property.name == "transparent_fresnel_boost")) {
        property.usage = PROPERTY_USAGE_NOEDITOR;
    } else if (!mesh_simplification_enabled &&
               (property.name == "mesh_simplification_mode" ||
                property.name == "mesh_simplification_ratio" ||
                property.name == "mesh_simplification_min_triangles" ||
                property.name == "mesh_simplification_max_error" ||
                property.name == "mesh_simplification_preserve_seams" ||
                property.name == "mesh_simplification_skinned" ||
                property.name == "mesh_simplification_gbuffer")) {
        property.usage = PROPERTY_USAGE_NOEDITOR;
    }
}

void RTReflections::_register_scene_hook() {
    if (hook_registered) {
        return;
    }
    GL43EffectHelpers::register_scene_hook(this, GL43EffectHelpers::HOOK_PRE_SCREEN_TEXTURE, 20, [this](const GL43EffectHelpers::SceneHookContext &p_ctx) {
        _on_scene_hook(p_ctx);
    });
    hook_registered = true;
}

void RTReflections::_unregister_scene_hook() {
    if (!hook_registered) {
        return;
    }
    GL43EffectHelpers::unregister_scene_hooks(this);
    hook_registered = false;
}

void RTReflections::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE: {
            _register_scene_hook();
            set_process(true);
            bvh_dirty = true;
            sky_sh_dirty = true;
            if (VisualServer::get_singleton() && !VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
                VisualServer::get_singleton()->connect("frame_pre_draw", this, "_on_frame_pre_draw");
            }
        } break;

        case NOTIFICATION_EXIT_TREE: {
            if (VisualServer::get_singleton() && VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
                VisualServer::get_singleton()->disconnect("frame_pre_draw", this, "_on_frame_pre_draw");
            }
            clear_attached_passes();
            _unregister_scene_hook();
        } break;

        case NOTIFICATION_PROCESS: {
            if (!enabled) {
                clear_attached_passes();
                MutexLock lock(scene_mutex);
                snapshot.valid = false;
                return;
            }
            if (Engine::get_singleton()->is_editor_hint()) return;
            process_frame_counter++;
            if (update_interval > 1 && (process_frame_counter % (uint64_t)update_interval) != 0) return;
            collect_scene_snapshot();
        } break;
    }
}

GL43Raytracer::SimplificationConfig RTReflections::_get_simplification_config() const {
    GL43Raytracer::SimplificationConfig cfg;
    cfg.enabled = mesh_simplification_enabled;
    cfg.mode = (mesh_simplification_mode == SIMPLIFY_VERTEX_CLUSTERING)
        ? GL43Raytracer::SIMPLIFY_VERTEX_CLUSTERING
        : GL43Raytracer::SIMPLIFY_QEM;
    cfg.ratio = mesh_simplification_ratio;
    cfg.min_triangles = mesh_simplification_min_triangles;
    cfg.max_error = mesh_simplification_max_error;
    cfg.preserve_seams = mesh_simplification_preserve_seams;
    cfg.simplify_skinned = mesh_simplification_skinned;
    cfg.simplify_gbuffer = mesh_simplification_gbuffer;
    return cfg;
}

void RTReflections::invalidate_blas_cache() {
    bvh_dirty = true;
    raytracer.invalidate_blas_cache();
}

void RTReflections::set_enabled(bool p_enabled) {
    enabled = p_enabled;
    if (!enabled) {
        clear_attached_passes();
        history_valid = false;
    }
}

void RTReflections::set_editor_preview(bool p_preview) {
    editor_preview = p_preview;
    if (!editor_preview && Engine::get_singleton()->is_editor_hint()) {
        clear_attached_passes();
        history_valid = false;
    }
}

void RTReflections::set_auto_composite(bool p_auto) {
    auto_composite = p_auto;
}

void RTReflections::set_auto_attach_materials(bool p_auto) {
    auto_attach_materials = p_auto;
    if (!auto_attach_materials) clear_attached_passes();
}

void RTReflections::set_reflection_intensity(float p_intensity) {
    reflection_intensity = MAX(p_intensity, 0.0f);
}

void RTReflections::set_reflection_scale(float p_scale) {
    float clamped = CLAMP(p_scale, 0.15f, 1.0f);
    if (std::abs(clamped - reflection_scale) > 0.001f) {
        reflection_scale = clamped;
        history_valid = false;
    }
}

void RTReflections::set_temporal_accumulation(bool p_enable) {
    if (temporal_accumulation != p_enable) {
        temporal_accumulation = p_enable;
        history_valid = false;
        _change_notify();
    }
}

void RTReflections::set_temporal_blend(float p_blend) {
    temporal_blend = CLAMP(p_blend, 0.02f, 1.0f);
}

void RTReflections::set_spatial_denoise_passes(int p_passes) {
    spatial_denoise_passes = CLAMP(p_passes, 0, 5);
}

void RTReflections::set_max_roughness(float p_rough) {
    max_roughness = CLAMP(p_rough, 0.05f, 1.0f);
}

void RTReflections::set_roughness_fade_range(float p_range) {
    roughness_fade_range = CLAMP(p_range, 0.0f, 0.5f);
}

void RTReflections::set_blur_strength(float p_strength) {
    blur_strength = MAX(p_strength, 0.0f);
}

void RTReflections::set_blur_lod_scale(float p_scale) {
    blur_lod_scale = CLAMP(p_scale, 0.0f, 8.0f);
}

void RTReflections::set_ray_samples_per_pixel(int p_samples) {
    ray_samples_per_pixel = CLAMP(p_samples, 1, 8);
}

void RTReflections::set_glossy_roughness_spread(float p_spread) {
    glossy_roughness_spread = CLAMP(p_spread, 0.0f, 1.0f);
}

void RTReflections::set_max_bounces(int p_bounces) {
    max_bounces = CLAMP(p_bounces, 1, 4);
}

void RTReflections::set_max_ray_distance(float p_dist) {
    max_ray_distance = MAX(p_dist, 1.0f);
}

void RTReflections::set_bounce_distance_falloff(float p_falloff) {
    bounce_distance_falloff = CLAMP(p_falloff, 0.1f, 1.0f);
}

void RTReflections::set_bounce_throughput_threshold(float p_thresh) {
    bounce_throughput_threshold = CLAMP(p_thresh, 0.0f, 0.25f);
}

void RTReflections::set_pbr_texture_bounces(int p_bounces) {
    pbr_texture_bounces = CLAMP(p_bounces, 0, 4);
}

void RTReflections::set_ray_origin_bias(float p_bias) {
    ray_origin_bias = CLAMP(p_bias, 0.0005f, 0.05f);
}

void RTReflections::set_custom_sky_mode(CustomSkyMode p_mode) {
    custom_sky_mode = p_mode;
    sky_sh_dirty = true;
    _change_notify();
}

void RTReflections::set_custom_sky(const Ref<Sky> &p_sky) {
    custom_sky = p_sky;
    sky_sh_dirty = true;
}

void RTReflections::set_custom_sky_color(const Color &p_color) {
    custom_sky_color = p_color;
    sky_sh_dirty = true;
}

void RTReflections::set_custom_sky_energy(float p_energy) {
    custom_sky_energy = CLAMP(p_energy, 0.0f, 16.0f);
    sky_sh_dirty = true;
}

void RTReflections::set_custom_sky_rotation(const Vector3 &p_rot_deg) {
    custom_sky_rotation = p_rot_deg;
}

void RTReflections::set_sky_reflection_energy(float p_energy) {
    sky_reflection_energy = CLAMP(p_energy, 0.0f, 8.0f);
}

void RTReflections::set_ambient_light_mode(AmbientLightMode p_mode) {
    ambient_light_mode = p_mode;
    _change_notify();
}

void RTReflections::set_ambient_light_energy(float p_energy) {
    ambient_light_energy = CLAMP(p_energy, 0.0f, 16.0f);
}

void RTReflections::set_ambient_light_sky_contribution(float p_contrib) {
    ambient_light_sky_contribution = CLAMP(p_contrib, 0.0f, 1.0f);
}

void RTReflections::set_ambient_radiance_boost(float p_boost) {
    ambient_radiance_boost = CLAMP(p_boost, 0.0f, 4.0f);
}

void RTReflections::set_ambient_bounce_approximation(float p_approx) {
    ambient_bounce_approximation = CLAMP(p_approx, 0.0f, 1.0f);
}

void RTReflections::set_ambient_occlusion_samples(int p_samples) {
    ambient_occlusion_samples = CLAMP(p_samples, 0, 4);
}

void RTReflections::set_ambient_occlusion_radius(float p_radius) {
    ambient_occlusion_radius = CLAMP(p_radius, 0.1f, 10.0f);
}

void RTReflections::set_ambient_occlusion_intensity(float p_intensity) {
    ambient_occlusion_intensity = CLAMP(p_intensity, 0.0f, 2.0f);
}

void RTReflections::set_alpha_consideration(bool p_enable) {
    alpha_consideration = p_enable;
    _change_notify();
}

void RTReflections::set_alpha_max_steps(int p_steps) {
    alpha_max_steps = CLAMP(p_steps, 1, 16);
}

void RTReflections::set_transparent_receiver_min_alpha(float p_alpha) {
    transparent_receiver_min_alpha = CLAMP(p_alpha, 0.0f, 1.0f);
}

void RTReflections::set_transparent_fresnel_boost(float p_boost) {
    transparent_fresnel_boost = CLAMP(p_boost, 0.0f, 1.0f);
}

void RTReflections::set_mesh_simplification_enabled(bool p_enable) {
    mesh_simplification_enabled = p_enable;
    invalidate_blas_cache();
    _change_notify();
}

void RTReflections::set_mesh_simplification_mode(SimplificationMode p_mode) {
    mesh_simplification_mode = p_mode;
    invalidate_blas_cache();
}

void RTReflections::set_mesh_simplification_ratio(float p_ratio) {
    mesh_simplification_ratio = CLAMP(p_ratio, 0.05f, 1.0f);
    invalidate_blas_cache();
}

void RTReflections::set_mesh_simplification_min_triangles(int p_min_tris) {
    mesh_simplification_min_triangles = MAX(p_min_tris, 0);
    invalidate_blas_cache();
}

void RTReflections::set_mesh_simplification_max_error(float p_max_error) {
    mesh_simplification_max_error = CLAMP(p_max_error, 0.0f, 1.0f);
    invalidate_blas_cache();
}

void RTReflections::set_mesh_simplification_preserve_seams(bool p_preserve) {
    mesh_simplification_preserve_seams = p_preserve;
    invalidate_blas_cache();
}

void RTReflections::set_mesh_simplification_skinned(bool p_enable) {
    mesh_simplification_skinned = p_enable;
    invalidate_blas_cache();
}

void RTReflections::set_mesh_simplification_gbuffer(bool p_enable) {
    mesh_simplification_gbuffer = p_enable;
    invalidate_blas_cache();
}

void RTReflections::set_shadow_bounces(int p_bounces) {
    shadow_bounces = CLAMP(p_bounces, 0, 4);
}

void RTReflections::set_shadow_max_distance(float p_dist) {
    shadow_max_distance = MAX(p_dist, 0.0f);
}

void RTReflections::set_shadow_ray_bias(float p_bias) {
    shadow_ray_bias = CLAMP(p_bias, 0.0005f, 0.05f);
}

void RTReflections::set_update_interval(int p_interval) {
    update_interval = CLAMP(p_interval, 1, 10);
}

void RTReflections::set_blas_builds_per_frame(int p_budget) {
    blas_builds_per_frame = CLAMP(p_budget, 0, 64);
}

void RTReflections::set_max_tlas_instances(int p_max) {
    max_tlas_instances = CLAMP(p_max, 0, 8192);
}

void RTReflections::set_texture_array_resolution(TextureResolution p_res) {
    if (texture_array_resolution != p_res) {
        texture_array_resolution = p_res;
        tex_pool.mark_reset_needed();
    }
}

void RTReflections::set_max_texture_layers(int p_layers) {
    int clamped = CLAMP(p_layers, 8, 256);
    if (max_texture_layers != clamped) {
        max_texture_layers = clamped;
        tex_pool.mark_reset_needed();
    }
}

void RTReflections::set_texture_lod_bias(float p_bias) {
    texture_lod_bias = CLAMP(p_bias, -2.0f, 6.0f);
}

void RTReflections::set_distance_lod_strength(float p_strength) {
    distance_lod_strength = CLAMP(p_strength, 0.0f, 4.0f);
}

void RTReflections::set_mesh_cull_distance(float p_dist) {
    mesh_cull_distance = MAX(p_dist, 1.0f);
}

void RTReflections::set_mesh_cull_min_angle(float p_angle) {
    mesh_cull_min_angle = CLAMP(p_angle, 0.0f, 0.2f);
}

void RTReflections::set_light_cull_distance(float p_dist) {
    light_cull_distance = MAX(p_dist, 1.0f);
}

void RTReflections::set_reflection_cull_mask(uint32_t p_mask) {
    reflection_cull_mask = p_mask;
}

void RTReflections::set_reflection_viewing_cull_mask(uint32_t p_mask) {
    reflection_viewing_cull_mask = p_mask;
    if (reflection_viewing_cull_mask == 0) {
        clear_attached_passes();
    }
}

void RTReflections::apply_quality_preset(QualityPreset p_preset) {
    switch (p_preset) {
        case PRESET_LOW: {
            reflection_scale = 0.25f;
            reflection_filtering = FILTER_BILINEAR;
            bilateral_upsample = false;
            temporal_accumulation = true;
            temporal_blend = 0.25f;
            checkerboard_tracing = true;
            spatial_denoise_passes = 1;
            ray_samples_per_pixel = 1;
            glossy_roughness_spread = 0.0f;
            max_bounces = 1;
            max_ray_distance = 100.0f;
            bounce_distance_falloff = 0.4f;
            bounce_throughput_threshold = 0.05f;
            bounce_local_lights = false;
            pbr_texture_bounces = 0;
            normal_maps_in_bounces = false;
            skinned_meshes_in_reflections = false;
            final_bounce_env_fallback = false;
            ambient_occlusion_samples = 0;
            alpha_ray_transparency = false;
            alpha_shadow_transmission = false;
            mesh_simplification_enabled = true;
            mesh_simplification_mode = SIMPLIFY_VERTEX_CLUSTERING;
            mesh_simplification_ratio = 0.35f;
            mesh_simplification_min_triangles = 64;
            trace_shadows = false;
            shadow_bounces = 0;
            shadow_max_distance = 0.0f;
            texture_array_resolution = TEX_RES_256;
            max_texture_layers = 32;
            blas_builds_per_frame = 4;
            texture_lod_bias = 1.0f;
            distance_lod_strength = 1.5f;
            mesh_cull_distance = 25.0f;
            mesh_cull_min_angle = 0.025f;
            light_cull_distance = 60.0f;
        } break;
        case PRESET_MEDIUM: {
            reflection_scale = 0.5f;
            reflection_filtering = FILTER_BICUBIC;
            bilateral_upsample = true;
            temporal_accumulation = true;
            temporal_blend = 0.18f;
            checkerboard_tracing = false;
            spatial_denoise_passes = 2;
            ray_samples_per_pixel = 1;
            glossy_roughness_spread = 0.15f;
            max_bounces = 1;
            max_ray_distance = 150.0f;
            bounce_distance_falloff = 0.5f;
            bounce_throughput_threshold = 0.03f;
            bounce_local_lights = false;
            pbr_texture_bounces = 1;
            normal_maps_in_bounces = true;
            skinned_meshes_in_reflections = true;
            final_bounce_env_fallback = true;
            ambient_occlusion_samples = 1;
            alpha_ray_transparency = true;
            alpha_max_steps = 4;
            alpha_shadow_transmission = true;
            mesh_simplification_enabled = true;
            mesh_simplification_mode = SIMPLIFY_QEM;
            mesh_simplification_ratio = 0.65f;
            mesh_simplification_min_triangles = 256;
            trace_shadows = true;
            shadow_bounces = 1;
            shadow_max_distance = 60.0f;
            texture_array_resolution = TEX_RES_512;
            max_texture_layers = 64;
            blas_builds_per_frame = 8;
            texture_lod_bias = 0.0f;
            distance_lod_strength = 1.0f;
            mesh_cull_distance = 40.0f;
            mesh_cull_min_angle = 0.015f;
            light_cull_distance = 100.0f;
        } break;
        case PRESET_HIGH: {
            reflection_scale = 0.75f;
            reflection_filtering = FILTER_BICUBIC;
            bilateral_upsample = true;
            temporal_accumulation = true;
            temporal_blend = 0.12f;
            checkerboard_tracing = false;
            spatial_denoise_passes = 2;
            ray_samples_per_pixel = 1;
            glossy_roughness_spread = 0.35f;
            max_bounces = 2;
            max_ray_distance = 250.0f;
            bounce_distance_falloff = 0.55f;
            bounce_throughput_threshold = 0.02f;
            bounce_local_lights = true;
            pbr_texture_bounces = 2;
            normal_maps_in_bounces = true;
            skinned_meshes_in_reflections = true;
            final_bounce_env_fallback = true;
            ambient_occlusion_samples = 2;
            alpha_ray_transparency = true;
            alpha_max_steps = 6;
            alpha_shadow_transmission = true;
            mesh_simplification_enabled = false;
            trace_shadows = true;
            shadow_bounces = 1;
            shadow_max_distance = 100.0f;
            texture_array_resolution = TEX_RES_512;
            max_texture_layers = 96;
            blas_builds_per_frame = 0;
            texture_lod_bias = 0.0f;
            distance_lod_strength = 0.85f;
            mesh_cull_distance = 75.0f;
            mesh_cull_min_angle = 0.008f;
            light_cull_distance = 160.0f;
        } break;
        case PRESET_ULTRA: {
            reflection_scale = 1.0f;
            reflection_filtering = FILTER_BICUBIC;
            bilateral_upsample = true;
            temporal_accumulation = true;
            temporal_blend = 0.10f;
            checkerboard_tracing = false;
            spatial_denoise_passes = 3;
            ray_samples_per_pixel = 2;
            glossy_roughness_spread = 0.6f;
            max_bounces = 3;
            max_ray_distance = 500.0f;
            bounce_distance_falloff = 0.65f;
            bounce_throughput_threshold = 0.01f;
            bounce_local_lights = true;
            pbr_texture_bounces = 3;
            normal_maps_in_bounces = true;
            skinned_meshes_in_reflections = true;
            final_bounce_env_fallback = true;
            ambient_occlusion_samples = 3;
            alpha_ray_transparency = true;
            alpha_max_steps = 8;
            alpha_shadow_transmission = true;
            mesh_simplification_enabled = false;
            trace_shadows = true;
            shadow_bounces = 2;
            shadow_max_distance = 180.0f;
            texture_array_resolution = TEX_RES_1024;
            max_texture_layers = 128;
            blas_builds_per_frame = 0;
            texture_lod_bias = 0.0f;
            distance_lod_strength = 0.6f;
            mesh_cull_distance = 150.0f;
            mesh_cull_min_angle = 0.0f;
            light_cull_distance = 300.0f;
        } break;
    }
    history_valid = false;
    tex_pool.mark_reset_needed();
    invalidate_blas_cache();
    _change_notify();
}



void RTReflections::detach_next_pass(uint32_t mat_id) {
    auto it = attached_passes.find(mat_id);
    if (it == attached_passes.end()) return;
    if (it->second.mat.is_valid() && VisualServer::get_singleton()) {
        Ref<Material> user_pass = it->second.mat->get_next_pass();
        VisualServer::get_singleton()->material_set_next_pass(it->second.mat->get_rid(), user_pass.is_valid() ? user_pass->get_rid() : RID());
    }
    attached_passes.erase(it);
}

void RTReflections::attach_next_pass(Ref<SpatialMaterial> smat) {
    if (smat.is_null() || !auto_attach_materials) return;

    uint32_t id = smat->get_rid().get_id();
    float rough = smat->get_roughness();
    float metal = smat->get_metallic();
    bool is_transparent = smat->get_feature(SpatialMaterial::FEATURE_TRANSPARENT);
    bool is_scissor = smat->get_flag(SpatialMaterial::FLAG_USE_ALPHA_SCISSOR);
    bool is_alpha_blend = smat->get_blend_mode() != SpatialMaterial::BLEND_MODE_MIX;
    bool is_alpha_material = (is_transparent && !is_scissor) || is_alpha_blend;
    bool is_unshaded = smat->get_flag(SpatialMaterial::FLAG_UNSHADED);

    // Opaque & alpha-scissor materials are composited onto the screen; only Alpha materials use next_pass.
    if (!is_alpha_material || !alpha_consideration || !transparent_receivers || is_unshaded) {
        detach_next_pass(id);
        return;
    }

    Ref<Texture> alb_tex = smat->get_texture(SpatialMaterial::TEXTURE_ALBEDO);
    Ref<Texture> rough_tex = smat->get_texture(SpatialMaterial::TEXTURE_ROUGHNESS);
    Ref<Texture> metal_tex = smat->get_texture(SpatialMaterial::TEXTURE_METALLIC);
    Ref<Texture> norm_tex = smat->get_feature(SpatialMaterial::FEATURE_NORMAL_MAPPING)
        ? smat->get_texture(SpatialMaterial::TEXTURE_NORMAL)
        : Ref<Texture>();

    Color col_srgb = smat->get_albedo();
    if (!alb_tex.is_valid() && col_srgb.a < transparent_receiver_min_alpha && transparent_fresnel_boost <= 0.001f) {
        detach_next_pass(id);
        return;
    }

    bool could_reflect = (rough_tex.is_valid() || metal_tex.is_valid() || rough <= max_roughness || metal >= 0.1f)
                      && (metal >= 0.1f || smat->get_specular() > 0.001f);
    if (!could_reflect || smat->get_next_pass().is_valid()) {
        detach_next_pass(id);
        return;
    }

    int cull_mode = respect_material_cull_mode ? CLAMP((int)smat->get_cull_mode(), 0, 2) : 0;
    if (shared_next_pass_shaders[cull_mode].is_null()) {
        static const char *cull_modes[3] = { "cull_back", "cull_front", "cull_disabled" };
        char code_buf[8192];
        std::snprintf(code_buf, sizeof(code_buf), GODOT_NEXT_PASS_SHADER_TEMPLATE, cull_modes[cull_mode]);
        shared_next_pass_shaders[cull_mode].instance();
        shared_next_pass_shaders[cull_mode]->set_code(String::utf8(code_buf));
    }

    Ref<ShaderMaterial> pass;
    auto it = attached_passes.find(id);
    if (it == attached_passes.end()) {
        pass.instance();
        pass->set_shader(shared_next_pass_shaders[cull_mode]);
        attached_passes[id] = { smat, pass, cull_mode, snapshot_counter };
        VisualServer::get_singleton()->material_set_next_pass(smat->get_rid(), pass->get_rid());
    } else {
        it->second.last_seen = snapshot_counter;
        pass = it->second.pass;
        if (it->second.cull_mode != cull_mode) {
            it->second.cull_mode = cull_mode;
            pass->set_shader(shared_next_pass_shaders[cull_mode]);
        }
        VisualServer::get_singleton()->material_set_next_pass(smat->get_rid(), pass->get_rid());
    }

    Color lin = col_srgb.to_linear();
    Vector3 uv_s = smat->get_uv1_scale();
    Vector3 uv_o = smat->get_uv1_offset();

    int alpha_mode = 0;
    if (is_scissor) alpha_mode = 1;
    else if (is_transparent) alpha_mode = 2;

    pass->set_shader_param("rt_refl_tex", refl_texture);
    pass->set_shader_param("rt_roughness", rough);
    pass->set_shader_param("rt_metallic", metal);
    pass->set_shader_param("rt_specular", smat->get_specular());
    pass->set_shader_param("rt_albedo", Vector3(lin.r, lin.g, lin.b));
    pass->set_shader_param("rt_albedo_alpha", col_srgb.a);
    pass->set_shader_param("rt_intensity", reflection_intensity);
    pass->set_shader_param("rt_max_roughness", max_roughness);
    pass->set_shader_param("rt_roughness_fade_range", roughness_fade_range);
    pass->set_shader_param("rt_blur_strength", blur_strength);
    pass->set_shader_param("rt_blur_lod_scale", blur_lod_scale);
    pass->set_shader_param("rt_bicubic_blur", reflection_filtering == FILTER_BICUBIC);
    pass->set_shader_param("rt_bilateral_upsample", bilateral_upsample);
    pass->set_shader_param("rt_uv1_scale", Vector2(uv_s.x, uv_s.y));
    pass->set_shader_param("rt_uv1_offset", Vector2(uv_o.x, uv_o.y));

    pass->set_shader_param("rt_has_albedo_tex", alb_tex.is_valid());
    pass->set_shader_param("rt_albedo_tex", alb_tex);

    pass->set_shader_param("rt_has_rough_tex", rough_tex.is_valid());
    pass->set_shader_param("rt_rough_tex", rough_tex);
    pass->set_shader_param("rt_rough_channel", (int)smat->get_roughness_texture_channel());

    pass->set_shader_param("rt_has_metal_tex", metal_tex.is_valid());
    pass->set_shader_param("rt_metal_tex", metal_tex);
    pass->set_shader_param("rt_metal_channel", (int)smat->get_metallic_texture_channel());

    pass->set_shader_param("rt_has_normal_tex", norm_tex.is_valid());
    pass->set_shader_param("rt_normal_tex", norm_tex);
    pass->set_shader_param("rt_normal_scale", smat->get_normal_scale());

    pass->set_shader_param("rt_alpha_consideration", alpha_consideration);
    pass->set_shader_param("rt_alpha_mode", alpha_mode);
    pass->set_shader_param("rt_alpha_scissor", smat->get_alpha_scissor_threshold());
    pass->set_shader_param("rt_transparent_fresnel_boost", transparent_fresnel_boost);
}

void RTReflections::clear_attached_passes() {
    for (auto &kv : attached_passes) {
        if (kv.second.mat.is_valid() && VisualServer::get_singleton()) {
            Ref<Material> user_pass = kv.second.mat->get_next_pass();
            VisualServer::get_singleton()->material_set_next_pass(
                kv.second.mat->get_rid(),
                user_pass.is_valid() ? user_pass->get_rid() : RID());
        }
    }
    attached_passes.clear();
    for (int i = 0; i < 3; i++) {
        shared_next_pass_shaders[i].unref();
    }
}

void RTReflections::collect_scene_snapshot() {
    SceneTree *tree = get_tree();
    Viewport *my_vp = get_viewport();
    if (!tree || !my_vp) return;

    Viewport *cam_vp = nullptr;
    Camera *cam = nullptr;
    if (!GL43EffectHelpers::find_primary_camera_and_viewport(this, my_vp, cam_vp, cam)) {
        clear_attached_passes();
        MutexLock lock(scene_mutex);
        snapshot.valid = false;
        return;
    }

    Size2 vp_size = cam_vp->get_size();
    if (vp_size.x < 4 || vp_size.y < 4) return;

    uint32_t cam_cull_mask = cam->get_cull_mask();
    uint32_t refl_mask = reflection_cull_mask;
    uint32_t view_mask = reflection_viewing_cull_mask & cam_cull_mask;

    std::vector<MeshInstance *> meshes;
    std::vector<MultiMeshInstance *> multimeshes;
    std::vector<Ref<Mesh>> extra_meshes;
    std::vector<Light *> scene_lights;
#ifdef MODULE_CSG_ENABLED
    struct CsgEntry {
        CSGShape *csg;
        Transform local_xf;
        Ref<Mesh> mesh;
    };
    std::vector<CsgEntry> csg_entries;
#endif

    struct Walker {
        static void visit(Node *n, uint32_t r_mask, uint32_t v_mask,
                std::vector<MeshInstance *> &out_m,
                std::vector<MultiMeshInstance *> &out_mm,
                std::vector<Ref<Mesh>> &out_em,
                std::vector<Light *> &out_l
#ifdef MODULE_CSG_ENABLED
                , std::vector<CsgEntry> &out_csg
#endif
                ) {
            if (MeshInstance *mi = Object::cast_to<MeshInstance>(n)) {
                if (mi->is_visible_in_tree() && mi->get_mesh().is_valid()) {
                    uint32_t lm = mi->get_layer_mask();
                    if ((lm & r_mask) != 0 || (lm & v_mask) != 0) {
                        out_m.push_back(mi);
                    }
                }
            } else if (MultiMeshInstance *mmi = Object::cast_to<MultiMeshInstance>(n)) {
                if (mmi->is_visible_in_tree() && mmi->get_multimesh().is_valid() && mmi->get_multimesh()->get_mesh().is_valid()) {
                    uint32_t lm = mmi->get_layer_mask();
                    if ((lm & r_mask) != 0 || (lm & v_mask) != 0) {
                        out_mm.push_back(mmi);
                        out_em.push_back(mmi->get_multimesh()->get_mesh());
                    }
                }
            }
#ifdef MODULE_CSG_ENABLED
            else if (CSGShape *csg = Object::cast_to<CSGShape>(n)) {
                if (csg->is_visible_in_tree() && csg->is_root_shape()) {
                    uint32_t lm = csg->get_layer_mask();
                    if ((lm & r_mask) != 0 || (lm & v_mask) != 0) {
                        Array csg_meshes = csg->get_meshes();
                        if (csg_meshes.size() >= 2) {
                            Ref<Mesh> m = csg_meshes[1];
                            if (m.is_valid()) {
                                Transform lxf = csg_meshes[0];
                                out_csg.push_back({ csg, lxf, m });
                                out_em.push_back(m);
                            }
                        }
                    }
                }
            }
#endif
            else if (Light *l = Object::cast_to<Light>(n)) {
                if (l->is_visible_in_tree())
                    out_l.push_back(l);
            }
            for (int i = 0; i < n->get_child_count(); i++)
                visit(n->get_child(i), r_mask, v_mask, out_m, out_mm, out_em, out_l
#ifdef MODULE_CSG_ENABLED
                        , out_csg
#endif
                );
        }
    };
    Walker::visit(my_vp, refl_mask, view_mask, meshes, multimeshes, extra_meshes, scene_lights
#ifdef MODULE_CSG_ENABLED
            , csg_entries
#endif
    );

    MutexLock lock(scene_mutex);
    snapshot_counter++;

    raytracer.rebuild_blas_cache(meshes, extra_meshes, _get_simplification_config(), blas_builds_per_frame);

    int target_w = MAX(1, (int)(vp_size.x * reflection_scale));
    int target_h = MAX(1, (int)(vp_size.y * reflection_scale));
    if (target_w != tex_w || target_h != tex_h) {
        tex_w = target_w;
        tex_h = target_h;
        history_valid = false;
        Ref<Image> init_img;
        init_img.instance();
        init_img->create(tex_w, tex_h, false, Image::FORMAT_RGBAH);
        init_img->fill(Color(0.0f, 0.0f, 0.0f, -1.0f));
        refl_texture->create_from_image(init_img, Texture::FLAG_MIPMAPS | Texture::FLAG_FILTER);

        Ref<Image> meta_img;
        meta_img.instance();
        meta_img->create(tex_w, tex_h, false, Image::FORMAT_RGBAH);
        meta_img->fill(Color(0.0f, 0.0f, 0.0f, -1.0f));
        composite_meta_texture->create_from_image(meta_img, Texture::FLAG_FILTER);
    }

    snapshot.viewport_w = tex_w;
    snapshot.viewport_h = tex_h;

    Transform cam_xform = cam->get_camera_transform();
    snapshot.cam_pos[0] = (float)cam_xform.origin.x;
    snapshot.cam_pos[1] = (float)cam_xform.origin.y;
    snapshot.cam_pos[2] = (float)cam_xform.origin.z;

    CameraMatrix view_cm, proj_cm, vp_cm, inv_proj_cm, inv_vp_cm;
    GL43EffectHelpers::compute_camera_matrices(cam, vp_size, view_cm, proj_cm, vp_cm, inv_proj_cm, inv_vp_cm);

    GL43EffectHelpers::camera_matrix_to_float16(view_cm, snapshot.view_mat);
    GL43EffectHelpers::camera_matrix_to_float16(proj_cm, snapshot.proj_mat);
    GL43EffectHelpers::camera_matrix_to_float16(vp_cm, snapshot.view_proj_mat);
    GL43EffectHelpers::camera_matrix_to_float16(inv_vp_cm, snapshot.inv_view_proj);

    float frustum_planes[6][4];
    GL43EffectHelpers::extract_frustum_planes(snapshot.view_proj_mat, frustum_planes);

    Ref<World> world = cam_vp->find_world();
    if (world.is_null()) world = my_vp->find_world();
    Ref<Environment> env = world.is_valid() ? world->get_environment() : Ref<Environment>();
    if (env.is_null() && world.is_valid()) env = world->get_fallback_environment();

    if (VisualServer::get_singleton()) {
        snapshot.color_tex_rid = VisualServer::get_singleton()->viewport_get_texture(cam_vp->get_viewport_rid(), VS::VIEWPORT_TEXTURE_BUFFER_COLOR);
    }

    // =========================================================
    // SKY RESOLUTION (Environment / Custom Sky / Custom Color)
    // =========================================================
    Color sky_top(0.376f, 0.672f, 0.880f);
    Color sky_hor(0.672f, 0.823f, 0.956f);
    Color sky_gnd_hor(0.153f, 0.127f, 0.105f);
    Color sky_gnd(0.021f, 0.018f, 0.014f);
    float sky_curve = 0.09f;
    float gnd_curve = 0.02f;
    float sky_energy = 1.0f;
    float gnd_energy = 1.0f;

    snapshot.use_panorama = false;
    snapshot.panorama_rid = RID();
    pinned_panorama.unref();
    snapshot.sky_energy = 1.0f;
    snapshot.sky_sun_energy = 0.0f;

    Basis sky_basis;
    auto apply_sky_resource = [&](const Ref<Sky> &sky_res, float bg_energy) {
        Ref<PanoramaSky> pan_sky = sky_res;
        Ref<ProceduralSky> psky = sky_res;
        if (pan_sky.is_valid() && pan_sky->get_panorama().is_valid()) {
            pinned_panorama = pan_sky->get_panorama();
            snapshot.use_panorama = true;
            snapshot.panorama_rid = pinned_panorama->get_rid();
            snapshot.sky_energy = bg_energy;
            snapshot.sky_sun_energy = 0.0f;
        } else if (psky.is_valid()) {
            sky_top = psky->get_sky_top_color().to_linear();
            sky_hor = psky->get_sky_horizon_color().to_linear();
            sky_gnd_hor = psky->get_ground_horizon_color().to_linear();
            sky_gnd = psky->get_ground_bottom_color().to_linear();
            sky_curve = psky->get_sky_curve();
            gnd_curve = psky->get_ground_curve();
            sky_energy = psky->get_sky_energy() * bg_energy;
            gnd_energy = psky->get_ground_energy() * bg_energy;

            float lat = Math::deg2rad(psky->get_sun_latitude());
            float lon = Math::deg2rad(psky->get_sun_longitude());
            Vector3 sdir = (Basis(Vector3(0, 1, 0), lon) * Basis(Vector3(1, 0, 0), lat)).xform(Vector3(0, 0, -1)).normalized();
            snapshot.sky_sun_dir[0] = (float)sdir.x;
            snapshot.sky_sun_dir[1] = (float)sdir.y;
            snapshot.sky_sun_dir[2] = (float)sdir.z;
            Color scol = psky->get_sun_color().to_linear();
            snapshot.sky_sun_color[0] = scol.r;
            snapshot.sky_sun_color[1] = scol.g;
            snapshot.sky_sun_color[2] = scol.b;
            snapshot.sky_sun_angle_min = psky->get_sun_angle_min();
            snapshot.sky_sun_angle_max = psky->get_sun_angle_max();
            snapshot.sky_sun_curve = psky->get_sun_curve();
            snapshot.sky_sun_energy = psky->get_sun_energy() * bg_energy;
        }
    };

    if (custom_sky_mode == SKY_MODE_CUSTOM_SKY && custom_sky.is_valid()) {
        Vector3 rad(Math::deg2rad(custom_sky_rotation.x), Math::deg2rad(custom_sky_rotation.y), Math::deg2rad(custom_sky_rotation.z));
        sky_basis.set_euler(rad);
        apply_sky_resource(custom_sky, custom_sky_energy);
    } else if (custom_sky_mode == SKY_MODE_COLOR) {
        Vector3 rad(Math::deg2rad(custom_sky_rotation.x), Math::deg2rad(custom_sky_rotation.y), Math::deg2rad(custom_sky_rotation.z));
        sky_basis.set_euler(rad);
        Color c = custom_sky_color.to_linear();
        sky_top = c; sky_hor = c; sky_gnd_hor = c; sky_gnd = c;
        sky_energy = custom_sky_energy;
        gnd_energy = custom_sky_energy;
        snapshot.sky_energy = custom_sky_energy;
    } else if (env.is_valid()) {
        sky_basis = env->get_sky_orientation();
        snapshot.sky_energy = env->get_bg_energy();
        if (env->get_background() == Environment::BG_SKY && env->get_sky().is_valid()) {
            apply_sky_resource(env->get_sky(), env->get_bg_energy());
        } else if (env->get_background() == Environment::BG_COLOR || env->get_background() == Environment::BG_CLEAR_COLOR) {
            Color bg = env->get_bg_color().to_linear();
            sky_top = bg; sky_hor = bg; sky_gnd_hor = bg; sky_gnd = bg;
            sky_energy = env->get_bg_energy();
            gnd_energy = env->get_bg_energy();
        }
    }

    Basis sky_inv = sky_basis.transposed();
    snapshot.sky_inv_basis[0] = (float)sky_inv.elements[0][0]; snapshot.sky_inv_basis[1] = (float)sky_inv.elements[1][0]; snapshot.sky_inv_basis[2] = (float)sky_inv.elements[2][0];
    snapshot.sky_inv_basis[3] = (float)sky_inv.elements[0][1]; snapshot.sky_inv_basis[4] = (float)sky_inv.elements[1][1]; snapshot.sky_inv_basis[5] = (float)sky_inv.elements[2][1];
    snapshot.sky_inv_basis[6] = (float)sky_inv.elements[0][2]; snapshot.sky_inv_basis[7] = (float)sky_inv.elements[1][2]; snapshot.sky_inv_basis[8] = (float)sky_inv.elements[2][2];

    snapshot.sky_top[0] = sky_top.r * sky_energy;
    snapshot.sky_top[1] = sky_top.g * sky_energy;
    snapshot.sky_top[2] = sky_top.b * sky_energy;
    snapshot.sky_horizon[0] = sky_hor.r * sky_energy;
    snapshot.sky_horizon[1] = sky_hor.g * sky_energy;
    snapshot.sky_horizon[2] = sky_hor.b * sky_energy;
    snapshot.sky_ground_horizon[0] = sky_gnd_hor.r * gnd_energy;
    snapshot.sky_ground_horizon[1] = sky_gnd_hor.g * gnd_energy;
    snapshot.sky_ground_horizon[2] = sky_gnd_hor.b * gnd_energy;
    snapshot.sky_ground[0] = sky_gnd.r * gnd_energy;
    snapshot.sky_ground[1] = sky_gnd.g * gnd_energy;
    snapshot.sky_ground[2] = sky_gnd.b * gnd_energy;
    snapshot.sky_curve = sky_curve;
    snapshot.ground_curve = gnd_curve;
    snapshot.sky_ground_energy = gnd_energy;

    uint64_t sh = 0xCBF29CE484222325ULL;
    sh = GL43EffectHelpers::hash_combine_u64(sh, snapshot.use_panorama ? 1ULL : 0ULL);
    sh = GL43EffectHelpers::hash_combine_u64(sh, snapshot.panorama_rid.is_valid() ? (uint64_t)snapshot.panorama_rid.get_id() : 0ULL);
    sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_energy);
    for (int i = 0; i < 3; i++) {
        sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_top[i]);
        sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_horizon[i]);
        sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_ground_horizon[i]);
        sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_ground[i]);
        sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_sun_dir[i]);
        sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_sun_color[i]);
    }
    sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_curve);
    sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.ground_curve);
    sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_sun_angle_min);
    sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_sun_angle_max);
    sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_sun_curve);
    sh = GL43EffectHelpers::hash_float_u64(sh, snapshot.sky_sun_energy);
    snapshot.sky_hash = sh;

    // =========================================================
    // AMBIENT LIGHT RESOLUTION
    // =========================================================
    if (ambient_light_mode == AMBIENT_MODE_DISABLED) {
        snapshot.ambient_color[0] = 0.0f;
        snapshot.ambient_color[1] = 0.0f;
        snapshot.ambient_color[2] = 0.0f;
        snapshot.ambient_energy = 0.0f;
        snapshot.sky_contrib = 0.0f;
    } else if (ambient_light_mode == AMBIENT_MODE_CUSTOM) {
        Color amb = ambient_light_color.to_linear();
        snapshot.ambient_color[0] = amb.r;
        snapshot.ambient_color[1] = amb.g;
        snapshot.ambient_color[2] = amb.b;
        snapshot.ambient_energy = ambient_light_energy;
        snapshot.sky_contrib = ambient_light_sky_contribution;
    } else if (env.is_valid()) {
        Color amb = env->get_ambient_light_color().to_linear();
        snapshot.ambient_color[0] = amb.r;
        snapshot.ambient_color[1] = amb.g;
        snapshot.ambient_color[2] = amb.b;
        snapshot.ambient_energy = env->get_ambient_light_energy();
        snapshot.sky_contrib = env->get_ambient_light_sky_contribution();
    } else {
        snapshot.ambient_color[0] = 0.05f;
        snapshot.ambient_color[1] = 0.05f;
        snapshot.ambient_color[2] = 0.06f;
        snapshot.ambient_energy = 1.0f;
        snapshot.sky_contrib = 1.0f;
    }

    // =========================================================
    // LIGHTS
    // =========================================================
    snapshot.lights.clear();
    const float pi = 3.14159265359f;
    const float max_local_light_dist = light_cull_distance * (0.5f + 0.5f * reflection_scale);
    const float max_local_shadow_dist = shadow_max_distance * (0.5f + 0.5f * reflection_scale);
    for (Light *l : scene_lights) {
        GpuLight gl = {};
        Transform lx = l->get_global_transform();
        Color lc = l->get_color().to_linear();
        float energy = l->get_param(Light::PARAM_ENERGY);
        if (energy <= 0.0001f) continue;

        gl.color_atten[0] = lc.r * energy * pi;
        gl.color_atten[1] = lc.g * energy * pi;
        gl.color_atten[2] = lc.b * energy * pi;
        gl.spot_params[2] = (trace_shadows && shadow_bounces > 0 && l->has_shadow()) ? 1.0f : 0.0f;

        if (Object::cast_to<DirectionalLight>(l)) {
            Vector3 to_light = lx.basis.get_axis(2).normalized();
            gl.dir_type[0] = (float)to_light.x;
            gl.dir_type[1] = (float)to_light.y;
            gl.dir_type[2] = (float)to_light.z;
            gl.dir_type[3] = 0.0f;
        } else {
            float range = l->get_param(Light::PARAM_RANGE);
            float dist_to_cam = MAX(0.0f, (float)lx.origin.distance_to(cam_xform.origin) - range);
            if (dist_to_cam > max_local_light_dist) continue;
            if (dist_to_cam > max_local_shadow_dist) gl.spot_params[2] = 0.0f;

            if (Object::cast_to<SpotLight>(l)) {
                Vector3 beam_dir = (-lx.basis.get_axis(2)).normalized();
                gl.pos_range[0] = (float)lx.origin.x;
                gl.pos_range[1] = (float)lx.origin.y;
                gl.pos_range[2] = (float)lx.origin.z;
                gl.pos_range[3] = range;
                gl.dir_type[0] = (float)beam_dir.x;
                gl.dir_type[1] = (float)beam_dir.y;
                gl.dir_type[2] = (float)beam_dir.z;
                gl.dir_type[3] = 2.0f;
                gl.color_atten[3] = l->get_param(Light::PARAM_ATTENUATION);
                gl.spot_params[0] = std::cos(Math::deg2rad(l->get_param(Light::PARAM_SPOT_ANGLE)));
                gl.spot_params[1] = l->get_param(Light::PARAM_SPOT_ATTENUATION);
            } else {
                gl.pos_range[0] = (float)lx.origin.x;
                gl.pos_range[1] = (float)lx.origin.y;
                gl.pos_range[2] = (float)lx.origin.z;
                gl.pos_range[3] = range;
                gl.dir_type[3] = 1.0f;
                gl.color_atten[3] = l->get_param(Light::PARAM_ATTENUATION);
            }
        }

        snapshot.lights.push_back(gl);
    }
    {
        auto dir_end = std::partition(snapshot.lights.begin(), snapshot.lights.end(), [](const GpuLight &l) {
            return l.dir_type[3] < 0.5f;
        });
        snapshot.num_dir_lights = (int)(dir_end - snapshot.lights.begin());
    }

    // =========================================================
    // INSTANCES & MATERIALS
    // =========================================================
    raytracer.begin_frame_instances();
    snapshot.materials.clear();
    snapshot.has_transparent_gbuf = false;
    snapshot.has_alpha_materials = false;

    auto push_surface_materials = [&](const Ref<Mesh> &mesh, auto get_mat_fn, bool allow_attach_pass, bool &out_any_opaque, bool &out_any_transparent) {
        out_any_opaque = false;
        out_any_transparent = false;

        int surf_count = MAX(1, mesh->get_surface_count());
        for (int s = 0; s < surf_count; s++) {
            GpuMaterial gm = {};
            gm.albedo[0] = 0.8f; gm.albedo[1] = 0.8f; gm.albedo[2] = 0.8f; gm.albedo[3] = 1.0f;
            gm.emission[0] = 0.0f; gm.emission[1] = 0.0f; gm.emission[2] = 0.0f; gm.emission[3] = 0.0f;
            gm.pbr_params[0] = 0.5f;
            gm.pbr_params[1] = 0.5f;
            gm.pbr_params[2] = 0.5f;
            gm.pbr_params[3] = 0.0f;
            gm.uv_scale_offset[0] = 1.0f;
            gm.uv_scale_offset[1] = 1.0f;
            gm.tex_layers[0] = -1; gm.tex_layers[1] = -1; gm.tex_layers[2] = -1; gm.tex_layers[3] = -1;
            gm.chan_normal[2] = -1;
            gm.chan_normal[3] = GL43EffectHelpers::float_to_int_bits(1.0f);
            gm.extra_tex[0] = -1;
            gm.extra_tex[1] = 0;
            gm.extra_tex[2] = 0;
            gm.extra_tex[3] = 0;

            Ref<Material> mat = get_mat_fn(s);
            Ref<SpatialMaterial> smat = mat;
            if (smat.is_valid()) {
                Color col_srgb = smat->get_albedo();
                Color col = col_srgb.to_linear();
                gm.albedo[0] = col.r;
                gm.albedo[1] = col.g;
                gm.albedo[2] = col.b;
                gm.albedo[3] = col_srgb.a;

                gm.pbr_params[0] = smat->get_roughness();
                gm.pbr_params[1] = smat->get_alpha_scissor_threshold();
                gm.pbr_params[2] = smat->get_specular();
                gm.pbr_params[3] = smat->get_ao_light_affect();
                gm.emission[3] = smat->get_metallic();

                Vector3 uvs = smat->get_uv1_scale();
                Vector3 uvo = smat->get_uv1_offset();
                gm.uv_scale_offset[0] = (float)uvs.x;
                gm.uv_scale_offset[1] = (float)uvs.y;
                gm.uv_scale_offset[2] = (float)uvo.x;
                gm.uv_scale_offset[3] = (float)uvo.y;

                gm.tex_layers[0] = tex_pool.register_layer(smat->get_texture(SpatialMaterial::TEXTURE_ALBEDO), true, max_texture_layers);
                gm.tex_layers[1] = tex_pool.register_layer(smat->get_texture(SpatialMaterial::TEXTURE_ROUGHNESS), false, max_texture_layers);
                gm.tex_layers[2] = tex_pool.register_layer(smat->get_texture(SpatialMaterial::TEXTURE_METALLIC), false, max_texture_layers);
                gm.chan_normal[0] = (int32_t)smat->get_roughness_texture_channel();
                gm.chan_normal[1] = (int32_t)smat->get_metallic_texture_channel();

                if (smat->get_feature(SpatialMaterial::FEATURE_NORMAL_MAPPING)) {
                    gm.chan_normal[2] = tex_pool.register_layer(smat->get_texture(SpatialMaterial::TEXTURE_NORMAL), false, max_texture_layers);
                    gm.chan_normal[3] = GL43EffectHelpers::float_to_int_bits(smat->get_normal_scale());
                }

                if (smat->get_feature(SpatialMaterial::FEATURE_EMISSION)) {
                    Color em = smat->get_emission().to_linear();
                    float en = smat->get_emission_energy();
                    gm.emission[0] = em.r * en;
                    gm.emission[1] = em.g * en;
                    gm.emission[2] = em.b * en;
                    gm.tex_layers[3] = tex_pool.register_layer(smat->get_texture(SpatialMaterial::TEXTURE_EMISSION), true, max_texture_layers);
                }

                if (smat->get_feature(SpatialMaterial::FEATURE_AMBIENT_OCCLUSION)) {
                    gm.extra_tex[0] = tex_pool.register_layer(smat->get_texture(SpatialMaterial::TEXTURE_AMBIENT_OCCLUSION), false, max_texture_layers);
                    gm.extra_tex[1] = (int32_t)smat->get_ao_texture_channel();
                }

                bool is_scissor = smat->get_flag(SpatialMaterial::FLAG_USE_ALPHA_SCISSOR);
                bool is_transparent = smat->get_feature(SpatialMaterial::FEATURE_TRANSPARENT);
                bool is_alpha_blend = smat->get_blend_mode() != SpatialMaterial::BLEND_MODE_MIX;
                int alpha_mode = is_scissor ? 1 : ((is_transparent || is_alpha_blend) ? 2 : 0);
                gm.extra_tex[2] = alpha_mode;
                if (alpha_mode != 0) snapshot.has_alpha_materials = true;

                int32_t mflags = CLAMP((int32_t)smat->get_cull_mode(), 0, 2);
                if (smat->get_flag(SpatialMaterial::FLAG_DISABLE_AMBIENT_LIGHT)) mflags |= 4;
                if (smat->get_flag(SpatialMaterial::FLAG_UNSHADED)) mflags |= 8;
                gm.extra_tex[3] = mflags;

                if (alpha_consideration && alpha_mode == 2) {
                    out_any_transparent = true;
                } else {
                    out_any_opaque = true;
                }

                if (allow_attach_pass) {
                    attach_next_pass(smat);
                }
            } else {
                out_any_opaque = true;
            }
            snapshot.materials.push_back(gm);
        }
    };

    for (MeshInstance *mi : meshes) {
        Ref<Mesh> mesh = mi->get_mesh();
        uint32_t mid = mesh->get_rid().get_id();
        const GL43Raytracer::MeshBLAS *blas = raytracer.get_blas(mid);
        if (!blas || (!skinned_meshes_in_reflections && blas->has_skin_weights)) {
            continue;
        }

        uint32_t layer_mask = mi->get_layer_mask();
        bool in_refl_mask = (layer_mask & reflection_cull_mask) != 0;
        bool in_view_mask = (layer_mask & reflection_viewing_cull_mask) != 0;
        bool cam_visible = (layer_mask & cam_cull_mask) != 0;
        bool shadows_only = (mi->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);

        int32_t inst_idx = -1;
        if (!raytracer.add_mesh_instance(
                mi,
                (int32_t)snapshot.materials.size(),
                cam_xform.origin,
                mesh_cull_distance,
                mesh_cull_min_angle,
                reflection_scale,
                skinned_meshes_in_reflections,
                in_refl_mask,
                cam_visible,
                0,
                &inst_idx)) {
            continue;
        }

        GL43Raytracer::InstanceEntry &inst = raytracer.get_instance_mutable(inst_idx);
        bool in_frustum = !frustum_cull_gbuffer || GL43EffectHelpers::aabb_in_frustum(inst.world_aabb, frustum_planes);
        inst.draw_in_gbuf = inst.draw_in_gbuf && in_frustum;

        bool any_opaque_or_cutout = false;
        bool any_transparent = false;
        push_surface_materials(mesh, [&](int s) { return mi->get_active_material(s); }, in_view_mask && cam_visible && !shadows_only, any_opaque_or_cutout, any_transparent);

        uint32_t uflags = 0;
        if (!in_view_mask || shadows_only) uflags |= INST_FLAG_SKIP_REFLECTION;
        if (any_opaque_or_cutout) uflags |= INST_FLAG_HAS_OPAQUE_CUTOUT;
        if (any_transparent && transparent_receivers) {
            uflags |= INST_FLAG_HAS_TRANSPARENT;
            if (inst.draw_in_gbuf) snapshot.has_transparent_gbuf = true;
        }
        inst.user_flags = uflags;
    }

    for (MultiMeshInstance *mmi : multimeshes) {
        Ref<MultiMesh> mm = mmi->get_multimesh();
        Ref<Mesh> mesh = mm->get_mesh();
        uint32_t mid = mesh->get_rid().get_id();
        const GL43Raytracer::MeshBLAS *blas = raytracer.get_blas(mid);
        if (!blas || (!skinned_meshes_in_reflections && blas->has_skin_weights)) {
            continue;
        }

        uint32_t layer_mask = mmi->get_layer_mask();
        bool in_refl_mask = (layer_mask & reflection_cull_mask) != 0;
        bool in_view_mask = (layer_mask & reflection_viewing_cull_mask) != 0;
        bool cam_visible = (layer_mask & cam_cull_mask) != 0;
        bool shadows_only = (mmi->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);

        size_t first_idx = raytracer.get_instances().size();
        if (!raytracer.add_multimesh_instance(
                mmi,
                (int32_t)snapshot.materials.size(),
                cam_xform.origin,
                mesh_cull_distance,
                mesh_cull_min_angle,
                reflection_scale,
                in_refl_mask,
                cam_visible,
                0)) {
            continue;
        }

        bool any_opaque_or_cutout = false;
        bool any_transparent = false;
        Ref<Material> override_mat = mmi->get_material_override();
        push_surface_materials(mesh, [&](int s) {
            return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s);
        }, in_view_mask && cam_visible && !shadows_only, any_opaque_or_cutout, any_transparent);

        uint32_t uflags = 0;
        if (!in_view_mask || shadows_only) uflags |= INST_FLAG_SKIP_REFLECTION;
        if (any_opaque_or_cutout) uflags |= INST_FLAG_HAS_OPAQUE_CUTOUT;
        if (any_transparent && transparent_receivers) uflags |= INST_FLAG_HAS_TRANSPARENT;

        size_t end_idx = raytracer.get_instances().size();
        for (size_t k = first_idx; k < end_idx; k++) {
            GL43Raytracer::InstanceEntry &inst = raytracer.get_instance_mutable(k);
            bool in_frustum = !frustum_cull_gbuffer || GL43EffectHelpers::aabb_in_frustum(inst.world_aabb, frustum_planes);
            inst.draw_in_gbuf = inst.draw_in_gbuf && in_frustum;
            inst.user_flags = uflags;
            if (inst.draw_in_gbuf && (uflags & INST_FLAG_HAS_TRANSPARENT)) {
                snapshot.has_transparent_gbuf = true;
            }
        }
    }

#ifdef MODULE_CSG_ENABLED
    for (const CsgEntry &entry : csg_entries) {
        CSGShape *csg = entry.csg;
        Ref<Mesh> mesh = entry.mesh;
        uint32_t mid = mesh->get_rid().get_id();
        const GL43Raytracer::MeshBLAS *blas = raytracer.get_blas(mid);
        if (!blas) continue;

        uint32_t layer_mask = csg->get_layer_mask();
        bool in_refl_mask = (layer_mask & reflection_cull_mask) != 0;
        bool in_view_mask = (layer_mask & reflection_viewing_cull_mask) != 0;
        bool cam_visible = (layer_mask & cam_cull_mask) != 0;
        bool shadows_only = (csg->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);

        int32_t inst_idx = -1;
        if (!raytracer.add_custom_mesh_instance(
                mesh,
                csg->get_global_transform() * entry.local_xf,
                layer_mask,
                csg->get_cast_shadows_setting(),
                (int32_t)snapshot.materials.size(),
                cam_xform.origin,
                mesh_cull_distance,
                mesh_cull_min_angle,
                reflection_scale,
                in_refl_mask,
                cam_visible,
                0,
                &inst_idx)) {
            continue;
        }

        GL43Raytracer::InstanceEntry &inst = raytracer.get_instance_mutable(inst_idx);
        bool in_frustum = !frustum_cull_gbuffer || GL43EffectHelpers::aabb_in_frustum(inst.world_aabb, frustum_planes);
        inst.draw_in_gbuf = inst.draw_in_gbuf && in_frustum;

        bool any_opaque_or_cutout = false;
        bool any_transparent = false;
        Ref<Material> override_mat = csg->get_material_override();
        push_surface_materials(mesh, [&](int s) {
            return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s);
        }, in_view_mask && cam_visible && !shadows_only, any_opaque_or_cutout, any_transparent);

        uint32_t uflags = 0;
        if (!in_view_mask || shadows_only) uflags |= INST_FLAG_SKIP_REFLECTION;
        if (any_opaque_or_cutout) uflags |= INST_FLAG_HAS_OPAQUE_CUTOUT;
        if (any_transparent && transparent_receivers) {
            uflags |= INST_FLAG_HAS_TRANSPARENT;
            if (inst.draw_in_gbuf) snapshot.has_transparent_gbuf = true;
        }
        inst.user_flags = uflags;
    }
#endif

    raytracer.build_tlas(max_tlas_instances);

    std::vector<uint32_t> stale_mat_ids;
    for (const auto &kv : attached_passes) {
        if (kv.second.last_seen != snapshot_counter) {
            stale_mat_ids.push_back(kv.first);
        }
    }
    for (uint32_t mat_id : stale_mat_ids) {
        detach_next_pass(mat_id);
    }

    snapshot.valid = !raytracer.is_empty();
}

bool RTReflections::init_gl_extensions_and_shaders() {
    if (gl_attempted) return gl_ready;
    gl_attempted = true;
    gl_thread_id = Thread::get_caller_id();

    if (!SharedGL43Context::get_singleton()->is_available()) {
        ERR_PRINT("RTReflections: Shared OpenGL 4.3 context not available.");
        return false;
    }

    if (!raytracer.init_gl()) {
        return false;
    }
    if (!tex_pool.init_gl("RTReflections")) {
        return false;
    }

    gbuf_program = GL43EffectHelpers::link_vf_program(GBUF_VERT_GLSL, GBUF_FRAG_GLSL, "RTReflections", "GBuffer");
    sh_program = GL43EffectHelpers::link_compute_program(SH_COMPUTE_GLSL, "RTReflections", "SkySH");
    String rt_src = String(RT_COMPUTE_HEADER_GLSL) + GL43Raytracer::get_glsl_common() + RT_COMPUTE_BODY_GLSL;
    rt_program = GL43EffectHelpers::link_compute_program(rt_src.utf8().get_data(), "RTReflections", "Compute");
    reconstruct_program = GL43EffectHelpers::link_compute_program(RECONSTRUCT_COMPUTE_GLSL, "RTReflections", "Reconstruct");
    denoise_program = GL43EffectHelpers::link_compute_program(DENOISE_COMPUTE_GLSL, "RTReflections", "Denoise");
    composite_program = GL43EffectHelpers::link_vf_program(GL43EffectHelpers::get_fullscreen_vert_glsl(), COMPOSITE_FRAG_GLSL, "RTReflections", "Composite");
    if (!gbuf_program || !sh_program || !rt_program || !reconstruct_program || !denoise_program || !composite_program) return false;

    glGenSamplers(1, &pano_sampler_mip);
    glSamplerParameteri(pano_sampler_mip, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glSamplerParameteri(pano_sampler_mip, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(pano_sampler_mip, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glSamplerParameteri(pano_sampler_mip, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenSamplers(1, &pano_sampler_nomip);
    glSamplerParameteri(pano_sampler_nomip, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glSamplerParameteri(pano_sampler_nomip, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(pano_sampler_nomip, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(pano_sampler_nomip, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    uint8_t white[4] = { 255, 255, 255, 255 };
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &dummy_white_tex);
    glBindTexture(GL_TEXTURE_2D, dummy_white_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenBuffers(1, &ssbo_materials);
    glGenBuffers(1, &ssbo_lights);
    glGenBuffers(1, &ssbo_sky_sh);

    float zero_sh[9 * 4] = {};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_sky_sh);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero_sh), zero_sh, GL_DYNAMIC_DRAW);

    gl_ready = true;
    return true;
}

void RTReflections::ensure_gbuffer_size(int w, int h) {
    if (gbuf_fbo != 0 && gbuf_w == w && gbuf_h == h) return;
    gbuf_w = w; gbuf_h = h;

    if (gbuf_fbo == 0) {
        glGenFramebuffers(1, &gbuf_fbo);
        glGenTextures(1, &gbuf_albedo_metal);
        glGenTextures(1, &gbuf_extra);
    }

    glActiveTexture(GL_TEXTURE0);
    GL43EffectHelpers::ensure_rgba16f_texture(gbuf_normal_rough, w, h, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, gbuf_albedo_metal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, gbuf_extra);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    GL43EffectHelpers::ensure_depth24_texture(gbuf_depth, w, h);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, gbuf_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gbuf_normal_rough, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gbuf_albedo_metal, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gbuf_extra, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, gbuf_depth, 0);
    GLenum bufs[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers(3, bufs);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RTReflections::ensure_work_textures(int w, int h, bool need_history) {
    bool size_changed = (work_w != w || work_h != h);
    if (size_changed) {
        work_w = w;
        work_h = h;
        history_valid = false;
    }

    glActiveTexture(GL_TEXTURE0);
    if (size_changed || rt_work_a == 0) GL43EffectHelpers::ensure_rgba16f_texture(rt_work_a, w, h, GL_LINEAR);
    if (size_changed || rt_work_b == 0) GL43EffectHelpers::ensure_rgba16f_texture(rt_work_b, w, h, GL_LINEAR);
    if (need_history && (size_changed || rt_history_tex[0] == 0 || rt_history_tex[1] == 0)) {
        GL43EffectHelpers::ensure_rgba16f_texture(rt_history_tex[0], w, h, GL_LINEAR);
        GL43EffectHelpers::ensure_rgba16f_texture(rt_history_tex[1], w, h, GL_LINEAR);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void RTReflections::_on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx) {
    if (!enabled || !auto_composite || !is_inside_tree() || !gl_ready || composite_program == 0) {
        return;
    }
    if (Engine::get_singleton()->is_editor_hint() && !editor_preview) {
        return;
    }

    {
        MutexLock lock(scene_mutex);
        if (!snapshot.valid || snapshot.color_tex_rid != p_ctx.color_tex_rid) {
            return;
        }
    }

    RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
    if (!rs) return;

    RasterizerStorageGLES3::Texture *godot_tex = rs->texture_owner.getornull(refl_texture->get_rid());
    RasterizerStorageGLES3::Texture *godot_meta_tex = rs->texture_owner.getornull(composite_meta_texture->get_rid());
    if (!godot_tex || godot_tex->tex_id == 0 || !godot_meta_tex || godot_meta_tex->tex_id == 0) return;

    glUseProgram(composite_program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, godot_tex->tex_id);
    glUniform1i(glGetUniformLocation(composite_program, "uReflTex"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, godot_meta_tex->tex_id);
    glUniform1i(glGetUniformLocation(composite_program, "uMetaTex"), 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, p_ctx.depth_tex_id);
    glUniform1i(glGetUniformLocation(composite_program, "uDepthTex"), 2);

    glUniform1f(glGetUniformLocation(composite_program, "uIntensity"), reflection_intensity);
    glUniform1f(glGetUniformLocation(composite_program, "uBlurStrength"), blur_strength);
    glUniform1f(glGetUniformLocation(composite_program, "uBlurLodScale"), blur_lod_scale);
    glUniform1i(glGetUniformLocation(composite_program, "uBicubicBlur"), reflection_filtering == FILTER_BICUBIC ? 1 : 0);
    glUniform1i(glGetUniformLocation(composite_program, "uBilateralUpsample"), bilateral_upsample ? 1 : 0);
    glUniformMatrix4fv(glGetUniformLocation(composite_program, "uInvProj"), 1, GL_FALSE, p_ctx.inv_proj_mat);

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);
}

void RTReflections::_on_frame_pre_draw() {
    if (!enabled) {
        return;
    }

    bool in_editor = Engine::get_singleton()->is_editor_hint();
    if (in_editor) {
        if (!editor_preview) {
            return;
        }
        process_frame_counter++;
        if (update_interval <= 1 || (process_frame_counter % (uint64_t)update_interval) == 0) {
            collect_scene_snapshot();
        }
    }

    if (update_interval > 1 && (process_frame_counter % (uint64_t)update_interval) != 0) {
        return;
    }

    RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
    if (!rs) return;

    MutexLock lock(scene_mutex);
    if (!snapshot.valid || snapshot.viewport_w <= 0 || snapshot.viewport_h <= 0) return;

    RasterizerStorageGLES3::Texture *godot_tex = rs->texture_owner.getornull(refl_texture->get_rid());
    RasterizerStorageGLES3::Texture *godot_meta_tex = rs->texture_owner.getornull(composite_meta_texture->get_rid());
    if (!godot_tex || godot_tex->tex_id == 0 || !godot_meta_tex || godot_meta_tex->tex_id == 0) return;
    if (godot_tex->width != snapshot.viewport_w || godot_tex->height != snapshot.viewport_h) return;
    if (godot_meta_tex->width != snapshot.viewport_w || godot_meta_tex->height != snapshot.viewport_h) return;

    SharedGL43Context::ScopedScope gl43;
    if (!gl43.is_active()) return;

    if (!init_gl_extensions_and_shaders()) return;

    glDisable(GL_FRAMEBUFFER_SRGB);
    tex_pool.flush_blits(rs, (int)texture_array_resolution, max_texture_layers, true);

    frame_counter++;
    raytracer.sync_gpu_buffers();
    bool has_tlas = raytracer.has_tlas();

    GL43EffectHelpers::upload_ssbo(ssbo_materials, snapshot.materials.data(), snapshot.materials.size() * sizeof(GpuMaterial), sizeof(GpuMaterial));
    GL43EffectHelpers::upload_ssbo(ssbo_lights, snapshot.lights.data(), snapshot.lights.size() * sizeof(GpuLight), sizeof(GpuLight));

    raytracer.bind_ssbos(0, 1, 2, 3, 4);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, ssbo_materials);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ssbo_lights);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssbo_sky_sh);

    // =========================================================
    // 1. G-BUFFER PASS (Pass 0: Opaque/Scissor, Pass 1: Transparent)
    // =========================================================
    ensure_gbuffer_size(snapshot.viewport_w, snapshot.viewport_h);
    glBindFramebuffer(GL_FRAMEBUFFER, gbuf_fbo);
    glViewport(0, 0, snapshot.viewport_w, snapshot.viewport_h);
    glFrontFace(GL_CW);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glClearDepth(1.0);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(gbuf_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex_pool.get_tex_array());
    glUniform1i(glGetUniformLocation(gbuf_program, "uTexArray"), 0);
    glUniformMatrix4fv(glGetUniformLocation(gbuf_program, "uViewProj"), 1, GL_FALSE, snapshot.view_proj_mat);
    glUniform1i(glGetUniformLocation(gbuf_program, "uRespectCullMode"), respect_material_cull_mode ? 1 : 0);
    glUniform1i(glGetUniformLocation(gbuf_program, "uAlphaConsideration"), alpha_consideration ? 1 : 0);
    glUniform1f(glGetUniformLocation(gbuf_program, "uTransparentMinAlpha"), transparent_receiver_min_alpha);

    GLint loc_inst_idx = glGetUniformLocation(gbuf_program, "uInstIdx");
    GLint loc_mat_off = glGetUniformLocation(gbuf_program, "uMatOffset");
    GLint loc_skip_refl = glGetUniformLocation(gbuf_program, "uSkipReflection");
    GLint loc_pass = glGetUniformLocation(gbuf_program, "uPass");
    GLint loc_flip = glGetUniformLocation(gbuf_program, "uFlipFacing");

    auto draw_gbuf_pass = [&](int pass_idx) {
        glUniform1i(loc_pass, pass_idx);
        const std::vector<GL43Raytracer::InstanceEntry> &insts = raytracer.get_instances();
        for (size_t i = 0; i < insts.size(); i++) {
            const GL43Raytracer::InstanceEntry &inst = insts[i];
            if (!inst.draw_in_gbuf) continue;
            if (pass_idx == 0 && !(inst.user_flags & INST_FLAG_HAS_OPAQUE_CUTOUT)) continue;
            if (pass_idx == 1 && !(inst.user_flags & INST_FLAG_HAS_TRANSPARENT)) continue;

            uint32_t vao = 0;
            int32_t vcount = 0;
            if (!raytracer.get_instance_draw_data(i, vao, vcount)) continue;
            glUniform1i(loc_inst_idx, (int)i);
            glUniform1i(loc_mat_off, inst.mat_offset);
            glUniform1i(loc_skip_refl, (inst.user_flags & INST_FLAG_SKIP_REFLECTION) ? 1 : 0);
            glUniform1i(loc_flip, inst.flip_facing ? 1 : 0);
            glBindVertexArray((GLuint)vao);
            glDrawArrays(GL_TRIANGLES, 0, vcount);
        }
    };

    draw_gbuf_pass(0);
    if (alpha_consideration && transparent_receivers && snapshot.has_transparent_gbuf) {
        draw_gbuf_pass(1);
    }

    // =========================================================
    // 2. RESOLVE PANORAMA BINDING & UPDATE SKY L2 SH IRRADIANCE
    // =========================================================
    int use_pano = 0;
    int pano_to_lin = 0;
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, dummy_white_tex);
    glBindSampler(4, pano_sampler_nomip);

    if (snapshot.use_panorama && snapshot.panorama_rid.is_valid()) {
        RasterizerStorageGLES3::Texture *pt = rs->texture_owner.getornull(snapshot.panorama_rid);
        if (pt && pt->tex_id != 0) {
            use_pano = 1;
            glBindTexture(GL_TEXTURE_2D, pt->tex_id);
            GLint int_fmt = 0, mip1_w = 0;
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &int_fmt);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 1, GL_TEXTURE_WIDTH, &mip1_w);
            bool is_hdr_fmt = (int_fmt == 0x881A || int_fmt == 0x881B || int_fmt == 0x8814 || int_fmt == 0x8815 || int_fmt == 0x8C3A || int_fmt == 0x8C3D);
            pano_to_lin = (!GL43EffectHelpers::is_srgb_gl_format((int32_t)int_fmt) && !is_hdr_fmt) ? 1 : 0;
            glBindSampler(4, mip1_w > 0 ? pano_sampler_mip : pano_sampler_nomip);
        }
    }

    if (sky_sh_dirty || last_sky_sh_hash != snapshot.sky_hash) {
        last_sky_sh_hash = snapshot.sky_hash;
        sky_sh_dirty = false;

        glUseProgram(sh_program);
        glUniform1i(glGetUniformLocation(sh_program, "uPanoramaTex"), 4);
        glUniform1i(glGetUniformLocation(sh_program, "uUsePanorama"), use_pano);
        glUniform1i(glGetUniformLocation(sh_program, "uPanoramaToLinear"), pano_to_lin);
        glUniform1f(glGetUniformLocation(sh_program, "uSkyEnergy"), snapshot.sky_energy);
        glUniform3f(glGetUniformLocation(sh_program, "uSkyTop"), snapshot.sky_top[0], snapshot.sky_top[1], snapshot.sky_top[2]);
        glUniform3f(glGetUniformLocation(sh_program, "uSkyHorizon"), snapshot.sky_horizon[0], snapshot.sky_horizon[1], snapshot.sky_horizon[2]);
        glUniform3f(glGetUniformLocation(sh_program, "uSkyGroundHorizon"), snapshot.sky_ground_horizon[0], snapshot.sky_ground_horizon[1], snapshot.sky_ground_horizon[2]);
        glUniform3f(glGetUniformLocation(sh_program, "uSkyGround"), snapshot.sky_ground[0], snapshot.sky_ground[1], snapshot.sky_ground[2]);
        glUniform1f(glGetUniformLocation(sh_program, "uSkyCurve"), snapshot.sky_curve);
        glUniform1f(glGetUniformLocation(sh_program, "uGroundCurve"), snapshot.ground_curve);
        glUniform3f(glGetUniformLocation(sh_program, "uSkySunDir"), snapshot.sky_sun_dir[0], snapshot.sky_sun_dir[1], snapshot.sky_sun_dir[2]);
        glUniform3f(glGetUniformLocation(sh_program, "uSkySunColor"), snapshot.sky_sun_color[0], snapshot.sky_sun_color[1], snapshot.sky_sun_color[2]);
        glUniform1f(glGetUniformLocation(sh_program, "uSkySunAngleMin"), snapshot.sky_sun_angle_min);
        glUniform1f(glGetUniformLocation(sh_program, "uSkySunAngleMax"), snapshot.sky_sun_angle_max);
        glUniform1f(glGetUniformLocation(sh_program, "uSkySunCurve"), snapshot.sky_sun_curve);
        glUniform1f(glGetUniformLocation(sh_program, "uSkySunEnergy"), snapshot.sky_sun_energy);

        SharedGL43Context::get_singleton()->dispatch_compute(1, 1, 1);
        SharedGL43Context::get_singleton()->memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // =========================================================
    // 3. RAY TRACING COMPUTE PASS
    // =========================================================
    bool need_reconstruct = checkerboard_tracing || temporal_accumulation;
    bool need_denoise = (spatial_denoise_passes > 0);
    bool need_post = need_reconstruct || need_denoise;
    bool allow_stochastic = (ray_samples_per_pixel > 1 || temporal_accumulation || need_denoise);

    if (need_post) {
        ensure_work_textures(snapshot.viewport_w, snapshot.viewport_h, temporal_accumulation);
    }

    GLuint rt_target_tex = need_post ? rt_work_a : godot_tex->tex_id;

    SharedGL43Context *gl43_ctx = SharedGL43Context::get_singleton();
    glUseProgram(rt_program);
    gl43_ctx->bind_image_texture(0, rt_target_tex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
    gl43_ctx->bind_image_texture(1, godot_meta_tex->tex_id, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gbuf_normal_rough);
    glUniform1i(glGetUniformLocation(rt_program, "uGNormalRough"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gbuf_albedo_metal);
    glUniform1i(glGetUniformLocation(rt_program, "uGAlbedoMetal"), 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, gbuf_depth);
    glUniform1i(glGetUniformLocation(rt_program, "uGDepth"), 2);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex_pool.get_tex_array());
    glUniform1i(glGetUniformLocation(rt_program, "uTexArray"), 3);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, gbuf_extra);
    glUniform1i(glGetUniformLocation(rt_program, "uGExtra"), 5);

    glUniform1i(glGetUniformLocation(rt_program, "uPanoramaTex"), 4);
    glUniform1i(glGetUniformLocation(rt_program, "uUsePanorama"), use_pano);
    glUniform1i(glGetUniformLocation(rt_program, "uPanoramaToLinear"), pano_to_lin);
    glUniform1f(glGetUniformLocation(rt_program, "uSkyEnergy"), snapshot.sky_energy);
    glUniform1f(glGetUniformLocation(rt_program, "uSkyReflectionEnergy"), sky_reflection_energy);
    glUniformMatrix3fv(glGetUniformLocation(rt_program, "uSkyInvBasis"), 1, GL_FALSE, snapshot.sky_inv_basis);

    glUniformMatrix4fv(glGetUniformLocation(rt_program, "uInvViewProj"), 1, GL_FALSE, snapshot.inv_view_proj);
    glUniform3f(glGetUniformLocation(rt_program, "uCamPos"), snapshot.cam_pos[0], snapshot.cam_pos[1], snapshot.cam_pos[2]);
    glUniform3f(glGetUniformLocation(rt_program, "uSkyTop"), snapshot.sky_top[0], snapshot.sky_top[1], snapshot.sky_top[2]);
    glUniform3f(glGetUniformLocation(rt_program, "uSkyHorizon"), snapshot.sky_horizon[0], snapshot.sky_horizon[1], snapshot.sky_horizon[2]);
    glUniform3f(glGetUniformLocation(rt_program, "uSkyGroundHorizon"), snapshot.sky_ground_horizon[0], snapshot.sky_ground_horizon[1], snapshot.sky_ground_horizon[2]);
    glUniform3f(glGetUniformLocation(rt_program, "uSkyGround"), snapshot.sky_ground[0], snapshot.sky_ground[1], snapshot.sky_ground[2]);
    glUniform1f(glGetUniformLocation(rt_program, "uSkyCurve"), snapshot.sky_curve);
    glUniform1f(glGetUniformLocation(rt_program, "uGroundCurve"), snapshot.ground_curve);
    glUniform1f(glGetUniformLocation(rt_program, "uSkyContrib"), snapshot.sky_contrib);
    glUniform3f(glGetUniformLocation(rt_program, "uAmbientColor"), snapshot.ambient_color[0], snapshot.ambient_color[1], snapshot.ambient_color[2]);
    glUniform1f(glGetUniformLocation(rt_program, "uAmbientEnergy"), snapshot.ambient_energy);
    glUniform1f(glGetUniformLocation(rt_program, "uAmbientRadianceBoost"), ambient_radiance_boost);
    glUniform1f(glGetUniformLocation(rt_program, "uAmbientBounceApprox"), ambient_bounce_approximation);
    glUniform1i(glGetUniformLocation(rt_program, "uAmbientAOSamples"), ambient_occlusion_samples);
    glUniform1f(glGetUniformLocation(rt_program, "uAmbientAORadius"), ambient_occlusion_radius);
    glUniform1f(glGetUniformLocation(rt_program, "uAmbientAOIntensity"), ambient_occlusion_intensity);
    glUniform3f(glGetUniformLocation(rt_program, "uSkySunDir"), snapshot.sky_sun_dir[0], snapshot.sky_sun_dir[1], snapshot.sky_sun_dir[2]);
    glUniform3f(glGetUniformLocation(rt_program, "uSkySunColor"), snapshot.sky_sun_color[0], snapshot.sky_sun_color[1], snapshot.sky_sun_color[2]);
    glUniform1f(glGetUniformLocation(rt_program, "uSkySunAngleMin"), snapshot.sky_sun_angle_min);
    glUniform1f(glGetUniformLocation(rt_program, "uSkySunAngleMax"), snapshot.sky_sun_angle_max);
    glUniform1f(glGetUniformLocation(rt_program, "uSkySunCurve"), snapshot.sky_sun_curve);
    glUniform1f(glGetUniformLocation(rt_program, "uSkySunEnergy"), snapshot.sky_sun_energy);

    float lod_bias = texture_lod_bias + std::max(0.0f, -std::log2(std::max(reflection_scale, 0.125f)));
    float eff_shadow_dist = (trace_shadows && shadow_bounces > 0) ? (shadow_max_distance * (0.35f + 0.65f * reflection_scale)) : 0.0f;
    float eff_ray_dist = max_ray_distance * (0.5f + 0.5f * reflection_scale);

    glUniform1i(glGetUniformLocation(rt_program, "uHasTLAS"), has_tlas ? 1 : 0);
    glUniform1i(glGetUniformLocation(rt_program, "uNumLights"), (int)snapshot.lights.size());
    glUniform1i(glGetUniformLocation(rt_program, "uNumDirLights"), snapshot.num_dir_lights);
    glUniform1i(glGetUniformLocation(rt_program, "uHasAlphaMaterials"), snapshot.has_alpha_materials ? 1 : 0);
    glUniform1i(glGetUniformLocation(rt_program, "uMaterialCount"), MAX(1, (int)snapshot.materials.size()));
    glUniform1i(glGetUniformLocation(rt_program, "uMaxBounces"), max_bounces);
    glUniform1f(glGetUniformLocation(rt_program, "uMaxRoughness"), max_roughness);
    glUniform1f(glGetUniformLocation(rt_program, "uRoughnessFadeRange"), roughness_fade_range);
    glUniform1f(glGetUniformLocation(rt_program, "uLodBias"), lod_bias);
    glUniform1f(glGetUniformLocation(rt_program, "uDistLodStrength"), distance_lod_strength);
    glUniform1f(glGetUniformLocation(rt_program, "uShadowMaxDist"), eff_shadow_dist);
    glUniform1i(glGetUniformLocation(rt_program, "uShadowBounces"), trace_shadows ? shadow_bounces : 0);
    glUniform1i(glGetUniformLocation(rt_program, "uPbrTexBounces"), pbr_texture_bounces);
    glUniform1i(glGetUniformLocation(rt_program, "uBounceLocalLights"), bounce_local_lights ? 1 : 0);
    glUniform1i(glGetUniformLocation(rt_program, "uNormalMapsInBounces"), normal_maps_in_bounces ? 1 : 0);
    glUniform1i(glGetUniformLocation(rt_program, "uRespectCullMode"), respect_material_cull_mode ? 1 : 0);
    glUniform1i(glGetUniformLocation(rt_program, "uEnvFallback"), final_bounce_env_fallback ? 1 : 0);
    glUniform1f(glGetUniformLocation(rt_program, "uMaxRayDistance"), eff_ray_dist);
    glUniform1f(glGetUniformLocation(rt_program, "uBounceDistFalloff"), bounce_distance_falloff);
    glUniform1f(glGetUniformLocation(rt_program, "uBounceThroughputCutoff"), bounce_throughput_threshold);
    glUniform1f(glGetUniformLocation(rt_program, "uRayBias"), ray_origin_bias);
    glUniform1f(glGetUniformLocation(rt_program, "uShadowBias"), shadow_ray_bias);
    glUniform1i(glGetUniformLocation(rt_program, "uSamplesPerPixel"), ray_samples_per_pixel);
    glUniform1f(glGetUniformLocation(rt_program, "uGlossySpread"), glossy_roughness_spread);
    glUniform1i(glGetUniformLocation(rt_program, "uAllowStochastic"), allow_stochastic ? 1 : 0);
    glUniform1i(glGetUniformLocation(rt_program, "uTemporalAccum"), temporal_accumulation ? 1 : 0);
    glUniform1i(glGetUniformLocation(rt_program, "uCheckerboard"), checkerboard_tracing ? 1 : 0);
    glUniform1i(glGetUniformLocation(rt_program, "uFrameIdx"), (int)(frame_counter & 0x7FFFFFFF));
    glUniform1i(glGetUniformLocation(rt_program, "uAlphaConsideration"), alpha_consideration ? 1 : 0);
    glUniform1i(glGetUniformLocation(rt_program, "uAlphaRayTransparency"), alpha_ray_transparency ? 1 : 0);
    glUniform1i(glGetUniformLocation(rt_program, "uAlphaMaxSteps"), alpha_max_steps);
    glUniform1i(glGetUniformLocation(rt_program, "uAlphaShadowTransmission"), alpha_shadow_transmission ? 1 : 0);

    GLuint gx = (GLuint)((snapshot.viewport_w + 7) / 8);
    GLuint gy = (GLuint)((snapshot.viewport_h + 7) / 8);
    gl43_ctx->dispatch_compute(gx, gy, 1);
    gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    // =========================================================
    // 4. CHECKERBOARD / TEMPORAL RECONSTRUCTION + SPATIAL DENOISE
    // =========================================================
    if (need_post) {
        GLuint cur_tex = rt_work_a;

        if (need_reconstruct) {
            int next_hist = 1 - history_idx;
            GLuint out_hist = temporal_accumulation ? rt_history_tex[next_hist] : rt_work_b;
            GLuint in_hist = (temporal_accumulation && history_valid) ? rt_history_tex[history_idx] : dummy_white_tex;

            glUseProgram(reconstruct_program);
            gl43_ctx->bind_image_texture(0, out_hist, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
            if (!need_denoise) {
                gl43_ctx->bind_image_texture(1, godot_tex->tex_id, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
            } else {
                gl43_ctx->bind_image_texture(1, rt_work_b, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
            }

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, cur_tex);
            glUniform1i(glGetUniformLocation(reconstruct_program, "uRawTex"), 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, in_hist);
            glUniform1i(glGetUniformLocation(reconstruct_program, "uHistoryTex"), 1);

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, gbuf_normal_rough);
            glUniform1i(glGetUniformLocation(reconstruct_program, "uGNormalRough"), 2);

            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, gbuf_depth);
            glUniform1i(glGetUniformLocation(reconstruct_program, "uGDepth"), 3);

            glUniformMatrix4fv(glGetUniformLocation(reconstruct_program, "uInvViewProj"), 1, GL_FALSE, snapshot.inv_view_proj);
            glUniformMatrix4fv(glGetUniformLocation(reconstruct_program, "uPrevViewProj"), 1, GL_FALSE, prev_view_proj);
            glUniform3f(glGetUniformLocation(reconstruct_program, "uCamPos"), snapshot.cam_pos[0], snapshot.cam_pos[1], snapshot.cam_pos[2]);
            glUniform3f(glGetUniformLocation(reconstruct_program, "uPrevCamPos"), prev_cam_pos[0], prev_cam_pos[1], prev_cam_pos[2]);
            glUniform1i(glGetUniformLocation(reconstruct_program, "uCheckerboard"), checkerboard_tracing ? 1 : 0);
            glUniform1i(glGetUniformLocation(reconstruct_program, "uTemporalAccum"), temporal_accumulation ? 1 : 0);
            glUniform1i(glGetUniformLocation(reconstruct_program, "uHistoryValid"), history_valid ? 1 : 0);
            glUniform1f(glGetUniformLocation(reconstruct_program, "uTemporalBlend"), temporal_blend);
            glUniform1i(glGetUniformLocation(reconstruct_program, "uWriteFinal"), need_denoise ? 0 : 1);

            gl43_ctx->dispatch_compute(gx, gy, 1);
            gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

            if (temporal_accumulation) {
                history_idx = next_hist;
                history_valid = true;
                std::memcpy(prev_view_proj, snapshot.view_proj_mat, sizeof(prev_view_proj));
                std::memcpy(prev_cam_pos, snapshot.cam_pos, sizeof(prev_cam_pos));
            }
            cur_tex = out_hist;
        }

        if (need_denoise) {
            glUseProgram(denoise_program);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, gbuf_normal_rough);
            glUniform1i(glGetUniformLocation(denoise_program, "uGNormalRough"), 1);

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, gbuf_depth);
            glUniform1i(glGetUniformLocation(denoise_program, "uGDepth"), 2);

            glUniformMatrix4fv(glGetUniformLocation(denoise_program, "uInvViewProj"), 1, GL_FALSE, snapshot.inv_view_proj);
            glUniform3f(glGetUniformLocation(denoise_program, "uCamPos"), snapshot.cam_pos[0], snapshot.cam_pos[1], snapshot.cam_pos[2]);

            for (int p = 0; p < spatial_denoise_passes; p++) {
                bool is_last = (p + 1 == spatial_denoise_passes);
                GLuint dst_tex = is_last ? godot_tex->tex_id : ((cur_tex == rt_work_a) ? rt_work_b : rt_work_a);
                gl43_ctx->bind_image_texture(0, dst_tex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, cur_tex);
                glUniform1i(glGetUniformLocation(denoise_program, "uSrcTex"), 0);
                glUniform1i(glGetUniformLocation(denoise_program, "uStepSize"), 1 << p);

                gl43_ctx->dispatch_compute(gx, gy, 1);
                gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
                cur_tex = dst_tex;
            }
        }
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, godot_tex->tex_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1000);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    for (int i = 0; i < 8; i++) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)i, 0);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindSampler(4, 0);
    for (int i = 0; i < 6; i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    }
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
