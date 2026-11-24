#ifndef RTAO_SHADERS_H
#define RTAO_SHADERS_H

namespace RTAOShaders {

static const char *GBUF_VERT_GLSL = R"GLSL(
#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUv;
layout (location = 3) in float aSurfIdx;

uniform mat4 uModel;
uniform mat4 uViewProj;
uniform int uMatOffset;

out vec3 vWorldNormal;
out vec2 vUv;
flat out int vMatId;

void main() {
	vec4 wp = uModel * vec4(aPos, 1.0);
	vWorldNormal = normalize(transpose(inverse(mat3(uModel))) * aNormal);
	vUv = aUv;
	vMatId = uMatOffset + int(aSurfIdx + 0.5);
	gl_Position = uViewProj * wp;
}
)GLSL";

static const char *GBUF_FRAG_GLSL = R"GLSL(
#version 430 core
in vec3 vWorldNormal;
in vec2 vUv;
flat in int vMatId;

layout (location = 0) out vec4 outNormalMask;
layout (location = 1) out uint outLayerMask;

uniform sampler2DArray uTexArray;
uniform int uSkipReceiver;
uniform int uRespectCullMode;
uniform int uFlipFacing;
uniform int uAlphaScissorSupport;
uniform uint uLayerMask;

struct RTAOMaterial {
	vec4 uvScaleOffset;
	vec4 params;      // x = alpha_base, y = alpha_scissor, z = 0, w = 0
	ivec4 flags;      // x = albedo_layer, y = alpha_mode (0=opaque,1=scissor,2=transp), z = cull_mode, w = 0
};

layout (std430, binding = 5) readonly buffer MaterialBuf {
	RTAOMaterial materials[];
};

void main() {
	RTAOMaterial m = materials[vMatId];
	int cullMode = m.flags.z & 3;
	bool frontFace = (uFlipFacing != 0) ? !gl_FrontFacing : gl_FrontFacing;
	if (uRespectCullMode != 0) {
		if (cullMode == 0 && !frontFace) discard;
		if (cullMode == 1 && frontFace) discard;
	}

	int alphaMode = m.flags.y;
	if (alphaMode == 2) discard;

	if (uAlphaScissorSupport != 0 && alphaMode == 1) {
		vec2 uv = vUv * m.uvScaleOffset.xy + m.uvScaleOffset.zw;
		float a = m.params.x;
		if (m.flags.x >= 0) {
			a *= texture(uTexArray, vec3(uv, float(m.flags.x))).a;
		}
		if (a < m.params.y) discard;
	}

	if (uSkipReceiver != 0) {
		outNormalMask = vec4(0.0);
		outLayerMask = 0u;
		return;
	}

	vec3 N = normalize(vWorldNormal);
	if (cullMode == 2 && !frontFace) N = -N;
	outNormalMask = vec4(N, 1.0);
	outLayerMask = uLayerMask;
}
)GLSL";

// Common BVH & Volume/Baked helpers shared across compute stages
// (Expects GL43Raytracer::get_glsl_common() prepended)
static const char *BVH_COMMON_GLSL = R"GLSL(
struct RTAOMaterial {
	vec4 uvScaleOffset;
	vec4 params;
	ivec4 flags;
};

struct LocalVolume {
	vec4 invRow0;
	vec4 invRow1;
	vec4 invRow2;
	vec4 colorOcc;    // rgb = ao_color, w = occlusion
	vec4 params;      // x = shape, y = mode, z = blend_mode, w = edge_falloff
	vec4 modParams;   // x = radius_mult, y = intensity_mult, z = noise_strength, w = noise_scale
	vec4 noiseOffset; // xyz = offset, w = 0
};

struct BakedVolume {
	vec4 invRow0;
	vec4 invRow1;
	vec4 invRow2;
	vec4 colorInt;    // rgb = ao_color, w = intensity
	vec4 params;      // x = blend_mode, y = edge_falloff, z = 0, w = 0
	uvec4 masks;      // x = viewing_cull_mask
};

layout (std430, binding = 5) readonly buffer MaterialBuf { RTAOMaterial materials[]; };
layout (std430, binding = 6) readonly buffer LocalVolBuf { LocalVolume localVols[]; };
layout (std430, binding = 7) readonly buffer BakedVolBuf { BakedVolume bakedVols[]; };

float ignNoise(vec2 p) {
	return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

vec2 r2Sample(int idx, vec2 jitter) {
	return fract(jitter + float(idx) * vec2(0.7548776662, 0.5698402910));
}

float hash31(vec3 p) {
	p = fract(p * 0.1031);
	p += dot(p, p.yzx + 33.33);
	return fract((p.x + p.y) * p.z);
}

float valueNoise3D(vec3 x) {
	vec3 i = floor(x);
	vec3 f = fract(x);
	f = f * f * (3.0 - 2.0 * f);

	float n000 = hash31(i + vec3(0.0, 0.0, 0.0));
	float n100 = hash31(i + vec3(1.0, 0.0, 0.0));
	float n010 = hash31(i + vec3(0.0, 1.0, 0.0));
	float n110 = hash31(i + vec3(1.0, 1.0, 0.0));
	float n001 = hash31(i + vec3(0.0, 0.0, 1.0));
	float n101 = hash31(i + vec3(1.0, 0.0, 1.0));
	float n011 = hash31(i + vec3(0.0, 1.0, 1.0));
	float n111 = hash31(i + vec3(1.0, 1.0, 1.0));

	float nx00 = mix(n000, n100, f.x);
	float nx10 = mix(n010, n110, f.x);
	float nx01 = mix(n001, n101, f.x);
	float nx11 = mix(n011, n111, f.x);
	return mix(mix(nx00, nx10, f.y), mix(nx01, nx11, f.y), f.z);
}

float fbm3D(vec3 p) {
	return valueNoise3D(p) * 0.65 + valueNoise3D(p * 2.03) * 0.35;
}

// cullMode: 0 = double-sided, 1 = AO ray culling, 2 = shadow ray culling (toward light)
bool intersectTriangleAO(RTTriangle tri, vec3 ro, vec3 rd, float maxT, int matOffset, int cullMode, bool alphaScissor, sampler2DArray texArr, out float outT) {
	float det, t, u, v;
	if (!rayTriangleGeom(tri, ro, rd, maxT, det, t, u, v)) return false;

	int mId = clamp(matOffset + floatBitsToInt(tri.p0.w), 0, materials.length() - 1);
	RTAOMaterial m = materials[mId];
	if (cullMode != 0) {
		int cm = m.flags.z & 3;
		if (cullMode == 1) {
			if (cm == 0 && det >= 0.0) return false;
			if (cm == 1 && det <= 0.0) return false;
		} else {
			if (cm == 0 && det <= 0.0) return false;
			if (cm == 1 && det >= 0.0) return false;
		}
	}
	int alphaMode = m.flags.y;
	if (alphaMode == 2) return false;
	if (alphaScissor && alphaMode == 1) {
		float w = 1.0 - u - v;
		vec2 uv = (w * tri.uv01.xy + u * tri.uv01.zw + v * tri.uv2pad.xy) * m.uvScaleOffset.xy + m.uvScaleOffset.zw;
		float a = m.params.x;
		if (m.flags.x >= 0) {
			a *= textureLod(texArr, vec3(uv, float(m.flags.x)), 1.0).a;
		}
		if (a < m.params.y) return false;
	}

	outT = t;
	return true;
}

void traverseBLASClosest(int rootNode, int matOffset, bool applyCull, bool alphaScissor, sampler2DArray texArr, vec3 ro, vec3 rd, vec3 invRd, inout float bestT) {
	if (intersectAABB(blasNodes[rootNode].aabbMin.xyz, blasNodes[rootNode].aabbMax.xyz, ro, invRd, bestT) >= bestT)
		return;

	int stackId[20];
	float stackT[20];
	int sp = 0;
	stackId[sp] = rootNode;
	stackT[sp] = 0.0;
	sp++;
	int cullMode = applyCull ? 1 : 0;

	while (sp > 0) {
		sp--;
		int nodeIdx = stackId[sp];
		if (stackT[sp] >= bestT) continue;

		BVHNode node = blasNodes[nodeIdx];
		int leftOrFirst = floatBitsToInt(node.aabbMin.w);
		int count = floatBitsToInt(node.aabbMax.w);

		if (count > 0) {
			for (int i = 0; i < count; i++) {
				float t;
				if (intersectTriangleAO(triangles[leftOrFirst + i], ro, rd, bestT, matOffset, cullMode, alphaScissor, texArr, t)) {
					bestT = t;
				}
			}
		} else {
			int left = nodeIdx + 1;
			int right = leftOrFirst;
			float tL = intersectAABB(blasNodes[left].aabbMin.xyz, blasNodes[left].aabbMax.xyz, ro, invRd, bestT);
			float tR = intersectAABB(blasNodes[right].aabbMin.xyz, blasNodes[right].aabbMax.xyz, ro, invRd, bestT);
			if (tL < bestT && tR < bestT) {
				if (tL < tR) {
					if (sp < 19) { stackId[sp] = right; stackT[sp] = tR; sp++; stackId[sp] = left; stackT[sp] = tL; sp++; }
				} else {
					if (sp < 19) { stackId[sp] = left; stackT[sp] = tL; sp++; stackId[sp] = right; stackT[sp] = tR; sp++; }
				}
			} else if (tL < bestT) {
				if (sp < 20) { stackId[sp] = left; stackT[sp] = tL; sp++; }
			} else if (tR < bestT) {
				if (sp < 20) { stackId[sp] = right; stackT[sp] = tR; sp++; }
			}
		}
	}
}

float traceOcclusionDist(vec3 ro, vec3 rd, float maxT, bool hasTLAS, bool applyCull, bool alphaScissor, sampler2DArray texArr) {
	if (!hasTLAS || tlasNodes.length() == 0) return maxT;

	vec3 invRd = safeInvDir(rd);
	float bestT = maxT;
	float tRoot = intersectAABB(tlasNodes[0].aabbMin.xyz, tlasNodes[0].aabbMax.xyz, ro, invRd, bestT);
	if (tRoot >= bestT) return maxT;

	int stackId[20];
	float stackT[20];
	int sp = 0;
	stackId[sp] = 0;
	stackT[sp] = tRoot;
	sp++;

	while (sp > 0) {
		sp--;
		int nodeIdx = stackId[sp];
		if (stackT[sp] >= bestT) continue;

		BVHNode node = tlasNodes[nodeIdx];
		int leftOrFirst = floatBitsToInt(node.aabbMin.w);
		int count = floatBitsToInt(node.aabbMax.w);

		if (count > 0) {
			for (int i = 0; i < count; i++) {
				RTInstance inst = instances[tlasIndices[leftOrFirst + i]];
				if (intersectAABB(inst.aabbMin.xyz, inst.aabbMax.xyz, ro, invRd, bestT) >= bestT)
					continue;
				vec3 lro = (inst.invModel * vec4(ro, 1.0)).xyz;
				vec3 lrd = (inst.invModel * vec4(rd, 0.0)).xyz;
				traverseBLASClosest(floatBitsToInt(inst.aabbMin.w), floatBitsToInt(inst.aabbMax.w), applyCull, alphaScissor, texArr, lro, lrd, safeInvDir(lrd), bestT);
			}
		} else {
			int left = nodeIdx + 1;
			int right = leftOrFirst;
			float tL = intersectAABB(tlasNodes[left].aabbMin.xyz, tlasNodes[left].aabbMax.xyz, ro, invRd, bestT);
			float tR = intersectAABB(tlasNodes[right].aabbMin.xyz, tlasNodes[right].aabbMax.xyz, ro, invRd, bestT);
			if (tL < bestT && tR < bestT) {
				if (tL < tR) {
					if (sp < 19) { stackId[sp] = right; stackT[sp] = tR; sp++; stackId[sp] = left; stackT[sp] = tL; sp++; }
				} else {
					if (sp < 19) { stackId[sp] = left; stackT[sp] = tL; sp++; stackId[sp] = right; stackT[sp] = tR; sp++; }
				}
			} else if (tL < bestT) {
				if (sp < 20) { stackId[sp] = left; stackT[sp] = tL; sp++; }
			} else if (tR < bestT) {
				if (sp < 20) { stackId[sp] = right; stackT[sp] = tR; sp++; }
			}
		}
	}
	return bestT;
}

bool traverseBLASShadow(int rootNode, int matOffset, bool applyCull, bool alphaScissor, sampler2DArray texArr, vec3 ro, vec3 rd, vec3 invRd, float maxT) {
	if (intersectAABB(blasNodes[rootNode].aabbMin.xyz, blasNodes[rootNode].aabbMax.xyz, ro, invRd, maxT) >= maxT)
		return false;

	int stackId[20];
	int sp = 0;
	stackId[sp++] = rootNode;
	int cullMode = applyCull ? 2 : 0;

	while (sp > 0) {
		int nodeIdx = stackId[--sp];
		BVHNode node = blasNodes[nodeIdx];
		int leftOrFirst = floatBitsToInt(node.aabbMin.w);
		int count = floatBitsToInt(node.aabbMax.w);

		if (count > 0) {
			for (int i = 0; i < count; i++) {
				float t;
				if (intersectTriangleAO(triangles[leftOrFirst + i], ro, rd, maxT, matOffset, cullMode, alphaScissor, texArr, t)) {
					return true;
				}
			}
		} else {
			int left = nodeIdx + 1;
			int right = leftOrFirst;
			if (intersectAABB(blasNodes[left].aabbMin.xyz, blasNodes[left].aabbMax.xyz, ro, invRd, maxT) < maxT && sp < 20)
				stackId[sp++] = left;
			if (intersectAABB(blasNodes[right].aabbMin.xyz, blasNodes[right].aabbMax.xyz, ro, invRd, maxT) < maxT && sp < 20)
				stackId[sp++] = right;
		}
	}
	return false;
}

bool traceOccludedAO(vec3 ro, vec3 rd, float maxT, bool hasTLAS, bool applyCull, bool alphaScissor, sampler2DArray texArr) {
	if (!hasTLAS || tlasNodes.length() == 0) return false;
	vec3 invRd = safeInvDir(rd);
	if (intersectAABB(tlasNodes[0].aabbMin.xyz, tlasNodes[0].aabbMax.xyz, ro, invRd, maxT) >= maxT)
		return false;

	int stackId[20];
	int sp = 0;
	stackId[sp++] = 0;

	while (sp > 0) {
		int nodeIdx = stackId[--sp];
		BVHNode node = tlasNodes[nodeIdx];
		int leftOrFirst = floatBitsToInt(node.aabbMin.w);
		int count = floatBitsToInt(node.aabbMax.w);

		if (count > 0) {
			for (int i = 0; i < count; i++) {
				RTInstance inst = instances[tlasIndices[leftOrFirst + i]];
				if (intersectAABB(inst.aabbMin.xyz, inst.aabbMax.xyz, ro, invRd, maxT) >= maxT)
					continue;
				vec3 lro = (inst.invModel * vec4(ro, 1.0)).xyz;
				vec3 lrd = (inst.invModel * vec4(rd, 0.0)).xyz;
				if (traverseBLASShadow(floatBitsToInt(inst.aabbMin.w), floatBitsToInt(inst.aabbMax.w), applyCull, alphaScissor, texArr, lro, lrd, safeInvDir(lrd), maxT)) {
					return true;
				}
			}
		} else {
			int left = nodeIdx + 1;
			int right = leftOrFirst;
			if (intersectAABB(tlasNodes[left].aabbMin.xyz, tlasNodes[left].aabbMax.xyz, ro, invRd, maxT) < maxT && sp < 20)
				stackId[sp++] = left;
			if (intersectAABB(tlasNodes[right].aabbMin.xyz, tlasNodes[right].aabbMax.xyz, ro, invRd, maxT) < maxT && sp < 20)
				stackId[sp++] = right;
		}
	}
	return false;
}

bool intersectTriangleAOVol(RTTriangle tri, vec3 ro, vec3 rd, float maxT, int matOffset, bool applyCull, bool alphaScissor, sampler2DArray texArr, out float outT, out bool outBackface) {
	float det, t, u, v;
	if (!rayTriangleGeom(tri, ro, rd, maxT, det, t, u, v)) return false;

	int mId = clamp(matOffset + floatBitsToInt(tri.p0.w), 0, materials.length() - 1);
	RTAOMaterial m = materials[mId];
	bool isBack = false;
	if (applyCull) {
		int cm = m.flags.z & 3;
		if (cm == 0 && det >= 0.0) isBack = true;
		if (cm == 1 && det <= 0.0) isBack = true;
	}
	int alphaMode = m.flags.y;
	if (alphaMode == 2) return false;
	if (alphaScissor && alphaMode == 1) {
		float w = 1.0 - u - v;
		vec2 uv = (w * tri.uv01.xy + u * tri.uv01.zw + v * tri.uv2pad.xy) * m.uvScaleOffset.xy + m.uvScaleOffset.zw;
		float a = m.params.x;
		if (m.flags.x >= 0) {
			a *= textureLod(texArr, vec3(uv, float(m.flags.x)), 1.0).a;
		}
		if (a < m.params.y) return false;
	}

	outT = t;
	outBackface = isBack;
	return true;
}

void traverseBLASClosestVol(int rootNode, int matOffset, bool applyCull, bool alphaScissor, sampler2DArray texArr, vec3 ro, vec3 rd, vec3 invRd, inout float bestFrontT, inout float bestBackT) {
	if (intersectAABB(blasNodes[rootNode].aabbMin.xyz, blasNodes[rootNode].aabbMax.xyz, ro, invRd, bestFrontT) >= bestFrontT)
		return;

	int stackId[20];
	float stackT[20];
	int sp = 0;
	stackId[sp] = rootNode;
	stackT[sp] = 0.0;
	sp++;

	while (sp > 0) {
		sp--;
		int nodeIdx = stackId[sp];
		if (stackT[sp] >= bestFrontT) continue;

		BVHNode node = blasNodes[nodeIdx];
		int leftOrFirst = floatBitsToInt(node.aabbMin.w);
		int count = floatBitsToInt(node.aabbMax.w);

		if (count > 0) {
			for (int i = 0; i < count; i++) {
				float t;
				bool isBack;
				if (intersectTriangleAOVol(triangles[leftOrFirst + i], ro, rd, bestFrontT, matOffset, applyCull, alphaScissor, texArr, t, isBack)) {
					if (isBack) {
						if (t < bestBackT) bestBackT = t;
					} else {
						if (t < bestFrontT) bestFrontT = t;
					}
				}
			}
		} else {
			int left = nodeIdx + 1;
			int right = leftOrFirst;
			float tL = intersectAABB(blasNodes[left].aabbMin.xyz, blasNodes[left].aabbMax.xyz, ro, invRd, bestFrontT);
			float tR = intersectAABB(blasNodes[right].aabbMin.xyz, blasNodes[right].aabbMax.xyz, ro, invRd, bestFrontT);
			if (tL < bestFrontT && tR < bestFrontT) {
				if (tL < tR) {
					if (sp < 19) { stackId[sp] = right; stackT[sp] = tR; sp++; stackId[sp] = left; stackT[sp] = tL; sp++; }
				} else {
					if (sp < 19) { stackId[sp] = left; stackT[sp] = tL; sp++; stackId[sp] = right; stackT[sp] = tR; sp++; }
				}
			} else if (tL < bestFrontT) {
				if (sp < 20) { stackId[sp] = left; stackT[sp] = tL; sp++; }
			} else if (tR < bestFrontT) {
				if (sp < 20) { stackId[sp] = right; stackT[sp] = tR; sp++; }
			}
		}
	}
}

float traceOcclusionDistVol(vec3 ro, vec3 rd, float maxT, bool hasTLAS, bool applyCull, bool alphaScissor, sampler2DArray texArr, out bool outBackface) {
	outBackface = false;
	if (!hasTLAS || tlasNodes.length() == 0) return maxT;

	vec3 invRd = safeInvDir(rd);
	float bestFrontT = maxT;
	float bestBackT = maxT;
	float tRoot = intersectAABB(tlasNodes[0].aabbMin.xyz, tlasNodes[0].aabbMax.xyz, ro, invRd, bestFrontT);
	if (tRoot >= bestFrontT) return maxT;

	int stackId[20];
	float stackT[20];
	int sp = 0;
	stackId[sp] = 0;
	stackT[sp] = tRoot;
	sp++;

	while (sp > 0) {
		sp--;
		int nodeIdx = stackId[sp];
		if (stackT[sp] >= bestFrontT) continue;

		BVHNode node = tlasNodes[nodeIdx];
		int leftOrFirst = floatBitsToInt(node.aabbMin.w);
		int count = floatBitsToInt(node.aabbMax.w);

		if (count > 0) {
			for (int i = 0; i < count; i++) {
				RTInstance inst = instances[tlasIndices[leftOrFirst + i]];
				if (intersectAABB(inst.aabbMin.xyz, inst.aabbMax.xyz, ro, invRd, bestFrontT) >= bestFrontT)
					continue;
				vec3 lro = (inst.invModel * vec4(ro, 1.0)).xyz;
				vec3 lrd = (inst.invModel * vec4(rd, 0.0)).xyz;
				traverseBLASClosestVol(floatBitsToInt(inst.aabbMin.w), floatBitsToInt(inst.aabbMax.w), applyCull, alphaScissor, texArr, lro, lrd, safeInvDir(lrd), bestFrontT, bestBackT);
			}
		} else {
			int left = nodeIdx + 1;
			int right = leftOrFirst;
			float tL = intersectAABB(tlasNodes[left].aabbMin.xyz, tlasNodes[left].aabbMax.xyz, ro, invRd, bestFrontT);
			float tR = intersectAABB(tlasNodes[right].aabbMin.xyz, tlasNodes[right].aabbMax.xyz, ro, invRd, bestFrontT);
			if (tL < bestFrontT && tR < bestFrontT) {
				if (tL < tR) {
					if (sp < 19) { stackId[sp] = right; stackT[sp] = tR; sp++; stackId[sp] = left; stackT[sp] = tL; sp++; }
				} else {
					if (sp < 19) { stackId[sp] = left; stackT[sp] = tL; sp++; stackId[sp] = right; stackT[sp] = tR; sp++; }
				}
			} else if (tL < bestFrontT) {
				if (sp < 20) { stackId[sp] = left; stackT[sp] = tL; sp++; }
			} else if (tR < bestFrontT) {
				if (sp < 20) { stackId[sp] = right; stackT[sp] = tR; sp++; }
			}
		}
	}
	outBackface = (bestBackT < bestFrontT);
	return bestFrontT;
}
)GLSL";

// Real-Time 3D AO Texture Compute Shader (stores L1 Spherical Harmonics visibility in GL_TEXTURE_3D)
static const char *VOLUME_COMPUTE_GLSL_BODY = R"GLSL(
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout (rgba16f, binding = 0) writeonly uniform image3D uOutAOVolume;
uniform sampler2DArray uTexArray;
uniform sampler3D uPrevHistVolume;

uniform vec3 uVolMin;
uniform vec3 uPrevVolMin;
uniform vec3 uVolSize;
uniform float uVolRadius;
uniform int uVolSamples;
uniform float uFalloffPower;
uniform float uRayBias;
uniform int uHasTLAS;
uniform int uRespectCullMode;
uniform int uAlphaScissorSupport;
uniform int uVolHistValid;
uniform int uVolFrame;

void main() {
	ivec3 vox = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 sz = imageSize(uOutAOVolume);
	if (vox.x >= sz.x || vox.y >= sz.y || vox.z >= sz.z) {
		return;
	}

	vec3 uvw = (vec3(vox) + 0.5) / vec3(sz);
	vec3 worldPos = uVolMin + uvw * uVolSize;
	vec3 voxelStep = max(uVolSize / vec3(sz), vec3(0.001));
	vec3 worldCell = floor(worldPos / voxelStep + 0.5);

	int samples = clamp(uVolSamples, 1, 16);
	float frameF = float(uVolFrame & 255);
	float jitPhi = fract(hash31(worldCell) + frameF * 0.61803398875) * 6.2831853;
	float jitY = fract(hash31(worldCell + vec3(17.0, 31.0, 47.0)) + frameF * 0.7548776662);
	const float goldenAngle = 2.399963229728653;

	float visSum = 0.0;
	vec3 bentSum = vec3(0.0);
	vec3 dirSum = vec3(0.0);
	int backfaceHits = 0;
	bool applyCull = (uRespectCullMode != 0);
	bool doScissor = (uAlphaScissorSupport != 0);
	bool hasTLAS = (uHasTLAS != 0);

	for (int s = 0; s < 16; s++) {
		if (s >= samples) break;
		float sy = 1.0 - (2.0 * (float(s) + jitY) / float(samples));
		float sr = sqrt(max(0.0, 1.0 - sy * sy));
		float phi = float(s) * goldenAngle + jitPhi;
		vec3 dir = vec3(cos(phi) * sr, sy, sin(phi) * sr);

		bool hitBackface = false;
		float hitT = traceOcclusionDistVol(worldPos + dir * max(uRayBias, 0.01), dir, uVolRadius, hasTLAS, applyCull, doScissor, uTexArray, hitBackface);
		if (hitBackface) {
			backfaceHits++;
		}
		float vis = 1.0;
		if (hitT < uVolRadius) {
			float occ = pow(clamp(1.0 - hitT / uVolRadius, 0.0, 1.0), uFalloffPower);
			vis = 1.0 - occ;
		}
		visSum += vis;
		bentSum += dir * vis;
		dirSum += dir;
	}

	vec4 curVal;
	if (backfaceHits * 2 >= samples) {
		curVal = vec4(0.0, 0.0, 0.0, -1.0);
	} else {
		float meanVis = clamp(visSum / float(samples), 0.0, 1.0);
		vec3 l1Bent = (bentSum - meanVis * dirSum) / float(samples);
		curVal = vec4(l1Bent, meanVis);
	}

	if (uVolHistValid != 0 && curVal.a >= 0.0) {
		ivec3 prevVox = vox + ivec3(round((uVolMin - uPrevVolMin) / voxelStep));
		if (all(greaterThanEqual(prevVox, ivec3(0))) && all(lessThan(prevVox, sz))) {
			vec4 prevVal = texelFetch(uPrevHistVolume, prevVox, 0);
			if (prevVal.a >= 0.0) {
				curVal = mix(prevVal, curVal, 0.2);
			}
		}
	}

	imageStore(uOutAOVolume, vox, curVal);
}
)GLSL";

static const char *VOLUME_FILTER_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout (rgba16f, binding = 0) writeonly uniform image3D uOutAOVolume;
uniform sampler3D uRawAOVolume;

void main() {
	ivec3 vox = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 sz = imageSize(uOutAOVolume);
	if (vox.x >= sz.x || vox.y >= sz.y || vox.z >= sz.z) {
		return;
	}

	vec4 center = texelFetch(uRawAOVolume, vox, 0);
	vec4 accum = vec4(0.0);
	float wSum = 0.0;

	if (center.a >= 0.0) {
		accum = center * 6.0;
		wSum = 6.0;
		const ivec3 offs[6] = ivec3[6](
			ivec3(-1, 0, 0), ivec3(1, 0, 0),
			ivec3(0, -1, 0), ivec3(0, 1, 0),
			ivec3(0, 0, -1), ivec3(0, 0, 1)
		);
		for (int i = 0; i < 6; i++) {
			ivec3 np = clamp(vox + offs[i], ivec3(0), sz - 1);
			vec4 s = texelFetch(uRawAOVolume, np, 0);
			if (s.a >= 0.0) {
				accum += s;
				wSum += 1.0;
			}
		}
	} else {
		for (int dz = -1; dz <= 1; dz++) {
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					if (dx == 0 && dy == 0 && dz == 0) continue;
					ivec3 np = clamp(vox + ivec3(dx, dy, dz), ivec3(0), sz - 1);
					vec4 s = texelFetch(uRawAOVolume, np, 0);
					if (s.a >= 0.0) {
						float w = (abs(dx) + abs(dy) + abs(dz) == 1) ? 2.0 : 1.0;
						accum += s * w;
						wSum += w;
					}
				}
			}
		}
	}

	vec4 outVal = (wSum > 0.001) ? (accum / wSum) : vec4(0.0, 0.0, 0.0, 1.0);
	imageStore(uOutAOVolume, vox, outVal);
}
)GLSL";

// Screen-Space Ray-Traced AO + 3D Volume / BakedRTAO / LocalRTAO Integration Compute Shader
static const char *RT_COMPUTE_GLSL_BODY = R"GLSL(
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutAO;

uniform sampler2D uGNormalMask;
uniform usampler2D uGLayerMask;
uniform sampler2D uGDepth;
uniform sampler2DArray uTexArray;
uniform sampler3D uRealtimeVolTex;
uniform sampler3D uBakedTex0;
uniform sampler3D uBakedTex1;
uniform sampler3D uBakedTex2;
uniform sampler3D uBakedTex3;

uniform mat4 uInvViewProj;
uniform vec3 uCamPos;
uniform int uHasTLAS;
uniform int uRespectCullMode;
uniform int uAlphaScissorSupport;

uniform int uSamplesPerPixel;
uniform float uRayRadius;
uniform float uFalloffPower;
uniform float uRayBias;
uniform float uIntensity;
uniform float uPower;
uniform vec3 uAOColor;

uniform int uVolumeEnabled;
uniform vec3 uVolMin;
uniform vec3 uVolSize;
uniform float uVolVoxelSize;
uniform float uVolBlend;

struct RTAOLight {
	vec4 posRange;
	vec4 dirType;
	vec4 colorAtten;
	vec4 spotParams;
};

layout (std140, binding = 0) uniform LightUBO {
	RTAOLight lights[32];
};

uniform int uLocalCount;
uniform int uBakedCount;
uniform int uNumLights;
uniform float uAmbientLum;
uniform float uDirectPreservation;
uniform int uUsePhysicalAtten;

uniform int uCheckerboard;
uniform int uTemporalAccum;
uniform int uFrameIdx;
uniform int uDebugMode; // 0=normal, 1=ao, 2=3d volume, 3=bent normal

const float PI = 3.14159265359;

vec3 sampleCosineHemisphere(vec2 xi, vec3 N) {
	float phi = 2.0 * PI * xi.x;
	float cosTheta = sqrt(max(0.0, 1.0 - xi.y));
	float sinTheta = sqrt(max(0.0, xi.y));
	vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 T = normalize(cross(up, N));
	vec3 B = cross(N, T);
	return normalize(T * (cos(phi) * sinTheta) + B * (sin(phi) * sinTheta) + N * cosTheta);
}

vec4 sampleBakedVolume(int idx, vec3 uvw) {
	if (idx == 0) return textureLod(uBakedTex0, uvw, 0.0);
	if (idx == 1) return textureLod(uBakedTex1, uvw, 0.0);
	if (idx == 2) return textureLod(uBakedTex2, uvw, 0.0);
	return textureLod(uBakedTex3, uvw, 0.0);
}

float evalLocalFactor(LocalVolume lv, vec3 worldPos) {
	vec4 wp4 = vec4(worldPos, 1.0);
	vec3 pl = vec3(dot(lv.invRow0, wp4), dot(lv.invRow1, wp4), dot(lv.invRow2, wp4));
	int shape = int(lv.params.x + 0.5);
	float d = 1.0;
	if (shape == 0) {
		d = max(abs(pl.x), max(abs(pl.y), abs(pl.z)));
	} else if (shape == 1 || shape == 2) {
		d = length(pl);
	} else if (shape == 3) {
		d = max(length(pl.xz), abs(pl.y));
	} else {
		float ty = clamp(0.5 - pl.y * 0.5, 0.0, 1.0);
		float rc = max(ty, 0.001);
		d = max(length(pl.xz) / rc, abs(pl.y));
	}
	if (d >= 1.0) return 0.0;
	float falloff = lv.params.w;
	float fade = (falloff > 0.001) ? smoothstep(1.0, max(0.0, 1.0 - falloff), d) : 1.0;
	if (lv.modParams.z > 0.001) {
		float n = fbm3D(worldPos * lv.modParams.w + lv.noiseOffset.xyz);
		fade *= mix(1.0, clamp(n * 1.4, 0.0, 1.0), lv.modParams.z);
	}
	return fade;
}

void main() {
	ivec2 px = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(uOutAO);
	if (px.x >= sz.x || px.y >= sz.y) return;

	if (uCheckerboard != 0 && (((px.x + px.y) & 1) != (uFrameIdx & 1))) {
		imageStore(uOutAO, px, vec4(1.0, 1.0, 1.0, -2.0));
		return;
	}

	float depth = texelFetch(uGDepth, px, 0).r;
	if (depth >= 0.99999) {
		imageStore(uOutAO, px, vec4(1.0, 1.0, 1.0, -1.0));
		return;
	}

	vec4 normMask = texelFetch(uGNormalMask, px, 0);
	if (dot(normMask.xyz, normMask.xyz) < 0.01 || normMask.w < 0.5) {
		imageStore(uOutAO, px, vec4(1.0, 1.0, 1.0, -1.0));
		return;
	}

	vec2 uv = (vec2(px) + 0.5) / vec2(sz);
	vec4 wp = uInvViewProj * vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
	vec3 worldPos = wp.xyz / wp.w;
	float dist = length(uCamPos - worldPos);
	vec3 N = normalize(normMask.xyz);

	// 1. Evaluate LocalRTAO trace modulation volumes
	float localRadiusMult = 1.0;
	float localIntensityMult = 1.0;
	for (int v = 0; v < uLocalCount; v++) {
		LocalVolume lv = localVols[v];
		if (int(lv.params.y + 0.5) != 1) continue; // MODE_MODULATE_TRACE
		float f = evalLocalFactor(lv, worldPos);
		if (f > 0.0001) {
			localRadiusMult *= mix(1.0, lv.modParams.x, f);
			localIntensityMult *= mix(1.0, lv.modParams.y, f);
		}
	}

	// 2. Ray-trace hemisphere AO against BVH
	float effRadius = max(uRayRadius * localRadiusMult, 0.01);
	int spp = clamp(uSamplesPerPixel, 1, 16);
	vec2 baseJitter = (uTemporalAccum != 0)
		? vec2(ignNoise(vec2(px) + float(uFrameIdx % 128) * 0.73), ignNoise(vec2(px.yx) + float((uFrameIdx + 31) % 128) * 1.19))
		: vec2(ignNoise(vec2(px)), ignNoise(vec2(px.yx) + vec2(13.7, 29.3)));

	vec3 rayOrigin = worldPos + N * max(uRayBias, dist * 0.0005);
	bool hasTLAS = (uHasTLAS != 0);
	bool applyCull = (uRespectCullMode != 0);
	bool doScissor = (uAlphaScissorSupport != 0);

	float occAccum = 0.0;
	vec3 bentAccum = vec3(0.0);
	for (int s = 0; s < 16; s++) {
		if (s >= spp) break;
		vec2 xi = r2Sample(s, baseJitter);
		vec3 dir = sampleCosineHemisphere(xi, N);
		float hitT = traceOcclusionDist(rayOrigin, dir, effRadius, hasTLAS, applyCull, doScissor, uTexArray);
		if (hitT < effRadius) {
			float w = pow(clamp(1.0 - hitT / effRadius, 0.0, 1.0), uFalloffPower);
			occAccum += w;
			bentAccum += dir * (1.0 - w);
		} else {
			bentAccum += dir;
		}
	}

	float rtAO = clamp(1.0 - occAccum / float(spp), 0.0, 1.0);
	vec3 bentN = (dot(bentAccum, bentAccum) > 1e-6) ? normalize(bentAccum) : N;

	// 3. Real-time 3D Texture AO (L1 SH hemisphere reconstruction)
	float volAO = 1.0;
	float volWeight = 0.0;
	if (uVolumeEnabled != 0 && uVolBlend > 0.001) {
		vec3 volSizeSafe = max(uVolSize, vec3(0.001));
		vec3 volUvw = (worldPos - uVolMin) / volSizeSafe;
		if (all(greaterThan(volUvw, vec3(0.0))) && all(lessThan(volUvw, vec3(1.0)))) {
			vec3 edgeDist = min(volUvw, 1.0 - volUvw);
			float edgeFade = smoothstep(0.0, 0.12, min(edgeDist.x, min(edgeDist.y, edgeDist.z)));
			vec3 volSamplePos = worldPos + N * (uVolVoxelSize * 0.45);
			vec3 volSampleUvw = clamp((volSamplePos - uVolMin) / volSizeSafe, vec3(0.001), vec3(0.999));
			vec4 vSample = textureLod(uRealtimeVolTex, volSampleUvw, 0.0);
			volAO = mix(1.0, clamp(vSample.a + 2.0 * dot(vSample.xyz, N), 0.0, 1.0), edgeFade);
			volWeight = edgeFade * uVolBlend;
			rtAO = mix(rtAO, min(rtAO, volAO), volWeight);
			if (dot(vSample.xyz, vSample.xyz) > 1e-5) {
				bentN = normalize(mix(bentN, normalize(vSample.xyz), volWeight * 0.5));
			}
		}
	}

	if (uDebugMode == 2) {
		imageStore(uOutAO, px, vec4(vec3(volAO), dist));
		return;
	}
	if (uDebugMode == 3) {
		imageStore(uOutAO, px, vec4(bentN * 0.5 + 0.5, dist));
		return;
	}

	// 4. BakedRTAO 3D Textures (L1 SH hemisphere reconstruction)
	vec3 finalRGB = mix(uAOColor, vec3(1.0), clamp(pow(rtAO, uPower), 0.0, 1.0));
	float effIntensity = uIntensity * localIntensityMult;
	finalRGB = mix(vec3(1.0), finalRGB, clamp(effIntensity, 0.0, 4.0));
	finalRGB = clamp(finalRGB, vec3(0.0), vec3(1.0));

	uint pixelLayerMask = texelFetch(uGLayerMask, px, 0).r;
	for (int b = 0; b < uBakedCount; b++) {
		BakedVolume bv = bakedVols[b];
		if ((pixelLayerMask & bv.masks.x) == 0u) continue;
		vec4 wp4 = vec4(worldPos, 1.0);
		vec3 uvw = vec3(dot(bv.invRow0, wp4), dot(bv.invRow1, wp4), dot(bv.invRow2, wp4));
		if (all(greaterThan(uvw, vec3(0.0))) && all(lessThan(uvw, vec3(1.0)))) {
			vec3 distEdge = abs(uvw - 0.5) * 2.0;
			float maxE = max(distEdge.x, max(distEdge.y, distEdge.z));
			float bFalloff = bv.params.y;
			float bFade = (bFalloff > 0.001) ? smoothstep(1.0, max(0.0, 1.0 - bFalloff), maxE) : 1.0;

			vec4 sp4 = vec4(worldPos + N * 0.25, 1.0);
			vec3 sampleUvw = clamp(vec3(dot(bv.invRow0, sp4), dot(bv.invRow1, sp4), dot(bv.invRow2, sp4)), vec3(0.001), vec3(0.999));
			vec4 bSample = sampleBakedVolume(b, sampleUvw);
			float hemiAO = clamp(bSample.a + 2.0 * dot(bSample.xyz, N), 0.0, 1.0);
			float bAO = mix(1.0, hemiAO, bFade);
			vec3 bRGB = mix(bv.colorInt.rgb, vec3(1.0), bAO);
			bRGB = mix(vec3(1.0), bRGB, clamp(bv.colorInt.w, 0.0, 4.0));
			bRGB = clamp(bRGB, vec3(0.0), vec3(1.0));

			int bmode = int(bv.params.x + 0.5);
			if (bmode == 0) finalRGB *= bRGB;
			else if (bmode == 1) finalRGB = min(finalRGB, bRGB);
			else finalRGB = mix(finalRGB, bRGB, bFade);
		}
	}

	// 5. Direct Light Preservation (evaluates actual scene lights & shadows against ambient)
	if (uDirectPreservation > 0.001 && uNumLights > 0) {
		vec3 directIrrad = vec3(0.0);
		vec3 shadowOrigin = worldPos + N * max(uRayBias, dist * 0.001);

		for (int li = 0; li < 32; li++) {
			if (li >= uNumLights) break;
			RTAOLight light = lights[li];
			int ltype = int(light.dirType.w + 0.5);

			vec3 lDir = vec3(0.0, 1.0, 0.0);
			float atten = 1.0;
			float maxShadowDist = light.spotParams.w;
			bool validLight = true;

			if (ltype == 0) {
				lDir = normalize(light.dirType.xyz);
			} else {
				vec3 toLight = light.posRange.xyz - worldPos;
				float distL = length(toLight);
				float range = max(light.posRange.w, 0.001);
				if (distL >= range || distL < 1e-5) {
					validLight = false;
				} else {
					lDir = toLight / distL;
					maxShadowDist = max(distL - max(uRayBias, 0.002), 0.01);
					float nd = distL / range;
					if (uUsePhysicalAtten != 0) {
						float nd4 = nd * nd;
						nd4 *= nd4;
						float fade = max(1.0 - nd4, 0.0);
						atten = (fade * fade) * pow(max(distL, 0.0001), -light.colorAtten.w);
					} else {
						atten = pow(max(1.0 - nd, 0.001), max(light.colorAtten.w, 0.01));
					}

					if (ltype == 2) {
						float scos = dot(-lDir, normalize(light.dirType.xyz));
						float spotMin = light.spotParams.x;
						if (scos <= spotMin) {
							validLight = false;
						} else {
							float spotRatio = clamp((scos - spotMin) / max(1.0 - spotMin, 1e-4), 0.0, 1.0);
							atten *= max(0.0, 1.0 - pow(1.0 - spotRatio, max(light.spotParams.y, 0.001)));
						}
					}
				}
			}

			if (!validLight || atten <= 0.001) continue;

			float NdotL = max(dot(N, lDir), 0.0);
			if (NdotL <= 0.0001) continue;

			vec3 contrib = light.colorAtten.rgb * (atten * NdotL * (1.0 / PI));
			float contribLum = dot(contrib, vec3(0.2126, 0.7152, 0.0722));
			if (contribLum <= 0.0005) continue;

			if (light.spotParams.z > 0.5) {
				if (traceOccludedAO(shadowOrigin, lDir, maxShadowDist, hasTLAS, applyCull, doScissor, uTexArray)) {
					continue;
				}
			}

			directIrrad += contrib;
		}

		float directLum = dot(directIrrad, vec3(0.2126, 0.7152, 0.0722));
		float ambLum = max(uAmbientLum, 0.05);
		float ratio = directLum / max(ambLum + directLum, 1e-4);
		float preserve = clamp(ratio * (clamp(uDirectPreservation, 0.0, 1.0) * 2.0), 0.0, 1.0);
		finalRGB = mix(finalRGB, vec3(1.0), preserve);
	}

	// 6. LocalRTAO Direct Occlusion Volumes
	for (int v = 0; v < uLocalCount; v++) {
		LocalVolume lv = localVols[v];
		if (int(lv.params.y + 0.5) != 0) continue; // MODE_DIRECT
		float f = evalLocalFactor(lv, worldPos);
		if (f <= 0.0001) continue;

		float targetAO = clamp(1.0 - lv.colorOcc.w * f, 0.0, 1.0);
		vec3 targetRGB = mix(lv.colorOcc.rgb, vec3(1.0), targetAO);
		int bmode = int(lv.params.z + 0.5);
		if (bmode == 0) {
			finalRGB *= targetRGB;
		} else if (bmode == 1) {
			finalRGB = min(finalRGB, targetRGB);
		} else if (bmode == 2) {
			finalRGB = max(finalRGB, targetRGB);
		} else {
			vec3 fullTarget = mix(lv.colorOcc.rgb, vec3(1.0), 1.0 - lv.colorOcc.w);
			finalRGB = mix(finalRGB, fullTarget, f);
		}
	}

	imageStore(uOutAO, px, vec4(clamp(finalRGB, vec3(0.0), vec3(1.0)), dist));
}
)GLSL";

static const char *RECONSTRUCT_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutWork;
layout (rgba16f, binding = 1) writeonly uniform image2D uOutFinal;

uniform sampler2D uRawTex;
uniform sampler2D uHistoryTex;
uniform sampler2D uGNormalMask;
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

void main() {
	ivec2 px = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(uOutWork);
	if (px.x >= sz.x || px.y >= sz.y) return;

	vec4 cur = texelFetch(uRawTex, px, 0);
	vec4 centerNR = texelFetch(uGNormalMask, px, 0);
	float centerDepth = texelFetch(uGDepth, px, 0).r;

	if (centerDepth >= 0.99999 || dot(centerNR.xyz, centerNR.xyz) < 0.01 || centerNR.w < 0.5) {
		vec4 skipVal = vec4(1.0, 1.0, 1.0, -1.0);
		imageStore(uOutWork, px, skipVal);
		if (uWriteFinal != 0) imageStore(uOutFinal, px, skipVal);
		return;
	}

	vec3 centerN = normalize(centerNR.xyz);
	vec2 uv = (vec2(px) + 0.5) / vec2(sz);
	vec4 wp = uInvViewProj * vec4(uv * 2.0 - 1.0, centerDepth * 2.0 - 1.0, 1.0);
	vec3 worldPos = wp.xyz / wp.w;
	float centerDist = length(uCamPos - worldPos);

	if (uCheckerboard != 0 && cur.a < -1.5) {
		vec3 sumRGB = vec3(0.0);
		float wSum = 0.0;
		const ivec2 offs[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
		for (int i = 0; i < 4; i++) {
			ivec2 np = clamp(px + offs[i], ivec2(0), sz - 1);
			vec4 nc = texelFetch(uRawTex, np, 0);
			if (nc.a >= 0.0) {
				vec4 nnr = texelFetch(uGNormalMask, np, 0);
				if (dot(nnr.xyz, nnr.xyz) > 0.01) {
					float nw = pow(max(0.0, dot(centerN, normalize(nnr.xyz))), 8.0) / (0.05 + abs(centerDist - nc.a));
					sumRGB += nc.rgb * nw;
					wSum += nw;
				}
			}
		}
		if (wSum > 1e-5) {
			cur = vec4(sumRGB / wSum, centerDist);
		} else {
			cur = vec4(1.0, 1.0, 1.0, centerDist);
		}
	}

	if (uTemporalAccum != 0 && uHistoryValid != 0 && cur.a >= 0.0) {
		vec4 prevClip = uPrevViewProj * vec4(worldPos, 1.0);
		if (prevClip.w > 0.001) {
			vec2 prevUv = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
			if (prevUv.x >= 0.0 && prevUv.x <= 1.0 && prevUv.y >= 0.0 && prevUv.y <= 1.0) {
				vec4 hist = textureLod(uHistoryTex, prevUv, 0.0);
				float prevDist = length(uPrevCamPos - worldPos);
				if (hist.a >= 0.0 && abs(prevDist - hist.a) < max(0.08, prevDist * 0.05)) {
					vec3 nMin = cur.rgb;
					vec3 nMax = cur.rgb;
					for (int dy = -1; dy <= 1; dy++) {
						for (int dx = -1; dx <= 1; dx++) {
							ivec2 np = clamp(px + ivec2(dx, dy), ivec2(0), sz - 1);
							vec4 nc = texelFetch(uRawTex, np, 0);
							if (nc.a >= 0.0) {
								nMin = min(nMin, nc.rgb);
								nMax = max(nMax, nc.rgb);
							}
						}
					}
					vec3 pad = max((nMax - nMin) * 0.2, vec3(0.015));
					vec3 clampedHist = clamp(hist.rgb, max(vec3(0.0), nMin - pad), min(vec3(1.0), nMax + pad));
					cur.rgb = mix(clampedHist, cur.rgb, clamp(uTemporalBlend, 0.02, 1.0));
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
uniform sampler2D uGNormalMask;
uniform int uStepSize;

void main() {
	ivec2 px = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(uOutTex);
	if (px.x >= sz.x || px.y >= sz.y) return;

	vec4 center = texelFetch(uSrcTex, px, 0);
	if (center.a < 0.0) {
		imageStore(uOutTex, px, center);
		return;
	}

	vec4 cNR = texelFetch(uGNormalMask, px, 0);
	if (dot(cNR.xyz, cNR.xyz) < 0.01) {
		imageStore(uOutTex, px, center);
		return;
	}

	vec3 cN = normalize(cNR.xyz);
	vec3 sumRGB = center.rgb * 0.25;
	float wSum = 0.25;

	const float kernel[3] = float[3](0.25, 0.5, 0.25);
	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			if (dx == 0 && dy == 0) continue;
			ivec2 np = clamp(px + ivec2(dx, dy) * uStepSize, ivec2(0), sz - 1);
			vec4 nc = texelFetch(uSrcTex, np, 0);
			if (nc.a < 0.0) continue;
			vec4 nNR = texelFetch(uGNormalMask, np, 0);
			if (dot(nNR.xyz, nNR.xyz) < 0.01) continue;

			float kw = kernel[dx + 1] * kernel[dy + 1];
			float nw = pow(max(0.0, dot(cN, normalize(nNR.xyz))), 16.0);
			float dw = exp(-abs(center.a - nc.a) * 3.5);
			float w = kw * nw * dw;

			sumRGB += nc.rgb * w;
			wSum += w;
		}
	}

	imageStore(uOutTex, px, vec4(sumRGB / max(wSum, 1e-5), center.a));
}
)GLSL";

static const char *COMPOSITE_CANVAS_SHADER = R"SHADER(
shader_type canvas_item;
render_mode unshaded, blend_mix;

uniform sampler2D u_ao_tex;
uniform float u_bilateral_upsample = 1.0;
uniform float u_keep_3d_linear = 0.0;
uniform float u_vflip = 0.0;
uniform int u_debug_mode = 0;

uniform vec4 u_inv_proj_0 = vec4(1.0, 0.0, 0.0, 0.0);
uniform vec4 u_inv_proj_1 = vec4(0.0, 1.0, 0.0, 0.0);
uniform vec4 u_inv_proj_2 = vec4(0.0, 0.0, 1.0, 0.0);
uniform vec4 u_inv_proj_3 = vec4(0.0, 0.0, 0.0, 1.0);

float linearize_dist(float raw_d, vec2 uv) {
	mat4 inv_proj = mat4(u_inv_proj_0, u_inv_proj_1, u_inv_proj_2, u_inv_proj_3);
	vec4 vp = inv_proj * vec4(uv * 2.0 - 1.0, min(raw_d, 0.99999) * 2.0 - 1.0, 1.0);
	return length(vp.xyz / vp.w);
}

vec3 srgb_to_linear(vec3 c) {
	c = max(c, vec3(0.0));
	return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

vec3 linear_to_srgb(vec3 c) {
	c = max(c, vec3(0.0));
	return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

void fragment() {
	vec2 m_uv = (u_vflip > 0.5) ? vec2(SCREEN_UV.x, 1.0 - SCREEN_UV.y) : SCREEN_UV;
	vec3 ao_rgb = vec3(1.0);
	bool valid = false;

	if (u_bilateral_upsample > 0.5) {
		float full_d = textureLod(DEPTH_TEXTURE, SCREEN_UV, 0.0).r;
		if (full_d < 0.99999) {
			float full_dist = linearize_dist(full_d, m_uv);
			vec2 tex_size = vec2(textureSize(u_ao_tex, 0));
			vec2 f_coord = m_uv * tex_size - 0.5;
			ivec2 base = ivec2(floor(f_coord));
			vec2 f = clamp(f_coord - vec2(base), vec2(0.0), vec2(1.0));
			ivec2 max_idx = ivec2(tex_size) - 1;

			vec4 s00 = texelFetch(u_ao_tex, clamp(base + ivec2(0, 0), ivec2(0), max_idx), 0);
			vec4 s10 = texelFetch(u_ao_tex, clamp(base + ivec2(1, 0), ivec2(0), max_idx), 0);
			vec4 s01 = texelFetch(u_ao_tex, clamp(base + ivec2(0, 1), ivec2(0), max_idx), 0);
			vec4 s11 = texelFetch(u_ao_tex, clamp(base + ivec2(1, 1), ivec2(0), max_idx), 0);

			float w00 = (s00.a >= 0.0) ? ((1.0 - f.x) * (1.0 - f.y) / (0.02 + abs(full_dist - s00.a) * 2.5)) : 0.0;
			float w10 = (s10.a >= 0.0) ? (      f.x  * (1.0 - f.y) / (0.02 + abs(full_dist - s10.a) * 2.5)) : 0.0;
			float w01 = (s01.a >= 0.0) ? ((1.0 - f.x) *       f.y  / (0.02 + abs(full_dist - s01.a) * 2.5)) : 0.0;
			float w11 = (s11.a >= 0.0) ? (      f.x  *       f.y  / (0.02 + abs(full_dist - s11.a) * 2.5)) : 0.0;

			float w_sum = w00 + w10 + w01 + w11;
			if (w_sum > 1e-5) {
				ao_rgb = (s00.rgb * w00 + s10.rgb * w10 + s01.rgb * w01 + s11.rgb * w11) / w_sum;
				valid = true;
			}
		}
	} else {
		vec4 s_fast = textureLod(u_ao_tex, m_uv, 0.0);
		if (s_fast.a >= 0.0) {
			ao_rgb = s_fast.rgb;
			valid = true;
		}
	}

	if (u_debug_mode != 0) {
		COLOR = vec4(valid ? ao_rgb : vec3(1.0), 1.0);
	} else {
		vec4 scene = textureLod(SCREEN_TEXTURE, SCREEN_UV, 0.0);
		if (!valid) {
			COLOR = vec4(scene.rgb, 1.0);
		} else {
			vec3 scene_lin = (u_keep_3d_linear > 0.5) ? max(scene.rgb, vec3(0.0)) : srgb_to_linear(scene.rgb);
			vec3 out_lin = scene_lin * ao_rgb;
			vec3 out_rgb = (u_keep_3d_linear > 0.5) ? out_lin : linear_to_srgb(out_lin);
			COLOR = vec4(out_rgb, 1.0);
		}
	}
}
)SHADER";

} // namespace RTAOShaders

#endif // RTAO_SHADERS_H
