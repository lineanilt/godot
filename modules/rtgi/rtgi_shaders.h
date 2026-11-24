#ifndef RTGI_SHADERS_H
#define RTGI_SHADERS_H

namespace RTGIShaders {

static const char *GBUF_VERT_GLSL = R"GLSL(
#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUv;
layout (location = 3) in float aSurfIdx;

uniform mat4 uModel;
uniform mat4 uViewProj;
uniform int uMatOffset;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 vUv;
flat out int vMatId;

void main() {
	vec4 wp = uModel * vec4(aPos, 1.0);
	vWorldPos = wp.xyz;
	vWorldNormal = normalize(transpose(inverse(mat3(uModel))) * aNormal);
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

layout (location = 0) out vec4 outNormalMask;
layout (location = 1) out vec4 outAlbedoMask;

uniform sampler2DArray uTexArray;
uniform int uSkipGI;
uniform int uRespectCullMode;
uniform int uFlipFacing;
uniform int uAlphaConsideration;

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
	if (alphaMode == 2) discard;

	vec2 uv = vUv * m.uvScaleOffset.xy + m.uvScaleOffset.zw;
	vec4 alb4 = m.albedoAlpha;
	if (m.texLayers.x >= 0) alb4 *= texture(uTexArray, vec3(uv, float(m.texLayers.x)));

	if (uAlphaConsideration != 0 && alphaMode == 1 && alb4.a < m.pbrParams.y) {
		discard;
	}

	bool isUnshaded = (m.extraTex.w & 8) != 0;
	if (uSkipGI != 0 || isUnshaded) {
		outNormalMask = vec4(0.0);
		outAlbedoMask = vec4(0.0);
		return;
	}

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

	vec3 diffAlbedo = clamp(alb4.rgb * (1.0 - clamp(metal, 0.0, 1.0)), vec3(0.0), vec3(1.0));
	outNormalMask = vec4(N, 1.0);
	outAlbedoMask = vec4(diffAlbedo, 1.0);
}
)GLSL";

static const char *DOWNSAMPLE_GBUF_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutLowNormalMask;
layout (r32f, binding = 1) writeonly uniform image2D uOutLowDepth;

uniform sampler2D uFullNormalMask;
uniform sampler2D uFullDepth;

void main() {
	ivec2 px = ivec2(gl_GlobalInvocationID.xy);
	ivec2 lowSz = imageSize(uOutLowNormalMask);
	if (px.x >= lowSz.x || px.y >= lowSz.y) return;

	ivec2 fullSz = textureSize(uFullNormalMask, 0);
	ivec2 srcPx = clamp(ivec2((vec2(px) + 0.5) * vec2(fullSz) / vec2(lowSz)), ivec2(0), fullSz - 1);

	vec4 nm = texelFetch(uFullNormalMask, srcPx, 0);
	float d = texelFetch(uFullDepth, srcPx, 0).r;

	imageStore(uOutLowNormalMask, px, nm);
	imageStore(uOutLowDepth, px, vec4(d, 0.0, 0.0, 0.0));
}
)GLSL";

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

// Common BVH & Shading code shared by 3D Volume and Screen-Space Path Tracer
// (Expects GL43Raytracer::get_glsl_common() prepended)
static const char *GI_TRACE_COMMON_GLSL = R"GLSL(
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

float hash31(vec3 p) {
	p = fract(p * 0.1031);
	p += dot(p, p.yzx + 33.33);
	return fract((p.x + p.y) * p.z);
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

bool intersectTriangleGI(RTTriangle tri, vec3 ro, vec3 rd, float maxT, int matOffset, int cullMode, out float outDet, out float outT, out float outU, out float outV) {
	if (!rayTriangleGeom(tri, ro, rd, maxT, outDet, outT, outU, outV)) return false;
	if (cullMode != 0) {
		int mId = clamp(matOffset + floatBitsToInt(tri.p0.w), 0, materials.length() - 1);
		int cm = materials[mId].extraTex.w & 3;
		if (cullMode == 1) {
			if (cm == 0 && outDet >= 0.0) return false;
			if (cm == 1 && outDet <= 0.0) return false;
		} else {
			if (cm == 0 && outDet <= 0.0) return false;
			if (cm == 1 && outDet >= 0.0) return false;
		}
	}
	return true;
}

struct HitInfo {
	bool hit;
	bool hitBackface;
	float t;
	vec3 pos;
	vec3 normal;
	vec2 uv;
	int matId;
	int instIdx;
	int triIdx;
};

void traverseBLAS(int rootNode, int matOffset, bool applyCull, vec3 ro, vec3 rd, vec3 invRd, inout float bestT, out int bestTri, out float bestU, out float bestV, out float bestDet) {
	bestTri = -1;
	bestU = 0.0;
	bestV = 0.0;
	bestDet = 0.0;

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
				float det, t, u, v;
				int triIdx = leftOrFirst + i;
				if (intersectTriangleGI(triangles[triIdx], ro, rd, bestT, matOffset, cullMode, det, t, u, v)) {
					bestT = t;
					bestTri = triIdx;
					bestU = u;
					bestV = v;
					bestDet = det;
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

bool traverseBLASShadow(int rootNode, int matOffset, int shadowCullMode, vec3 ro, vec3 rd, vec3 invRd, float maxT, bool checkAlpha, sampler2DArray texArr, inout float vis) {
	if (intersectAABB(blasNodes[rootNode].aabbMin.xyz, blasNodes[rootNode].aabbMax.xyz, ro, invRd, maxT) >= maxT)
		return true;

	int stackId[20];
	int sp = 0;
	stackId[sp++] = rootNode;

	while (sp > 0) {
		int nodeIdx = stackId[--sp];
		BVHNode node = blasNodes[nodeIdx];
		int leftOrFirst = floatBitsToInt(node.aabbMin.w);
		int count = floatBitsToInt(node.aabbMax.w);

		if (count > 0) {
			for (int i = 0; i < count; i++) {
				float det, t, u, v;
				RTTriangle tri = triangles[leftOrFirst + i];
				if (intersectTriangleGI(tri, ro, rd, maxT, matOffset, shadowCullMode, det, t, u, v)) {
					if (!checkAlpha) {
						vis = 0.0;
						return false;
					}
					int mId = clamp(matOffset + floatBitsToInt(tri.p0.w), 0, materials.length() - 1);
					RTMaterial hm = materials[mId];
					int alphaMode = hm.extraTex.z;
					if (alphaMode == 0) {
						vis = 0.0;
						return false;
					}
					float w = 1.0 - u - v;
					vec2 uv = (w * tri.uv01.xy + u * tri.uv01.zw + v * tri.uv2pad.xy) * hm.uvScaleOffset.xy + hm.uvScaleOffset.zw;
					float texA = (hm.texLayers.x >= 0) ? textureLod(texArr, vec3(uv, float(hm.texLayers.x)), 2.0).a : 1.0;
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
			if (intersectAABB(blasNodes[left].aabbMin.xyz, blasNodes[left].aabbMax.xyz, ro, invRd, maxT) < maxT && sp < 20)
				stackId[sp++] = left;
			if (intersectAABB(blasNodes[right].aabbMin.xyz, blasNodes[right].aabbMax.xyz, ro, invRd, maxT) < maxT && sp < 20)
				stackId[sp++] = right;
		}
	}
	return true;
}

HitInfo traceScene(vec3 ro, vec3 rd, float maxT, bool hasTLAS, bool applyCull) {
	HitInfo res;
	res.hit = false;
	res.hitBackface = false;
	res.t = maxT;
	res.matId = 0;
	res.instIdx = -1;
	res.triIdx = -1;
	res.uv = vec2(0.0);

	if (!hasTLAS || tlasNodes.length() == 0) return res;

	vec3 invRd = safeInvDir(rd);
	int stackId[20];
	float stackT[20];
	int sp = 0;

	float tRoot = intersectAABB(tlasNodes[0].aabbMin.xyz, tlasNodes[0].aabbMax.xyz, ro, invRd, res.t);
	if (tRoot >= res.t) return res;

	stackId[sp] = 0;
	stackT[sp] = tRoot;
	sp++;

	int winInst = -1;
	int winTri = -1;
	float winU = 0.0, winV = 0.0, winDet = 0.0;

	while (sp > 0) {
		sp--;
		int nodeIdx = stackId[sp];
		if (stackT[sp] >= res.t) continue;

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

				int tri; float u, v, det;
				float prevT = res.t;
				traverseBLAS(blasRoot, matOffset, applyCull, lro, lrd, invLrd, res.t, tri, u, v, det);
				if (tri >= 0 && res.t < prevT) {
					winInst = instIdx;
					winTri = tri;
					winU = u;
					winV = v;
					winDet = det;
				}
			}
		} else {
			int left = nodeIdx + 1;
			int right = leftOrFirst;
			float tL = intersectAABB(tlasNodes[left].aabbMin.xyz, tlasNodes[left].aabbMax.xyz, ro, invRd, res.t);
			float tR = intersectAABB(tlasNodes[right].aabbMin.xyz, tlasNodes[right].aabbMax.xyz, ro, invRd, res.t);
			if (tL < res.t && tR < res.t) {
				if (tL < tR) {
					if (sp < 19) { stackId[sp] = right; stackT[sp] = tR; sp++; stackId[sp] = left; stackT[sp] = tL; sp++; }
				} else {
					if (sp < 19) { stackId[sp] = left; stackT[sp] = tL; sp++; stackId[sp] = right; stackT[sp] = tR; sp++; }
				}
			} else if (tL < res.t) {
				if (sp < 20) { stackId[sp] = left; stackT[sp] = tL; sp++; }
			} else if (tR < res.t) {
				if (sp < 20) { stackId[sp] = right; stackT[sp] = tR; sp++; }
			}
		}
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
		if (dot(worldN, rd) > 0.0) {
			res.hitBackface = true;
			worldN = -worldN;
		}
		res.normal = worldN;
		res.uv = w * tri.uv01.xy + winU * tri.uv01.zw + winV * tri.uv2pad.xy;
		res.matId = clamp(floatBitsToInt(inst.aabbMax.w) + floatBitsToInt(tri.p0.w), 0, materials.length() - 1);
	}
	return res;
}

float traceShadow(vec3 ro, vec3 lightDir, float maxDist, bool hasTLAS, bool applyCull, bool checkAlpha, sampler2DArray texArr) {
	if (!hasTLAS || tlasNodes.length() == 0) return 1.0;
	vec3 invRd = safeInvDir(lightDir);
	int stackId[20];
	int sp = 0;
	if (intersectAABB(tlasNodes[0].aabbMin.xyz, tlasNodes[0].aabbMax.xyz, ro, invRd, maxDist) >= maxDist)
		return 1.0;

	float vis = 1.0;
	stackId[sp++] = 0;
	while (sp > 0) {
		int nodeIdx = stackId[--sp];
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
				int shadowCull = (applyCull && (inst.flags.x & 4) == 0) ? 2 : 0;
				vec3 lro = (inst.invModel * vec4(ro, 1.0)).xyz;
				vec3 lrd = (inst.invModel * vec4(lightDir, 0.0)).xyz;
				if (!traverseBLASShadow(floatBitsToInt(inst.aabbMin.w), floatBitsToInt(inst.aabbMax.w), shadowCull, lro, lrd, safeInvDir(lrd), maxDist, checkAlpha, texArr, vis))
					return 0.0;
			}
		} else {
			int left = nodeIdx + 1;
			int right = leftOrFirst;
			if (intersectAABB(tlasNodes[left].aabbMin.xyz, tlasNodes[left].aabbMax.xyz, ro, invRd, maxDist) < maxDist && sp < 20)
				stackId[sp++] = left;
			if (intersectAABB(tlasNodes[right].aabbMin.xyz, tlasNodes[right].aabbMax.xyz, ro, invRd, maxDist) < maxDist && sp < 20)
				stackId[sp++] = right;
		}
	}
	return vis;
}

vec3 evalSkySH(vec3 worldN, mat3 skyInvBasis) {
	vec3 n = normalize(skyInvBasis * worldN);
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

float sampleChanLod(sampler2DArray texArr, int layer, int ch, vec2 uv, float lod) {
	vec4 t = textureLod(texArr, vec3(uv, float(layer)), lod);
	if (ch == 0) return t.r;
	if (ch == 1) return t.g;
	if (ch == 2) return t.b;
	if (ch == 3) return t.a;
	return dot(t.rgb, vec3(0.299, 0.587, 0.114));
}

vec3 shadeDirectGI(
		RTMaterial mat,
		vec3 hitPos,
		vec3 N,
		vec3 albedo,
		float metallic,
		vec3 emission,
		float matAO,
		int numLights,
		bool hasTLAS,
		bool applyCull,
		bool doTraceShadows,
		bool checkAlphaShadow,
		bool allowLocalLights,
		float shadowBias,
		float emissiveIntensity,
		float ambientBounceApprox,
		mat3 skyInvBasis,
		float skyIndirectEnergy,
		sampler2DArray texArr) {
	int mflags = mat.extraTex.w;
	vec3 emissive = emission * emissiveIntensity;
	if ((mflags & 8) != 0) {
		return albedo + emissive;
	}

	vec3 col = emissive;
	float diffuseScale = 1.0 - metallic;
	if (diffuseScale <= 0.001) {
		return col;
	}

	float aoLightScale = mix(1.0, matAO, mat.pbrParams.w);
	vec3 skyIrradiance = evalSkySH(N, skyInvBasis) * skyIndirectEnergy;

	if (ambientBounceApprox > 0.001) {
		for (int i = 0; i < numLights; i++) {
			if (int(lights[i].dirType.w + 0.5) == 0) {
				vec3 L = normalize(lights[i].dirType.xyz);
				float wrap = clamp(0.45 - 0.35 * dot(N, L), 0.0, 1.0);
				col += (lights[i].colorAtten.rgb / PI) * skyIrradiance * wrap * 0.15 * ambientBounceApprox * matAO * diffuseScale * albedo;
			}
		}
	}

	vec3 shadowRo = hitPos + N * max(shadowBias, 0.002);
	for (int i = 0; i < numLights; i++) {
		RTLight light = lights[i];
		int ltype = int(light.dirType.w + 0.5);
		if (!allowLocalLights && ltype != 0) continue;

		vec3 qDir = N;
		float qMaxDist = 200.0;
		float atten = 1.0;
		bool validLight = true;

		if (ltype == 0) {
			qDir = normalize(light.dirType.xyz);
		} else {
			vec3 toLight = light.posRange.xyz - hitPos;
			float dist = length(toLight);
			float range = max(light.posRange.w, 0.001);
			if (dist >= range || dist < 1e-5) {
				validLight = false;
			} else {
				qDir = toLight / dist;
				qMaxDist = dist - max(shadowBias, 0.002);
				float normDist = clamp(1.0 - dist / range, 0.0, 1.0);
				atten = pow(normDist, max(light.colorAtten.w, 0.01));
				if (ltype == 2) {
					float scos = dot(-qDir, normalize(light.dirType.xyz));
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
		float NdotL = max(dot(N, qDir), 0.0);
		if (NdotL <= 0.0) continue;

		vec3 contrib = (diffuseScale * albedo / PI) * light.colorAtten.rgb * atten * NdotL * aoLightScale;
		if (dot(contrib, vec3(0.2126, 0.7152, 0.0722)) <= 0.0002) continue;

		float vis = (doTraceShadows && light.spotParams.z > 0.5)
			? traceShadow(shadowRo, qDir, qMaxDist, hasTLAS, applyCull, checkAlphaShadow, texArr)
			: 1.0;
		col += contrib * vis;
	}

	return col;
}
)GLSL";

// Real-Time 3D Irradiance Volume Compute Shader (camera-centered multi-bounce cache)
static const char *VOLUME_COMPUTE_GLSL_BODY = R"GLSL(
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout (rgba16f, binding = 0) writeonly uniform image3D uOutGIVolume;
uniform sampler2DArray uTexArray;
uniform sampler3D uPrevHistVolume;

uniform vec3 uVolMin;
uniform vec3 uPrevVolMin;
uniform vec3 uVolSize;
uniform float uVolRayDist;
uniform int uVolSamples;
uniform float uRayBias;
uniform float uShadowBias;
uniform int uHasTLAS;
uniform int uNumLights;
uniform int uRespectCullMode;
uniform int uTraceShadows;
uniform int uSkyIndirectEnabled;
uniform float uSkyIndirectEnergy;
uniform mat3 uSkyInvBasis;
uniform float uEmissiveIntensity;
uniform float uVolumeFeedback;
uniform float uFireflyClamp;
uniform int uVolHistValid;
uniform int uTemporalNoise;
uniform int uVolFrame;

void main() {
	ivec3 vox = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 sz = imageSize(uOutGIVolume);
	if (vox.x >= sz.x || vox.y >= sz.y || vox.z >= sz.z) {
		return;
	}

	vec3 uvw = (vec3(vox) + 0.5) / vec3(sz);
	vec3 worldPos = uVolMin + uvw * uVolSize;
	vec3 voxelStep = max(uVolSize / vec3(sz), vec3(0.001));
	vec3 worldCell = floor(worldPos / voxelStep + 0.5);

	int samples = clamp(uVolSamples, 1, 32);
	float frameF = (uTemporalNoise != 0) ? float(uVolFrame & 255) : 0.0;
	float jitPhi = fract(hash31(worldCell) + frameF * 0.61803398875) * 6.2831853;
	float jitY = fract(hash31(worldCell + vec3(17.0, 31.0, 47.0)) + frameF * 0.7548776662);
	const float goldenAngle = 2.399963229728653;

	bool hasTLAS = (uHasTLAS != 0);
	bool applyCull = (uRespectCullMode != 0);
	bool doShadows = (uTraceShadows != 0);

	vec3 irradSum = vec3(0.0);
	int backfaceHits = 0;

	for (int s = 0; s < 32; s++) {
		if (s >= samples) break;
		float sy = 1.0 - (2.0 * (float(s) + jitY) / float(samples));
		float sr = sqrt(max(0.0, 1.0 - sy * sy));
		float phi = float(s) * goldenAngle + jitPhi;
		vec3 dir = vec3(cos(phi) * sr, sy, sin(phi) * sr);

		HitInfo h = traceScene(worldPos + dir * max(uRayBias, 0.01), dir, uVolRayDist, hasTLAS, applyCull);
		vec3 rad = vec3(0.0);
		if (!h.hit) {
			if (uSkyIndirectEnabled != 0) {
				rad = evalSkySH(dir, uSkyInvBasis) * uSkyIndirectEnergy;
			}
		} else {
			if (h.hitBackface) {
				backfaceHits++;
			}
			RTMaterial hm = materials[h.matId];
			vec2 huv = h.uv * hm.uvScaleOffset.xy + hm.uvScaleOffset.zw;
			vec3 hAlb = hm.albedoAlpha.rgb;
			if (hm.texLayers.x >= 0) hAlb *= textureLod(uTexArray, vec3(huv, float(hm.texLayers.x)), 2.5).rgb;
			float hMetal = hm.emissionMetal.w;
			vec3 hEmis = hm.emissionMetal.rgb;
			if (hm.texLayers.w >= 0) hEmis *= textureLod(uTexArray, vec3(huv, float(hm.texLayers.w)), 2.5).rgb;

			rad = shadeDirectGI(
				hm, h.pos, h.normal, hAlb, hMetal, hEmis, 1.0,
				uNumLights, hasTLAS, applyCull, doShadows, false, false,
				uShadowBias, uEmissiveIntensity, 0.0, uSkyInvBasis, uSkyIndirectEnergy, uTexArray);

			if (uVolHistValid != 0 && uVolumeFeedback > 0.001) {
				vec3 hUvw = (h.pos + h.normal * max(voxelStep.x, 0.1) - uPrevVolMin) / max(uVolSize, vec3(0.001));
				if (all(greaterThan(hUvw, vec3(0.0))) && all(lessThan(hUvw, vec3(1.0)))) {
					vec4 prevBounce = textureLod(uPrevHistVolume, hUvw, 0.0);
					if (prevBounce.a >= 0.0) {
						rad += prevBounce.rgb * hAlb * (1.0 - hMetal) * uVolumeFeedback;
					}
				}
			}
		}

		if (uFireflyClamp > 0.01) {
			float lum = dot(rad, vec3(0.2126, 0.7152, 0.0722));
			if (lum > uFireflyClamp) {
				rad *= uFireflyClamp / lum;
			}
		}
		irradSum += rad;
	}

	vec4 curVal;
	if (backfaceHits * 2 >= samples) {
		curVal = vec4(0.0, 0.0, 0.0, -1.0);
	} else {
		curVal = vec4(irradSum / float(samples), 1.0);
	}

	if (uVolHistValid != 0 && curVal.a >= 0.0) {
		ivec3 prevVox = vox + ivec3(round((uVolMin - uPrevVolMin) / voxelStep));
		if (all(greaterThanEqual(prevVox, ivec3(0))) && all(lessThan(prevVox, sz))) {
			vec4 prevVal = texelFetch(uPrevHistVolume, prevVox, 0);
			if (prevVal.a >= 0.0) {
				curVal = mix(prevVal, curVal, 0.18);
			}
		}
	}

	imageStore(uOutGIVolume, vox, curVal);
}
)GLSL";

static const char *VOLUME_FILTER_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout (rgba16f, binding = 0) writeonly uniform image3D uOutGIVolume;
uniform sampler3D uRawGIVolume;

void main() {
	ivec3 vox = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 sz = imageSize(uOutGIVolume);
	if (vox.x >= sz.x || vox.y >= sz.y || vox.z >= sz.z) {
		return;
	}

	vec4 center = texelFetch(uRawGIVolume, vox, 0);
	vec3 accum = vec3(0.0);
	float wSum = 0.0;

	if (center.a >= 0.0) {
		accum = center.rgb * 4.0;
		wSum = 4.0;
		const ivec3 offs[6] = ivec3[6](
			ivec3(-1, 0, 0), ivec3(1, 0, 0),
			ivec3(0, -1, 0), ivec3(0, 1, 0),
			ivec3(0, 0, -1), ivec3(0, 0, 1)
		);
		for (int i = 0; i < 6; i++) {
			ivec3 np = clamp(vox + offs[i], ivec3(0), sz - 1);
			vec4 s = texelFetch(uRawGIVolume, np, 0);
			if (s.a >= 0.0) {
				accum += s.rgb;
				wSum += 1.0;
			}
		}
	} else {
		for (int dz = -1; dz <= 1; dz++) {
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					if (dx == 0 && dy == 0 && dz == 0) continue;
					ivec3 np = clamp(vox + ivec3(dx, dy, dz), ivec3(0), sz - 1);
					vec4 s = texelFetch(uRawGIVolume, np, 0);
					if (s.a >= 0.0) {
						float w = (abs(dx) + abs(dy) + abs(dz) == 1) ? 2.0 : 1.0;
						accum += s.rgb * w;
						wSum += w;
					}
				}
			}
		}
	}

	vec4 outVal = (wSum > 0.001) ? vec4(accum / wSum, 1.0) : vec4(0.0, 0.0, 0.0, 0.0);
	imageStore(uOutGIVolume, vox, outVal);
}
)GLSL";

// Main Screen-Space Diffuse Path Tracing Compute Shader
static const char *RT_COMPUTE_HEADER_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutGI;

uniform sampler2D uGNormalMask;
uniform sampler2D uGDepth;
uniform sampler2DArray uTexArray;
uniform sampler2D uPanoramaTex;
uniform sampler3D uGIVolumeTex;

uniform mat4 uInvViewProj;
uniform vec3 uCamPos;
uniform int uUsePanorama;
uniform int uPanoramaToLinear;
uniform float uSkyEnergy;
uniform mat3 uSkyInvBasis;
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

uniform int uHasTLAS;
uniform int uNumLights;
uniform int uRaysPerPixel;
uniform int uMaxBounces;
uniform float uMaxRayDistance;
uniform float uBounceDistFalloff;
uniform float uBounceEnergyBoost;
uniform float uBounceThroughputCutoff;
uniform float uEmissiveIntensity;
uniform float uAmbientBounceApprox;
uniform int uSkyIndirectEnabled;
uniform float uSkyIndirectEnergy;
uniform int uBounceLocalLights;
uniform int uNormalMapsInBounces;
uniform int uRespectCullMode;
uniform int uTraceShadows;
uniform int uShadowBounces;
uniform float uShadowMaxDist;
uniform float uRayBias;
uniform float uShadowBias;
uniform float uLodBias;
uniform float uDistLodStrength;
uniform float uFireflyClamp;

uniform int uAlphaConsideration;
uniform int uAlphaRayTransparency;
uniform int uAlphaMaxSteps;
uniform int uAlphaShadowTransmission;

uniform int uVolumeEnabled;
uniform vec3 uVolMin;
uniform vec3 uVolSize;
uniform float uVolVoxelSize;
uniform float uVolumeFeedback;

uniform int uCheckerboard;
uniform int uTemporalNoise;
uniform int uFrameIdx;
uniform int uDebugMode;
)GLSL";

static const char *RT_COMPUTE_BODY_GLSL = R"GLSL(
vec3 samplePanorama(vec3 dir, float lod) {
	vec3 d = normalize(uSkyInvBasis * dir);
	float u = atan(d.x, -d.z) / (2.0 * PI) + 0.5;
	float v = acos(clamp(d.y, -1.0, 1.0)) / PI;
	vec3 c = textureLod(uPanoramaTex, vec2(u, clamp(v, 0.001, 0.999)), lod).rgb;
	if (uPanoramaToLinear == 1) {
		c = pow(max(c, vec3(0.0)), vec3(2.2));
	}
	return c * uSkyEnergy;
}

vec3 sampleProceduralSky(vec3 dir) {
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
		float maxAng = uSkySunAngleMax + 15.0;
		if (angDeg < maxAng) {
			float c2 = clamp((angDeg - uSkySunAngleMin) / max(maxAng - uSkySunAngleMin, 0.001), 0.0, 1.0);
			float sunAmt = (angDeg <= uSkySunAngleMin) ? 1.0 : pow(1.0 - c2, 1.0 / max(uSkySunCurve, 0.001));
			col = mix(col, max(col, uSkySunColor * uSkySunEnergy), sunAmt * 0.5);
		}
	}
	return col;
}

vec3 sampleSkyDiffuseRay(vec3 dir) {
	vec3 col;
	if (uUsePanorama == 1) {
		float maxLod = max(float(textureQueryLevels(uPanoramaTex)) - 1.0, 0.0);
		col = mix(samplePanorama(dir, maxLod * 0.4), evalSkySH(dir, uSkyInvBasis), 0.35);
	} else {
		col = mix(sampleProceduralSky(dir), evalSkySH(dir, uSkyInvBasis), 0.25);
	}
	return col * uSkyIndirectEnergy;
}

vec3 sampleGIVolume(vec3 worldPos, vec3 N) {
	vec3 volSizeSafe = max(uVolSize, vec3(0.001));
	vec3 samplePos = worldPos + N * (uVolVoxelSize * 0.45);
	vec3 uvw = (samplePos - uVolMin) / volSizeSafe;
	if (any(lessThanEqual(uvw, vec3(0.0))) || any(greaterThanEqual(uvw, vec3(1.0)))) {
		return vec3(0.0);
	}
	vec3 edgeDist = min(uvw, 1.0 - uvw);
	float edgeFade = smoothstep(0.0, 0.12, min(edgeDist.x, min(edgeDist.y, edgeDist.z)));
	vec4 vSample = textureLod(uGIVolumeTex, clamp(uvw, vec3(0.001), vec3(0.999)), 0.0);
	return max(vSample.rgb, vec3(0.0)) * edgeFade;
}

void evalMaterialAtHit(RTMaterial mat, HitInfo hit, float texLod, out vec3 outNormal, out vec3 albedo, out float metallic, out vec3 emission, out float matAO) {
	vec2 uv = hit.uv * mat.uvScaleOffset.xy + mat.uvScaleOffset.zw;
	albedo = mat.albedoAlpha.rgb;
	if (mat.texLayers.x >= 0) albedo *= textureLod(uTexArray, vec3(uv, float(mat.texLayers.x)), texLod).rgb;

	metallic = mat.emissionMetal.w;
	if (texLod < 5.0 && mat.texLayers.z >= 0) {
		metallic *= sampleChanLod(uTexArray, mat.texLayers.z, mat.chanNormal.y, uv, texLod);
	}

	emission = mat.emissionMetal.rgb;
	if (mat.texLayers.w >= 0) emission *= textureLod(uTexArray, vec3(uv, float(mat.texLayers.w)), texLod).rgb;

	matAO = 1.0;
	if (texLod < 5.0 && mat.extraTex.x >= 0) {
		matAO = sampleChanLod(uTexArray, mat.extraTex.x, mat.extraTex.y, uv, texLod);
	}

	outNormal = hit.normal;
	if (uNormalMapsInBounces != 0 && texLod < 4.5 && mat.chanNormal.z >= 0 && hit.triIdx >= 0 && hit.instIdx >= 0) {
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

void main() {
	ivec2 px = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(uOutGI);
	if (px.x >= sz.x || px.y >= sz.y) return;

	if (uCheckerboard != 0 && (((px.x + px.y) & 1) != (uFrameIdx & 1))) {
		imageStore(uOutGI, px, vec4(0.0, 0.0, 0.0, -2.0));
		return;
	}

	float depth = texelFetch(uGDepth, px, 0).r;
	vec4 normMask = texelFetch(uGNormalMask, px, 0);
	if (depth >= 0.99999 || dot(normMask.xyz, normMask.xyz) < 0.01 || normMask.w < 0.5) {
		imageStore(uOutGI, px, vec4(0.0, 0.0, 0.0, -1.0));
		return;
	}

	vec2 uv = (vec2(px) + 0.5) / vec2(sz);
	vec4 wp = uInvViewProj * vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
	vec3 worldPos = wp.xyz / wp.w;
	float dist = length(uCamPos - worldPos);
	vec3 N = normalize(normMask.xyz);

	if (uDebugMode == 4) {
		vec3 volCol = (uVolumeEnabled != 0) ? sampleGIVolume(worldPos, N) : vec3(0.0);
		imageStore(uOutGI, px, vec4(volCol, dist));
		return;
	}

	int spp = clamp(uRaysPerPixel, 1, 16);
	float jt0 = ignNoise(vec2(px));
	float jt1 = ignNoise(vec2(px.yx) + vec2(19.3, 41.7));
	vec2 baseJitter = (uTemporalNoise != 0)
		? vec2(fract(jt0 + float(uFrameIdx & 255) * 0.61803398875),
			   fract(jt1 + float(uFrameIdx & 255) * 0.7548776662))
		: vec2(jt0, jt1);

	bool hasTLAS = (uHasTLAS != 0);
	bool applyCull = (uRespectCullMode != 0);
	bool checkAlphaShadow = (uAlphaConsideration != 0 && uAlphaShadowTransmission != 0);
	int maxAlphaSteps = (uAlphaConsideration != 0 && uAlphaRayTransparency != 0) ? max(1, uAlphaMaxSteps) : 1;

	vec3 totalAccum = vec3(0.0);
	vec3 rayOrigin = worldPos + N * max(uRayBias, dist * 0.001);
	vec3 rayDir = sampleCosineHemisphere(r2Sample(0, baseJitter), N);
	vec3 throughput = vec3(1.0);
	vec3 sampleAccum = vec3(0.0);
	float accumDist = dist;
	float bounceRange = uMaxRayDistance;
	float remT = bounceRange;
	int s = 0;
	int bounce = 0;
	int astep = 0;
	int maxTotalIters = spp * uMaxBounces * maxAlphaSteps;

	for (int iter = 0; iter < maxTotalIters; iter++) {
		HitInfo h = traceScene(rayOrigin, rayDir, remT, hasTLAS, applyCull);
		bool startNextSample = false;

		if (!h.hit) {
			if (uSkyIndirectEnabled != 0) {
				sampleAccum += throughput * sampleSkyDiffuseRay(rayDir);
			}
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
						vec3 tEmis = hm.emissionMetal.rgb * uEmissiveIntensity * hitAlpha;
						sampleAccum += throughput * tEmis;
						throughput *= mix(vec3(1.0), max(albSample.rgb, vec3(0.25)), hitAlpha * 0.5) * (1.0 - hitAlpha);
					}
					if (max(max(throughput.r, throughput.g), throughput.b) < uBounceThroughputCutoff) {
						startNextSample = true;
					} else {
						float adv = max(h.t, 0.002);
						accumDist += adv;
						remT -= adv;
						if (remT <= 0.01) {
							if (uSkyIndirectEnabled != 0) {
								sampleAccum += throughput * sampleSkyDiffuseRay(rayDir);
							}
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
				float texLod = clamp(distLod + uLodBias + float(bounce) * 1.5 + 1.0, 0.0, 8.0);

				vec3 hN, hAlb, hEmis;
				float hMetal, hAO;
				evalMaterialAtHit(hm, h, texLod, hN, hAlb, hMetal, hEmis, hAO);

				bool doShadows = (uTraceShadows != 0) && (bounce < uShadowBounces) && (accumDist < uShadowMaxDist);
				bool allowLocal = (bounce == 0) || (uBounceLocalLights != 0 && bounce == 1);
				vec3 direct = shadeDirectGI(
					hm, h.pos, hN, hAlb, hMetal, hEmis, hAO,
					uNumLights, hasTLAS, applyCull, doShadows, checkAlphaShadow, allowLocal,
					uShadowBias, uEmissiveIntensity, uAmbientBounceApprox,
					uSkyInvBasis, uSkyIndirectEnergy, uTexArray);

				sampleAccum += throughput * direct;
				throughput *= hAlb * (1.0 - hMetal) * uBounceEnergyBoost;

				bool isUnshaded = (hm.extraTex.w & 8) != 0;
				if (isUnshaded || bounce + 1 >= uMaxBounces) {
					if (!isUnshaded && uVolumeEnabled != 0 && uVolumeFeedback > 0.001) {
						sampleAccum += throughput * sampleGIVolume(h.pos, hN) * uVolumeFeedback;
					}
					startNextSample = true;
				} else if (max(max(throughput.r, throughput.g), throughput.b) < uBounceThroughputCutoff * float(bounce + 1)) {
					startNextSample = true;
				} else {
					bounceRange *= uBounceDistFalloff;
					remT = bounceRange;
					rayOrigin = h.pos + hN * max(uRayBias, accumDist * 0.001);
					vec2 bxi = r2Sample(s * 4 + bounce + 1, baseJitter);
					rayDir = sampleCosineHemisphere(bxi, hN);
					bounce++;
					astep = 0;
				}
			}
		}

		if (startNextSample) {
			if (uFireflyClamp > 0.01) {
				float sLum = dot(sampleAccum, vec3(0.2126, 0.7152, 0.0722));
				if (sLum > uFireflyClamp) {
					sampleAccum *= uFireflyClamp / sLum;
				}
			}
			totalAccum += sampleAccum;
			s++;
			if (s >= spp) break;
			sampleAccum = vec3(0.0);
			throughput = vec3(1.0);
			accumDist = dist;
			bounceRange = uMaxRayDistance;
			remT = bounceRange;
			bounce = 0;
			astep = 0;
			rayOrigin = worldPos + N * max(uRayBias, dist * 0.001);
			rayDir = sampleCosineHemisphere(r2Sample(s * 4, baseJitter), N);
		}
	}

	imageStore(uOutGI, px, vec4(totalAccum / float(spp), dist));
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
	vec4 centerNR = texelFetch(uGNormalMask, px, 0);
	float centerDepth = texelFetch(uGDepth, px, 0).r;

	if (centerDepth >= 0.99999 || dot(centerNR.xyz, centerNR.xyz) < 0.01 || centerNR.w < 0.5) {
		vec4 skipVal = vec4(0.0, 0.0, 0.0, -1.0);
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
		vec3 sumC = vec3(0.0);
		float wSum = 0.0;
		const ivec2 offs[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
		for (int i = 0; i < 4; i++) {
			ivec2 np = clamp(px + offs[i], ivec2(0), sz - 1);
			vec4 nc = texelFetch(uRawTex, np, 0);
			if (nc.a >= 0.0) {
				vec4 nnr = texelFetch(uGNormalMask, np, 0);
				if (dot(nnr.xyz, nnr.xyz) > 0.01) {
					float relD = abs(centerDist - nc.a) / max(centerDist, 0.25);
					float nw = pow(max(0.0, dot(centerN, normalize(nnr.xyz))), 16.0) * exp(-relD * relD * 250.0);
					sumC += compressHDR(nc.rgb) * nw;
					wSum += nw;
				}
			}
		}
		if (wSum > 1e-5) {
			cur = vec4(uncompressHDR(sumC / wSum), centerDist);
		} else {
			vec3 bestC = vec3(0.0);
			float bestScore = -1.0;
			for (int i = 0; i < 4; i++) {
				ivec2 np = clamp(px + offs[i], ivec2(0), sz - 1);
				vec4 nc = texelFetch(uRawTex, np, 0);
				if (nc.a >= 0.0) {
					vec4 nnr = texelFetch(uGNormalMask, np, 0);
					float nd = (dot(nnr.xyz, nnr.xyz) > 0.01) ? max(0.0, dot(centerN, normalize(nnr.xyz))) : 0.0;
					float relD = abs(centerDist - nc.a) / max(centerDist, 0.25);
					float score = (0.1 + nd) / (0.01 + relD);
					if (score > bestScore) {
						bestScore = score;
						bestC = nc.rgb;
					}
				}
			}
			cur = vec4(bestC, centerDist);
		}
	}

	if (cur.a >= 0.0) {
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

			if (cnt >= 3.0) {
				float maxAllowedY = mean.x + max(stddev.x * 2.5, 0.05);
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
							vec3 boxMin = max(nMin - vec3(0.015), mean - stddev * 1.75 - vec3(0.02));
							vec3 boxMax = min(nMax + vec3(0.015), mean + stddev * 1.75 + vec3(0.02));

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
uniform sampler2D uGNormalMask;
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

	vec4 cNR = texelFetch(uGNormalMask, px, 0);
	float cDepth = texelFetch(uGDepth, px, 0).r;
	if (cDepth >= 0.99999 || dot(cNR.xyz, cNR.xyz) < 0.01 || cNR.w < 0.5) {
		imageStore(uOutTex, px, center);
		return;
	}

	vec3 cN = normalize(cNR.xyz);
	vec2 cuv = (vec2(px) + 0.5) / vec2(sz);
	vec4 cwp = uInvViewProj * vec4(cuv * 2.0 - 1.0, cDepth * 2.0 - 1.0, 1.0);
	vec3 cPos = cwp.xyz / cwp.w;

	vec3 cComp = compressHDR(center.rgb);
	float cLum = hdrLum(cComp);

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

	float sigmaL = max(0.25, sqrt(lVar) * 3.5);
	float sigmaPlane = max(0.015, center.a * 0.008);
	float invTwoSigmaL2 = 0.5 / (sigmaL * sigmaL);
	float invTwoSigmaPlane2 = 0.5 / (sigmaPlane * sigmaPlane);

	const float k1d[5] = float[5](0.0625, 0.25, 0.375, 0.25, 0.0625);
	float centerW = k1d[2] * k1d[2];
	vec3 sumC = cComp * centerW;
	float wSum = centerW;

	for (int dy = -2; dy <= 2; dy++) {
		for (int dx = -2; dx <= 2; dx++) {
			if (dx == 0 && dy == 0) continue;
			ivec2 np = clamp(px + ivec2(dx, dy) * uStepSize, ivec2(0), sz - 1);
			vec4 nc = texelFetch(uSrcTex, np, 0);
			if (nc.a < 0.0) continue;

			vec4 nNR = texelFetch(uGNormalMask, np, 0);
			float nDepth = texelFetch(uGDepth, np, 0).r;
			if (nDepth >= 0.99999 || dot(nNR.xyz, nNR.xyz) < 0.01 || nNR.w < 0.5) continue;

			vec2 nuv = (vec2(np) + 0.5) / vec2(sz);
			vec4 nwp = uInvViewProj * vec4(nuv * 2.0 - 1.0, nDepth * 2.0 - 1.0, 1.0);
			vec3 nPos = nwp.xyz / nwp.w;

			float kw = k1d[dx + 2] * k1d[dy + 2];

			float nd = max(0.0, dot(cN, normalize(nNR.xyz)));
			float nw = pow(nd, 24.0);

			float planeDist = abs(dot(nPos - cPos, cN));
			float relD = abs(center.a - nc.a) / max(center.a, 0.25);
			float dw = exp(-(planeDist * planeDist) * invTwoSigmaPlane2 - relD * relD * 250.0);

			vec3 nComp = compressHDR(nc.rgb);
			float dl = cLum - hdrLum(nComp);
			float lw = exp(-(dl * dl) * invTwoSigmaL2);

			float w = kw * nw * dw * lw;
			sumC += nComp * w;
			wSum += w;
		}
	}

	imageStore(uOutTex, px, vec4(uncompressHDR(sumC / wSum), center.a));
}
)GLSL";

static const char *COMPOSITE_CANVAS_SHADER = R"SHADER(
shader_type canvas_item;
render_mode unshaded, blend_mix;

uniform sampler2D u_gi_tex;
uniform sampler2D u_albedo_tex;
uniform sampler2D u_full_normal_tex;
uniform float u_intensity = 1.0;
uniform float u_bilateral_upsample = 1.0;
uniform float u_keep_3d_linear = 0.0;
uniform float u_vflip = 0.0;
uniform int u_tonemap_mode = 0;
uniform float u_tonemap_exposure = 1.0;
uniform float u_tonemap_white = 1.0;
uniform int u_hdr = 1;
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

vec3 tonemap_filmic(vec3 x) {
	const float A = 0.15;
	const float B = 0.50;
	const float C = 0.10;
	const float D = 0.20;
	const float E = 0.02;
	const float F = 0.30;
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 tonemap_aces(vec3 color) {
	const float A = 2.51;
	const float B = 0.03;
	const float C = 2.43;
	const float D = 0.59;
	const float E = 0.14;
	return clamp((color * (A * color + B)) / (color * (C * color + D) + E), vec3(0.0), vec3(1.0));
}

vec3 apply_tonemap(vec3 c) {
	c = max(c, vec3(0.0)) * u_tonemap_exposure;
	if (u_hdr == 0) {
		return clamp(c, vec3(0.0), vec3(1.0));
	}
	if (u_tonemap_mode == 1) {
		float w2 = max(u_tonemap_white * u_tonemap_white, 0.001);
		return clamp((c * (1.0 + c / w2)) / (1.0 + c), vec3(0.0), vec3(1.0));
	} else if (u_tonemap_mode == 2) {
		vec3 num = tonemap_filmic(c);
		vec3 den = max(tonemap_filmic(vec3(u_tonemap_white)), vec3(0.001));
		return clamp(num / den, vec3(0.0), vec3(1.0));
	} else if (u_tonemap_mode == 3) {
		return tonemap_aces(c);
	}
	return c;
}

void fragment() {
	vec4 scene = textureLod(SCREEN_TEXTURE, SCREEN_UV, 0.0);
	float full_d = textureLod(DEPTH_TEXTURE, SCREEN_UV, 0.0).r;

	vec3 irrad = vec3(0.0);
	vec3 recv_albedo = vec3(0.0);
	bool valid = false;

	if (full_d < 0.99999) {
		vec2 m_uv = (u_vflip > 0.5) ? vec2(SCREEN_UV.x, 1.0 - SCREEN_UV.y) : SCREEN_UV;
		ivec2 full_sz = textureSize(u_albedo_tex, 0);
		ivec2 full_px = clamp(ivec2(m_uv * vec2(full_sz)), ivec2(0), full_sz - 1);
		vec4 full_alb = texelFetch(u_albedo_tex, full_px, 0);
		vec4 full_nm = texelFetch(u_full_normal_tex, full_px, 0);

		if (full_alb.a > 0.5 && dot(full_nm.xyz, full_nm.xyz) > 0.01) {
			recv_albedo = full_alb.rgb;
			vec3 full_N = normalize(full_nm.xyz);

			if (u_bilateral_upsample > 0.5) {
				float full_dist = linearize_dist(full_d, m_uv);
				ivec2 low_sz = textureSize(u_gi_tex, 0);
				vec2 f_coord = ((vec2(full_px) + 0.5) / vec2(full_sz)) * vec2(low_sz) - 0.5;
				ivec2 base = ivec2(floor(f_coord));
				vec2 f = clamp(f_coord - vec2(base), vec2(0.0), vec2(1.0));
				ivec2 max_idx = low_sz - 1;

				float inv_d = 1.0 / max(full_dist, 0.25);
				vec3 irrad_sum = vec3(0.0);
				float w_sum = 0.0;
				vec3 best_irrad = vec3(0.0);
				float best_score = -1.0;

				for (int dy = 0; dy <= 1; dy++) {
					float wy = (dy == 0) ? (1.0 - f.y) : f.y;
					for (int dx = 0; dx <= 1; dx++) {
						float wx = (dx == 0) ? (1.0 - f.x) : f.x;
						ivec2 c = clamp(base + ivec2(dx, dy), ivec2(0), max_idx);
						vec4 s = texelFetch(u_gi_tex, c, 0);
						if (s.a < 0.0) continue;

						ivec2 src_px = clamp(ivec2((vec2(c) + 0.5) * vec2(full_sz) / vec2(low_sz)), ivec2(0), full_sz - 1);
						vec4 lnm = texelFetch(u_full_normal_tex, src_px, 0);
						if (lnm.w < 0.5 || dot(lnm.xyz, lnm.xyz) < 0.01) continue;

						float rd = abs(full_dist - s.a) * inv_d;
						float dw = exp(-rd * rd * 180.0);
						float nd = max(0.0, dot(full_N, normalize(lnm.xyz)));
						float nw = pow(nd, 8.0);
						float w = wx * wy * dw * nw;

						irrad_sum += s.rgb * w;
						w_sum += w;

						float score = (0.05 + nw) / (0.01 + rd);
						if (score > best_score) {
							best_score = score;
							best_irrad = s.rgb;
						}
					}
				}

				if (w_sum > 1e-4) {
					irrad = irrad_sum / w_sum;
					valid = true;
				} else if (best_score > 0.0) {
					irrad = best_irrad;
					valid = true;
				} else {
					for (int oy = -1; oy <= 1; oy++) {
						for (int ox = -1; ox <= 1; ox++) {
							ivec2 c = clamp(base + ivec2(ox, oy), ivec2(0), max_idx);
							vec4 s = texelFetch(u_gi_tex, c, 0);
							if (s.a < 0.0) continue;
							ivec2 src_px = clamp(ivec2((vec2(c) + 0.5) * vec2(full_sz) / vec2(low_sz)), ivec2(0), full_sz - 1);
							vec4 lnm = texelFetch(u_full_normal_tex, src_px, 0);
							if (lnm.w < 0.5 || dot(lnm.xyz, lnm.xyz) < 0.01) continue;
							float rd = abs(full_dist - s.a) * inv_d;
							float nd = max(0.0, dot(full_N, normalize(lnm.xyz)));
							float score = (0.05 + pow(nd, 8.0)) / (0.01 + rd);
							if (score > best_score) {
								best_score = score;
								best_irrad = s.rgb;
							}
						}
					}
					if (best_score > 0.0) {
						irrad = best_irrad;
						valid = true;
					}
				}
			} else {
				vec4 s_fast = textureLod(u_gi_tex, m_uv, 0.0);
				if (s_fast.a >= 0.0) {
					irrad = s_fast.rgb;
					valid = true;
				}
			}
		}
	}

	if (u_debug_mode == 1 || u_debug_mode == 4) {
		vec3 dbg_lin = valid ? apply_tonemap(irrad * u_intensity) : vec3(0.0);
		vec3 dbg_rgb = (u_keep_3d_linear > 0.5) ? dbg_lin : linear_to_srgb(dbg_lin);
		COLOR = vec4(dbg_rgb, 1.0);
	} else if (u_debug_mode == 2) {
		vec3 dbg_lin = valid ? apply_tonemap(irrad * recv_albedo * u_intensity) : vec3(0.0);
		vec3 dbg_rgb = (u_keep_3d_linear > 0.5) ? dbg_lin : linear_to_srgb(dbg_lin);
		COLOR = vec4(dbg_rgb, 1.0);
	} else if (u_debug_mode == 3) {
		vec3 dbg_rgb = valid ? ((u_keep_3d_linear > 0.5) ? recv_albedo : linear_to_srgb(recv_albedo)) : vec3(0.0);
		COLOR = vec4(dbg_rgb, 1.0);
	} else if (!valid) {
		COLOR = scene;
	} else {
		vec3 scene_lin = (u_keep_3d_linear > 0.5) ? max(scene.rgb, vec3(0.0)) : srgb_to_linear(scene.rgb);
		vec3 gi_lin = apply_tonemap(max(irrad, vec3(0.0)) * recv_albedo * u_intensity);
		vec3 out_lin = scene_lin + gi_lin;
		if (u_hdr == 0) {
			out_lin = clamp(out_lin, vec3(0.0), vec3(1.0));
		}
		vec3 out_rgb = (u_keep_3d_linear > 0.5) ? out_lin : linear_to_srgb(out_lin);
		COLOR = vec4(out_rgb, 1.0);
	}
}
)SHADER";

} // namespace RTGIShaders

#endif // RTGI_SHADERS_H
