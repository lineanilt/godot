#include "baked_rtao.h"
#include "ao_bake_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define AO_BAKE_X86_SIMD 1
#include <emmintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#include "core/engine.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "drivers/gles3/rasterizer_storage_gles3.h"
#include "drivers/gles3/shared_gl43_context.h"
#include "local_rtao.h"
#include "modules/modules_enabled.gen.h"
#include "rtao.h"
#include "scene/3d/camera.h"

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/multimesh_instance.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/resources/environment.h"
#include "scene/resources/mesh.h"
#include "scene/resources/sky.h"
#include "scene/resources/world.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual_server.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace {

static const char *BAKED_MASK_VERT_GLSL = R"GLSL(
#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUv;
layout (location = 3) in float aSurfIdx;

uniform mat4 uModel;
uniform mat4 uMVP;
uniform int uMatOffset;
out vec3 vWorldNormal;
out vec2 vUv;
flat out int vMatId;

void main() {
	vWorldNormal = normalize(transpose(inverse(mat3(uModel))) * aNormal);
	vUv = aUv;
	vMatId = uMatOffset + int(aSurfIdx + 0.5);
	gl_Position = uMVP * vec4(aPos, 1.0);
}
)GLSL";

static const char *BAKED_MASK_FRAG_GLSL = R"GLSL(
#version 430 core
in vec3 vWorldNormal;
in vec2 vUv;
flat in int vMatId;
layout (location = 0) out vec4 outNormalMask;

uniform sampler2DArray uTexArray;
uniform float uMaskVal;
uniform int uFlipFacing;

struct MaskMaterial {
	vec4 uvScaleOffset;
	vec4 params;
	ivec4 flags;
};

layout (std430, binding = 5) readonly buffer MaterialBuf {
	MaskMaterial materials[];
};

void main() {
	MaskMaterial m = materials[vMatId];
	int cullMode = m.flags.z & 3;
	bool frontFace = (uFlipFacing != 0) ? !gl_FrontFacing : gl_FrontFacing;
	if (cullMode == 0 && !frontFace) discard;
	if (cullMode == 1 && frontFace) discard;

	int alphaMode = m.flags.y;
	if (alphaMode == 2) discard;
	if (alphaMode == 1) {
		vec2 uv = vUv * m.uvScaleOffset.xy + m.uvScaleOffset.zw;
		float a = m.params.x;
		if (m.flags.x >= 0) {
			a *= texture(uTexArray, vec3(uv, float(m.flags.x))).a;
		}
		if (a < m.params.y) discard;
	}

	vec3 N = normalize(vWorldNormal);
	if (cullMode == 2 && !frontFace) {
		N = -N;
	}
	outNormalMask = vec4(N, uMaskVal);
}
)GLSL";

static const char *BAKED_STANDALONE_FRAG_GLSL = R"GLSL(
#version 430 core
in vec2 vUv;
layout (location = 0) out vec4 outColor;

uniform sampler3D uBakedTex;
uniform sampler2D uViewingMaskTex;
uniform sampler2D uDepthTex;
uniform int uHasNormalMaskTex;
uniform int uUseViewingMask;
uniform mat4 uInvXform;
uniform vec3 uExtents;
uniform float uVoxelSize;
uniform float uIntensity;
uniform vec3 uAOColor;
uniform float uEdgeFalloff;
uniform float uDirectPreservation;
uniform int uDebugMode;

uniform int uNumLights;
uniform float uAmbientLum;
uniform int uUsePhysicalAtten;
uniform vec4 uLightPosRange[32];
uniform vec4 uLightDirType[32];
uniform vec4 uLightColorAtten[32];
uniform vec4 uLightSpotParams[32];

uniform mat4 uCamMatrix;
uniform mat4 uInvProj;
uniform vec2 uPixelSize;

vec3 getViewPos(vec2 suv, float d) {
	vec4 vp = uInvProj * vec4(suv * 2.0 - 1.0, min(d, 0.99999) * 2.0 - 1.0, 1.0);
	return vp.xyz / vp.w;
}

void main() {
	float rawD = textureLod(uDepthTex, vUv, 0.0).r;

	vec3 aoRGB = vec3(1.0);
	vec3 bentVis = vec3(0.5, 1.0, 0.5);
	bool valid = false;

	vec4 gbufNM = vec4(0.0);
	bool hasGbufNM = false;
	bool maskAllowed = true;
	if (uHasNormalMaskTex != 0) {
		gbufNM = textureLod(uViewingMaskTex, vUv, 0.0);
		if (dot(gbufNM.xyz, gbufNM.xyz) > 0.01) {
			hasGbufNM = true;
		}
		if (uUseViewingMask != 0 && gbufNM.a < 0.5) {
			maskAllowed = false;
		}
	}

	if (maskAllowed && rawD < 0.99999) {
		vec2 px = uPixelSize;
		float dL = textureLod(uDepthTex, vUv - vec2(px.x, 0.0), 0.0).r;
		float dR = textureLod(uDepthTex, vUv + vec2(px.x, 0.0), 0.0).r;
		float dD = textureLod(uDepthTex, vUv - vec2(0.0, px.y), 0.0).r;
		float dU = textureLod(uDepthTex, vUv + vec2(0.0, px.y), 0.0).r;

		vec3 pC = getViewPos(vUv, rawD);
		vec3 pL = getViewPos(vUv - vec2(px.x, 0.0), dL);
		vec3 pR = getViewPos(vUv + vec2(px.x, 0.0), dR);
		vec3 pD = getViewPos(vUv - vec2(0.0, px.y), dD);
		vec3 pU = getViewPos(vUv + vec2(0.0, px.y), dU);

		float diffL = (dL < 0.99999) ? abs(pC.z - pL.z) : 1e20;
		float diffR = (dR < 0.99999) ? abs(pR.z - pC.z) : 1e20;
		float diffD = (dD < 0.99999) ? abs(pC.z - pD.z) : 1e20;
		float diffU = (dU < 0.99999) ? abs(pU.z - pC.z) : 1e20;

		vec3 dX = (diffR < diffL) ? (pR - pC) : (pC - pL);
		vec3 dY = (diffU < diffD) ? (pU - pC) : (pC - pD);
		float len2X = dot(dX, dX);
		float len2Y = dot(dY, dY);
		vec3 toCamView = normalize(-pC);
		vec3 viewN = toCamView;
		if (len2X > 1e-16 && len2Y > 1e-16) {
			vec3 viewNRaw = cross(dX * inversesqrt(len2X), dY * inversesqrt(len2Y));
			float vnLen2 = dot(viewNRaw, viewNRaw);
			if (vnLen2 > 1e-8) {
				viewN = viewNRaw * inversesqrt(vnLen2);
			}
		}
		if (dot(viewN, toCamView) < 0.0) {
			viewN = -viewN;
		}

		vec3 N = hasGbufNM ? normalize(gbufNM.xyz) : normalize(mat3(uCamMatrix) * viewN);
		vec3 worldPos = (uCamMatrix * vec4(pC, 1.0)).xyz;
		vec3 boxSize = 2.0 * max(uExtents, vec3(0.0001));

		vec3 localPos = (uInvXform * vec4(worldPos, 1.0)).xyz;
		vec3 uvw = localPos / boxSize + 0.5;
		if (all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0)))) {
			vec3 distEdge = abs(uvw - 0.5) * 2.0;
			float maxE = max(distEdge.x, max(distEdge.y, distEdge.z));
			float edgeFade = (uEdgeFalloff > 0.001) ? smoothstep(1.0, max(0.0, 1.0 - uEdgeFalloff), maxE) : 1.0;

			if (edgeFade > 0.0001) {
				vec3 samplePos = worldPos + N * min(uVoxelSize * 0.5, 0.35);
				vec3 sampleLocal = (uInvXform * vec4(samplePos, 1.0)).xyz;
				vec3 sampleUvw = clamp(sampleLocal / boxSize + 0.5, vec3(0.001), vec3(0.999));
				vec4 sampleVal = texture(uBakedTex, sampleUvw);
				float hemiAO = clamp(sampleVal.a + 2.0 * dot(sampleVal.xyz, N), 0.0, 1.0);
				float ao = mix(1.0, hemiAO, edgeFade);

				vec3 tinted = mix(uAOColor, vec3(1.0), ao);
				aoRGB = clamp(mix(vec3(1.0), tinted, clamp(uIntensity, 0.0, 4.0)), vec3(0.0), vec3(1.0));

				if (uDirectPreservation > 0.001 && uNumLights > 0) {
					vec3 directIrrad = vec3(0.0);
					for (int li = 0; li < 32; li++) {
						if (li >= uNumLights) break;
						vec4 posRange = uLightPosRange[li];
						vec4 dirType = uLightDirType[li];
						vec4 colorAtten = uLightColorAtten[li];
						vec4 spotParams = uLightSpotParams[li];
						int ltype = int(dirType.w + 0.5);

						vec3 lDir = vec3(0.0, 1.0, 0.0);
						float atten = 1.0;
						bool validLight = true;

						if (ltype == 0) {
							lDir = normalize(dirType.xyz);
						} else {
							vec3 toLight = posRange.xyz - worldPos;
							float distL = length(toLight);
							float range = max(posRange.w, 0.001);
							if (distL >= range || distL < 1e-5) {
								validLight = false;
							} else {
								lDir = toLight / distL;
								float nd = distL / range;
								if (uUsePhysicalAtten != 0) {
									float nd4 = nd * nd;
									nd4 *= nd4;
									float fade = max(1.0 - nd4, 0.0);
									atten = (fade * fade) * pow(max(distL, 0.0001), -colorAtten.w);
								} else {
									atten = pow(max(1.0 - nd, 0.001), max(colorAtten.w, 0.01));
								}

								if (ltype == 2) {
									float scos = dot(-lDir, normalize(dirType.xyz));
									float spotMin = spotParams.x;
									if (scos <= spotMin) {
										validLight = false;
									} else {
										float spotRatio = clamp((scos - spotMin) / max(1.0 - spotMin, 1e-4), 0.0, 1.0);
										atten *= max(0.0, 1.0 - pow(1.0 - spotRatio, max(spotParams.y, 0.001)));
									}
								}
							}
						}

						if (!validLight || atten <= 0.001) continue;
						float ndotl = max(dot(N, lDir), 0.0);
						if (ndotl <= 0.0001) continue;

						float vis = 1.0;
						if (spotParams.z > 0.5) {
							vis = mix(1.0, clamp(sampleVal.a + 2.0 * dot(sampleVal.xyz, lDir), 0.0, 1.0), edgeFade);
						}
						directIrrad += colorAtten.rgb * (atten * ndotl * 0.31830988618 * vis);
					}

					float directLum = dot(directIrrad, vec3(0.2126, 0.7152, 0.0722));
					float ambLum = max(uAmbientLum, 0.05);
					float ratio = directLum / max(ambLum + directLum, 1e-4);
					float preserve = clamp(ratio * (clamp(uDirectPreservation, 0.0, 1.0) * 2.0), 0.0, 1.0);
					aoRGB = mix(aoRGB, vec3(1.0), preserve);
				}

				if (dot(sampleVal.xyz, sampleVal.xyz) > 1e-6) {
					bentVis = normalize(sampleVal.xyz) * 0.5 + 0.5;
				}
				valid = true;
			}
		}
	}

	if (uDebugMode == 1) {
		outColor = vec4(aoRGB, 1.0);
	} else if (uDebugMode == 2) {
		outColor = vec4(valid ? bentVis : vec3(0.5), 1.0);
	} else {
		if (!valid) discard;
		outColor = vec4(aoRGB, 1.0);
	}
}
)GLSL";

struct AOBakeTri {
	Vector3 p0, p1, p2;
	Vector3 normal;
	Vector3 albedo;
	Vector3 emission;
};

struct AOBakeBVH2Node {
	AABB bounds;
	int left_or_first = 0;
	int count = 0;
};

static bool ao_cpu_has_avx1() {
#if defined(AO_BAKE_X86_SIMD)
#if defined(_MSC_VER)
	int info[4] = { 0 };
	__cpuid(info, 1);
	if ((info[2] & (1 << 27)) == 0 || (info[2] & (1 << 28)) == 0) {
		return false;
	}
	return (_xgetbv(0) & 0x6) == 0x6;
#elif defined(__GNUC__) || defined(__clang__)
	return __builtin_cpu_supports("avx");
#endif
#endif
	return false;
}

#if defined(AO_BAKE_X86_SIMD)
static _FORCE_INLINE_ __m128 ao_sse2_select(__m128 p_mask, __m128 p_a, __m128 p_b) {
	return _mm_or_ps(_mm_and_ps(p_mask, p_a), _mm_andnot_ps(p_mask, p_b));
}

static float ao_trace_bvh4_sse2(
		const AOBVH4Node *p_nodes,
		const AOTri8 *p_blocks,
		float p_rox, float p_roy, float p_roz,
		float p_rdx, float p_rdy, float p_rdz,
		float p_tmin, float p_tmax,
		int32_t *r_hit_tri = nullptr) {
	float best_t = p_tmax;
	int32_t best_tri = -1;

	const __m128 rox4 = _mm_set1_ps(p_rox);
	const __m128 roy4 = _mm_set1_ps(p_roy);
	const __m128 roz4 = _mm_set1_ps(p_roz);
	const __m128 rdx4 = _mm_set1_ps(p_rdx);
	const __m128 rdy4 = _mm_set1_ps(p_rdy);
	const __m128 rdz4 = _mm_set1_ps(p_rdz);

	const __m128 invx4 = _mm_set1_ps(1.0f / (ABS(p_rdx) < 1e-8f ? (p_rdx < 0.0f ? -1e-8f : 1e-8f) : p_rdx));
	const __m128 invy4 = _mm_set1_ps(1.0f / (ABS(p_rdy) < 1e-8f ? (p_rdy < 0.0f ? -1e-8f : 1e-8f) : p_rdy));
	const __m128 invz4 = _mm_set1_ps(1.0f / (ABS(p_rdz) < 1e-8f ? (p_rdz < 0.0f ? -1e-8f : 1e-8f) : p_rdz));

	const __m128 zeros4 = _mm_setzero_ps();
	const __m128 ones4 = _mm_set1_ps(1.0f);
	const __m128 eps4 = _mm_set1_ps(1e-7f);
	const __m128 neg_zero4 = _mm_set1_ps(-0.0f);
	const __m128 tmin_v = _mm_set1_ps(p_tmin);
	__m128 best4 = _mm_set1_ps(best_t);

	int stack_idx[64];
	float stack_t[64];
	stack_idx[0] = 0;
	stack_t[0] = 0.0f;
	int sp = 1;

	while (sp > 0) {
		--sp;
		if (stack_t[sp] >= best_t) {
			continue;
		}

		const AOBVH4Node &node = p_nodes[stack_idx[sp]];

		__m128 tx0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_x), rox4), invx4);
		__m128 tx1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_x), rox4), invx4);
		__m128 tmin4 = _mm_min_ps(tx0, tx1);
		__m128 tmax4 = _mm_max_ps(tx0, tx1);

		__m128 ty0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_y), roy4), invy4);
		__m128 ty1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_y), roy4), invy4);
		tmin4 = _mm_max_ps(tmin4, _mm_min_ps(ty0, ty1));
		tmax4 = _mm_min_ps(tmax4, _mm_max_ps(ty0, ty1));

		__m128 tz0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_z), roz4), invz4);
		__m128 tz1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_z), roz4), invz4);
		tmin4 = _mm_max_ps(tmin4, _mm_min_ps(tz0, tz1));
		tmax4 = _mm_min_ps(tmax4, _mm_max_ps(tz0, tz1));

		tmin4 = _mm_max_ps(tmin4, tmin_v);
		__m128 hit4 = _mm_and_ps(_mm_cmpge_ps(tmax4, tmin4), _mm_cmplt_ps(tmin4, best4));
		int hit_mask = _mm_movemask_ps(hit4) & (int)node.valid_mask;
		if (hit_mask == 0) {
			continue;
		}

		alignas(16) float tmin_arr[4];
		_mm_store_ps(tmin_arr, tmin4);

		int internal_c[4];
		float internal_t[4];
		int n_internal = 0;

		for (int k = 0; k < 4; k++) {
			if (!((hit_mask >> k) & 1)) {
				continue;
			}
			if (node.leaf_blocks[k] > 0) {
				if (tmin_arr[k] >= best_t) {
					continue;
				}
				const int first_blk = node.child[k];
				const int blk_count = node.leaf_blocks[k];
				for (int b = 0; b < blk_count; b++) {
					const AOTri8 &tri = p_blocks[first_blk + b];
					for (int half = 0; half < 8; half += 4) {
						__m128 e1x = _mm_load_ps(&tri.e1x[half]);
						__m128 e1y = _mm_load_ps(&tri.e1y[half]);
						__m128 e1z = _mm_load_ps(&tri.e1z[half]);
						__m128 e2x = _mm_load_ps(&tri.e2x[half]);
						__m128 e2y = _mm_load_ps(&tri.e2y[half]);
						__m128 e2z = _mm_load_ps(&tri.e2z[half]);

						__m128 pvx = _mm_sub_ps(_mm_mul_ps(rdy4, e2z), _mm_mul_ps(rdz4, e2y));
						__m128 pvy = _mm_sub_ps(_mm_mul_ps(rdz4, e2x), _mm_mul_ps(rdx4, e2z));
						__m128 pvz = _mm_sub_ps(_mm_mul_ps(rdx4, e2y), _mm_mul_ps(rdy4, e2x));

						__m128 det = _mm_add_ps(_mm_add_ps(_mm_mul_ps(e1x, pvx), _mm_mul_ps(e1y, pvy)), _mm_mul_ps(e1z, pvz));
						__m128 valid = _mm_cmpge_ps(_mm_andnot_ps(neg_zero4, det), eps4);
						if (_mm_movemask_ps(valid) == 0) {
							continue;
						}

						__m128 safe_det = ao_sse2_select(valid, det, ones4);
						__m128 inv_det = _mm_div_ps(ones4, safe_det);

						__m128 tvx = _mm_sub_ps(rox4, _mm_load_ps(&tri.p0x[half]));
						__m128 tvy = _mm_sub_ps(roy4, _mm_load_ps(&tri.p0y[half]));
						__m128 tvz = _mm_sub_ps(roz4, _mm_load_ps(&tri.p0z[half]));

						__m128 u = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(tvx, pvx), _mm_mul_ps(tvy, pvy)), _mm_mul_ps(tvz, pvz)), inv_det);
						valid = _mm_and_ps(valid, _mm_and_ps(_mm_cmpge_ps(u, zeros4), _mm_cmple_ps(u, ones4)));
						if (_mm_movemask_ps(valid) == 0) {
							continue;
						}

						__m128 qvx = _mm_sub_ps(_mm_mul_ps(tvy, e1z), _mm_mul_ps(tvz, e1y));
						__m128 qvy = _mm_sub_ps(_mm_mul_ps(tvz, e1x), _mm_mul_ps(tvx, e1z));
						__m128 qvz = _mm_sub_ps(_mm_mul_ps(tvx, e1y), _mm_mul_ps(tvy, e1x));

						__m128 v = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(rdx4, qvx), _mm_mul_ps(rdy4, qvy)), _mm_mul_ps(rdz4, qvz)), inv_det);
						valid = _mm_and_ps(valid, _mm_and_ps(_mm_cmpge_ps(v, zeros4), _mm_cmple_ps(_mm_add_ps(u, v), ones4)));
						if (_mm_movemask_ps(valid) == 0) {
							continue;
						}

						__m128 t = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(e2x, qvx), _mm_mul_ps(e2y, qvy)), _mm_mul_ps(e2z, qvz)), inv_det);
						valid = _mm_and_ps(valid, _mm_and_ps(_mm_cmpgt_ps(t, tmin_v), _mm_cmplt_ps(t, best4)));

						if (_mm_movemask_ps(valid) != 0) {
							best4 = ao_sse2_select(valid, t, best4);
							__m128 m2 = _mm_min_ps(best4, _mm_movehl_ps(best4, best4));
							__m128 m1 = _mm_min_ss(m2, _mm_shuffle_ps(m2, m2, 1));
							best_t = _mm_cvtss_f32(m1);
							best4 = _mm_set1_ps(best_t);
							if (r_hit_tri) {
								int hit_lane_mask = _mm_movemask_ps(_mm_and_ps(valid, _mm_cmpeq_ps(t, best4)));
								if (hit_lane_mask != 0) {
									for (int l = 0; l < 4; l++) {
										if ((hit_lane_mask >> l) & 1) {
											best_tri = tri.tri_idx[half + l];
											break;
										}
									}
								}
							}
						}
					}
				}
			} else {
				internal_c[n_internal] = node.child[k];
				internal_t[n_internal] = tmin_arr[k];
				n_internal++;
			}
		}

		for (int i = 0; i < n_internal - 1; i++) {
			for (int j = i + 1; j < n_internal; j++) {
				if (internal_t[i] < internal_t[j]) {
					int tc = internal_c[i]; internal_c[i] = internal_c[j]; internal_c[j] = tc;
					float tt = internal_t[i]; internal_t[i] = internal_t[j]; internal_t[j] = tt;
				}
			}
		}
		for (int i = 0; i < n_internal; i++) {
			if (sp < 64) {
				stack_idx[sp] = internal_c[i];
				stack_t[sp] = internal_t[i];
				sp++;
			}
		}
	}

	if (r_hit_tri) {
		*r_hit_tri = best_tri;
	}
	return best_t;
}

static bool ao_occluded_bvh4_sse2(
		const AOBVH4Node *p_nodes,
		const AOTri8 *p_blocks,
		float p_rox, float p_roy, float p_roz,
		float p_rdx, float p_rdy, float p_rdz,
		float p_tmin, float p_tmax) {
	const __m128 rox4 = _mm_set1_ps(p_rox);
	const __m128 roy4 = _mm_set1_ps(p_roy);
	const __m128 roz4 = _mm_set1_ps(p_roz);
	const __m128 rdx4 = _mm_set1_ps(p_rdx);
	const __m128 rdy4 = _mm_set1_ps(p_rdy);
	const __m128 rdz4 = _mm_set1_ps(p_rdz);

	const __m128 invx4 = _mm_set1_ps(1.0f / (ABS(p_rdx) < 1e-8f ? (p_rdx < 0.0f ? -1e-8f : 1e-8f) : p_rdx));
	const __m128 invy4 = _mm_set1_ps(1.0f / (ABS(p_rdy) < 1e-8f ? (p_rdy < 0.0f ? -1e-8f : 1e-8f) : p_rdy));
	const __m128 invz4 = _mm_set1_ps(1.0f / (ABS(p_rdz) < 1e-8f ? (p_rdz < 0.0f ? -1e-8f : 1e-8f) : p_rdz));

	const __m128 zeros4 = _mm_setzero_ps();
	const __m128 ones4 = _mm_set1_ps(1.0f);
	const __m128 eps4 = _mm_set1_ps(1e-7f);
	const __m128 neg_zero4 = _mm_set1_ps(-0.0f);
	const __m128 tmin_v = _mm_set1_ps(p_tmin);
	const __m128 tmax_v = _mm_set1_ps(p_tmax);

	int stack_idx[64];
	stack_idx[0] = 0;
	int sp = 1;

	while (sp > 0) {
		const AOBVH4Node &node = p_nodes[stack_idx[--sp]];

		__m128 tx0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_x), rox4), invx4);
		__m128 tx1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_x), rox4), invx4);
		__m128 tmin4 = _mm_min_ps(tx0, tx1);
		__m128 tmax4 = _mm_max_ps(tx0, tx1);

		__m128 ty0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_y), roy4), invy4);
		__m128 ty1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_y), roy4), invy4);
		tmin4 = _mm_max_ps(tmin4, _mm_min_ps(ty0, ty1));
		tmax4 = _mm_min_ps(tmax4, _mm_max_ps(ty0, ty1));

		__m128 tz0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_z), roz4), invz4);
		__m128 tz1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_z), roz4), invz4);
		tmin4 = _mm_max_ps(tmin4, _mm_min_ps(tz0, tz1));
		tmax4 = _mm_min_ps(tmax4, _mm_max_ps(tz0, tz1));

		tmin4 = _mm_max_ps(tmin4, tmin_v);
		__m128 hit4 = _mm_and_ps(_mm_cmpge_ps(tmax4, tmin4), _mm_cmplt_ps(tmin4, tmax_v));
		int hit_mask = _mm_movemask_ps(hit4) & (int)node.valid_mask;
		if (hit_mask == 0) {
			continue;
		}

		for (int k = 0; k < 4; k++) {
			if (!((hit_mask >> k) & 1)) {
				continue;
			}
			if (node.leaf_blocks[k] > 0) {
				const int first_blk = node.child[k];
				const int blk_count = node.leaf_blocks[k];
				for (int b = 0; b < blk_count; b++) {
					const AOTri8 &tri = p_blocks[first_blk + b];
					for (int half = 0; half < 8; half += 4) {
						__m128 e1x = _mm_load_ps(&tri.e1x[half]);
						__m128 e1y = _mm_load_ps(&tri.e1y[half]);
						__m128 e1z = _mm_load_ps(&tri.e1z[half]);
						__m128 e2x = _mm_load_ps(&tri.e2x[half]);
						__m128 e2y = _mm_load_ps(&tri.e2y[half]);
						__m128 e2z = _mm_load_ps(&tri.e2z[half]);

						__m128 pvx = _mm_sub_ps(_mm_mul_ps(rdy4, e2z), _mm_mul_ps(rdz4, e2y));
						__m128 pvy = _mm_sub_ps(_mm_mul_ps(rdz4, e2x), _mm_mul_ps(rdx4, e2z));
						__m128 pvz = _mm_sub_ps(_mm_mul_ps(rdx4, e2y), _mm_mul_ps(rdy4, e2x));

						__m128 det = _mm_add_ps(_mm_add_ps(_mm_mul_ps(e1x, pvx), _mm_mul_ps(e1y, pvy)), _mm_mul_ps(e1z, pvz));
						__m128 valid = _mm_cmpge_ps(_mm_andnot_ps(neg_zero4, det), eps4);
						if (_mm_movemask_ps(valid) == 0) {
							continue;
						}

						__m128 safe_det = ao_sse2_select(valid, det, ones4);
						__m128 inv_det = _mm_div_ps(ones4, safe_det);

						__m128 tvx = _mm_sub_ps(rox4, _mm_load_ps(&tri.p0x[half]));
						__m128 tvy = _mm_sub_ps(roy4, _mm_load_ps(&tri.p0y[half]));
						__m128 tvz = _mm_sub_ps(roz4, _mm_load_ps(&tri.p0z[half]));

						__m128 u = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(tvx, pvx), _mm_mul_ps(tvy, pvy)), _mm_mul_ps(tvz, pvz)), inv_det);
						valid = _mm_and_ps(valid, _mm_and_ps(_mm_cmpge_ps(u, zeros4), _mm_cmple_ps(u, ones4)));
						if (_mm_movemask_ps(valid) == 0) {
							continue;
						}

						__m128 qvx = _mm_sub_ps(_mm_mul_ps(tvy, e1z), _mm_mul_ps(tvz, e1y));
						__m128 qvy = _mm_sub_ps(_mm_mul_ps(tvz, e1x), _mm_mul_ps(tvx, e1z));
						__m128 qvz = _mm_sub_ps(_mm_mul_ps(tvx, e1y), _mm_mul_ps(tvy, e1x));

						__m128 v = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(rdx4, qvx), _mm_mul_ps(rdy4, qvy)), _mm_mul_ps(rdz4, qvz)), inv_det);
						valid = _mm_and_ps(valid, _mm_and_ps(_mm_cmpge_ps(v, zeros4), _mm_cmple_ps(_mm_add_ps(u, v), ones4)));
						if (_mm_movemask_ps(valid) == 0) {
							continue;
						}

						__m128 t = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(e2x, qvx), _mm_mul_ps(e2y, qvy)), _mm_mul_ps(e2z, qvz)), inv_det);
						valid = _mm_and_ps(valid, _mm_and_ps(_mm_cmpgt_ps(t, tmin_v), _mm_cmplt_ps(t, tmax_v)));
						if (_mm_movemask_ps(valid) != 0) {
							return true;
						}
					}
				}
			} else if (sp < 64) {
				stack_idx[sp++] = node.child[k];
			}
		}
	}

	return false;
}
#endif

static _FORCE_INLINE_ float ao_point_tri_dist_sq(const Vector3 &p, const Vector3 &a, const Vector3 &b, const Vector3 &c) {
	Vector3 ab = b - a;
	Vector3 ac = c - a;
	Vector3 ap = p - a;
	float d1 = ab.dot(ap);
	float d2 = ac.dot(ap);
	if (d1 <= 0.0f && d2 <= 0.0f) return ap.length_squared();

	Vector3 bp = p - b;
	float d3 = ab.dot(bp);
	float d4 = ac.dot(bp);
	if (d3 >= 0.0f && d4 <= d3) return bp.length_squared();

	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		float v = d1 / (d1 - d3);
		return (p - (a + ab * v)).length_squared();
	}

	Vector3 cp = p - c;
	float d5 = ab.dot(cp);
	float d6 = ac.dot(cp);
	if (d6 >= 0.0f && d5 <= d6) return cp.length_squared();

	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		float w = d2 / (d2 - d6);
		return (p - (a + ac * w)).length_squared();
	}

	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return (p - (b + (c - b) * w)).length_squared();
	}

	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;
	return (p - (a + ab * v + ac * w)).length_squared();
}

class AOBakeRaycaster {
	std::vector<AOBVH4Node> bvh4_nodes;
	std::vector<AOTri8> tri_blocks;
	std::vector<AOBakeTri> ordered_tris;
	bool use_avx1 = false;

	int build_bin_recursive(std::vector<AOBakeBVH2Node> &r_bin, std::vector<AABB> &p_aabbs, std::vector<int> &p_indices, int p_first, int p_count, int p_depth) {
		int node_idx = (int)r_bin.size();
		r_bin.push_back(AOBakeBVH2Node());

		AABB bounds = p_aabbs[p_indices[p_first]];
		AABB cent_bounds(bounds.position + bounds.size * 0.5f, Vector3());
		for (int i = 1; i < p_count; i++) {
			const AABB &b = p_aabbs[p_indices[p_first + i]];
			bounds.merge_with(b);
			cent_bounds.expand_to(b.position + b.size * 0.5f);
		}
		r_bin[node_idx].bounds = bounds;

		int longest_axis = cent_bounds.get_longest_axis_index();
		if (p_count <= 8 || p_depth >= 24 || cent_bounds.size[longest_axis] < 1e-5f) {
			r_bin[node_idx].left_or_first = p_first;
			r_bin[node_idx].count = p_count;
			return node_idx;
		}

		const int NUM_BINS = 12;
		int best_axis = -1;
		int best_split_bin = -1;
		float best_cost = 1e30f;

		for (int axis = 0; axis < 3; axis++) {
			float cmin = cent_bounds.position[axis];
			float cext = cent_bounds.size[axis];
			if (cext < 1e-5f) {
				continue;
			}
			float inv_ext = (float)NUM_BINS / cext;

			AABB bin_box[NUM_BINS];
			int bin_cnt[NUM_BINS] = {};
			for (int i = 0; i < p_count; i++) {
				const AABB &tb = p_aabbs[p_indices[p_first + i]];
				float c = tb.position[axis] + tb.size[axis] * 0.5f;
				int b = CLAMP((int)((c - cmin) * inv_ext), 0, NUM_BINS - 1);
				if (bin_cnt[b] == 0) {
					bin_box[b] = tb;
				} else {
					bin_box[b].merge_with(tb);
				}
				bin_cnt[b]++;
			}

			AABB left_box[NUM_BINS - 1];
			int left_cnt[NUM_BINS - 1];
			AABB cur_l;
			int cur_lc = 0;
			for (int b = 0; b < NUM_BINS - 1; b++) {
				if (bin_cnt[b] > 0) {
					if (cur_lc == 0) cur_l = bin_box[b];
					else cur_l.merge_with(bin_box[b]);
					cur_lc += bin_cnt[b];
				}
				left_box[b] = cur_l;
				left_cnt[b] = cur_lc;
			}

			AABB cur_r;
			int cur_rc = 0;
			for (int b = NUM_BINS - 2; b >= 0; b--) {
				if (bin_cnt[b + 1] > 0) {
					if (cur_rc == 0) cur_r = bin_box[b + 1];
					else cur_r.merge_with(bin_box[b + 1]);
					cur_rc += bin_cnt[b + 1];
				}
				if (left_cnt[b] == 0 || cur_rc == 0) {
					continue;
				}
				Vector3 ls = left_box[b].size;
				Vector3 rs = cur_r.size;
				float l_area = ls.x * ls.y + ls.y * ls.z + ls.z * ls.x;
				float r_area = rs.x * rs.y + rs.y * rs.z + rs.z * rs.x;
				float cost = l_area * (float)left_cnt[b] + r_area * (float)cur_rc;
				if (cost < best_cost) {
					best_cost = cost;
					best_axis = axis;
					best_split_bin = b;
				}
			}
		}

		int mid = p_first + p_count / 2;
		if (best_axis >= 0) {
			float cmin = cent_bounds.position[best_axis];
			float inv_ext = (float)NUM_BINS / cent_bounds.size[best_axis];
			auto it = std::partition(p_indices.begin() + p_first, p_indices.begin() + p_first + p_count, [&](int idx) {
				float c = p_aabbs[idx].position[best_axis] + p_aabbs[idx].size[best_axis] * 0.5f;
				int b = CLAMP((int)((c - cmin) * inv_ext), 0, NUM_BINS - 1);
				return b <= best_split_bin;
			});
			mid = (int)(it - p_indices.begin());
			if (mid == p_first || mid == p_first + p_count) {
				mid = p_first + p_count / 2;
			}
		}

		build_bin_recursive(r_bin, p_aabbs, p_indices, p_first, mid - p_first, p_depth + 1);
		int right_idx = build_bin_recursive(r_bin, p_aabbs, p_indices, mid, p_first + p_count - mid, p_depth + 1);
		r_bin[node_idx].left_or_first = right_idx;
		r_bin[node_idx].count = 0;
		return node_idx;
	}

	int pack_leaf_blocks(const std::vector<AOBakeTri> &p_tris, int p_first, int p_count, int &r_block_count) {
		int first_blk = (int)tri_blocks.size();
		r_block_count = (p_count + 7) / 8;
		for (int b = 0; b < r_block_count; b++) {
			AOTri8 blk = {};
			for (int k = 0; k < 8; k++) {
				int idx = b * 8 + k;
				if (idx < p_count) {
					const AOBakeTri &t = p_tris[p_first + idx];
					Vector3 e1 = t.p1 - t.p0;
					Vector3 e2 = t.p2 - t.p0;
					blk.p0x[k] = t.p0.x; blk.p0y[k] = t.p0.y; blk.p0z[k] = t.p0.z;
					blk.e1x[k] = e1.x;   blk.e1y[k] = e1.y;   blk.e1z[k] = e1.z;
					blk.e2x[k] = e2.x;   blk.e2y[k] = e2.y;   blk.e2z[k] = e2.z;
					blk.tri_idx[k] = p_first + idx;
				} else {
					blk.tri_idx[k] = -1;
				}
			}
			tri_blocks.push_back(blk);
		}
		return first_blk;
	}

	int collapse_to_bvh4(const std::vector<AOBakeBVH2Node> &p_bin, const std::vector<AOBakeTri> &p_tris, int p_bin_idx) {
		int node4_idx = (int)bvh4_nodes.size();
		bvh4_nodes.push_back(AOBVH4Node());

		int children[4];
		int nc = 0;
		if (p_bin[p_bin_idx].count > 0) {
			children[0] = p_bin_idx;
			nc = 1;
		} else {
			children[0] = p_bin_idx + 1;
			children[1] = p_bin[p_bin_idx].left_or_first;
			nc = 2;
			while (nc < 4) {
				int best_slot = -1;
				float best_area = -1.0f;
				for (int i = 0; i < nc; i++) {
					const AOBakeBVH2Node &cn = p_bin[children[i]];
					if (cn.count == 0) {
						Vector3 s = cn.bounds.size;
						float area = s.x * s.y + s.y * s.z + s.z * s.x;
						if (area > best_area) {
							best_area = area;
							best_slot = i;
						}
					}
				}
				if (best_slot < 0) {
					break;
				}
				int split_idx = children[best_slot];
				children[best_slot] = split_idx + 1;
				children[nc++] = p_bin[split_idx].left_or_first;
			}
		}

		uint32_t valid_mask = 0;
		for (int k = 0; k < nc; k++) {
			const AOBakeBVH2Node &cn = p_bin[children[k]];
			Vector3 bmin = cn.bounds.position;
			Vector3 bmax = bmin + cn.bounds.size;

			bvh4_nodes[node4_idx].min_x[k] = bmin.x;
			bvh4_nodes[node4_idx].min_y[k] = bmin.y;
			bvh4_nodes[node4_idx].min_z[k] = bmin.z;
			bvh4_nodes[node4_idx].max_x[k] = bmax.x;
			bvh4_nodes[node4_idx].max_y[k] = bmax.y;
			bvh4_nodes[node4_idx].max_z[k] = bmax.z;
			valid_mask |= (1u << k);

			if (cn.count > 0) {
				int blk_cnt = 0;
				int first_blk = pack_leaf_blocks(p_tris, cn.left_or_first, cn.count, blk_cnt);
				bvh4_nodes[node4_idx].child[k] = first_blk;
				bvh4_nodes[node4_idx].leaf_blocks[k] = blk_cnt;
			} else {
				int sub = collapse_to_bvh4(p_bin, p_tris, children[k]);
				bvh4_nodes[node4_idx].child[k] = sub;
				bvh4_nodes[node4_idx].leaf_blocks[k] = 0;
			}
		}
		bvh4_nodes[node4_idx].valid_mask = valid_mask;
		return node4_idx;
	}

	struct SurfaceMatInfo {
		Vector3 albedo = Vector3(0.8f, 0.8f, 0.8f);
		Vector3 emission = Vector3(0.0f, 0.0f, 0.0f);
	};

	static Vector3 sample_tex_avg_linear(const Ref<Texture> &p_tex, std::unordered_map<uint32_t, Vector3> &r_cache, bool p_srgb) {
		if (p_tex.is_null() || !p_tex->get_rid().is_valid()) {
			return Vector3(1.0f, 1.0f, 1.0f);
		}
		uint32_t tid = p_tex->get_rid().get_id();
		auto it = r_cache.find(tid);
		if (it != r_cache.end()) {
			return it->second;
		}
		Vector3 avg(1.0f, 1.0f, 1.0f);
		Ref<Image> img = p_tex->get_data();
		if (img.is_valid() && !img->empty()) {
			Ref<Image> work = img;
			if (work->is_compressed()) {
				work = work->duplicate();
				if (work->decompress() != OK) {
					work.unref();
				}
			}
			if (work.is_valid() && !work->is_compressed()) {
				int w = work->get_width();
				int h = work->get_height();
				if (w > 0 && h > 0) {
					work->lock();
					int sx = MAX(1, w / 8);
					int sy = MAX(1, h / 8);
					double r = 0.0, g = 0.0, b = 0.0;
					int cnt = 0;
					for (int y = 0; y < h; y += sy) {
						for (int x = 0; x < w; x += sx) {
							Color c = work->get_pixel(x, y);
							if (p_srgb) {
								c = c.to_linear();
							}
							r += c.r;
							g += c.g;
							b += c.b;
							cnt++;
						}
					}
					work->unlock();
					if (cnt > 0) {
						avg = Vector3((float)(r / cnt), (float)(g / cnt), (float)(b / cnt));
					}
				}
			}
		}
		r_cache[tid] = avg;
		return avg;
	}

	static SurfaceMatInfo extract_surface_mat(const Ref<Material> &p_mat, std::unordered_map<uint32_t, Vector3> &r_tex_cache) {
		SurfaceMatInfo info;
		Ref<SpatialMaterial> smat = p_mat;
		if (smat.is_valid()) {
			Color col = smat->get_albedo().to_linear();
			Vector3 tex_alb = sample_tex_avg_linear(smat->get_texture(SpatialMaterial::TEXTURE_ALBEDO), r_tex_cache, true);
			info.albedo = Vector3(col.r * tex_alb.x, col.g * tex_alb.y, col.b * tex_alb.z);
			if (smat->get_feature(SpatialMaterial::FEATURE_EMISSION)) {
				Color em = smat->get_emission().to_linear();
				float en = smat->get_emission_energy();
				Vector3 tex_em = sample_tex_avg_linear(smat->get_texture(SpatialMaterial::TEXTURE_EMISSION), r_tex_cache, true);
				info.emission = Vector3(em.r * en * tex_em.x, em.g * en * tex_em.y, em.b * en * tex_em.z);
			}
		}
		return info;
	}

	void append_mesh(const Ref<Mesh> &p_mesh, const Transform &p_xf, const std::vector<SurfaceMatInfo> &p_surf_mats, std::vector<AOBakeTri> &r_raw, std::vector<AABB> &r_aabbs) {
		if (p_mesh.is_null()) {
			return;
		}
		int sc = p_mesh->get_surface_count();
		for (int s = 0; s < sc; s++) {
			if (p_mesh->surface_get_primitive_type(s) != Mesh::PRIMITIVE_TRIANGLES) {
				continue;
			}
			Array arrays = p_mesh->surface_get_arrays(s);
			if (arrays.size() <= Mesh::ARRAY_VERTEX) {
				continue;
			}
			PoolVector<Vector3> verts = arrays[Mesh::ARRAY_VERTEX];
			int vcount = verts.size();
			if (vcount < 3) {
				continue;
			}
			PoolVector<int> indices;
			bool has_indices = arrays.size() > Mesh::ARRAY_INDEX && arrays[Mesh::ARRAY_INDEX].get_type() == Variant::POOL_INT_ARRAY;
			if (has_indices) {
				indices = arrays[Mesh::ARRAY_INDEX];
			}
			PoolVector<Vector3>::Read vr = verts.read();
			PoolVector<int>::Read ir = indices.read();
			int count = has_indices ? indices.size() : vcount;

			SurfaceMatInfo mat = (s < (int)p_surf_mats.size()) ? p_surf_mats[s] : SurfaceMatInfo();

			for (int i = 0; i + 2 < count; i += 3) {
				int i0 = has_indices ? ir[i + 0] : (i + 0);
				int i1 = has_indices ? ir[i + 1] : (i + 1);
				int i2 = has_indices ? ir[i + 2] : (i + 2);
				if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= vcount || i1 >= vcount || i2 >= vcount) {
					continue;
				}
				AOBakeTri t;
				t.p0 = p_xf.xform(vr[i0]);
				t.p1 = p_xf.xform(vr[i1]);
				t.p2 = p_xf.xform(vr[i2]);
				Vector3 fn = (t.p2 - t.p0).cross(t.p1 - t.p0);
				float fl2 = fn.length_squared();
				if (fl2 < 1e-12f) {
					continue;
				}
				t.normal = fn / Math::sqrt(fl2);
				t.albedo = mat.albedo;
				t.emission = mat.emission;

				AABB tb(t.p0, Vector3());
				tb.expand_to(t.p1);
				tb.expand_to(t.p2);
				tb.grow_by(0.002f);
				r_raw.push_back(t);
				r_aabbs.push_back(tb);
			}
		}
	}

public:
	void build(const std::vector<MeshInstance *> &p_meshes, const std::vector<MultiMeshInstance *> &p_multimeshes,
#ifdef MODULE_CSG_ENABLED
			const std::vector<CSGShape *> &p_csgs,
#endif
			uint32_t p_shadow_mask, bool p_collect_materials) {
		use_avx1 = ao_cpu_has_avx1();

		std::vector<AOBakeTri> raw_tris;
		std::vector<AABB> aabbs;
		std::unordered_map<uint32_t, Vector3> tex_cache;

		for (size_t m = 0; m < p_meshes.size(); m++) {
			MeshInstance *mi = p_meshes[m];
			if (!mi->is_visible_in_tree() || !(mi->get_layer_mask() & p_shadow_mask)) {
				continue;
			}
			if (mi->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_OFF) {
				continue;
			}
			Ref<Mesh> mesh = mi->get_mesh();
			if (mesh.is_null()) {
				continue;
			}
			std::vector<SurfaceMatInfo> surf_mats;
			if (p_collect_materials) {
				int sc = mesh->get_surface_count();
				surf_mats.resize(sc);
				for (int s = 0; s < sc; s++) {
					surf_mats[s] = extract_surface_mat(mi->get_active_material(s), tex_cache);
				}
			}
			append_mesh(mesh, mi->get_global_transform(), surf_mats, raw_tris, aabbs);
		}

		for (size_t m = 0; m < p_multimeshes.size(); m++) {
			MultiMeshInstance *mmi = p_multimeshes[m];
			if (!mmi->is_visible_in_tree() || !(mmi->get_layer_mask() & p_shadow_mask)) {
				continue;
			}
			if (mmi->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_OFF) {
				continue;
			}
			Ref<MultiMesh> mm = mmi->get_multimesh();
			if (mm.is_null() || mm->get_transform_format() != MultiMesh::TRANSFORM_3D) {
				continue;
			}
			Ref<Mesh> mesh = mm->get_mesh();
			if (mesh.is_null()) {
				continue;
			}
			std::vector<SurfaceMatInfo> surf_mats;
			if (p_collect_materials) {
				int sc = mesh->get_surface_count();
				surf_mats.resize(sc);
				Ref<Material> override_mat = mmi->get_material_override();
				for (int s = 0; s < sc; s++) {
					Ref<Material> mref = override_mat.is_valid() ? override_mat : mesh->surface_get_material(s);
					surf_mats[s] = extract_surface_mat(mref, tex_cache);
				}
			}
			Transform base_xf = mmi->get_global_transform();
			int count = mm->get_visible_instance_count();
			if (count < 0) {
				count = mm->get_instance_count();
			}
			for (int i = 0; i < count; i++) {
				append_mesh(mesh, base_xf * mm->get_instance_transform(i), surf_mats, raw_tris, aabbs);
			}
		}

#ifdef MODULE_CSG_ENABLED
		for (size_t m = 0; m < p_csgs.size(); m++) {
			CSGShape *csg = p_csgs[m];
			if (!csg->is_visible_in_tree() || !csg->is_root_shape() || !(csg->get_layer_mask() & p_shadow_mask)) {
				continue;
			}
			if (csg->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_OFF) {
				continue;
			}
			Array csg_meshes = csg->get_meshes();
			if (csg_meshes.size() < 2) {
				continue;
			}
			Ref<Mesh> mesh = csg_meshes[1];
			if (mesh.is_null()) {
				continue;
			}
			Transform local_xf = csg_meshes[0];
			std::vector<SurfaceMatInfo> surf_mats;
			if (p_collect_materials) {
				int sc = mesh->get_surface_count();
				surf_mats.resize(sc);
				Ref<Material> override_mat = csg->get_material_override();
				for (int s = 0; s < sc; s++) {
					Ref<Material> mref = override_mat.is_valid() ? override_mat : mesh->surface_get_material(s);
					surf_mats[s] = extract_surface_mat(mref, tex_cache);
				}
			}
			append_mesh(mesh, csg->get_global_transform() * local_xf, surf_mats, raw_tris, aabbs);
		}
#endif

		if (raw_tris.empty()) {
			return;
		}
		std::vector<int> order(raw_tris.size());
		for (size_t i = 0; i < order.size(); i++) {
			order[i] = (int)i;
		}

		std::vector<AOBakeBVH2Node> bin_nodes;
		bin_nodes.reserve(raw_tris.size());
		build_bin_recursive(bin_nodes, aabbs, order, 0, (int)order.size(), 0);

		ordered_tris.resize(order.size());
		for (size_t i = 0; i < order.size(); i++) {
			ordered_tris[i] = raw_tris[order[i]];
		}

		bvh4_nodes.clear();
		tri_blocks.clear();
		bvh4_nodes.reserve(bin_nodes.size() / 2 + 1);
		collapse_to_bvh4(bin_nodes, ordered_tris, 0);
	}

	const AOBakeTri &get_triangle(int32_t p_idx) const {
		return ordered_tris[p_idx];
	}

	float trace_distance(const Vector3 &p_ro, const Vector3 &p_rd, float p_tmin, float p_tmax, int32_t *r_hit_tri = nullptr) const {
		if (bvh4_nodes.empty()) {
			if (r_hit_tri) {
				*r_hit_tri = -1;
			}
			return p_tmax;
		}
#if defined(AO_BAKE_X86_SIMD)
		if (use_avx1) {
			return ao_trace_bvh4_avx(bvh4_nodes.data(), tri_blocks.data(), p_ro.x, p_ro.y, p_ro.z, p_rd.x, p_rd.y, p_rd.z, p_tmin, p_tmax, r_hit_tri);
		}
		return ao_trace_bvh4_sse2(bvh4_nodes.data(), tri_blocks.data(), p_ro.x, p_ro.y, p_ro.z, p_rd.x, p_rd.y, p_rd.z, p_tmin, p_tmax, r_hit_tri);
#else
		if (r_hit_tri) {
			*r_hit_tri = -1;
		}
		return p_tmax;
#endif
	}

	bool is_occluded(const Vector3 &p_ro, const Vector3 &p_rd, float p_tmin, float p_tmax) const {
		if (bvh4_nodes.empty()) {
			return false;
		}
#if defined(AO_BAKE_X86_SIMD)
		if (use_avx1) {
			return ao_occluded_bvh4_avx(bvh4_nodes.data(), tri_blocks.data(), p_ro.x, p_ro.y, p_ro.z, p_rd.x, p_rd.y, p_rd.z, p_tmin, p_tmax);
		}
		return ao_occluded_bvh4_sse2(bvh4_nodes.data(), tri_blocks.data(), p_ro.x, p_ro.y, p_ro.z, p_rd.x, p_rd.y, p_rd.z, p_tmin, p_tmax);
#else
		return false;
#endif
	}

	bool overlaps_sphere(const Vector3 &p_center, float p_radius) const {
		if (bvh4_nodes.empty()) {
			return false;
		}
		const float r2 = p_radius * p_radius;
#if defined(AO_BAKE_X86_SIMD)
		const __m128 cx4 = _mm_set1_ps(p_center.x);
		const __m128 cy4 = _mm_set1_ps(p_center.y);
		const __m128 cz4 = _mm_set1_ps(p_center.z);
		const __m128 r2_4 = _mm_set1_ps(r2);
		const __m128 zeros4 = _mm_setzero_ps();

		int stack[64];
		stack[0] = 0;
		int sp = 1;
		while (sp > 0) {
			const AOBVH4Node &node = bvh4_nodes[stack[--sp]];
			__m128 dx = _mm_add_ps(_mm_max_ps(_mm_sub_ps(_mm_load_ps(node.min_x), cx4), zeros4), _mm_max_ps(_mm_sub_ps(cx4, _mm_load_ps(node.max_x)), zeros4));
			__m128 dy = _mm_add_ps(_mm_max_ps(_mm_sub_ps(_mm_load_ps(node.min_y), cy4), zeros4), _mm_max_ps(_mm_sub_ps(cy4, _mm_load_ps(node.max_y)), zeros4));
			__m128 dz = _mm_add_ps(_mm_max_ps(_mm_sub_ps(_mm_load_ps(node.min_z), cz4), zeros4), _mm_max_ps(_mm_sub_ps(cz4, _mm_load_ps(node.max_z)), zeros4));
			__m128 d2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)), _mm_mul_ps(dz, dz));
			int hit_mask = _mm_movemask_ps(_mm_cmple_ps(d2, r2_4)) & (int)node.valid_mask;
			if (hit_mask == 0) {
				continue;
			}
			for (int k = 0; k < 4; k++) {
				if (!((hit_mask >> k) & 1)) {
					continue;
				}
				if (node.leaf_blocks[k] > 0) {
					const int first_blk = node.child[k];
					const int blk_count = node.leaf_blocks[k];
					for (int b = 0; b < blk_count; b++) {
						const AOTri8 &blk = tri_blocks[first_blk + b];
						for (int t = 0; t < 8; t++) {
							int32_t tidx = blk.tri_idx[t];
							if (tidx < 0) {
								break;
							}
							const AOBakeTri &tri = ordered_tris[tidx];
							if (ao_point_tri_dist_sq(p_center, tri.p0, tri.p1, tri.p2) <= r2) {
								return true;
							}
						}
					}
				} else if (sp < 64) {
					stack[sp++] = node.child[k];
				}
			}
		}
		return false;
#else
		return true;
#endif
	}
};

static float ao_hash31(int x, int y, int z) {
	uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + (uint32_t)z * 2147483647u;
	h = (h ^ (h >> 13)) * 1274126177u;
	return (float)(h & 0xFFFFu) / 65535.0f;
}

static _FORCE_INLINE_ float ao_luminance(const Vector3 &p_c) {
	return MAX(0.0f, p_c.x * 0.2126f + p_c.y * 0.7152f + p_c.z * 0.0722f);
}

struct BakeLight {
	enum Type {
		LIGHT_DIRECTIONAL = 0,
		LIGHT_OMNI = 1,
		LIGHT_SPOT = 2,
	};

	Type type = LIGHT_DIRECTIONAL;
	Vector3 pos;
	Vector3 dir; // Directional: toward light; Spot: forward beam direction
	Transform inv_xform;
	Vector3 color; // linear color * energy
	float range = 10.0f;
	float attenuation = 1.0f;
	float spot_cutoff = 0.7f;
	float spot_atten = 1.0f;
	bool is_square = false;
	bool has_shadow = false;
	bool use_physical_atten = false;

	bool evaluate_at(const Vector3 &p_pos, Vector3 &r_to_light, Vector3 &r_radiance, float &r_max_dist) const {
		if (type == LIGHT_DIRECTIONAL) {
			r_to_light = dir;
			r_radiance = color;
			r_max_dist = 500.0f;
			return true;
		}

		Vector3 to_light = pos - p_pos;
		float dist = to_light.length();
		if (dist < 1e-5f) {
			return false;
		}

		auto eval_dist_atten = [&](float p_d) -> float {
			if (use_physical_atten) {
				float nd = p_d / range;
				float nd4 = nd * nd;
				nd4 *= nd4;
				float fade = MAX(1.0f - nd4, 0.0f);
				return (fade * fade) * Math::pow(MAX(p_d, 0.0001f), -attenuation);
			}
			return Math::pow(MAX(1.0f - p_d / range, 0.0f), attenuation);
		};

		float atten = 1.0f;
		if (type == LIGHT_OMNI) {
			if (is_square) {
				Vector3 lp = inv_xform.xform(p_pos);
				float sq_dist = MAX(ABS(lp.x), MAX(ABS(lp.y), ABS(lp.z)));
				if (sq_dist >= range) {
					return false;
				}
				atten = eval_dist_atten(sq_dist);
			} else {
				if (dist >= range) {
					return false;
				}
				atten = eval_dist_atten(dist);
			}
			r_to_light = to_light / dist;
		} else {
			if (dist >= range) {
				return false;
			}
			Vector3 l_local = inv_xform.xform(p_pos);
			if (l_local.z >= -0.001f) {
				return false;
			}
			r_to_light = to_light / dist;
			atten = eval_dist_atten(dist);
			float rim = 1.0f;
			if (is_square) {
				float tan_a = Math::sqrt(MAX(0.0f, 1.0f - spot_cutoff * spot_cutoff)) / MAX(0.001f, spot_cutoff);
				Vector2 suv(l_local.x / (-l_local.z * tan_a), l_local.y / (-l_local.z * tan_a));
				if (MAX(ABS(suv.x), ABS(suv.y)) > 1.0f) {
					return false;
				}
				float ux2 = suv.x * suv.x;
				float uy2 = suv.y * suv.y;
				rim = CLAMP(ux2 + uy2 - ux2 * uy2, 0.0001f, 1.0f);
			} else {
				float scos = (-r_to_light).dot(dir);
				if (scos <= spot_cutoff) {
					return false;
				}
				rim = CLAMP((1.0f - scos) / MAX(1.0f - spot_cutoff, 0.0001f), 0.0001f, 1.0f);
			}
			atten *= MAX(0.0f, 1.0f - Math::pow(rim, MAX(spot_atten, 0.001f)));
		}

		if (atten <= 0.0001f) {
			return false;
		}
		r_radiance = color * atten;
		r_max_dist = dist;
		return true;
	}
};

struct BakeEnvironment {
	Basis sky_inv_basis;
	bool has_sky = false;
	bool use_panorama = false;
	int pano_w = 0;
	int pano_h = 0;
	std::vector<Vector3> pano_pixels;

	Vector3 sky_top = Vector3(0.376f, 0.672f, 0.880f);
	Vector3 sky_horizon = Vector3(0.672f, 0.823f, 0.956f);
	Vector3 ground_horizon = Vector3(0.153f, 0.127f, 0.105f);
	Vector3 ground_bottom = Vector3(0.021f, 0.018f, 0.014f);
	float sky_curve = 0.09f;
	float ground_curve = 0.02f;
	float sky_energy = 1.0f;
	float ground_energy = 1.0f;

	Vector3 sun_dir = Vector3(0.0f, 0.5736f, -0.8192f);
	Vector3 sun_color = Vector3(1.0f, 1.0f, 1.0f);
	float sun_angle_min = 1.0f;
	float sun_angle_max = 100.0f;
	float sun_curve = 0.05f;
	float sun_energy = 0.0f;

	Vector3 ambient_irradiance = Vector3(0.08f, 0.08f, 0.10f);

	void init_from_environment(const Ref<Environment> &p_env) {
		if (p_env.is_null()) {
			ambient_irradiance = Vector3(0.12f, 0.12f, 0.14f);
			return;
		}

		sky_inv_basis = p_env->get_sky_orientation().transposed();
		float bg_energy = p_env->get_bg_energy();

		if (p_env->get_background() == Environment::BG_SKY && p_env->get_sky().is_valid()) {
			has_sky = true;
			Ref<Sky> sky = p_env->get_sky();
			Ref<PanoramaSky> pan_sky = sky;
			Ref<ProceduralSky> psky = sky;

			if (pan_sky.is_valid() && pan_sky->get_panorama().is_valid()) {
				Ref<Image> img = pan_sky->get_panorama()->get_data();
				if (img.is_valid() && !img->empty()) {
					Ref<Image> work = img->duplicate();
					if (work->is_compressed()) {
						work->decompress();
					}
					if (!work->is_compressed()) {
						work->resize(64, 32, Image::INTERPOLATE_BILINEAR);
						pano_w = work->get_width();
						pano_h = work->get_height();
						pano_pixels.resize((size_t)pano_w * (size_t)pano_h);
						bool convert_srgb = (work->get_format() != Image::FORMAT_RGBF && work->get_format() != Image::FORMAT_RGBAF && work->get_format() != Image::FORMAT_RGBH && work->get_format() != Image::FORMAT_RGBAH);
						work->lock();
						for (int py = 0; py < pano_h; py++) {
							for (int px = 0; px < pano_w; px++) {
								Color c = work->get_pixel(px, py);
								if (convert_srgb) {
									c = c.to_linear();
								}
								pano_pixels[(size_t)py * (size_t)pano_w + (size_t)px] = Vector3(c.r, c.g, c.b) * bg_energy;
							}
						}
						work->unlock();
						use_panorama = true;
					}
				}
			} else if (psky.is_valid()) {
				Color st = psky->get_sky_top_color().to_linear();
				Color sh = psky->get_sky_horizon_color().to_linear();
				Color gh = psky->get_ground_horizon_color().to_linear();
				Color gb = psky->get_ground_bottom_color().to_linear();
				sky_curve = psky->get_sky_curve();
				ground_curve = psky->get_ground_curve();
				sky_energy = psky->get_sky_energy() * bg_energy;
				ground_energy = psky->get_ground_energy() * bg_energy;
				sky_top = Vector3(st.r, st.g, st.b) * sky_energy;
				sky_horizon = Vector3(sh.r, sh.g, sh.b) * sky_energy;
				ground_horizon = Vector3(gh.r, gh.g, gh.b) * ground_energy;
				ground_bottom = Vector3(gb.r, gb.g, gb.b) * ground_energy;

				float lat = Math::deg2rad(psky->get_sun_latitude());
				float lon = Math::deg2rad(psky->get_sun_longitude());
				sun_dir = (Basis(Vector3(0, 1, 0), lon) * Basis(Vector3(1, 0, 0), lat)).xform(Vector3(0, 0, -1)).normalized();
				Color sc = psky->get_sun_color().to_linear();
				sun_color = Vector3(sc.r, sc.g, sc.b);
				sun_angle_min = psky->get_sun_angle_min();
				sun_angle_max = psky->get_sun_angle_max();
				sun_curve = psky->get_sun_curve();
				sun_energy = psky->get_sun_energy() * bg_energy;
			}
		} else if (p_env->get_background() == Environment::BG_COLOR || p_env->get_background() == Environment::BG_CLEAR_COLOR) {
			has_sky = true;
			Color bg = p_env->get_bg_color().to_linear();
			Vector3 c = Vector3(bg.r, bg.g, bg.b) * bg_energy;
			sky_top = c;
			sky_horizon = c;
			ground_horizon = c;
			ground_bottom = c;
		}

		Vector3 mean_sky(0.0f, 0.0f, 0.0f);
		if (has_sky) {
			const float golden = 2.399963229728653f;
			for (int i = 0; i < 64; i++) {
				float y = 1.0f - (2.0f * ((float)i + 0.5f) / 64.0f);
				float r = Math::sqrt(MAX(0.0f, 1.0f - y * y));
				float phi = (float)i * golden;
				mean_sky += eval_sky(Vector3(Math::cos(phi) * r, y, Math::sin(phi) * r));
			}
			mean_sky /= 64.0f;
		}

		Color amb = p_env->get_ambient_light_color().to_linear();
		float amb_e = p_env->get_ambient_light_energy();
		float sky_c = has_sky ? p_env->get_ambient_light_sky_contribution() : 0.0f;
		Vector3 amb_v(amb.r, amb.g, amb.b);
		ambient_irradiance = amb_v.linear_interpolate(mean_sky, sky_c) * amb_e;
	}

	Vector3 eval_sky(const Vector3 &p_dir) const {
		if (!has_sky) {
			return ambient_irradiance;
		}
		Vector3 d = sky_inv_basis.xform(p_dir).normalized();

		if (use_panorama && pano_w > 0 && pano_h > 0) {
			float u = Math::atan2(d.x, -d.z) / (float)Math_TAU + 0.5f;
			float v = Math::acos(CLAMP(d.y, -1.0f, 1.0f)) / (float)Math_PI;
			u = u - Math::floor(u);
			v = CLAMP(v, 0.0f, 0.9999f);
			int px = CLAMP((int)(u * (float)pano_w), 0, pano_w - 1);
			int py = CLAMP((int)(v * (float)pano_h), 0, pano_h - 1);
			return pano_pixels[(size_t)py * (size_t)pano_w + (size_t)px];
		}

		float h = CLAMP(d.y, -1.0f, 1.0f);
		Vector3 col;
		if (h >= 0.0f) {
			float c = CLAMP(1.0f - Math::acos(h) / ((float)Math_PI * 0.5f), 0.0f, 1.0f);
			col = sky_horizon.linear_interpolate(sky_top, Math::pow(MAX(c, 1e-4f), MAX(sky_curve, 0.005f)));
		} else {
			float c = CLAMP(1.0f - Math::acos(-h) / ((float)Math_PI * 0.5f), 0.0f, 1.0f);
			col = ground_horizon.linear_interpolate(ground_bottom, Math::pow(MAX(c, 1e-4f), MAX(ground_curve, 0.005f)));
		}

		if (sun_energy > 0.001f) {
			float ang_deg = Math::rad2deg(Math::acos(CLAMP(d.dot(sun_dir), -1.0f, 1.0f)));
			if (ang_deg < sun_angle_max) {
				float c2 = CLAMP((ang_deg - sun_angle_min) / MAX(sun_angle_max - sun_angle_min, 0.001f), 0.0f, 1.0f);
				float sun_amt = (ang_deg <= sun_angle_min) ? 1.0f : Math::pow(1.0f - c2, 1.0f / MAX(sun_curve, 0.001f));
				Vector3 s_peak = sun_color * sun_energy;
				Vector3 s_max(MAX(col.x, s_peak.x), MAX(col.y, s_peak.y), MAX(col.z, s_peak.z));
				col = col.linear_interpolate(s_max, sun_amt);
			}
		}
		return col;
	}
};

struct BakeLocalVolume {
	Transform inv_xform;
	LocalRTAO::Mode mode = LocalRTAO::MODE_DIRECT;
	LocalRTAO::Shape shape = LocalRTAO::SHAPE_BOX;
	Vector3 extents;
	float edge_falloff = 0.25f;
	float occlusion = 0.5f;
	float radius_multiplier = 1.0f;
	LocalRTAO::BlendMode blend_mode = LocalRTAO::BLEND_MULTIPLY;

	float sample_shape_factor(const Vector3 &p_world_pos) const {
		Vector3 local = inv_xform.xform(p_world_pos);
		Vector3 e = extents;
		if (shape == LocalRTAO::SHAPE_SPHERE) {
			float r = MAX(e.x, MAX(e.y, e.z));
			e = Vector3(r, r, r);
		}
		Vector3 n(
				local.x / MAX(e.x, 0.0001f),
				local.y / MAX(e.y, 0.0001f),
				local.z / MAX(e.z, 0.0001f));

		float d = 1.0f;
		switch (shape) {
			case LocalRTAO::SHAPE_BOX:
				d = MAX(ABS(n.x), MAX(ABS(n.y), ABS(n.z)));
				break;
			case LocalRTAO::SHAPE_SPHERE:
			case LocalRTAO::SHAPE_ELLIPSOID:
				d = n.length();
				break;
			case LocalRTAO::SHAPE_CYLINDER:
				d = MAX(Math::sqrt(n.x * n.x + n.z * n.z), ABS(n.y));
				break;
			case LocalRTAO::SHAPE_CONE: {
				float ty = CLAMP(0.5f - n.y * 0.5f, 0.0f, 1.0f);
				float r_cone = MAX(ty, 0.001f);
				d = MAX(Math::sqrt(n.x * n.x + n.z * n.z) / r_cone, ABS(n.y));
			} break;
		}

		if (d >= 1.0f) {
			return 0.0f;
		}
		if (edge_falloff <= 0.001f) {
			return 1.0f;
		}
		float edge0 = MAX(0.0f, 1.0f - edge_falloff);
		return 1.0f - Math::smoothstep(edge0, 1.0f, d);
	}
};

} // namespace

BakedRTAO::BakedRTAO() {
}

BakedRTAO::~BakedRTAO() {
	_unregister_scene_hook();
	_cleanup_gl_mask_resources();
}

void BakedRTAO::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &BakedRTAO::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &BakedRTAO::is_enabled);

	ClassDB::bind_method(D_METHOD("set_editor_preview", "preview"), &BakedRTAO::set_editor_preview);
	ClassDB::bind_method(D_METHOD("get_editor_preview"), &BakedRTAO::get_editor_preview);

	ClassDB::bind_method(D_METHOD("set_auto_composite", "auto_composite"), &BakedRTAO::set_auto_composite);
	ClassDB::bind_method(D_METHOD("get_auto_composite"), &BakedRTAO::get_auto_composite);

	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &BakedRTAO::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &BakedRTAO::get_extents);

	ClassDB::bind_method(D_METHOD("set_resolution", "resolution"), &BakedRTAO::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution"), &BakedRTAO::get_resolution);

	ClassDB::bind_method(D_METHOD("set_baked_texture", "texture"), &BakedRTAO::set_baked_texture);
	ClassDB::bind_method(D_METHOD("get_baked_texture"), &BakedRTAO::get_baked_texture);

	ClassDB::bind_method(D_METHOD("set_baked_data", "data"), &BakedRTAO::set_baked_data);
	ClassDB::bind_method(D_METHOD("get_baked_data"), &BakedRTAO::get_baked_data);

	ClassDB::bind_method(D_METHOD("set_baked_data_resolution", "resolution"), &BakedRTAO::set_baked_data_resolution);
	ClassDB::bind_method(D_METHOD("get_baked_data_resolution"), &BakedRTAO::get_baked_data_resolution);

	ClassDB::bind_method(D_METHOD("set_intensity", "intensity"), &BakedRTAO::set_intensity);
	ClassDB::bind_method(D_METHOD("get_intensity"), &BakedRTAO::get_intensity);

	ClassDB::bind_method(D_METHOD("set_ao_color", "color"), &BakedRTAO::set_ao_color);
	ClassDB::bind_method(D_METHOD("get_ao_color"), &BakedRTAO::get_ao_color);

	ClassDB::bind_method(D_METHOD("set_edge_falloff", "falloff"), &BakedRTAO::set_edge_falloff);
	ClassDB::bind_method(D_METHOD("get_edge_falloff"), &BakedRTAO::get_edge_falloff);

	ClassDB::bind_method(D_METHOD("set_direct_light_preservation", "preservation"), &BakedRTAO::set_direct_light_preservation);
	ClassDB::bind_method(D_METHOD("get_direct_light_preservation"), &BakedRTAO::get_direct_light_preservation);

	ClassDB::bind_method(D_METHOD("set_blend_mode", "mode"), &BakedRTAO::set_blend_mode);
	ClassDB::bind_method(D_METHOD("get_blend_mode"), &BakedRTAO::get_blend_mode);

	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &BakedRTAO::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &BakedRTAO::get_cull_mask);

	ClassDB::bind_method(D_METHOD("set_viewing_cull_mask", "mask"), &BakedRTAO::set_viewing_cull_mask);
	ClassDB::bind_method(D_METHOD("get_viewing_cull_mask"), &BakedRTAO::get_viewing_cull_mask);

	ClassDB::bind_method(D_METHOD("set_debug_mode", "mode"), &BakedRTAO::set_debug_mode);
	ClassDB::bind_method(D_METHOD("get_debug_mode"), &BakedRTAO::get_debug_mode);

	ClassDB::bind_method(D_METHOD("set_bake_samples", "samples"), &BakedRTAO::set_bake_samples);
	ClassDB::bind_method(D_METHOD("get_bake_samples"), &BakedRTAO::get_bake_samples);

	ClassDB::bind_method(D_METHOD("set_bake_radius", "radius"), &BakedRTAO::set_bake_radius);
	ClassDB::bind_method(D_METHOD("get_bake_radius"), &BakedRTAO::get_bake_radius);

	ClassDB::bind_method(D_METHOD("set_bake_ray_bias", "bias"), &BakedRTAO::set_bake_ray_bias);
	ClassDB::bind_method(D_METHOD("get_bake_ray_bias"), &BakedRTAO::get_bake_ray_bias);

	ClassDB::bind_method(D_METHOD("set_bake_power", "power"), &BakedRTAO::set_bake_power);
	ClassDB::bind_method(D_METHOD("get_bake_power"), &BakedRTAO::get_bake_power);

	ClassDB::bind_method(D_METHOD("set_bake_local_volumes", "enable"), &BakedRTAO::set_bake_local_volumes);
	ClassDB::bind_method(D_METHOD("get_bake_local_volumes"), &BakedRTAO::get_bake_local_volumes);

	ClassDB::bind_method(D_METHOD("set_bake_directional_gi", "enable"), &BakedRTAO::set_bake_directional_gi);
	ClassDB::bind_method(D_METHOD("get_bake_directional_gi"), &BakedRTAO::get_bake_directional_gi);

	ClassDB::bind_method(D_METHOD("set_bake_gi_direction_strength", "strength"), &BakedRTAO::set_bake_gi_direction_strength);
	ClassDB::bind_method(D_METHOD("get_bake_gi_direction_strength"), &BakedRTAO::get_bake_gi_direction_strength);

	ClassDB::bind_method(D_METHOD("set_bake_denoise_passes", "passes"), &BakedRTAO::set_bake_denoise_passes);
	ClassDB::bind_method(D_METHOD("get_bake_denoise_passes"), &BakedRTAO::get_bake_denoise_passes);

	ClassDB::bind_method(D_METHOD("set_bake_filter", "filter"), &BakedRTAO::set_bake_filter);
	ClassDB::bind_method(D_METHOD("get_bake_filter"), &BakedRTAO::get_bake_filter);

	ClassDB::bind_method(D_METHOD("set_shadow_cull_mask", "mask"), &BakedRTAO::set_shadow_cull_mask);
	ClassDB::bind_method(D_METHOD("get_shadow_cull_mask"), &BakedRTAO::get_shadow_cull_mask);

	ClassDB::bind_method(D_METHOD("bake"), &BakedRTAO::bake);
	ClassDB::bind_method(D_METHOD("_on_frame_pre_draw"), &BakedRTAO::_on_frame_pre_draw);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_preview"), "set_editor_preview", "get_editor_preview");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_composite"), "set_auto_composite", "get_auto_composite");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents"), "set_extents", "get_extents");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "resolution"), "set_resolution", "get_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "baked_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture3D", PROPERTY_USAGE_EDITOR), "set_baked_texture", "get_baked_texture");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "baked_data_resolution", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_baked_data_resolution", "get_baked_data_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_BYTE_ARRAY, "baked_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_baked_data", "get_baked_data");

	ADD_GROUP("Playback", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "intensity", PROPERTY_HINT_RANGE, "0,4,0.05,or_greater"), "set_intensity", "get_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ao_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_ao_color", "get_ao_color");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "edge_falloff", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_edge_falloff", "get_edge_falloff");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "direct_light_preservation", PROPERTY_HINT_RANGE, "0,1,0.02"), "set_direct_light_preservation", "get_direct_light_preservation");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "blend_mode", PROPERTY_HINT_ENUM, "Multiply,Min,Override"), "set_blend_mode", "get_blend_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "viewing_cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_viewing_cull_mask", "get_viewing_cull_mask");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_mode", PROPERTY_HINT_ENUM, "None,Baked AO,Bent Normal"), "set_debug_mode", "get_debug_mode");

	ADD_GROUP("Bake Settings", "bake_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bake_samples", PROPERTY_HINT_RANGE, "1,256,1"), "set_bake_samples", "get_bake_samples");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_radius", PROPERTY_HINT_RANGE, "0.1,50,0.1,or_greater"), "set_bake_radius", "get_bake_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_ray_bias", PROPERTY_HINT_RANGE, "0.001,0.5,0.005"), "set_bake_ray_bias", "get_bake_ray_bias");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_power", PROPERTY_HINT_RANGE, "0.1,4,0.05"), "set_bake_power", "get_bake_power");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bake_local_volumes"), "set_bake_local_volumes", "get_bake_local_volumes");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bake_directional_gi"), "set_bake_directional_gi", "get_bake_directional_gi");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_gi_direction_strength", PROPERTY_HINT_RANGE, "0,1,0.02"), "set_bake_gi_direction_strength", "get_bake_gi_direction_strength");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bake_denoise_passes", PROPERTY_HINT_RANGE, "0,5,1"), "set_bake_denoise_passes", "get_bake_denoise_passes");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bake_filter"), "set_bake_filter", "get_bake_filter");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_shadow_cull_mask", "get_shadow_cull_mask");

	BIND_ENUM_CONSTANT(BLEND_MULTIPLY);
	BIND_ENUM_CONSTANT(BLEND_MIN);
	BIND_ENUM_CONSTANT(BLEND_OVERRIDE);

	BIND_ENUM_CONSTANT(DEBUG_NONE);
	BIND_ENUM_CONSTANT(DEBUG_AO);
	BIND_ENUM_CONSTANT(DEBUG_BENT_NORMAL);
}

void BakedRTAO::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_register_scene_hook();
			set_process(true);
			set_process_priority(10000);
			if (VisualServer::get_singleton() && !VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
				VisualServer::get_singleton()->connect("frame_pre_draw", this, "_on_frame_pre_draw");
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible_in_tree()) {
				_detach_all_viewports();
			}
		} break;

		case NOTIFICATION_EXIT_WORLD: {
			_detach_all_viewports();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (VisualServer::get_singleton() && VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
				VisualServer::get_singleton()->disconnect("frame_pre_draw", this, "_on_frame_pre_draw");
			}
			_unregister_scene_hook();
		} break;

		case NOTIFICATION_PROCESS: {
			if (!Engine::get_singleton()->is_editor_hint() && OS::get_singleton()->get_render_thread_mode() == OS::RENDER_SEPARATE_THREAD) {
				_update_standalone_composite();
			}
		} break;
	}
}

void BakedRTAO::_on_frame_pre_draw() {
	if (Engine::get_singleton()->is_editor_hint() || OS::get_singleton()->get_render_thread_mode() != OS::RENDER_SEPARATE_THREAD) {
		_update_standalone_composite();
	}
	_render_standalone_viewing_mask();
}

void BakedRTAO::_register_scene_hook() {
	if (hook_registered) {
		return;
	}
	GL43EffectHelpers::register_scene_hook(this, GL43EffectHelpers::HOOK_PRE_SCREEN_TEXTURE, 10, [this](const GL43EffectHelpers::SceneHookContext &p_ctx) {
		_on_scene_hook(p_ctx);
	});
	hook_registered = true;
}

void BakedRTAO::_unregister_scene_hook() {
	if (!hook_registered) {
		return;
	}
	GL43EffectHelpers::unregister_scene_hooks(this);
	hook_registered = false;
	_detach_all_viewports();
}

void BakedRTAO::_detach_all_viewports() {
	MutexLock lock(mask_mutex);
	for (auto &kv : viewport_states) {
		kv.second.active_composite = false;
	}
}

BakedRTAO::ViewportCompositeState &BakedRTAO::_get_or_create_vp_state(Viewport *p_vp) {
	uint32_t key = p_vp->get_viewport_rid().get_id();
	auto it = viewport_states.find(key);
	if (it == viewport_states.end()) {
		ViewportCompositeState st;
		st.mask_texture.instance();
		it = viewport_states.emplace(key, std::move(st)).first;
	}
	if (VisualServer::get_singleton()) {
		it->second.color_tex_rid = VisualServer::get_singleton()->viewport_get_texture(p_vp->get_viewport_rid(), VS::VIEWPORT_TEXTURE_BUFFER_COLOR);
	}
	it->second.active_composite = auto_composite;
	it->second.last_seen = vp_collect_counter;
	return it->second;
}

void BakedRTAO::_prune_stale_viewports() {
	for (auto it = viewport_states.begin(); it != viewport_states.end(); ) {
		if (it->second.last_seen != vp_collect_counter) {
			it->second.active_composite = false;
			if (vp_collect_counter - it->second.last_seen > 60ULL) {
				if (gl_ready && gl_thread_id == Thread::get_caller_id()) {
					SharedGL43Context::ScopedScope gl43;
					if (gl43.is_active()) {
						if (it->second.mask_fbo) {
							GLuint f = (GLuint)it->second.mask_fbo;
							glDeleteFramebuffers(1, &f);
						}
						if (it->second.mask_depth_tex) {
							GLuint t = (GLuint)it->second.mask_depth_tex;
							glDeleteTextures(1, &t);
						}
					}
				}
				it = viewport_states.erase(it);
				continue;
			}
		}
		++it;
	}
}

void BakedRTAO::_collect_mask_snapshot(const std::vector<std::pair<Viewport *, Camera *>> &p_active_vps) {
	Node *search_root = get_viewport() ? (Node *)get_viewport() : (Node *)get_tree()->get_root();
	std::vector<MeshInstance *> meshes;
	std::vector<MultiMeshInstance *> multimeshes;
	std::vector<Ref<Mesh>> extra_meshes;
#ifdef MODULE_CSG_ENABLED
	struct CsgEntry {
		CSGShape *csg;
		Transform local_xf;
		Ref<Mesh> mesh;
	};
	std::vector<CsgEntry> csg_entries;
#endif

	struct MaskCollector {
		static void visit(Node *n,
				std::vector<MeshInstance *> &r_m,
				std::vector<MultiMeshInstance *> &r_mm,
				std::vector<Ref<Mesh>> &r_em
#ifdef MODULE_CSG_ENABLED
				, std::vector<CsgEntry> &r_csg
#endif
				) {
			if (MeshInstance *mi = Object::cast_to<MeshInstance>(n)) {
				if (mi->is_visible_in_tree() &&
						mi->get_cast_shadows_setting() != GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY &&
						mi->get_mesh().is_valid()) {
					r_m.push_back(mi);
				}
			} else if (MultiMeshInstance *mmi = Object::cast_to<MultiMeshInstance>(n)) {
				if (mmi->is_visible_in_tree() &&
						mmi->get_cast_shadows_setting() != GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY &&
						mmi->get_multimesh().is_valid() && mmi->get_multimesh()->get_mesh().is_valid()) {
					r_mm.push_back(mmi);
					r_em.push_back(mmi->get_multimesh()->get_mesh());
				}
			}
#ifdef MODULE_CSG_ENABLED
			else if (CSGShape *csg = Object::cast_to<CSGShape>(n)) {
				if (csg->is_visible_in_tree() && csg->is_root_shape() &&
						csg->get_cast_shadows_setting() != GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY) {
					Array csg_meshes = csg->get_meshes();
					if (csg_meshes.size() >= 2) {
						Ref<Mesh> m = csg_meshes[1];
						if (m.is_valid()) {
							Transform lxf = csg_meshes[0];
							r_csg.push_back({ csg, lxf, m });
							r_em.push_back(m);
						}
					}
				}
			}
#endif
			for (int i = 0; i < n->get_child_count(); i++) {
				visit(n->get_child(i), r_m, r_mm, r_em
#ifdef MODULE_CSG_ENABLED
						, r_csg
#endif
				);
			}
		}
	};

	if (search_root) {
		MaskCollector::visit(search_root, meshes, multimeshes, extra_meshes
#ifdef MODULE_CSG_ENABLED
				, csg_entries
#endif
		);
	}

	MutexLock lock(mask_mutex);
	mask_raytracer.rebuild_blas_cache(meshes, extra_meshes, GL43Raytracer::SimplificationConfig(), 0);

	mask_snapshot.viewports.clear();
	for (size_t vi = 0; vi < p_active_vps.size(); vi++) {
		Viewport *vp = p_active_vps[vi].first;
		Camera *cam = p_active_vps[vi].second;
		if (!(cam->get_cull_mask() & cull_mask)) continue;

		Size2 vsz = vp->get_size();
		int w = MAX(1, (int)vsz.x);
		int h = MAX(1, (int)vsz.y);

		ViewportCompositeState &vst = viewport_states[vp->get_viewport_rid().get_id()];
		if (vst.mask_w != w || vst.mask_h != h) {
			vst.mask_w = w;
			vst.mask_h = h;
			Ref<Image> mimg;
			mimg.instance();
			mimg->create(w, h, false, Image::FORMAT_RGBAH);
			mimg->fill(Color(0.0f, 0.0f, 0.0f, 0.0f));
			vst.mask_texture->create_from_image(mimg, 0);
		}

		CameraMatrix v_view, v_proj, v_vp, v_inv_proj, v_inv_vp;
		GL43EffectHelpers::compute_camera_matrices(cam, vsz, v_view, v_proj, v_vp, v_inv_proj, v_inv_vp);

		MaskViewportTarget vt;
		vt.vp_id = vp->get_viewport_rid().get_id();
		vt.width = w;
		vt.height = h;
		vt.cam_cull_mask = cam->get_cull_mask();
		vt.viewing_mask = viewing_cull_mask;
		vt.mask_tex_rid = vst.mask_texture->get_rid();
		GL43EffectHelpers::camera_matrix_to_float16(v_vp, vt.vp_mat);
		GL43EffectHelpers::extract_frustum_planes(v_vp, vt.frustum_planes);
		mask_snapshot.viewports.push_back(vt);
	}

	mask_raytracer.begin_frame_instances();
	mask_snapshot.materials.clear();

	auto push_materials = [&](const Ref<Mesh> &mesh, auto get_mat_fn) {
		int surf_count = MAX(1, mesh->get_surface_count());
		for (int s = 0; s < surf_count; s++) {
			MaskMaterial mm = {};
			mm.uv_scale_offset[0] = 1.0f;
			mm.uv_scale_offset[1] = 1.0f;
			mm.params[0] = 1.0f;
			mm.params[1] = 0.5f;
			mm.flags[0] = -1;

			Ref<SpatialMaterial> smat = get_mat_fn(s);
			if (smat.is_valid()) {
				Vector3 uvs = smat->get_uv1_scale();
				Vector3 uvo = smat->get_uv1_offset();
				mm.uv_scale_offset[0] = (float)uvs.x;
				mm.uv_scale_offset[1] = (float)uvs.y;
				mm.uv_scale_offset[2] = (float)uvo.x;
				mm.uv_scale_offset[3] = (float)uvo.y;
				mm.params[0] = smat->get_albedo().a;
				mm.params[1] = smat->get_alpha_scissor_threshold();

				bool is_scissor = smat->get_flag(SpatialMaterial::FLAG_USE_ALPHA_SCISSOR);
				bool is_transp = smat->get_feature(SpatialMaterial::FEATURE_TRANSPARENT);
				mm.flags[1] = is_scissor ? 1 : (is_transp ? 2 : 0);
				mm.flags[2] = CLAMP((int)smat->get_cull_mode(), 0, 2);
				if (is_scissor) {
					mm.flags[0] = mask_tex_pool.register_layer(smat->get_texture(SpatialMaterial::TEXTURE_ALBEDO), false, 64);
				}
			}
			mask_snapshot.materials.push_back(mm);
		}
	};

	for (MeshInstance *mi : meshes) {
		Ref<Mesh> mesh = mi->get_mesh();
		if (mask_raytracer.add_mesh_instance(mi, (int32_t)mask_snapshot.materials.size(), Vector3(), 0.0f, 0.0f, 1.0f, true, false, true, 0)) {
			push_materials(mesh, [&](int s) { return mi->get_active_material(s); });
		}
	}

	for (MultiMeshInstance *mmi : multimeshes) {
		Ref<MultiMesh> mm = mmi->get_multimesh();
		Ref<Mesh> mesh = mm->get_mesh();
		if (mask_raytracer.add_multimesh_instance(mmi, (int32_t)mask_snapshot.materials.size(), Vector3(), 0.0f, 0.0f, 1.0f, false, true, 0)) {
			Ref<Material> override_mat = mmi->get_material_override();
			push_materials(mesh, [&](int s) { return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s); });
		}
	}

#ifdef MODULE_CSG_ENABLED
	for (const CsgEntry &entry : csg_entries) {
		Ref<Mesh> mesh = entry.mesh;
		if (mask_raytracer.add_custom_mesh_instance(
					mesh,
					entry.csg->get_global_transform() * entry.local_xf,
					entry.csg->get_layer_mask(),
					entry.csg->get_cast_shadows_setting(),
					(int32_t)mask_snapshot.materials.size(),
					Vector3(), 0.0f, 0.0f, 1.0f, false, true, 0)) {
			Ref<Material> override_mat = entry.csg->get_material_override();
			push_materials(mesh, [&](int s) { return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s); });
		}
	}
#endif

	mask_snapshot.valid = !mask_snapshot.viewports.empty();
}

bool BakedRTAO::_init_gl_mask_resources() {
	if (gl_attempted) return gl_ready;
	gl_attempted = true;
	gl_thread_id = Thread::get_caller_id();

	if (!SharedGL43Context::get_singleton()->is_available()) {
		return false;
	}
	if (!mask_raytracer.init_gl() || !mask_tex_pool.init_gl("BakedRTAO")) {
		return false;
	}
	mask_program = GL43EffectHelpers::link_vf_program(BAKED_MASK_VERT_GLSL, BAKED_MASK_FRAG_GLSL, "BakedRTAO", "ViewingMask");
	composite_program = GL43EffectHelpers::link_vf_program(GL43EffectHelpers::get_fullscreen_vert_glsl(), BAKED_STANDALONE_FRAG_GLSL, "BakedRTAO", "Composite");
	if (!mask_program || !composite_program) {
		return false;
	}
	GLuint mbuf = 0;
	glGenBuffers(1, &mbuf);
	mask_ssbo_materials = (uint32_t)mbuf;
	gl_ready = true;
	return true;
}

void BakedRTAO::_cleanup_gl_mask_resources() {
	if (!gl_ready || gl_thread_id != Thread::get_caller_id()) {
		return;
	}

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	mask_raytracer.cleanup_gl();
	mask_tex_pool.cleanup_gl();

	for (auto &kv : viewport_states) {
		if (kv.second.mask_fbo) {
			GLuint f = (GLuint)kv.second.mask_fbo;
			glDeleteFramebuffers(1, &f);
			kv.second.mask_fbo = 0;
		}
		if (kv.second.mask_depth_tex) {
			GLuint t = (GLuint)kv.second.mask_depth_tex;
			glDeleteTextures(1, &t);
			kv.second.mask_depth_tex = 0;
		}
	}

	if (mask_ssbo_materials) {
		GLuint b = (GLuint)mask_ssbo_materials;
		glDeleteBuffers(1, &b);
		mask_ssbo_materials = 0;
	}
	if (mask_program) {
		glDeleteProgram((GLuint)mask_program);
		mask_program = 0;
	}
	if (composite_program) {
		glDeleteProgram((GLuint)composite_program);
		composite_program = 0;
	}

	gl_ready = false;
	gl_attempted = false;
}

void BakedRTAO::_ensure_mask_fbo_depth(ViewportCompositeState &r_state, int p_w, int p_h) {
	if (r_state.mask_fbo != 0 && r_state.alloc_depth_w == p_w && r_state.alloc_depth_h == p_h) {
		return;
	}
	r_state.alloc_depth_w = p_w;
	r_state.alloc_depth_h = p_h;
	if (r_state.mask_fbo == 0) {
		GLuint f = 0;
		glGenFramebuffers(1, &f);
		r_state.mask_fbo = (uint32_t)f;
	}
	glActiveTexture(GL_TEXTURE0);
	GL43EffectHelpers::ensure_depth24_texture(r_state.mask_depth_tex, p_w, p_h);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void BakedRTAO::_render_standalone_viewing_mask() {
	if (!enabled || !is_inside_tree() || !is_visible_in_tree() || baked_texture.is_null() || !auto_composite || viewing_cull_mask == 0) {
		return;
	}

	RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
	if (!rs) return;

	MutexLock lock(mask_mutex);
	if (!mask_snapshot.valid || mask_snapshot.viewports.empty()) {
		return;
	}

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}
	if (!_init_gl_mask_resources()) {
		return;
	}

	glDisable(0x8DB9); // GL_FRAMEBUFFER_SRGB
	mask_tex_pool.flush_blits(rs, 256, 64, false);
	mask_raytracer.sync_gpu_buffers();
	GL43EffectHelpers::upload_ssbo(mask_ssbo_materials, mask_snapshot.materials.data(), mask_snapshot.materials.size() * sizeof(MaskMaterial), sizeof(MaskMaterial));
	glBindBufferBase(0x90D2, 5, (GLuint)mask_ssbo_materials); // GL_SHADER_STORAGE_BUFFER

	glUseProgram((GLuint)mask_program);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(0x8C1A, (GLuint)mask_tex_pool.get_tex_array()); // GL_TEXTURE_2D_ARRAY
	GLint loc_model = glGetUniformLocation((GLuint)mask_program, "uModel");
	GLint loc_mvp = glGetUniformLocation((GLuint)mask_program, "uMVP");
	GLint loc_mat_off = glGetUniformLocation((GLuint)mask_program, "uMatOffset");
	GLint loc_mask_val = glGetUniformLocation((GLuint)mask_program, "uMaskVal");
	GLint loc_flip = glGetUniformLocation((GLuint)mask_program, "uFlipFacing");
	glUniform1i(glGetUniformLocation((GLuint)mask_program, "uTexArray"), 0);

	const std::vector<GL43Raytracer::InstanceEntry> &insts = mask_raytracer.get_instances();

	for (size_t vi = 0; vi < mask_snapshot.viewports.size(); vi++) {
		const MaskViewportTarget &vt = mask_snapshot.viewports[vi];
		auto st_it = viewport_states.find(vt.vp_id);
		if (st_it == viewport_states.end()) continue;
		ViewportCompositeState &vst = st_it->second;

		RasterizerStorageGLES3::Texture *godot_mask_tex = rs->texture_owner.getornull(vt.mask_tex_rid);
		if (!godot_mask_tex || godot_mask_tex->tex_id == 0) continue;
		if (godot_mask_tex->width != vt.width || godot_mask_tex->height != vt.height) continue;

		_ensure_mask_fbo_depth(vst, vt.width, vt.height);
		glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)vst.mask_fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, godot_mask_tex->tex_id, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, (GLuint)vst.mask_depth_tex, 0);
		GLenum buf = GL_COLOR_ATTACHMENT0;
		glDrawBuffers(1, &buf);

		glViewport(0, 0, vt.width, vt.height);
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

		for (size_t i = 0; i < insts.size(); i++) {
			const GL43Raytracer::InstanceEntry &inst = insts[i];
			if (!inst.draw_in_gbuf) continue;
			if (!(inst.layer_mask & vt.cam_cull_mask)) continue;
			if (!GL43EffectHelpers::aabb_in_frustum(inst.world_aabb, vt.frustum_planes)) continue;

			uint32_t vao = 0;
			int32_t vcount = 0;
			if (!mask_raytracer.get_instance_draw_data(i, vao, vcount)) continue;

			float mvp[16];
			GL43EffectHelpers::mat4_mul(vt.vp_mat, inst.model_mat, mvp);
			glUniformMatrix4fv(loc_model, 1, GL_FALSE, inst.model_mat);
			glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp);
			glUniform1i(loc_mat_off, inst.mat_offset);
			glUniform1f(loc_mask_val, ((inst.layer_mask & vt.viewing_mask) != 0) ? 1.0f : 0.0f);
			glUniform1i(loc_flip, inst.flip_facing ? 1 : 0);
			glBindVertexArray((GLuint)vao);
			glDrawArrays(GL_TRIANGLES, 0, vcount);
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, godot_mask_tex->tex_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	glBindBufferBase(0x90D2, 5, 0);
	glBindBuffer(0x90D2, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(0x8C1A, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

bool BakedRTAO::_is_rtao_active() const {
	Viewport *vp = get_viewport();
	if (!vp) return false;
	Node *root = vp ? (Node *)vp : (Node *)get_tree()->get_root();
	if (!root) return false;

	struct RTAOFinder {
		static bool check(Node *n) {
			if (RTAO *r = Object::cast_to<RTAO>(n)) {
				if (r->is_enabled() && r->is_visible_in_tree()) {
					return true;
				}
			}
			for (int i = 0; i < n->get_child_count(); i++) {
				if (check(n->get_child(i))) return true;
			}
			return false;
		}
	};

	return RTAOFinder::check(root);
}

void BakedRTAO::_update_standalone_composite() {
	if (!enabled || !is_inside_tree() || !is_visible_in_tree() || baked_texture.is_null() || !auto_composite || viewing_cull_mask == 0) {
		_detach_all_viewports();
		MutexLock lock(mask_mutex);
		mask_snapshot.valid = false;
		return;
	}

	bool in_editor = Engine::get_singleton()->is_editor_hint();
	if (in_editor && !editor_preview) {
		_detach_all_viewports();
		MutexLock lock(mask_mutex);
		mask_snapshot.valid = false;
		return;
	}

	if (_is_rtao_active()) {
		_detach_all_viewports();
		MutexLock lock(mask_mutex);
		mask_snapshot.valid = false;
		return;
	}

	vp_collect_counter++;
	std::vector<std::pair<Viewport *, Camera *>> active_vps;
	GL43EffectHelpers::find_active_viewports(this, get_viewport(), active_vps);
	if (active_vps.empty()) {
		_prune_stale_viewports();
		MutexLock lock(mask_mutex);
		mask_snapshot.valid = false;
		return;
	}

	std::vector<Light *> scene_lights;
	if (direct_light_preservation > 0.001f) {
		Node *search_root = get_viewport() ? (Node *)get_viewport() : (Node *)get_tree()->get_root();
		struct LightCollector {
			static void visit(Node *n, std::vector<Light *> &out_l) {
				if (Light *l = Object::cast_to<Light>(n)) {
					if (l->is_visible_in_tree() && !l->is_negative() && l->get_param(Light::PARAM_ENERGY) > 0.0001f) {
						out_l.push_back(l);
					}
				}
				for (int i = 0; i < n->get_child_count(); i++) {
					visit(n->get_child(i), out_l);
				}
			}
		};
		if (search_root) {
			LightCollector::visit(search_root, scene_lights);
		}
	}

	bool use_physical_atten = ProjectSettings::get_singleton()->has_setting("rendering/quality/shading/use_physical_light_attenuation")
			? bool(ProjectSettings::get_singleton()->get("rendering/quality/shading/use_physical_light_attenuation"))
			: false;

	Transform inv_xform = get_global_transform().affine_inverse();
	Color ac = ao_color.to_linear();
	float rx = MAX((float)baked_texture->get_width(), 1.0f);
	float ry = MAX((float)baked_texture->get_height(), 1.0f);
	float rz = MAX((float)baked_texture->get_depth(), 1.0f);
	float voxel_size = MAX((2.0f * extents.x) / rx, MAX((2.0f * extents.y) / ry, (2.0f * extents.z) / rz));

	const bool use_viewing_mask = (viewing_cull_mask != 0xFFFFFFFF);
	{
		MutexLock lock(mask_mutex);
		for (size_t vi = 0; vi < active_vps.size(); vi++) {
			if (active_vps[vi].second->get_cull_mask() & cull_mask) {
				_get_or_create_vp_state(active_vps[vi].first);
			}
		}
	}
	_collect_mask_snapshot(active_vps);

	MutexLock lock(mask_mutex);
	for (size_t vi = 0; vi < active_vps.size(); vi++) {
		Viewport *active_vp = active_vps[vi].first;
		Camera *cam = active_vps[vi].second;
		if (!(cam->get_cull_mask() & cull_mask)) {
			auto st_it = viewport_states.find(active_vp->get_viewport_rid().get_id());
			if (st_it != viewport_states.end()) {
				st_it->second.active_composite = false;
			}
			continue;
		}
		Size2 vp_size = active_vp->get_size();
		ViewportCompositeState &vst = _get_or_create_vp_state(active_vp);

		Transform cam_xf = cam->get_camera_transform();
		CameraMatrix cam_view, cam_proj, cam_vp, cam_inv_proj, cam_inv_vp;
		GL43EffectHelpers::compute_camera_matrices(cam, vp_size, cam_view, cam_proj, cam_vp, cam_inv_proj, cam_inv_vp);

		float fplanes[6][4];
		GL43EffectHelpers::extract_frustum_planes(cam_vp, fplanes);
		uint32_t cam_cull_mask = cam->get_cull_mask();

		Ref<World> world = active_vp->find_world();
		if (world.is_null() && get_viewport()) world = get_viewport()->find_world();
		Ref<Environment> env = world.is_valid() ? world->get_environment() : Ref<Environment>();
		if (env.is_null() && world.is_valid()) env = world->get_fallback_environment();

		float amb_lum = 0.15f;
		if (env.is_valid()) {
			Color amb = env->get_ambient_light_color().to_linear();
			float amb_e = env->get_ambient_light_energy();
			float sky_c = env->get_ambient_light_sky_contribution();
			float bg_e = env->get_bg_energy();
			Color sky_avg(0.35f, 0.45f, 0.55f);
			bool has_sky_bg = false;

			if (env->get_background() == Environment::BG_SKY && env->get_sky().is_valid()) {
				has_sky_bg = true;
				Ref<ProceduralSky> psky = env->get_sky();
				if (psky.is_valid()) {
					Color st = psky->get_sky_top_color().to_linear() * psky->get_sky_energy();
					Color sh = psky->get_sky_horizon_color().to_linear() * psky->get_sky_energy();
					Color gh = psky->get_ground_horizon_color().to_linear() * psky->get_ground_energy();
					Color gb = psky->get_ground_bottom_color().to_linear() * psky->get_ground_energy();
					sky_avg = Color(
							(st.r + sh.r + gh.r + gb.r) * 0.25f * bg_e,
							(st.g + sh.g + gh.g + gb.g) * 0.25f * bg_e,
							(st.b + sh.b + gh.b + gb.b) * 0.25f * bg_e);
				} else {
					sky_avg = Color(0.5f * bg_e, 0.5f * bg_e, 0.5f * bg_e);
				}
			} else if (env->get_background() == Environment::BG_COLOR || env->get_background() == Environment::BG_CLEAR_COLOR) {
				has_sky_bg = true;
				Color bg = env->get_bg_color().to_linear();
				sky_avg = Color(bg.r * bg_e, bg.g * bg_e, bg.b * bg_e);
			} else {
				sky_c = 0.0f;
			}

			Color eff_amb = has_sky_bg ? amb.linear_interpolate(sky_avg, sky_c) : amb;
			eff_amb = Color(eff_amb.r * amb_e, eff_amb.g * amb_e, eff_amb.b * amb_e);
			amb_lum = MAX(eff_amb.r * 0.2126f + eff_amb.g * 0.7152f + eff_amb.b * 0.0722f, 0.05f);
		}

		struct LightCand {
			Plane pos_range;
			Plane dir_type;
			Plane color_atten;
			Plane spot_params;
			int type_order;
			float sort_dist;
		};
		std::vector<LightCand> light_cands;
		light_cands.reserve(scene_lights.size());

		for (Light *l : scene_lights) {
			if (!(l->get_cull_mask() & cam_cull_mask)) continue;
			Transform lx = l->get_global_transform();
			Color lc = l->get_color().to_linear();
			float energy = l->get_param(Light::PARAM_ENERGY);
			float has_shad = l->has_shadow() ? 1.0f : 0.0f;

			if (Object::cast_to<DirectionalLight>(l)) {
				Vector3 to_light = lx.basis.get_axis(2).normalized();
				light_cands.push_back({
						Plane(0, 0, 0, 0),
						Plane((float)to_light.x, (float)to_light.y, (float)to_light.z, 0.0f),
						Plane(lc.r * energy, lc.g * energy, lc.b * energy, 1.0f),
						Plane(0, 0, has_shad, 0),
						0,
						0.0f });
			} else {
				float range = MAX(l->get_param(Light::PARAM_RANGE), 0.01f);
				if (!GL43EffectHelpers::sphere_in_frustum(lx.origin, range, fplanes)) continue;
				float d = MAX(0.0f, (float)lx.origin.distance_to(cam_xf.origin) - range);
				float atten_p = l->get_param(Light::PARAM_ATTENUATION);

				if (Object::cast_to<SpotLight>(l)) {
					Vector3 beam_dir = (-lx.basis.get_axis(2)).normalized();
					light_cands.push_back({
							Plane((float)lx.origin.x, (float)lx.origin.y, (float)lx.origin.z, range),
							Plane((float)beam_dir.x, (float)beam_dir.y, (float)beam_dir.z, 2.0f),
							Plane(lc.r * energy, lc.g * energy, lc.b * energy, atten_p),
							Plane(Math::cos(Math::deg2rad(l->get_param(Light::PARAM_SPOT_ANGLE))), l->get_param(Light::PARAM_SPOT_ATTENUATION), has_shad, 0),
							1,
							d });
				} else {
					light_cands.push_back({
							Plane((float)lx.origin.x, (float)lx.origin.y, (float)lx.origin.z, range),
							Plane(0, 0, 0, 1.0f),
							Plane(lc.r * energy, lc.g * energy, lc.b * energy, atten_p),
							Plane(0, 0, has_shad, 0),
							1,
							d });
				}
			}
		}

		std::sort(light_cands.begin(), light_cands.end(), [](const LightCand &a, const LightCand &b) {
			if (a.type_order != b.type_order) return a.type_order < b.type_order;
			return a.sort_dist < b.sort_dist;
		});

		int num_lights = MIN((int)light_cands.size(), 32);
		vst.num_lights = num_lights;
		vst.ambient_lum = amb_lum;
		vst.use_physical_atten = use_physical_atten;
		vst.use_viewing_mask = use_viewing_mask;
		vst.voxel_size = voxel_size;
		GL43EffectHelpers::transform_to_float16(inv_xform, vst.inv_xform);
		GL43EffectHelpers::transform_to_float16(cam_xf, vst.cam_matrix);

		for (int li = 0; li < num_lights; li++) {
			const Plane &pr = light_cands[li].pos_range;
			const Plane &dt = light_cands[li].dir_type;
			const Plane &ca = light_cands[li].color_atten;
			const Plane &sp = light_cands[li].spot_params;
			vst.light_pos_range[li * 4 + 0] = pr.normal.x; vst.light_pos_range[li * 4 + 1] = pr.normal.y; vst.light_pos_range[li * 4 + 2] = pr.normal.z; vst.light_pos_range[li * 4 + 3] = pr.d;
			vst.light_dir_type[li * 4 + 0] = dt.normal.x;  vst.light_dir_type[li * 4 + 1] = dt.normal.y;  vst.light_dir_type[li * 4 + 2] = dt.normal.z;  vst.light_dir_type[li * 4 + 3] = dt.d;
			vst.light_color_atten[li * 4 + 0] = ca.normal.x; vst.light_color_atten[li * 4 + 1] = ca.normal.y; vst.light_color_atten[li * 4 + 2] = ca.normal.z; vst.light_color_atten[li * 4 + 3] = ca.d;
			vst.light_spot_params[li * 4 + 0] = sp.normal.x; vst.light_spot_params[li * 4 + 1] = sp.normal.y; vst.light_spot_params[li * 4 + 2] = sp.normal.z; vst.light_spot_params[li * 4 + 3] = sp.d;
		}
	}
	_prune_stale_viewports();
}

void BakedRTAO::_on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx) {
	if (!enabled || !auto_composite || !is_inside_tree() || !is_visible_in_tree() || baked_texture.is_null() || viewing_cull_mask == 0 || !gl_ready || composite_program == 0) {
		return;
	}
	if (Engine::get_singleton()->is_editor_hint() && !editor_preview) {
		return;
	}

	RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
	if (!rs) return;

	ViewportCompositeState st_copy;
	bool found = false;
	{
		MutexLock lock(mask_mutex);
		if (!mask_snapshot.valid) return;
		for (const auto &kv : viewport_states) {
			if (kv.second.color_tex_rid == p_ctx.color_tex_rid && kv.second.active_composite) {
				st_copy = kv.second;
				found = true;
				break;
			}
		}
	}
	if (!found) return;

	RasterizerStorageGLES3::Texture *bt = rs->texture_owner.getornull(baked_texture->get_rid());
	if (bt) bt = bt->get_ptr();
	if (!bt || bt->tex_id == 0) return;

	GLuint mask_tex_id = 0;
	if (st_copy.mask_texture.is_valid()) {
		RasterizerStorageGLES3::Texture *mt = rs->texture_owner.getornull(st_copy.mask_texture->get_rid());
		if (mt) mt = mt->get_ptr();
		if (mt) mask_tex_id = mt->tex_id;
	}

	glUseProgram((GLuint)composite_program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(0x806F, bt->tex_id); // GL_TEXTURE_3D
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uBakedTex"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, p_ctx.depth_tex_id);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uDepthTex"), 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, mask_tex_id);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uViewingMaskTex"), 2);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uHasNormalMaskTex"), mask_tex_id != 0 ? 1 : 0);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uUseViewingMask"), st_copy.use_viewing_mask ? 1 : 0);

	Color ac = ao_color.to_linear();
	glUniformMatrix4fv(glGetUniformLocation((GLuint)composite_program, "uInvXform"), 1, GL_FALSE, st_copy.inv_xform);
	glUniform3f(glGetUniformLocation((GLuint)composite_program, "uExtents"), (float)extents.x, (float)extents.y, (float)extents.z);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uVoxelSize"), st_copy.voxel_size);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uIntensity"), intensity);
	glUniform3f(glGetUniformLocation((GLuint)composite_program, "uAOColor"), ac.r, ac.g, ac.b);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uEdgeFalloff"), edge_falloff);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uDirectPreservation"), direct_light_preservation);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uDebugMode"), (int)debug_mode);

	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uNumLights"), st_copy.num_lights);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uAmbientLum"), st_copy.ambient_lum);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uUsePhysicalAtten"), st_copy.use_physical_atten ? 1 : 0);
	if (st_copy.num_lights > 0) {
		glUniform4fv(glGetUniformLocation((GLuint)composite_program, "uLightPosRange"), st_copy.num_lights, st_copy.light_pos_range);
		glUniform4fv(glGetUniformLocation((GLuint)composite_program, "uLightDirType"), st_copy.num_lights, st_copy.light_dir_type);
		glUniform4fv(glGetUniformLocation((GLuint)composite_program, "uLightColorAtten"), st_copy.num_lights, st_copy.light_color_atten);
		glUniform4fv(glGetUniformLocation((GLuint)composite_program, "uLightSpotParams"), st_copy.num_lights, st_copy.light_spot_params);
	}

	glUniformMatrix4fv(glGetUniformLocation((GLuint)composite_program, "uCamMatrix"), 1, GL_FALSE, st_copy.cam_matrix);
	glUniformMatrix4fv(glGetUniformLocation((GLuint)composite_program, "uInvProj"), 1, GL_FALSE, p_ctx.inv_proj_mat);
	glUniform2f(glGetUniformLocation((GLuint)composite_program, "uPixelSize"), 1.0f / (float)p_ctx.width, 1.0f / (float)p_ctx.height);

	if (debug_mode != DEBUG_NONE) {
		glDisable(GL_BLEND);
	} else {
		glEnable(GL_BLEND);
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_DST_COLOR, GL_ZERO);
	}

	glDrawArrays(GL_TRIANGLES, 0, 3);
	glDisable(GL_BLEND);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(0x806F, 0);
}

void BakedRTAO::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	if (!enabled) {
		_detach_all_viewports();
	}
	update_gizmo();
}

void BakedRTAO::set_editor_preview(bool p_preview) {
	editor_preview = p_preview;
	if (!editor_preview && Engine::get_singleton()->is_editor_hint()) {
		_detach_all_viewports();
	}
}

void BakedRTAO::set_auto_composite(bool p_auto) {
	auto_composite = p_auto;
	if (!auto_composite) {
		_detach_all_viewports();
	}
}

void BakedRTAO::set_extents(const Vector3 &p_extents) {
	extents = Vector3(MAX(p_extents.x, 0.1f), MAX(p_extents.y, 0.1f), MAX(p_extents.z, 0.1f));
	update_gizmo();
}

void BakedRTAO::set_resolution(const Vector3 &p_res) {
	resolution = Vector3(
			CLAMP((int)Math::round(p_res.x), 4, 128),
			CLAMP((int)Math::round(p_res.y), 4, 128),
			CLAMP((int)Math::round(p_res.z), 4, 128));
}

void BakedRTAO::set_intensity(float p_intensity) {
	intensity = MAX(p_intensity, 0.0f);
}

void BakedRTAO::set_edge_falloff(float p_falloff) {
	edge_falloff = CLAMP(p_falloff, 0.0f, 1.0f);
}

void BakedRTAO::set_direct_light_preservation(float p_val) {
	direct_light_preservation = CLAMP(p_val, 0.0f, 1.0f);
}

void BakedRTAO::set_bake_samples(int p_samples) {
	bake_samples = CLAMP(p_samples, 1, 256);
}

void BakedRTAO::set_bake_radius(float p_radius) {
	bake_radius = MAX(p_radius, 0.05f);
}

void BakedRTAO::set_bake_ray_bias(float p_bias) {
	bake_ray_bias = MAX(p_bias, 0.001f);
}

void BakedRTAO::set_bake_power(float p_power) {
	bake_power = MAX(p_power, 0.05f);
}

void BakedRTAO::set_bake_gi_direction_strength(float p_strength) {
	bake_gi_direction_strength = CLAMP(p_strength, 0.0f, 1.0f);
}

void BakedRTAO::set_bake_denoise_passes(int p_passes) {
	bake_denoise_passes = CLAMP(p_passes, 0, 5);
}

void BakedRTAO::set_baked_texture(const Ref<Texture3D> &p_tex) {
	baked_texture = p_tex;
	if (baked_texture.is_null()) {
		baked_data.resize(0);
		baked_data_resolution = Vector3(0, 0, 0);
	}
}

void BakedRTAO::set_baked_data(const PoolByteArray &p_data) {
	baked_data = p_data;
	_rebuild_texture_from_baked_data();
}

void BakedRTAO::set_baked_data_resolution(const Vector3 &p_res) {
	baked_data_resolution = p_res;
	_rebuild_texture_from_baked_data();
}

void BakedRTAO::_rebuild_texture_from_baked_data() {
	int rx = (int)Math::round(baked_data_resolution.x);
	int ry = (int)Math::round(baked_data_resolution.y);
	int rz = (int)Math::round(baked_data_resolution.z);
	if (rx < 1 || ry < 1 || rz < 1) {
		return;
	}
	int slice_bytes = rx * ry * 8; // Image::FORMAT_RGBAH = 4 * 2 bytes per pixel
	if (baked_data.size() != slice_bytes * rz) {
		return;
	}

	Ref<Texture3D> tex;
	tex.instance();
	tex->create(rx, ry, rz, Image::FORMAT_RGBAH, bake_filter ? Texture::FLAG_FILTER : 0);

	PoolByteArray::Read r = baked_data.read();
	for (int z = 0; z < rz; z++) {
		PoolByteArray slice_buf;
		slice_buf.resize(slice_bytes);
		{
			PoolByteArray::Write w = slice_buf.write();
			memcpy(w.ptr(), r.ptr() + z * slice_bytes, slice_bytes);
		}
		Ref<Image> slice;
		slice.instance();
		slice->create(rx, ry, false, Image::FORMAT_RGBAH, slice_buf);
		tex->set_layer_data(slice, z);
	}

	baked_texture = tex;
	_change_notify();
}

AABB BakedRTAO::get_world_aabb() const {
	AABB local(-extents, extents * 2.0f);
	return get_global_transform().xform(local);
}

bool BakedRTAO::bake() {
	ERR_FAIL_COND_V_MSG(!is_inside_tree(), false, "BakedRTAO must be inside the SceneTree to bake.");

	Viewport *vp = get_viewport();
	Node *search_root = vp ? (Node *)vp : (Node *)get_tree()->get_root();
	ERR_FAIL_COND_V(!search_root, false);

	std::vector<LocalRTAO *> local_vols;
	std::vector<MeshInstance *> meshes;
	std::vector<MultiMeshInstance *> multimeshes;
	std::vector<Light *> lights;
#ifdef MODULE_CSG_ENABLED
	std::vector<CSGShape *> csg_shapes;
#endif

	struct Collector {
		static void visit(Node *n, bool collect_vols, bool collect_lights,
				std::vector<LocalRTAO *> &out_vols,
				std::vector<MeshInstance *> &out_meshes,
				std::vector<MultiMeshInstance *> &out_multimeshes,
				std::vector<Light *> &out_lights
#ifdef MODULE_CSG_ENABLED
				, std::vector<CSGShape *> &out_csgs
#endif
				) {
			if (collect_vols) {
				if (LocalRTAO *lv = Object::cast_to<LocalRTAO>(n)) {
					if (lv->is_enabled() && lv->is_visible_in_tree()) {
						out_vols.push_back(lv);
					}
				}
			}
			if (collect_lights) {
				if (Light *l = Object::cast_to<Light>(n)) {
					if (l->is_visible_in_tree() && !l->is_negative() && l->get_param(Light::PARAM_ENERGY) > 0.0001f) {
						out_lights.push_back(l);
					}
				}
			}
			if (MeshInstance *mi = Object::cast_to<MeshInstance>(n)) {
				if (mi->is_visible_in_tree() && mi->get_mesh().is_valid()) {
					out_meshes.push_back(mi);
				}
			} else if (MultiMeshInstance *mmi = Object::cast_to<MultiMeshInstance>(n)) {
				if (mmi->is_visible_in_tree() && mmi->get_multimesh().is_valid() && mmi->get_multimesh()->get_mesh().is_valid()) {
					out_multimeshes.push_back(mmi);
				}
			}
#ifdef MODULE_CSG_ENABLED
			else if (CSGShape *csg = Object::cast_to<CSGShape>(n)) {
				if (csg->is_visible_in_tree() && csg->is_root_shape()) {
					out_csgs.push_back(csg);
				}
			}
#endif
			for (int i = 0; i < n->get_child_count(); i++) {
				visit(n->get_child(i), collect_vols, collect_lights, out_vols, out_meshes, out_multimeshes, out_lights
#ifdef MODULE_CSG_ENABLED
						, out_csgs
#endif
				);
			}
		}
	};

	const bool use_directional_gi = bake_directional_gi && bake_gi_direction_strength > 0.0001f;
	Collector::visit(search_root, bake_local_volumes, use_directional_gi, local_vols, meshes, multimeshes, lights
#ifdef MODULE_CSG_ENABLED
			, csg_shapes
#endif
	);

	AOBakeRaycaster raycaster;
	raycaster.build(meshes, multimeshes,
#ifdef MODULE_CSG_ENABLED
			csg_shapes,
#endif
			shadow_cull_mask, use_directional_gi);

	bool use_physical_atten = ProjectSettings::get_singleton()->has_setting("rendering/quality/shading/use_physical_light_attenuation")
			? bool(ProjectSettings::get_singleton()->get("rendering/quality/shading/use_physical_light_attenuation"))
			: false;

	BakeEnvironment bake_env;
	std::vector<BakeLight> bake_lights;
	if (use_directional_gi) {
		Ref<World> world = vp ? vp->find_world() : Ref<World>();
		Ref<Environment> env = world.is_valid() ? world->get_environment() : Ref<Environment>();
		if (env.is_null() && world.is_valid()) {
			env = world->get_fallback_environment();
		}
		bake_env.init_from_environment(env);

		for (size_t i = 0; i < lights.size(); i++) {
			Light *l = lights[i];
			if (!(l->get_cull_mask() & cull_mask)) {
				continue;
			}
			Transform lxf = l->get_global_transform();
			Color lc = l->get_color().to_linear();
			float energy = l->get_param(Light::PARAM_ENERGY);

			BakeLight bl;
			bl.color = Vector3(lc.r, lc.g, lc.b) * energy;
			bl.has_shadow = l->has_shadow();
			bl.use_physical_atten = use_physical_atten;

			if (Object::cast_to<DirectionalLight>(l)) {
				bl.type = BakeLight::LIGHT_DIRECTIONAL;
				bl.dir = lxf.basis.get_axis(2).normalized();
				bake_lights.push_back(bl);
			} else if (OmniLight *ol = Object::cast_to<OmniLight>(l)) {
				bl.type = BakeLight::LIGHT_OMNI;
				bl.pos = lxf.origin;
				bl.inv_xform = lxf.affine_inverse();
				bl.range = MAX(ol->get_param(Light::PARAM_RANGE), 0.01f);
				bl.attenuation = MAX(ol->get_param(Light::PARAM_ATTENUATION), 0.01f);
				bl.is_square = ol->is_square_shape();
				bake_lights.push_back(bl);
			} else if (SpotLight *sl = Object::cast_to<SpotLight>(l)) {
				bl.type = BakeLight::LIGHT_SPOT;
				bl.pos = lxf.origin;
				bl.inv_xform = lxf.affine_inverse();
				bl.dir = (-lxf.basis.get_axis(2)).normalized();
				bl.range = MAX(sl->get_param(Light::PARAM_RANGE), 0.01f);
				bl.attenuation = MAX(sl->get_param(Light::PARAM_ATTENUATION), 0.01f);
				bl.spot_cutoff = Math::cos(Math::deg2rad(sl->get_param(Light::PARAM_SPOT_ANGLE)));
				bl.spot_atten = sl->get_param(Light::PARAM_SPOT_ATTENUATION);
				bl.is_square = sl->is_square_shape();
				bake_lights.push_back(bl);
			}
		}
	}

	std::vector<BakeLocalVolume> baked_local_vols;
	baked_local_vols.reserve(local_vols.size());
	for (size_t i = 0; i < local_vols.size(); i++) {
		LocalRTAO *lv = local_vols[i];
		BakeLocalVolume blv;
		blv.inv_xform = lv->get_global_transform().affine_inverse();
		blv.mode = lv->get_mode();
		blv.shape = lv->get_shape();
		blv.extents = lv->get_extents();
		blv.edge_falloff = lv->get_edge_falloff();
		blv.occlusion = lv->get_occlusion();
		blv.radius_multiplier = lv->get_radius_multiplier();
		blv.blend_mode = lv->get_blend_mode();
		baked_local_vols.push_back(blv);
	}

	int rx = CLAMP((int)Math::round(resolution.x), 4, 128);
	int ry = CLAMP((int)Math::round(resolution.y), 4, 128);
	int rz = CLAMP((int)Math::round(resolution.z), 4, 128);

	std::vector<Color> voxels((size_t)rx * (size_t)ry * (size_t)rz);

	const Transform global_xf = get_global_transform();
	const Vector3 bake_extents = extents;
	const int samples = bake_samples;
	const float radius = bake_radius;
	const float ray_bias = bake_ray_bias;
	const float power = bake_power;
	const float golden_angle = 2.399963229728653f;

	struct FibDir {
		float bx, by, bz;
	};
	std::vector<FibDir> fib_dirs((size_t)samples);
	for (int s = 0; s < samples; s++) {
		float sy = 1.0f - (2.0f * ((float)s + 0.5f) / (float)samples);
		float sr = Math::sqrt(MAX(0.0f, 1.0f - sy * sy));
		float phi = (float)s * golden_angle;
		fib_dirs[s] = { Math::cos(phi) * sr, sy, Math::sin(phi) * sr };
	}

	const int total_rows = ry * rz;
	std::atomic<int> next_row(0);

	auto worker_fn = [&]() {
		struct VoxelDirectLight {
			Vector3 dir;
			float lum;
		};
		struct RayHitSample {
			Vector3 dir;
			float hit_t;
			float vis;
			int32_t hit_tri;
		};
		std::vector<VoxelDirectLight> voxel_lights;
		voxel_lights.reserve(bake_lights.size());
		std::vector<RayHitSample> hit_samples((size_t)samples);

		while (true) {
			int row = next_row.fetch_add(1, std::memory_order_relaxed);
			if (row >= total_rows) {
				break;
			}
			int z = row / ry;
			int y = row % ry;

			float uz = ((float)z + 0.5f) / (float)rz - 0.5f;
			float uy = ((float)y + 0.5f) / (float)ry - 0.5f;
			Color *dst_row = &voxels[(size_t)row * (size_t)rx];

			for (int x = 0; x < rx; x++) {
				float ux = ((float)x + 0.5f) / (float)rx - 0.5f;
				Vector3 local_pos(ux * 2.0f * bake_extents.x, uy * 2.0f * bake_extents.y, uz * 2.0f * bake_extents.z);
				Vector3 world_pos = global_xf.xform(local_pos);

				float local_radius_mult = 1.0f;
				for (size_t i = 0; i < baked_local_vols.size(); i++) {
					const BakeLocalVolume &lv = baked_local_vols[i];
					if (lv.mode != LocalRTAO::MODE_MODULATE_TRACE) {
						continue;
					}
					float factor = lv.sample_shape_factor(world_pos);
					if (factor > 0.0001f) {
						local_radius_mult *= Math::lerp(1.0f, lv.radius_multiplier, factor);
					}
				}
				float eff_radius = MAX(radius * local_radius_mult, 0.01f);

				float mean_vis = 1.0f;
				Vector3 l1_bent(0.0f, 0.0f, 0.0f);

				// Fast O(log N) empty-space rejection: skip all rays if no geometry is within eff_radius + ray_bias
				if (raycaster.overlaps_sphere(world_pos, eff_radius + ray_bias)) {
					// 3D R3 quasi-random phase distributes angles evenly across adjacent voxels without white-noise clumping
					float phase_jitter = Math::fmod((float)x * 0.8191725134f + (float)y * 0.6710436067f + (float)z * 0.5497004779f, 1.0f) * (float)Math_TAU;
					float cj = Math::cos(phase_jitter);
					float sj = Math::sin(phase_jitter);
					float vis_sum = 0.0f;
					Vector3 geom_bent_sum(0.0f, 0.0f, 0.0f);

					// Phase 1: Short AO rays up to eff_radius only
					for (int s = 0; s < samples; s++) {
						const FibDir &fd = fib_dirs[s];
						Vector3 dir(fd.bx * cj - fd.bz * sj, fd.by, fd.bz * cj + fd.bx * sj);

						int32_t hit_tri = -1;
						float hit_t = raycaster.trace_distance(world_pos, dir, ray_bias + 0.001f, eff_radius + ray_bias, use_directional_gi ? &hit_tri : nullptr) - ray_bias;
						float vis = 1.0f;
						if (hit_t < eff_radius) {
							float occ = Math::pow(CLAMP(1.0f - hit_t / eff_radius, 0.0f, 1.0f), power);
							vis = 1.0f - occ;
						}
						vis_sum += vis;
						geom_bent_sum += dir * vis;
						hit_samples[s] = { dir, hit_t, vis, hit_tri };
					}

					mean_vis = vis_sum / (float)samples;
					Vector3 b_geom = geom_bent_sum / (float)samples;
					l1_bent = b_geom;

					// Phase 2: Only compute GI + light direction if voxel has partial occlusion (max_b > 0)
					if (use_directional_gi && mean_vis > 1e-5f && mean_vis < 0.9999f) {
						voxel_lights.clear();
						for (size_t li = 0; li < bake_lights.size(); li++) {
							const BakeLight &bl = bake_lights[li];
							Vector3 l_dir, l_rad;
							float l_max_dist = 0.0f;
							if (!bl.evaluate_at(world_pos, l_dir, l_rad, l_max_dist)) {
								continue;
							}
							float lum = ao_luminance(l_rad);
							if (lum <= 1e-5f) {
								continue;
							}
							if (bl.has_shadow && l_max_dist > ray_bias + 0.002f) {
								if (raycaster.is_occluded(world_pos, l_dir, ray_bias + 0.001f, l_max_dist)) {
									continue;
								}
							}
							voxel_lights.push_back({ l_dir, lum });
						}

						const float gi_trace_max = MAX(eff_radius * 4.0f, 32.0f);
						const float inv_pi = 1.0f / (float)Math_PI;
						float gi_l0 = 0.0f;
						Vector3 gi_l1(0.0f, 0.0f, 0.0f);

						for (int s = 0; s < samples; s++) {
							const RayHitSample &hs = hit_samples[s];
							if (hs.vis <= 0.001f) {
								continue;
							}

							int32_t hit_tri = hs.hit_tri;
							float hit_t = hs.hit_t;
							if (hit_tri < 0 && gi_trace_max > eff_radius) {
								float cont_tmin = MAX(ray_bias + 0.001f, eff_radius + ray_bias - 0.001f);
								hit_t = raycaster.trace_distance(world_pos, hs.dir, cont_tmin, gi_trace_max + ray_bias, &hit_tri) - ray_bias;
							}

							Vector3 gi_rad(0.0f, 0.0f, 0.0f);
							if (hit_tri >= 0) {
								const AOBakeTri &ht = raycaster.get_triangle(hit_tri);
								if (ao_luminance(ht.albedo) > 1e-4f) {
									Vector3 hit_pos = world_pos + hs.dir * hit_t;
									Vector3 hit_n = ht.normal;
									if (hit_n.dot(hs.dir) > 0.0f) {
										hit_n = -hit_n;
									}
									Vector3 hit_ro = hit_pos + hit_n * ray_bias;
									Vector3 hit_irrad = bake_env.ambient_irradiance;

									for (size_t li = 0; li < bake_lights.size(); li++) {
										const BakeLight &bl = bake_lights[li];
										Vector3 l_dir, l_rad;
										float l_max_dist = 0.0f;
										if (!bl.evaluate_at(hit_pos, l_dir, l_rad, l_max_dist)) {
											continue;
										}
										float ndotl = hit_n.dot(l_dir);
										if (ndotl <= 0.0f) {
											continue;
										}
										if (ao_luminance(l_rad) * ndotl <= 1e-4f) {
											continue;
										}
										if (bl.has_shadow && l_max_dist > ray_bias + 0.002f) {
											if (raycaster.is_occluded(hit_ro, l_dir, 0.001f, l_max_dist - ray_bias)) {
												continue;
											}
										}
										hit_irrad += l_rad * ndotl;
									}

									gi_rad = Vector3(
											ht.emission.x + ht.albedo.x * hit_irrad.x * inv_pi,
											ht.emission.y + ht.albedo.y * hit_irrad.y * inv_pi,
											ht.emission.z + ht.albedo.z * hit_irrad.z * inv_pi);
								} else {
									gi_rad = ht.emission;
								}
							} else {
								gi_rad = bake_env.eval_sky(hs.dir);
							}

							float g_lum = ao_luminance(gi_rad) * hs.vis;
							gi_l0 += g_lum;
							gi_l1 += hs.dir * g_lum;
						}

						float gi_0 = gi_l0 / (float)samples;
						Vector3 gi_1 = gi_l1 / (float)samples;
						float amb_lum = MAX(ao_luminance(bake_env.ambient_irradiance), 0.05f);
						float gi_strength = CLAMP(bake_gi_direction_strength, 0.0f, 1.0f);

						float lifted_vis_sum = 0.0f;
						Vector3 lifted_dir_sum(0.0f, 0.0f, 0.0f);
						Vector3 dir_sum(0.0f, 0.0f, 0.0f);

						for (int s = 0; s < samples; s++) {
							const Vector3 &n_dir = hit_samples[s].dir;
							float geom_ao = CLAMP(mean_vis + 2.0f * b_geom.dot(n_dir), 0.0f, 1.0f);

							float direct_lum_n = 0.0f;
							for (size_t li = 0; li < voxel_lights.size(); li++) {
								float ndotl = n_dir.dot(voxel_lights[li].dir);
								if (ndotl > 0.0f) {
									direct_lum_n += voxel_lights[li].lum * ndotl * inv_pi;
								}
							}

							float gi_lum_n = MAX(0.0f, gi_0 + 2.0f * gi_1.dot(n_dir));
							float total_rad_lum = direct_lum_n + gi_lum_n * 0.5f;
							float ratio = total_rad_lum / MAX(amb_lum + total_rad_lum, 1e-4f);
							float lift = CLAMP(ratio * (gi_strength * 1.5f), 0.0f, 1.0f);
							float eff_ao = Math::lerp(geom_ao, 1.0f, lift);

							lifted_vis_sum += eff_ao;
							lifted_dir_sum += n_dir * eff_ao;
							dir_sum += n_dir;
						}

						mean_vis = CLAMP(lifted_vis_sum / (float)samples, 0.0f, 1.0f);
						l1_bent = ((lifted_dir_sum - dir_sum * mean_vis) / (float)samples) * 1.5f;
						float len_bent = l1_bent.length();
						if (len_bent > 0.5f && len_bent > 1e-6f) {
							l1_bent *= (0.5f / len_bent);
						}
					}
				}

				// Store L1 Spherical Harmonics visibility: RGB = mean(vis * dir), A = mean(vis)
				dst_row[x] = Color(l1_bent.x, l1_bent.y, l1_bent.z, CLAMP(mean_vis, 0.0f, 1.0f));
			}
		}
	};

	int thread_count = MAX(1, OS::get_singleton()->get_processor_count());
	thread_count = MIN(thread_count, total_rows);
	if (thread_count > 1) {
		std::vector<std::thread> workers;
		workers.reserve((size_t)(thread_count - 1));
		for (int t = 0; t < thread_count - 1; t++) {
			workers.emplace_back(worker_fn);
		}
		worker_fn();
		for (size_t t = 0; t < workers.size(); t++) {
			workers[t].join();
		}
	} else {
		worker_fn();
	}

	// Separable 3D edge-aware bilateral denoise pass (smooths AO & GI direction without bleeding across walls)
	if (bake_denoise_passes > 0) {
		std::vector<Color> denoise_tmp(voxels.size());
		auto filter_axis = [&](const std::vector<Color> &p_src, std::vector<Color> &r_dst, int p_axis) {
			const int stride_y = rx;
			const int stride_z = rx * ry;
			const int stride = (p_axis == 0) ? 1 : ((p_axis == 1) ? stride_y : stride_z);
			const int limit = (p_axis == 0) ? rx : ((p_axis == 1) ? ry : rz);

			for (int z = 0; z < rz; z++) {
				for (int y = 0; y < ry; y++) {
					for (int x = 0; x < rx; x++) {
						int coord = (p_axis == 0) ? x : ((p_axis == 1) ? y : z);
						int idx = z * stride_z + y * stride_y + x;
						const Color &c = p_src[idx];

						float sum_r = c.r * 0.5f;
						float sum_g = c.g * 0.5f;
						float sum_b = c.b * 0.5f;
						float sum_a = c.a * 0.5f;
						float w_sum = 0.5f;

						if (coord > 0) {
							const Color &n = p_src[idx - stride];
							float da = n.a - c.a;
							float w = 0.25f * std::exp(-da * da * 20.0f);
							sum_r += n.r * w;
							sum_g += n.g * w;
							sum_b += n.b * w;
							sum_a += n.a * w;
							w_sum += w;
						}
						if (coord + 1 < limit) {
							const Color &n = p_src[idx + stride];
							float da = n.a - c.a;
							float w = 0.25f * std::exp(-da * da * 20.0f);
							sum_r += n.r * w;
							sum_g += n.g * w;
							sum_b += n.b * w;
							sum_a += n.a * w;
							w_sum += w;
						}

						float inv_w = 1.0f / w_sum;
						float vis = CLAMP(sum_a * inv_w, 0.0f, 1.0f);
						Vector3 bent(sum_r * inv_w, sum_g * inv_w, sum_b * inv_w);
						float max_b = 0.5f;
						float len_b = bent.length();
						if (len_b > max_b && len_b > 1e-6f) {
							bent *= (max_b / len_b);
						}
						r_dst[idx] = Color(bent.x, bent.y, bent.z, vis);
					}
				}
			}
		};

		for (int p = 0; p < bake_denoise_passes; p++) {
			filter_axis(voxels, denoise_tmp, 0);
			filter_axis(denoise_tmp, voxels, 1);
			filter_axis(voxels, denoise_tmp, 2);
			voxels.swap(denoise_tmp);
		}
	}

	bool has_direct_vols = false;
	for (size_t i = 0; i < baked_local_vols.size(); i++) {
		if (baked_local_vols[i].mode == LocalRTAO::MODE_DIRECT) {
			has_direct_vols = true;
			break;
		}
	}

	Ref<Texture3D> tex;
	tex.instance();
	tex->create(rx, ry, rz, Image::FORMAT_RGBAH, bake_filter ? Texture::FLAG_FILTER : 0);

	int slice_bytes = rx * ry * 8;
	PoolByteArray raw_bytes;
	raw_bytes.resize(slice_bytes * rz);
	PoolByteArray::Write raw_w = raw_bytes.write();

	for (int z = 0; z < rz; z++) {
		Ref<Image> slice;
		slice.instance();
		slice->create(rx, ry, false, Image::FORMAT_RGBAH);
		slice->lock();

		float uz = ((float)z + 0.5f) / (float)rz - 0.5f;
		const Color *slice_src = &voxels[(size_t)z * (size_t)ry * (size_t)rx];
		for (int y = 0; y < ry; y++) {
			float uy = ((float)y + 0.5f) / (float)ry - 0.5f;
			const Color *row_src = slice_src + (size_t)y * (size_t)rx;
			for (int x = 0; x < rx; x++) {
				Color out_c = row_src[x];
				if (has_direct_vols) {
					float ux = ((float)x + 0.5f) / (float)rx - 0.5f;
					Vector3 world_pos = global_xf.xform(Vector3(ux * 2.0f * bake_extents.x, uy * 2.0f * bake_extents.y, uz * 2.0f * bake_extents.z));
					float mean_vis = out_c.a;
					Vector3 l1_bent(out_c.r, out_c.g, out_c.b);

					for (size_t i = 0; i < baked_local_vols.size(); i++) {
						const BakeLocalVolume &lv = baked_local_vols[i];
						if (lv.mode != LocalRTAO::MODE_DIRECT) {
							continue;
						}
						float factor = lv.sample_shape_factor(world_pos);
						if (factor <= 0.0001f) {
							continue;
						}
						float mult = 1.0f - lv.occlusion * factor;
						switch (lv.blend_mode) {
							case LocalRTAO::BLEND_MULTIPLY:
								mean_vis *= mult;
								l1_bent *= mult;
								break;
							case LocalRTAO::BLEND_MIN:
								if (mult < mean_vis) {
									float ratio = mult / MAX(mean_vis, 1e-4f);
									mean_vis = mult;
									l1_bent *= ratio;
								}
								break;
							case LocalRTAO::BLEND_MAX:
								if (mult > mean_vis) {
									mean_vis = mult;
								}
								break;
							case LocalRTAO::BLEND_OVERRIDE:
								mean_vis = Math::lerp(mean_vis, 1.0f - lv.occlusion, factor);
								l1_bent *= (1.0f - factor);
								break;
						}
					}
					out_c = Color(l1_bent.x, l1_bent.y, l1_bent.z, CLAMP(mean_vis, 0.0f, 1.0f));
				}
				slice->set_pixel(x, y, out_c);
			}
		}

		slice->unlock();
		PoolByteArray slice_data = slice->get_data();
		if (slice_data.size() == slice_bytes) {
			PoolByteArray::Read sr = slice_data.read();
			memcpy(raw_w.ptr() + z * slice_bytes, sr.ptr(), slice_bytes);
		}
		tex->set_layer_data(slice, z);
	}

	raw_w.release();
	baked_data_resolution = Vector3((float)rx, (float)ry, (float)rz);
	baked_data = raw_bytes;
	baked_texture = tex;
	_change_notify();
	return true;
}
