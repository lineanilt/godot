#ifndef VOLUMETRICS_SHADERS_H
#define VOLUMETRICS_SHADERS_H

namespace VolumetricsShaders {

// Raw OpenGL 4.3 Depth-Only Rasterization Shaders (for Camera & Light Depth Maps)
static const char *DEPTH_VERT_GLSL = R"GLSL(
#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aUv;

uniform mat4 uMVP;
out vec2 vUv;

void main() {
	vUv = aUv;
	gl_Position = uMVP * vec4(aPos, 1.0);
}
)GLSL";

static const char *DEPTH_FRAG_GLSL = R"GLSL(
#version 430 core
in vec2 vUv;

uniform sampler2D uAlbedoTex;
uniform int uUseAlphaScissor;
uniform float uAlphaThreshold;

void main() {
	if (uUseAlphaScissor != 0) {
		float a = texture(uAlbedoTex, vUv).a;
		if (a < uAlphaThreshold) {
			discard;
		}
	}
}
)GLSL";

// Raw OpenGL 4.3 Raymarching Compute Shader
static const char *RAYMARCH_COMPUTE_GLSL_FULL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutFog;
layout (r32f, binding = 1) writeonly uniform image2D uOutLowDepth;

uniform sampler2D uCamDepthTex;
uniform sampler2DArray uDirDepthArray;
uniform sampler2DArray uSpotDepthArray;
uniform sampler2D uDirProj0;
uniform sampler3D uBakedTex0;
uniform sampler3D uBakedTex1;
uniform sampler3D uBakedTex2;
uniform sampler3D uBakedTex3;

uniform mat4 uProj;
uniform mat4 uInvProj;
uniform mat4 uCamMatrix;

uniform int uSteps;
uniform float uMaxDistance;
uniform float uDistanceFade;
uniform int uStepDistribution;
uniform float uJitter;
uniform float uTemporalJitter;
uniform float uSkyAffect;

uniform float uGlobalDensity;
uniform vec3 uGlobalAlbedo;
uniform vec3 uGlobalEmission;
uniform vec3 uAmbientColor;
uniform float uAnisotropy;
uniform float uLightFogRatio;

uniform int uHeightFogEnabled;
uniform float uHeightFogBase;
uniform float uHeightFogFalloff;

uniform int uNoiseEnabled;
uniform float uNoiseScale;
uniform float uNoiseStrength;
uniform vec3 uNoiseOffset;

uniform int uFocusActive;
uniform vec3 uFocusPos;
uniform float uFocusRadius;
uniform float uFocusFalloff;
uniform float uFocusDensityMult;

uniform int uDirCount;
uniform int uDir0HasProj;
uniform int uDir0ProjToLinear;
uniform int uDirRes;

uniform int uSpotCount;
uniform float uSpotNearPlane;
uniform int uSpotRes;

uniform int uOmniCount;
uniform int uOmniSSShadows;
uniform int uOmniSSSteps;
uniform float uOmniSSThickness;
uniform float uOmniSSSoftness;
uniform float uOmniSSStrength;

uniform int uLocalCount;
uniform int uBakedCount;

struct DirLight {
	mat4 vp;
	vec4 dirBias;
	vec4 colorFlags;
	vec4 projColor;
	vec4 projUV;
	vec4 projExtra;
};

struct SpotLight {
	mat4 vp;
	vec4 posRange;
	vec4 dirAtten;
	vec4 colorCone;
	vec4 params;
};

struct OmniLight {
	vec4 posRange;
	vec4 viewAtten;
	vec4 colorFlags;
};

struct LocalVolume {
	vec4 invRow0;
	vec4 invRow1;
	vec4 invRow2;
	vec4 albedoDensity;
	vec4 emissionFalloff;
	vec4 params;
	vec4 noiseOffset;
};

struct BakedVolume {
	vec4 invRow0;
	vec4 invRow1;
	vec4 invRow2;
	vec4 tintDens;
	vec4 params;
};

layout (std430, binding = 0) readonly buffer DirLightBuf { DirLight dirLights[]; };
layout (std430, binding = 1) readonly buffer SpotLightBuf { SpotLight spotLights[]; };
layout (std430, binding = 2) readonly buffer OmniLightBuf { OmniLight omniLights[]; };
layout (std430, binding = 3) readonly buffer LocalVolBuf { LocalVolume localVols[]; };
layout (std430, binding = 4) readonly buffer BakedVolBuf { BakedVolume bakedVols[]; };

float ignNoise(vec2 p) {
	return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
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

float hgPhase(float cosTheta, float g) {
	float g2 = g * g;
	float denom = max(1.0 + g2 - 2.0 * g * cosTheta, 0.002);
	float p = (1.0 - g2) / (12.56637061436 * pow(denom, 1.5));
	return min(p, 16.0);
}

vec2 xformProjUV(vec2 uv, vec2 scale, vec2 offset, float rot) {
	vec2 p = uv - 0.5;
	float c = cos(rot);
	float s = sin(rot);
	p = vec2(c * p.x - s * p.y, s * p.x + c * p.y);
	return p * scale + offset + 0.5;
}

float sampleDirShadowPCF(int layer, vec3 uvd, float bias) {
	float invRes = 1.0 / float(max(uDirRes, 1));
	vec2 offsets[4] = vec2[4](
		vec2(-0.5, -0.5) * invRes,
		vec2( 0.5, -0.5) * invRes,
		vec2(-0.5,  0.5) * invRes,
		vec2( 0.5,  0.5) * invRes
	);
	float sum = 0.0;
	for (int k = 0; k < 4; k++) {
		float dm = textureLod(uDirDepthArray, vec3(uvd.xy + offsets[k], float(layer)), 0.0).r;
		sum += (uvd.z - bias <= dm) ? 1.0 : 0.0;
	}
	return sum * 0.25;
}

float spotLinearZ(float d, float zn, float zf) {
	return (2.0 * zn * zf) / max(zf + zn - (d * 2.0 - 1.0) * (zf - zn), 0.0001);
}

float sampleSpotShadowPCF(int layer, vec2 uv, float sampleZ, float bias, float zn, float range) {
	float invRes = 1.0 / float(max(uSpotRes, 1));
	vec2 offsets[4] = vec2[4](
		vec2(-0.5, -0.5) * invRes,
		vec2( 0.5, -0.5) * invRes,
		vec2(-0.5,  0.5) * invRes,
		vec2( 0.5,  0.5) * invRes
	);
	float sum = 0.0;
	for (int k = 0; k < 4; k++) {
		float dm = textureLod(uSpotDepthArray, vec3(uv + offsets[k], float(layer)), 0.0).r;
		float mapZ = spotLinearZ(dm, zn, range);
		sum += (sampleZ - bias <= mapZ) ? 1.0 : 0.0;
	}
	return sum * 0.25;
}

vec4 sampleBakedVolume(int idx, vec3 uvw) {
	if (idx == 0) return textureLod(uBakedTex0, uvw, 0.0);
	if (idx == 1) return textureLod(uBakedTex1, uvw, 0.0);
	if (idx == 2) return textureLod(uBakedTex2, uvw, 0.0);
	return textureLod(uBakedTex3, uvw, 0.0);
}

void main() {
	ivec2 px = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(uOutFog);
	if (px.x >= sz.x || px.y >= sz.y) {
		return;
	}

	vec2 uv = (vec2(px) + 0.5) / vec2(sz);
	float rawDepth = texelFetch(uCamDepthTex, px, 0).r;

	vec4 nearVP = uInvProj * vec4(uv * 2.0 - 1.0, -1.0, 1.0);
	nearVP.xyz /= nearVP.w;

	float clampedDepth = min(rawDepth, 0.99999);
	vec4 farVP = uInvProj * vec4(uv * 2.0 - 1.0, clampedDepth * 2.0 - 1.0, 1.0);
	farVP.xyz /= farVP.w;

	float linearZ = max(-farVP.z, 0.001);
	imageStore(uOutLowDepth, px, vec4(linearZ, 0.0, 0.0, 0.0));

	vec3 roView = nearVP.xyz;
	vec3 diffView = farVP.xyz - nearVP.xyz;
	float sceneDist = length(diffView);
	vec3 rdView = diffView / max(sceneDist, 0.0001);

	bool isSky = (rawDepth >= 0.99999);
	float marchDist = isSky ? uMaxDistance : min(sceneDist, uMaxDistance);
	if (marchDist <= 0.001) {
		imageStore(uOutFog, px, vec4(0.0, 0.0, 0.0, 1.0));
		return;
	}

	vec3 roWorld = (uCamMatrix * vec4(roView, 1.0)).xyz;
	vec3 rdWorld = normalize((uCamMatrix * vec4(rdView, 0.0)).xyz);

	float jRaw = fract(ignNoise(vec2(px)) + uTemporalJitter);
	float j = jRaw * uJitter;
	float sampleFrac = mix(0.5, jRaw, uJitter);
	vec3 accumLight = vec3(0.0);
	float transmittance = 1.0;

	int steps = max(uSteps, 1);
	float invSteps = 1.0 / float(steps);

	for (int i = 0; i < 128; i++) {
		if (i >= steps) {
			break;
		}

		float u0 = float(i) * invSteps;
		float u1 = float(i + 1) * invSteps;

		float t0 = (uStepDistribution == 1) ? (u0 * u0) : u0;
		float t1 = (uStepDistribution == 1) ? (u1 * u1) : u1;
		float d0 = t0 * marchDist;
		float d1 = t1 * marchDist;
		float dt = d1 - d0;
		if (dt <= 0.0001) {
			continue;
		}
		float dMid = mix(d0, d1, sampleFrac);

		vec3 pWorld = roWorld + rdWorld * dMid;
		vec3 pView = roView + rdView * dMid;

		// 1. Global Density & Height Fog
		float localDensity = uGlobalDensity;
		if (uHeightFogEnabled != 0 && localDensity > 0.0) {
			float hDiff = max(0.0, pWorld.y - uHeightFogBase);
			localDensity *= exp(-uHeightFogFalloff * hDiff);
		}
		if (uNoiseEnabled != 0 && uNoiseStrength > 0.001 && localDensity > 0.0001) {
			float n = fbm3D(pWorld * uNoiseScale + uNoiseOffset);
			localDensity *= mix(1.0, n * 1.6, uNoiseStrength);
		}

		if (uFocusActive != 0) {
			float fd = length(pWorld - uFocusPos) / max(uFocusRadius, 0.001);
			if (fd < 1.0) {
				float fEdge = max(0.0, 1.0 - uFocusFalloff);
				float fw = (uFocusFalloff > 0.001) ? smoothstep(1.0, fEdge, fd) : 1.0;
				localDensity *= mix(1.0, uFocusDensityMult, fw);
			}
		}

		vec3 localAlbedo = uGlobalAlbedo;
		vec3 localEmission = uGlobalEmission;

		// 2. LocalFogVolumes
		for (int v = 0; v < uLocalCount; v++) {
			LocalVolume lv = localVols[v];
			vec4 wp4 = vec4(pWorld, 1.0);
			vec3 pl = vec3(
				dot(lv.invRow0, wp4),
				dot(lv.invRow1, wp4),
				dot(lv.invRow2, wp4)
			);

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

			if (d < 1.0) {
				float falloff = lv.emissionFalloff.w;
				float fade = (falloff > 0.001) ? smoothstep(1.0, max(0.0, 1.0 - falloff), d) : 1.0;
				float vd = lv.albedoDensity.w * fade;
				if (lv.params.z > 0.001) {
					float vn = fbm3D(pWorld * lv.params.w + lv.noiseOffset.xyz);
					vd *= mix(1.0, vn * 1.6, lv.params.z);
				}

				int bmode = int(lv.params.y + 0.5);
				vec3 valb = lv.albedoDensity.rgb;
				vec3 vemi = lv.emissionFalloff.rgb * fade;

				if (bmode == 0) {
					localDensity += vd;
					localAlbedo = mix(localAlbedo, valb, clamp(vd / max(localDensity, 0.0001), 0.0, 1.0));
					localEmission += vemi;
				} else if (bmode == 1) {
					if (vd > localDensity) {
						localDensity = vd;
						localAlbedo = valb;
					}
					localEmission = max(localEmission, vemi);
				} else if (bmode == 2) {
					localDensity *= mix(1.0, lv.albedoDensity.w, fade);
				} else {
					localDensity = max(0.0, localDensity - vd);
				}
			}
		}

		// 3. BakedFogVolumes
		vec3 bakedRadiance = vec3(0.0);
		float bakedDensity = 0.0;
		for (int b = 0; b < uBakedCount; b++) {
			BakedVolume bv = bakedVols[b];
			vec4 wp4 = vec4(pWorld, 1.0);
			vec3 uvw = vec3(
				dot(bv.invRow0, wp4),
				dot(bv.invRow1, wp4),
				dot(bv.invRow2, wp4)
			);

			if (uvw.x > 0.0 && uvw.x < 1.0 && uvw.y > 0.0 && uvw.y < 1.0 && uvw.z > 0.0 && uvw.z < 1.0) {
				vec3 distEdge = abs(uvw - 0.5) * 2.0;
				float maxE = max(distEdge.x, max(distEdge.y, distEdge.z));
				float bFalloff = bv.params.y;
				float bFade = (bFalloff > 0.001) ? smoothstep(1.0, max(0.0, 1.0 - bFalloff), maxE) : 1.0;

				vec4 bSample = sampleBakedVolume(b, uvw);
				float bd = max(0.0, bSample.a) * bv.tintDens.w * bFade;
				bakedDensity += bd;
				bakedRadiance += max(bSample.rgb, vec3(0.0)) * bv.tintDens.rgb * bv.params.x * bd;
			}
		}

		float totalDensity = localDensity + bakedDensity;

		float distFade = 1.0;
		if (uDistanceFade > 0.001) {
			float fadeStart = uMaxDistance * (1.0 - clamp(uDistanceFade, 0.0, 0.99));
			distFade = 1.0 - smoothstep(fadeStart, uMaxDistance, dMid);
		}
		totalDensity *= distFade;

		if (totalDensity <= 0.00001 && dot(localEmission, vec3(1.0)) <= 0.00001) {
			continue;
		}

		float ratioNorm = clamp(uLightFogRatio * 0.1, -1.0, 1.0);
		float lightFactor = 1.0 + ratioNorm;
		float fogFactor = 1.0 - ratioNorm;
		vec3 directInscatter = vec3(0.0);

		if (localDensity > 0.00001) {
			// 4. Directional Lights
			for (int d = 0; d < uDirCount; d++) {
				DirLight dl = dirLights[d];
				vec3 lDir = normalize(dl.dirBias.xyz);
				vec3 lCol = dl.colorFlags.rgb;
				float vis = 1.0;

				vec4 lp = dl.vp * vec4(pWorld, 1.0);
				vec3 uvd = (lp.xyz / lp.w) * 0.5 + 0.5;
				bool inBox = (uvd.x >= 0.0 && uvd.x <= 1.0 && uvd.y >= 0.0 && uvd.y <= 1.0 && uvd.z >= 0.0 && uvd.z <= 1.0);

				if (dl.colorFlags.w > 0.5 && inBox) {
					float shad = sampleDirShadowPCF(d, uvd, dl.dirBias.w);
					float edgeD = max(abs(uvd.x - 0.5), max(abs(uvd.y - 0.5), abs(uvd.z - 0.5))) * 2.0;
					float edgeBlend = smoothstep(0.85, 1.0, edgeD);
					vis = mix(shad, 1.0, edgeBlend);
				}

				if (d == 0 && uDir0HasProj != 0) {
					vec2 puv = xformProjUV(uvd.xy, dl.projUV.xy, dl.projUV.zw, dl.projExtra.x);
					if (dl.projExtra.y > 0.5) {
						puv = fract(puv);
					}
					if (dl.projExtra.y > 0.5 || (puv.x >= 0.0 && puv.x <= 1.0 && puv.y >= 0.0 && puv.y <= 1.0)) {
						vec4 gobo = textureLod(uDirProj0, puv, dl.projExtra.z);
						if (uDir0ProjToLinear != 0) {
							gobo.rgb = pow(max(gobo.rgb, vec3(0.0)), vec3(2.2));
						}
						lCol = mix(dl.projColor.rgb, lCol * gobo.rgb, gobo.a);
					} else {
						lCol = dl.projColor.rgb;
					}
				}

				float phase = hgPhase(dot(rdWorld, lDir), uAnisotropy);
				directInscatter += lCol * vis * phase;
			}

			// 5. Spot Lights
			for (int s = 0; s < uSpotCount; s++) {
				SpotLight sl = spotLights[s];
				vec3 toLight = sl.posRange.xyz - pWorld;
				float range = sl.posRange.w;
				float dist = length(toLight);
				if (dist >= range || dist <= 0.0001) {
					continue;
				}

				vec3 lDir = toLight / dist;
				vec3 spotFwd = normalize(sl.dirAtten.xyz);
				float spotCutoff = sl.colorCone.w;
				float spotAttenP = sl.params.x;

				vec4 lp = sl.vp * vec4(pWorld, 1.0);
				if (lp.w <= 0.001) {
					continue;
				}
				vec3 ndcL = lp.xyz / lp.w;
				vec3 uvd = ndcL * 0.5 + 0.5;
				if (uvd.x < 0.0 || uvd.x > 1.0 || uvd.y < 0.0 || uvd.y > 1.0 || uvd.z < 0.0 || uvd.z > 1.0) {
					continue;
				}

				float rim = 1.0;
				if (sl.params.z > 0.5) {
					vec2 suv = ndcL.xy;
					if (max(abs(suv.x), abs(suv.y)) > 1.0) {
						continue;
					}
					vec2 suv2 = suv * suv;
					rim = max(0.0001, suv2.x + suv2.y - suv2.x * suv2.y);
				} else {
					float scos = dot(-lDir, spotFwd);
					if (scos <= spotCutoff) {
						continue;
					}
					rim = max(0.0001, (1.0 - scos) / max(1.0 - spotCutoff, 0.0001));
				}

				float distAtten = pow(max(1.0 - dist / range, 0.0), max(sl.dirAtten.w, 0.01));
				float coneAtten = max(0.0, 1.0 - pow(rim, max(spotAttenP, 0.001)));
				float atten = distAtten * coneAtten;
				if (atten <= 0.0001) {
					continue;
				}

				float vis = 1.0;
				if (sl.params.w > 0.5) {
					float zn = min(uSpotNearPlane, range * 0.25);
					float sampleZ = dot(pWorld - sl.posRange.xyz, spotFwd);
					float linBias = sl.params.y * (1.0 + sampleZ * 0.08);
					vis = sampleSpotShadowPCF(s, uvd.xy, sampleZ, linBias, zn, range);
				}

				float phase = hgPhase(dot(rdWorld, lDir), uAnisotropy);
				directInscatter += sl.colorCone.rgb * atten * vis * phase;
			}

			// 6. Screen-Space Omni Lights (sub-integrated across [d0, d1] for low-step stability)
			for (int o = 0; o < uOmniCount; o++) {
				OmniLight ol = omniLights[o];
				float range = ol.posRange.w;
				float flags = ol.colorFlags.w;
				bool isSquare = (mod(flags, 2.0) > 0.5);
				bool hasShadow = (flags >= 1.5);
				float decay = max(ol.viewAtten.w, 0.01);

				// Fast ray-sphere / ray-box bound check over [d0, d1]
				vec3 segOrigin = roWorld + rdWorld * d0;
				float tProj = clamp(dot(ol.posRange.xyz - segOrigin, rdWorld), 0.0, dt);
				vec3 closestPt = segOrigin + rdWorld * tProj;
				float boundR = isSquare ? (range * 1.73205) : range;
				if (length(ol.posRange.xyz - closestPt) >= boundR) {
					continue;
				}

				// Sub-integrate attenuation * phase over 4 points in [d0, d1]
				float scatterAccum = 0.0;
				for (int sub = 0; sub < 4; sub++) {
					float dSub = mix(d0, d1, (float(sub) + sampleFrac) * 0.25);
					vec3 pSub = roWorld + rdWorld * dSub;
					vec3 toL = ol.posRange.xyz - pSub;
					float distSub = length(toL);
					float evalD = isSquare ? max(abs(toL.x), max(abs(toL.y), abs(toL.z))) : distSub;
					if (evalD < range && distSub > 0.0001) {
						float aSub = pow(max(1.0 - evalD / range, 0.0), decay);
						float phSub = hgPhase(dot(rdWorld, toL / distSub), uAnisotropy);
						scatterAccum += aSub * phSub;
					}
				}
				float scatterAvg = scatterAccum * 0.25;
				if (scatterAvg <= 0.0001) {
					continue;
				}

				float vis = 1.0;
				float lum = dot(ol.colorFlags.rgb, vec3(0.3333)) * scatterAvg;
				if (uOmniSSShadows != 0 && hasShadow && uOmniSSSteps > 0 && lum > 0.002) {
					vec3 pShadowView = roView + rdView * (d0 + tProj);
					vec3 lView = ol.viewAtten.xyz;
					vec3 endView = lView;
					if (endView.z > -0.05) {
						float denom = lView.z - pShadowView.z;
						if (abs(denom) > 0.0001) {
							float tClip = clamp((-0.05 - pShadowView.z) / denom, 0.0, 1.0);
							endView = mix(pShadowView, lView, tClip);
						} else {
							endView = pShadowView;
						}
					}

					float occ = 0.0;
					int ssSteps = min(uOmniSSSteps, 16);
					for (int k = 1; k <= 16; k++) {
						if (k > ssSteps) {
							break;
						}
						float st = (float(k) - 0.5 + j * 0.5) / float(ssSteps);
						vec3 sv = mix(pShadowView, endView, clamp(st, 0.02, 0.98));
						if (sv.z >= -0.05) {
							continue;
						}
						vec4 sc = uProj * vec4(sv, 1.0);
						if (sc.w <= 0.0001) {
							continue;
						}
						vec2 suv = (sc.xy / sc.w) * 0.5 + 0.5;
						if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) {
							continue;
						}

						float sd = textureLod(uCamDepthTex, suv, 0.0).r;
						if (sd >= 0.99999) {
							continue;
						}
						vec4 sp = uInvProj * vec4(suv * 2.0 - 1.0, sd * 2.0 - 1.0, 1.0);
						float sceneZ = sp.z / sp.w;
						float depthDiff = sceneZ - sv.z;
						float effThickness = uOmniSSThickness * max(1.0, -sv.z * 0.15);
						if (depthDiff > 0.03 && depthDiff < effThickness) {
							float softW = smoothstep(0.03, 0.03 + uOmniSSSoftness, depthDiff) *
								(1.0 - smoothstep(max(0.03, effThickness - uOmniSSSoftness), effThickness, depthDiff));
							vec2 edge2 = min(suv, 1.0 - suv);
							float edgeFade = smoothstep(0.0, 0.05, min(edge2.x, edge2.y));
							occ = max(occ, softW * edgeFade);
							if (occ >= 0.98) {
								break;
							}
						}
					}
					vis = 1.0 - occ * uOmniSSStrength;
				}

				directInscatter += ol.colorFlags.rgb * scatterAvg * vis;
			}
		}

		vec3 lightInscatter = uAmbientColor * fogFactor + directInscatter * lightFactor;
		vec3 stepInscatter = (localAlbedo * lightInscatter + localEmission) * (localDensity * distFade) + bakedRadiance * (lightFactor * distFade);
		float sigmaT = max(totalDensity * fogFactor, 0.0);
		float stepTrans = exp(-sigmaT * dt);

		vec3 stepInt = (sigmaT * dt > 0.0001)
			? (stepInscatter - stepInscatter * stepTrans) / sigmaT
			: (stepInscatter * dt);

		accumLight += transmittance * stepInt;
		transmittance *= stepTrans;

		if (transmittance < 0.005) {
			transmittance = 0.0;
			break;
		}
	}

	if (isSky) {
		accumLight *= uSkyAffect;
		transmittance = mix(1.0, transmittance, uSkyAffect);
	}

	imageStore(uOutFog, px, vec4(accumLight, clamp(transmittance, 0.0, 1.0)));
}
)GLSL";

// Raw OpenGL 4.3 Temporal Reprojection Compute Shader
static const char *TEMPORAL_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutFog;
layout (r32f, binding = 1) writeonly uniform image2D uOutPrevDepth;

uniform sampler2D uCurrentFog;
uniform sampler2D uHistoryFog;
uniform sampler2D uCamDepthTex;
uniform sampler2D uCurLinearDepth;
uniform sampler2D uPrevLinearDepth;

uniform mat4 uInvViewProj;
uniform mat4 uPrevViewProj;
uniform mat4 uPrevView;
uniform int uHistoryValid;
uniform float uBlend;

void main() {
	ivec2 px = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(uOutFog);
	if (px.x >= sz.x || px.y >= sz.y) {
		return;
	}

	vec2 uv = (vec2(px) + 0.5) / vec2(sz);
	vec4 cur = texelFetch(uCurrentFog, px, 0);
	float curLinZ = texelFetch(uCurLinearDepth, px, 0).r;
	imageStore(uOutPrevDepth, px, vec4(curLinZ, 0.0, 0.0, 0.0));

	if (uHistoryValid == 0) {
		imageStore(uOutFog, px, cur);
		return;
	}

	float rawD = texelFetch(uCamDepthTex, px, 0).r;
	vec4 wp = uInvViewProj * vec4(uv * 2.0 - 1.0, min(rawD, 0.99999) * 2.0 - 1.0, 1.0);
	wp.xyz /= wp.w;

	vec4 prevClip = uPrevViewProj * vec4(wp.xyz, 1.0);
	if (prevClip.w <= 0.001) {
		imageStore(uOutFog, px, cur);
		return;
	}

	vec2 prevUv = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
	if (prevUv.x < 0.0 || prevUv.x > 1.0 || prevUv.y < 0.0 || prevUv.y > 1.0) {
		imageStore(uOutFog, px, cur);
		return;
	}

	vec4 prevVP = uPrevView * vec4(wp.xyz, 1.0);
	float expectedPrevZ = max(-prevVP.z, 0.001);
	float sampledPrevZ = textureLod(uPrevLinearDepth, prevUv, 0.0).r;
	if (abs(expectedPrevZ - sampledPrevZ) > max(0.25, expectedPrevZ * 0.12)) {
		imageStore(uOutFog, px, cur);
		return;
	}

	vec4 m1 = vec4(0.0);
	vec4 m2 = vec4(0.0);
	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			ivec2 np = clamp(px + ivec2(dx, dy), ivec2(0), sz - 1);
			vec4 s = texelFetch(uCurrentFog, np, 0);
			m1 += s;
			m2 += s * s;
		}
	}
	vec4 mean = m1 / 9.0;
	vec4 stddev = sqrt(max(m2 / 9.0 - mean * mean, vec4(0.0)));
	vec4 cMin = max(vec4(0.0), mean - stddev * 1.25 - vec4(0.004));
	vec4 cMax = mean + stddev * 1.25 + vec4(0.004);

	vec4 hist = textureLod(uHistoryFog, prevUv, 0.0);
	hist = clamp(hist, cMin, cMax);

	vec4 outCol = mix(hist, cur, clamp(uBlend, 0.02, 1.0));
	imageStore(uOutFog, px, outCol);
}
)GLSL";

// Raw OpenGL 4.3 Edge-Aware Spatial Denoise Compute Shader (A-Trous Bilateral)
static const char *DENOISE_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutFog;

uniform sampler2D uSrcFog;
uniform sampler2D uLinearDepth;
uniform int uStepSize;

void main() {
	ivec2 px = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(uOutFog);
	if (px.x >= sz.x || px.y >= sz.y) {
		return;
	}

	vec4 center = texelFetch(uSrcFog, px, 0);
	float zC = texelFetch(uLinearDepth, px, 0).r;
	float sigmaZ = max(0.05, zC * 0.025);

	const float kernel[3] = float[3](0.25, 0.5, 0.25);
	vec4 sum = center * 0.25;
	float wSum = 0.25;

	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			if (dx == 0 && dy == 0) {
				continue;
			}
			ivec2 np = clamp(px + ivec2(dx, dy) * uStepSize, ivec2(0), sz - 1);
			vec4 ns = texelFetch(uSrcFog, np, 0);
			float zN = texelFetch(uLinearDepth, np, 0).r;

			float kw = kernel[dx + 1] * kernel[dy + 1];
			float dz = (zC - zN) / sigmaZ;
			float wz = exp(-0.5 * dz * dz);
			float w = kw * wz;

			sum += ns * w;
			wSum += w;
		}
	}

	imageStore(uOutFog, px, sum / max(wSum, 1e-5));
}
)GLSL";

// Raw OpenGL 4.3 Finalize (Tonemap + Linear->sRGB + Premultiply Alpha) Compute Shader
static const char *FINALIZE_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutFinalFog;

uniform sampler2D uSrcFog;
uniform int uTonemapMode;
uniform float uExposure;
uniform float uWhite;
uniform int uHdr;
uniform int uKeep3DLinear;

vec3 tonemapFilmic(vec3 x) {
	const float A = 0.15;
	const float B = 0.50;
	const float C = 0.10;
	const float D = 0.20;
	const float E = 0.02;
	const float F = 0.30;
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 tonemapACES(vec3 color) {
	const float A = 2.51;
	const float B = 0.03;
	const float C = 2.43;
	const float D = 0.59;
	const float E = 0.14;
	return clamp((color * (A * color + B)) / (color * (C * color + D) + E), vec3(0.0), vec3(1.0));
}

vec3 applyTonemap(vec3 c) {
	c = max(c, vec3(0.0)) * uExposure;
	if (uHdr == 0) {
		return clamp(c, vec3(0.0), vec3(1.0));
	}
	if (uTonemapMode == 1) {
		float w2 = max(uWhite * uWhite, 0.001);
		return clamp((c * (1.0 + c / w2)) / (1.0 + c), vec3(0.0), vec3(1.0));
	} else if (uTonemapMode == 2) {
		vec3 num = tonemapFilmic(c);
		vec3 den = max(tonemapFilmic(vec3(uWhite)), vec3(0.001));
		return clamp(num / den, vec3(0.0), vec3(1.0));
	} else if (uTonemapMode == 3) {
		return tonemapACES(c);
	}
	return c;
}

vec3 linearToSrgb(vec3 c) {
	c = max(c, vec3(0.0));
	return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

void main() {
	ivec2 px = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(uOutFinalFog);
	if (px.x >= sz.x || px.y >= sz.y) {
		return;
	}

	vec4 s = texelFetch(uSrcFog, px, 0);
	vec3 inscatter = max(s.rgb, vec3(0.0));
	float transmittance = clamp(s.a, 0.0, 1.0);

	vec3 outFog = applyTonemap(inscatter);
	imageStore(uOutFinalFog, px, vec4(outFog, transmittance));
}
)GLSL";

// Minimal 2D CanvasItem Composite Shader (NO spatial shader -> zero conflict with user 3D spatial shaders!)
static const char *COMPOSITE_CANVAS_SHADER = R"SHADER(
shader_type canvas_item;
render_mode unshaded, blend_mix;

uniform sampler2D u_fog_tex;
uniform sampler2D u_low_depth_tex;
uniform float u_bilateral_upsample = 1.0;
uniform float u_keep_3d_linear = 0.0;
uniform vec4 u_inv_proj_0 = vec4(1.0, 0.0, 0.0, 0.0);
uniform vec4 u_inv_proj_1 = vec4(0.0, 1.0, 0.0, 0.0);
uniform vec4 u_inv_proj_2 = vec4(0.0, 0.0, 1.0, 0.0);
uniform vec4 u_inv_proj_3 = vec4(0.0, 0.0, 0.0, 1.0);

float linearize_depth(float raw_d, vec2 uv) {
	mat4 inv_proj = mat4(u_inv_proj_0, u_inv_proj_1, u_inv_proj_2, u_inv_proj_3);
	vec4 vp = inv_proj * vec4(uv * 2.0 - 1.0, min(raw_d, 0.99999) * 2.0 - 1.0, 1.0);
	return max(-vp.z / vp.w, 0.001);
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
	vec4 fog;
	if (u_bilateral_upsample > 0.5) {
		float full_d = textureLod(DEPTH_TEXTURE, SCREEN_UV, 0.0).r;
		float full_z = linearize_depth(full_d, SCREEN_UV);

		vec2 low_size = vec2(textureSize(u_fog_tex, 0));
		vec2 inv_low = 1.0 / max(low_size, vec2(1.0));
		vec2 fcoord = SCREEN_UV * low_size - 0.5;
		vec2 base = floor(fcoord);
		vec2 f = clamp(fcoord - base, vec2(0.0), vec2(1.0));

		vec2 uv00 = clamp((base + vec2(0.5, 0.5)) * inv_low, vec2(0.0), vec2(1.0));
		vec2 uv10 = clamp((base + vec2(1.5, 0.5)) * inv_low, vec2(0.0), vec2(1.0));
		vec2 uv01 = clamp((base + vec2(0.5, 1.5)) * inv_low, vec2(0.0), vec2(1.0));
		vec2 uv11 = clamp((base + vec2(1.5, 1.5)) * inv_low, vec2(0.0), vec2(1.0));

		float z00 = textureLod(u_low_depth_tex, uv00, 0.0).r;
		float z10 = textureLod(u_low_depth_tex, uv10, 0.0).r;
		float z01 = textureLod(u_low_depth_tex, uv01, 0.0).r;
		float z11 = textureLod(u_low_depth_tex, uv11, 0.0).r;

		float sigma = max(0.05, full_z * 0.025);
		float d00 = (full_z - z00) / sigma;
		float d10 = (full_z - z10) / sigma;
		float d01 = (full_z - z01) / sigma;
		float d11 = (full_z - z11) / sigma;
		float w00 = ((1.0 - f.x) * (1.0 - f.y)) * exp(-0.5 * d00 * d00);
		float w10 = (f.x * (1.0 - f.y)) * exp(-0.5 * d10 * d10);
		float w01 = ((1.0 - f.x) * f.y) * exp(-0.5 * d01 * d01);
		float w11 = (f.x * f.y) * exp(-0.5 * d11 * d11);

		float w_sum = w00 + w10 + w01 + w11;
		vec4 s00 = textureLod(u_fog_tex, uv00, 0.0);
		vec4 s10 = textureLod(u_fog_tex, uv10, 0.0);
		vec4 s01 = textureLod(u_fog_tex, uv01, 0.0);
		vec4 s11 = textureLod(u_fog_tex, uv11, 0.0);

		fog = (w_sum > 0.000001)
			? (s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11) / w_sum
			: textureLod(u_fog_tex, SCREEN_UV, 0.0);
	} else {
		fog = textureLod(u_fog_tex, SCREEN_UV, 0.0);
	}

	vec4 scene = textureLod(SCREEN_TEXTURE, SCREEN_UV, 0.0);
	vec3 scene_lin = (u_keep_3d_linear > 0.5) ? scene.rgb : srgb_to_linear(scene.rgb);
	float transmittance = clamp(fog.a, 0.0, 1.0);
	vec3 combined_lin = scene_lin * transmittance + max(fog.rgb, vec3(0.0));
	vec3 out_rgb = (u_keep_3d_linear > 0.5) ? combined_lin : linear_to_srgb(combined_lin);

	COLOR = vec4(out_rgb, 1.0);
}
)SHADER";

} // namespace VolumetricsShaders

#endif // VOLUMETRICS_SHADERS_H
