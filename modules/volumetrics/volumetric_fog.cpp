#include "volumetric_fog.h"

#include "baked_fog_volume.h"
#include "local_fog_volume.h"
#include "volumetric_fog_focus.h"
#include "volumetrics_shaders.h"

#include "core/engine.h"
#include "core/math/math_funcs.h"
#include "drivers/gles3/rasterizer_storage_gles3.h"
#include "drivers/gles3/shared_gl43_context.h"
#include "modules/modules_enabled.gen.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/multimesh_instance.h"

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/resources/environment.h"
#include "scene/resources/world.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual_server.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <array>

#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT 0x00000020
#endif
#ifndef GL_TEXTURE_FETCH_BARRIER_BIT
#define GL_TEXTURE_FETCH_BARRIER_BIT 0x00000008
#endif
#ifndef GL_WRITE_ONLY
#define GL_WRITE_ONLY 0x88B9
#endif
#ifndef GL_FRAMEBUFFER_SRGB
#define GL_FRAMEBUFFER_SRGB 0x8DB9
#endif
#ifndef GL_DEPTH_CLAMP
#define GL_DEPTH_CLAMP 0x864F
#endif
#ifndef GL_RGBA16F
#define GL_RGBA16F 0x881A
#endif
#ifndef GL_R32F
#define GL_R32F 0x822E
#endif
#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif
#ifndef GL_TEXTURE_3D
#define GL_TEXTURE_3D 0x806F
#endif

namespace {

static const char *VOLUMETRICS_COMPOSITE_FRAG_GLSL = R"GLSL(
#version 430 core
in vec2 vUv;
layout (location = 0) out vec4 outColor;

uniform sampler2D uFogTex;
uniform sampler2D uLowDepthTex;
uniform sampler2D uDepthTex;
uniform int uBilateralUpsample;
uniform mat4 uInvProj;

float linearizeDepth(float rawD, vec2 uv) {
	vec4 vp = uInvProj * vec4(uv * 2.0 - 1.0, min(rawD, 0.99999) * 2.0 - 1.0, 1.0);
	return max(-vp.z / vp.w, 0.001);
}

void main() {
	vec4 fog;
	if (uBilateralUpsample != 0) {
		float fullD = textureLod(uDepthTex, vUv, 0.0).r;
		float fullZ = linearizeDepth(fullD, vUv);

		vec2 lowSize = vec2(textureSize(uFogTex, 0));
		vec2 invLow = 1.0 / max(lowSize, vec2(1.0));
		vec2 fcoord = vUv * lowSize - 0.5;
		vec2 base = floor(fcoord);
		vec2 f = clamp(fcoord - base, vec2(0.0), vec2(1.0));

		vec2 uv00 = clamp((base + vec2(0.5, 0.5)) * invLow, vec2(0.0), vec2(1.0));
		vec2 uv10 = clamp((base + vec2(1.5, 0.5)) * invLow, vec2(0.0), vec2(1.0));
		vec2 uv01 = clamp((base + vec2(0.5, 1.5)) * invLow, vec2(0.0), vec2(1.0));
		vec2 uv11 = clamp((base + vec2(1.5, 1.5)) * invLow, vec2(0.0), vec2(1.0));

		float z00 = textureLod(uLowDepthTex, uv00, 0.0).r;
		float z10 = textureLod(uLowDepthTex, uv10, 0.0).r;
		float z01 = textureLod(uLowDepthTex, uv01, 0.0).r;
		float z11 = textureLod(uLowDepthTex, uv11, 0.0).r;

		float sigma = max(0.05, fullZ * 0.025);
		float d00 = (fullZ - z00) / sigma;
		float d10 = (fullZ - z10) / sigma;
		float d01 = (fullZ - z01) / sigma;
		float d11 = (fullZ - z11) / sigma;
		float w00 = ((1.0 - f.x) * (1.0 - f.y)) * exp(-0.5 * d00 * d00);
		float w10 = (f.x * (1.0 - f.y)) * exp(-0.5 * d10 * d10);
		float w01 = ((1.0 - f.x) * f.y) * exp(-0.5 * d01 * d01);
		float w11 = (f.x * f.y) * exp(-0.5 * d11 * d11);

		float wSum = w00 + w10 + w01 + w11;
		vec4 s00 = textureLod(uFogTex, uv00, 0.0);
		vec4 s10 = textureLod(uFogTex, uv10, 0.0);
		vec4 s01 = textureLod(uFogTex, uv01, 0.0);
		vec4 s11 = textureLod(uFogTex, uv11, 0.0);

		fog = (wSum > 0.000001)
			? (s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11) / wSum
			: textureLod(uFogTex, vUv, 0.0);
	} else {
		fog = textureLod(uFogTex, vUv, 0.0);
	}

	outColor = vec4(max(fog.rgb, vec3(0.0)), clamp(fog.a, 0.0, 1.0));
}
)GLSL";

} // namespace

VolumetricFog::VolumetricFog() {
	fog_texture.instance();
	low_depth_texture.instance();
}

VolumetricFog::~VolumetricFog() {
	_unregister_scene_hook();
	_cleanup_gl_resources();
}

void VolumetricFog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &VolumetricFog::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &VolumetricFog::is_enabled);

	ClassDB::bind_method(D_METHOD("set_editor_preview", "preview"), &VolumetricFog::set_editor_preview);
	ClassDB::bind_method(D_METHOD("get_editor_preview"), &VolumetricFog::get_editor_preview);

	ClassDB::bind_method(D_METHOD("set_auto_composite", "auto_composite"), &VolumetricFog::set_auto_composite);
	ClassDB::bind_method(D_METHOD("get_auto_composite"), &VolumetricFog::get_auto_composite);

	ClassDB::bind_method(D_METHOD("set_density", "density"), &VolumetricFog::set_density);
	ClassDB::bind_method(D_METHOD("get_density"), &VolumetricFog::get_density);

	ClassDB::bind_method(D_METHOD("set_albedo", "albedo"), &VolumetricFog::set_albedo);
	ClassDB::bind_method(D_METHOD("get_albedo"), &VolumetricFog::get_albedo);

	ClassDB::bind_method(D_METHOD("set_emission", "emission"), &VolumetricFog::set_emission);
	ClassDB::bind_method(D_METHOD("get_emission"), &VolumetricFog::get_emission);

	ClassDB::bind_method(D_METHOD("set_emission_energy", "energy"), &VolumetricFog::set_emission_energy);
	ClassDB::bind_method(D_METHOD("get_emission_energy"), &VolumetricFog::get_emission_energy);

	ClassDB::bind_method(D_METHOD("set_anisotropy", "anisotropy"), &VolumetricFog::set_anisotropy);
	ClassDB::bind_method(D_METHOD("get_anisotropy"), &VolumetricFog::get_anisotropy);

	ClassDB::bind_method(D_METHOD("set_light_fog_ratio", "ratio"), &VolumetricFog::set_light_fog_ratio);
	ClassDB::bind_method(D_METHOD("get_light_fog_ratio"), &VolumetricFog::get_light_fog_ratio);

	ClassDB::bind_method(D_METHOD("set_use_environment_ambient", "use"), &VolumetricFog::set_use_environment_ambient);
	ClassDB::bind_method(D_METHOD("get_use_environment_ambient"), &VolumetricFog::get_use_environment_ambient);

	ClassDB::bind_method(D_METHOD("set_ambient_light_color", "color"), &VolumetricFog::set_ambient_light_color);
	ClassDB::bind_method(D_METHOD("get_ambient_light_color"), &VolumetricFog::get_ambient_light_color);

	ClassDB::bind_method(D_METHOD("set_ambient_light_energy", "energy"), &VolumetricFog::set_ambient_light_energy);
	ClassDB::bind_method(D_METHOD("get_ambient_light_energy"), &VolumetricFog::get_ambient_light_energy);

	ClassDB::bind_method(D_METHOD("set_sky_affect", "affect"), &VolumetricFog::set_sky_affect);
	ClassDB::bind_method(D_METHOD("get_sky_affect"), &VolumetricFog::get_sky_affect);

	ClassDB::bind_method(D_METHOD("set_height_fog_enabled", "enable"), &VolumetricFog::set_height_fog_enabled);
	ClassDB::bind_method(D_METHOD("is_height_fog_enabled"), &VolumetricFog::is_height_fog_enabled);

	ClassDB::bind_method(D_METHOD("set_height_fog_base_height", "height"), &VolumetricFog::set_height_fog_base_height);
	ClassDB::bind_method(D_METHOD("get_height_fog_base_height"), &VolumetricFog::get_height_fog_base_height);

	ClassDB::bind_method(D_METHOD("set_height_fog_falloff", "falloff"), &VolumetricFog::set_height_fog_falloff);
	ClassDB::bind_method(D_METHOD("get_height_fog_falloff"), &VolumetricFog::get_height_fog_falloff);

	ClassDB::bind_method(D_METHOD("set_noise_enabled", "enable"), &VolumetricFog::set_noise_enabled);
	ClassDB::bind_method(D_METHOD("is_noise_enabled"), &VolumetricFog::is_noise_enabled);

	ClassDB::bind_method(D_METHOD("set_noise_scale", "scale"), &VolumetricFog::set_noise_scale);
	ClassDB::bind_method(D_METHOD("get_noise_scale"), &VolumetricFog::get_noise_scale);

	ClassDB::bind_method(D_METHOD("set_noise_strength", "strength"), &VolumetricFog::set_noise_strength);
	ClassDB::bind_method(D_METHOD("get_noise_strength"), &VolumetricFog::get_noise_strength);

	ClassDB::bind_method(D_METHOD("set_noise_wind", "wind"), &VolumetricFog::set_noise_wind);
	ClassDB::bind_method(D_METHOD("get_noise_wind"), &VolumetricFog::get_noise_wind);

	ClassDB::bind_method(D_METHOD("set_steps", "steps"), &VolumetricFog::set_steps);
	ClassDB::bind_method(D_METHOD("get_steps"), &VolumetricFog::get_steps);

	ClassDB::bind_method(D_METHOD("set_editor_max_steps", "steps"), &VolumetricFog::set_editor_max_steps);
	ClassDB::bind_method(D_METHOD("get_editor_max_steps"), &VolumetricFog::get_editor_max_steps);

	ClassDB::bind_method(D_METHOD("set_max_distance", "distance"), &VolumetricFog::set_max_distance);
	ClassDB::bind_method(D_METHOD("get_max_distance"), &VolumetricFog::get_max_distance);

	ClassDB::bind_method(D_METHOD("set_distance_fade", "fade"), &VolumetricFog::set_distance_fade);
	ClassDB::bind_method(D_METHOD("get_distance_fade"), &VolumetricFog::get_distance_fade);

	ClassDB::bind_method(D_METHOD("set_step_distribution", "distribution"), &VolumetricFog::set_step_distribution);
	ClassDB::bind_method(D_METHOD("get_step_distribution"), &VolumetricFog::get_step_distribution);

	ClassDB::bind_method(D_METHOD("set_jitter", "jitter"), &VolumetricFog::set_jitter);
	ClassDB::bind_method(D_METHOD("get_jitter"), &VolumetricFog::get_jitter);

	ClassDB::bind_method(D_METHOD("set_resolution_scale", "scale"), &VolumetricFog::set_resolution_scale);
	ClassDB::bind_method(D_METHOD("get_resolution_scale"), &VolumetricFog::get_resolution_scale);

	ClassDB::bind_method(D_METHOD("set_temporal_reprojection", "enable"), &VolumetricFog::set_temporal_reprojection);
	ClassDB::bind_method(D_METHOD("get_temporal_reprojection"), &VolumetricFog::get_temporal_reprojection);

	ClassDB::bind_method(D_METHOD("set_temporal_blend", "blend"), &VolumetricFog::set_temporal_blend);
	ClassDB::bind_method(D_METHOD("get_temporal_blend"), &VolumetricFog::get_temporal_blend);

	ClassDB::bind_method(D_METHOD("set_spatial_denoise_passes", "passes"), &VolumetricFog::set_spatial_denoise_passes);
	ClassDB::bind_method(D_METHOD("get_spatial_denoise_passes"), &VolumetricFog::get_spatial_denoise_passes);

	ClassDB::bind_method(D_METHOD("set_bilateral_upsample", "enable"), &VolumetricFog::set_bilateral_upsample);
	ClassDB::bind_method(D_METHOD("get_bilateral_upsample"), &VolumetricFog::get_bilateral_upsample);

	ClassDB::bind_method(D_METHOD("set_directional_lights_enabled", "enable"), &VolumetricFog::set_directional_lights_enabled);
	ClassDB::bind_method(D_METHOD("are_directional_lights_enabled"), &VolumetricFog::are_directional_lights_enabled);

	ClassDB::bind_method(D_METHOD("set_directional_depth_resolution", "resolution"), &VolumetricFog::set_directional_depth_resolution);
	ClassDB::bind_method(D_METHOD("get_directional_depth_resolution"), &VolumetricFog::get_directional_depth_resolution);

	ClassDB::bind_method(D_METHOD("set_directional_shadow_distance", "distance"), &VolumetricFog::set_directional_shadow_distance);
	ClassDB::bind_method(D_METHOD("get_directional_shadow_distance"), &VolumetricFog::get_directional_shadow_distance);

	ClassDB::bind_method(D_METHOD("set_directional_depth_range", "range"), &VolumetricFog::set_directional_depth_range);
	ClassDB::bind_method(D_METHOD("get_directional_depth_range"), &VolumetricFog::get_directional_depth_range);

	ClassDB::bind_method(D_METHOD("set_directional_depth_bias", "bias"), &VolumetricFog::set_directional_depth_bias);
	ClassDB::bind_method(D_METHOD("get_directional_depth_bias"), &VolumetricFog::get_directional_depth_bias);

	ClassDB::bind_method(D_METHOD("set_directional_update_interval", "interval"), &VolumetricFog::set_directional_update_interval);
	ClassDB::bind_method(D_METHOD("get_directional_update_interval"), &VolumetricFog::get_directional_update_interval);

	ClassDB::bind_method(D_METHOD("set_directional_cull_mask_override", "mask"), &VolumetricFog::set_directional_cull_mask_override);
	ClassDB::bind_method(D_METHOD("get_directional_cull_mask_override"), &VolumetricFog::get_directional_cull_mask_override);

	ClassDB::bind_method(D_METHOD("set_spot_lights_enabled", "enable"), &VolumetricFog::set_spot_lights_enabled);
	ClassDB::bind_method(D_METHOD("are_spot_lights_enabled"), &VolumetricFog::are_spot_lights_enabled);

	ClassDB::bind_method(D_METHOD("set_max_spot_lights", "max"), &VolumetricFog::set_max_spot_lights);
	ClassDB::bind_method(D_METHOD("get_max_spot_lights"), &VolumetricFog::get_max_spot_lights);

	ClassDB::bind_method(D_METHOD("set_spot_depth_resolution", "resolution"), &VolumetricFog::set_spot_depth_resolution);
	ClassDB::bind_method(D_METHOD("get_spot_depth_resolution"), &VolumetricFog::get_spot_depth_resolution);

	ClassDB::bind_method(D_METHOD("set_spot_depth_bias", "bias"), &VolumetricFog::set_spot_depth_bias);
	ClassDB::bind_method(D_METHOD("get_spot_depth_bias"), &VolumetricFog::get_spot_depth_bias);

	ClassDB::bind_method(D_METHOD("set_spot_near_plane", "near_plane"), &VolumetricFog::set_spot_near_plane);
	ClassDB::bind_method(D_METHOD("get_spot_near_plane"), &VolumetricFog::get_spot_near_plane);

	ClassDB::bind_method(D_METHOD("set_spot_update_interval", "interval"), &VolumetricFog::set_spot_update_interval);
	ClassDB::bind_method(D_METHOD("get_spot_update_interval"), &VolumetricFog::get_spot_update_interval);

	ClassDB::bind_method(D_METHOD("set_spot_cull_mask_override", "mask"), &VolumetricFog::set_spot_cull_mask_override);
	ClassDB::bind_method(D_METHOD("get_spot_cull_mask_override"), &VolumetricFog::get_spot_cull_mask_override);

	ClassDB::bind_method(D_METHOD("set_omni_lights_enabled", "enable"), &VolumetricFog::set_omni_lights_enabled);
	ClassDB::bind_method(D_METHOD("are_omni_lights_enabled"), &VolumetricFog::are_omni_lights_enabled);

	ClassDB::bind_method(D_METHOD("set_max_omni_lights", "max"), &VolumetricFog::set_max_omni_lights);
	ClassDB::bind_method(D_METHOD("get_max_omni_lights"), &VolumetricFog::get_max_omni_lights);

	ClassDB::bind_method(D_METHOD("set_omni_screen_space_shadows", "enable"), &VolumetricFog::set_omni_screen_space_shadows);
	ClassDB::bind_method(D_METHOD("get_omni_screen_space_shadows"), &VolumetricFog::get_omni_screen_space_shadows);

	ClassDB::bind_method(D_METHOD("set_omni_ss_shadow_steps", "steps"), &VolumetricFog::set_omni_ss_shadow_steps);
	ClassDB::bind_method(D_METHOD("get_omni_ss_shadow_steps"), &VolumetricFog::get_omni_ss_shadow_steps);

	ClassDB::bind_method(D_METHOD("set_omni_ss_shadow_thickness", "thickness"), &VolumetricFog::set_omni_ss_shadow_thickness);
	ClassDB::bind_method(D_METHOD("get_omni_ss_shadow_thickness"), &VolumetricFog::get_omni_ss_shadow_thickness);

	ClassDB::bind_method(D_METHOD("set_omni_ss_shadow_softness", "softness"), &VolumetricFog::set_omni_ss_shadow_softness);
	ClassDB::bind_method(D_METHOD("get_omni_ss_shadow_softness"), &VolumetricFog::get_omni_ss_shadow_softness);

	ClassDB::bind_method(D_METHOD("set_omni_ss_shadow_strength", "strength"), &VolumetricFog::set_omni_ss_shadow_strength);
	ClassDB::bind_method(D_METHOD("get_omni_ss_shadow_strength"), &VolumetricFog::get_omni_ss_shadow_strength);

	ClassDB::bind_method(D_METHOD("get_fog_texture"), &VolumetricFog::get_fog_texture);
	ClassDB::bind_method(D_METHOD("_on_frame_pre_draw"), &VolumetricFog::_on_frame_pre_draw);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_preview"), "set_editor_preview", "get_editor_preview");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_composite"), "set_auto_composite", "get_auto_composite");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "density", PROPERTY_HINT_RANGE, "0,1,0.002,or_greater"), "set_density", "get_density");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "albedo", PROPERTY_HINT_COLOR_NO_ALPHA), "set_albedo", "get_albedo");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "emission", PROPERTY_HINT_COLOR_NO_ALPHA), "set_emission", "get_emission");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "emission_energy", PROPERTY_HINT_RANGE, "0,8,0.05,or_greater"), "set_emission_energy", "get_emission_energy");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "anisotropy", PROPERTY_HINT_RANGE, "-0.9,0.9,0.02"), "set_anisotropy", "get_anisotropy");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "light_fog_ratio", PROPERTY_HINT_RANGE, "-10,10,0.01"), "set_light_fog_ratio", "get_light_fog_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_environment_ambient"), "set_use_environment_ambient", "get_use_environment_ambient");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ambient_light_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_ambient_light_color", "get_ambient_light_color");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ambient_light_energy", PROPERTY_HINT_RANGE, "0,8,0.02,or_greater"), "set_ambient_light_energy", "get_ambient_light_energy");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "sky_affect", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_sky_affect", "get_sky_affect");

	ADD_GROUP("Height Fog", "height_fog_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "height_fog_enabled"), "set_height_fog_enabled", "is_height_fog_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height_fog_base_height"), "set_height_fog_base_height", "get_height_fog_base_height");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height_fog_falloff", PROPERTY_HINT_RANGE, "0,4,0.01,or_greater"), "set_height_fog_falloff", "get_height_fog_falloff");

	ADD_GROUP("Noise", "noise_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "noise_enabled"), "set_noise_enabled", "is_noise_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "noise_scale", PROPERTY_HINT_RANGE, "0.005,5,0.005,or_greater"), "set_noise_scale", "get_noise_scale");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "noise_strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_noise_strength", "get_noise_strength");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "noise_wind"), "set_noise_wind", "get_noise_wind");

	ADD_GROUP("Raymarching & Quality", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "steps", PROPERTY_HINT_RANGE, "4,128,1"), "set_steps", "get_steps");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "editor_max_steps", PROPERTY_HINT_RANGE, "4,64,1"), "set_editor_max_steps", "get_editor_max_steps");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_distance", PROPERTY_HINT_RANGE, "5,1000,1,or_greater"), "set_max_distance", "get_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "distance_fade", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_distance_fade", "get_distance_fade");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "step_distribution", PROPERTY_HINT_ENUM, "Linear,Quadratic"), "set_step_distribution", "get_step_distribution");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "jitter", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_jitter", "get_jitter");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "resolution_scale", PROPERTY_HINT_RANGE, "0.25,1.0,0.05"), "set_resolution_scale", "get_resolution_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "temporal_reprojection"), "set_temporal_reprojection", "get_temporal_reprojection");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "temporal_blend", PROPERTY_HINT_RANGE, "0.02,1.0,0.01"), "set_temporal_blend", "get_temporal_blend");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spatial_denoise_passes", PROPERTY_HINT_RANGE, "0,5,1"), "set_spatial_denoise_passes", "get_spatial_denoise_passes");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bilateral_upsample"), "set_bilateral_upsample", "get_bilateral_upsample");

	ADD_GROUP("Directional Lights (Depth Texture)", "directional_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "directional_lights_enabled"), "set_directional_lights_enabled", "are_directional_lights_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "directional_depth_resolution", PROPERTY_HINT_ENUM, "256:256,512:512,1024:1024,2048:2048"), "set_directional_depth_resolution", "get_directional_depth_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "directional_shadow_distance", PROPERTY_HINT_RANGE, "5,500,1,or_greater"), "set_directional_shadow_distance", "get_directional_shadow_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "directional_depth_range", PROPERTY_HINT_RANGE, "10,1000,5,or_greater"), "set_directional_depth_range", "get_directional_depth_range");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "directional_depth_bias", PROPERTY_HINT_RANGE, "0,0.05,0.0005"), "set_directional_depth_bias", "get_directional_depth_bias");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "directional_update_interval", PROPERTY_HINT_RANGE, "1,10,1"), "set_directional_update_interval", "get_directional_update_interval");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "directional_cull_mask_override", PROPERTY_HINT_LAYERS_3D_RENDER), "set_directional_cull_mask_override", "get_directional_cull_mask_override");

	ADD_GROUP("Spot Lights (Depth Texture)", "spot_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "spot_lights_enabled"), "set_spot_lights_enabled", "are_spot_lights_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_spot_lights", PROPERTY_HINT_RANGE, "0,6,1"), "set_max_spot_lights", "get_max_spot_lights");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spot_depth_resolution", PROPERTY_HINT_ENUM, "128:128,256:256,512:512,1024:1024"), "set_spot_depth_resolution", "get_spot_depth_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "spot_depth_bias", PROPERTY_HINT_RANGE, "0,2.0,0.01"), "set_spot_depth_bias", "get_spot_depth_bias");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "spot_near_plane", PROPERTY_HINT_RANGE, "0.01,1.0,0.01"), "set_spot_near_plane", "get_spot_near_plane");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spot_update_interval", PROPERTY_HINT_RANGE, "1,10,1"), "set_spot_update_interval", "get_spot_update_interval");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spot_cull_mask_override", PROPERTY_HINT_LAYERS_3D_RENDER), "set_spot_cull_mask_override", "get_spot_cull_mask_override");

	ADD_GROUP("Omni Lights (Screen-Space)", "omni_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "omni_lights_enabled"), "set_omni_lights_enabled", "are_omni_lights_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_omni_lights", PROPERTY_HINT_RANGE, "0,16,1"), "set_max_omni_lights", "get_max_omni_lights");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "omni_screen_space_shadows"), "set_omni_screen_space_shadows", "get_omni_screen_space_shadows");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "omni_ss_shadow_steps", PROPERTY_HINT_RANGE, "2,16,1"), "set_omni_ss_shadow_steps", "get_omni_ss_shadow_steps");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "omni_ss_shadow_thickness", PROPERTY_HINT_RANGE, "0.05,5.0,0.05"), "set_omni_ss_shadow_thickness", "get_omni_ss_shadow_thickness");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "omni_ss_shadow_softness", PROPERTY_HINT_RANGE, "0.01,2.0,0.02"), "set_omni_ss_shadow_softness", "get_omni_ss_shadow_softness");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "omni_ss_shadow_strength", PROPERTY_HINT_RANGE, "0,1,0.02"), "set_omni_ss_shadow_strength", "get_omni_ss_shadow_strength");

	BIND_ENUM_CONSTANT(STEP_LINEAR);
	BIND_ENUM_CONSTANT(STEP_QUADRATIC);
}

void VolumetricFog::_validate_property(PropertyInfo &property) const {
	if (use_environment_ambient && property.name == "ambient_light_color") {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
	if (!height_fog_enabled && (property.name == "height_fog_base_height" || property.name == "height_fog_falloff")) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
	if (!noise_enabled && (property.name == "noise_scale" || property.name == "noise_strength" || property.name == "noise_wind")) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
	if (!temporal_reprojection && property.name == "temporal_blend") {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
}

void VolumetricFog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_register_scene_hook();
			set_process(true);
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
			noise_time += get_process_delta_time();
			if (!enabled || !is_inside_tree() || !is_visible_in_tree()) {
				_detach_all_viewports();
				MutexLock lock(scene_mutex);
				snapshot.valid = false;
				return;
			}
			if (!Engine::get_singleton()->is_editor_hint()) {
				_collect_scene_snapshot();
			}
		} break;
	}
}

void VolumetricFog::_register_scene_hook() {
	if (hook_registered) {
		return;
	}
	GL43EffectHelpers::register_scene_hook(this, GL43EffectHelpers::HOOK_PRE_SCREEN_TEXTURE, 30, [this](const GL43EffectHelpers::SceneHookContext &p_ctx) {
		_on_scene_hook(p_ctx);
	});
	hook_registered = true;
}

void VolumetricFog::_unregister_scene_hook() {
	if (!hook_registered) {
		return;
	}
	GL43EffectHelpers::unregister_scene_hooks(this);
	hook_registered = false;
	_detach_all_viewports();
}

void VolumetricFog::_detach_all_viewports() {
	MutexLock lock(scene_mutex);
	for (auto &kv : viewport_states) {
		kv.second.active_composite = false;
		kv.second.history_valid = false;
	}
}

VolumetricFog::ViewportRenderState &VolumetricFog::_get_or_create_vp_state(Viewport *p_vp) {
	uint32_t key = p_vp->get_viewport_rid().get_id();
	auto it = viewport_states.find(key);
	if (it == viewport_states.end()) {
		ViewportRenderState st;
		st.fog_texture.instance();
		st.low_depth_texture.instance();
		it = viewport_states.emplace(key, std::move(st)).first;
	}
	if (VisualServer::get_singleton()) {
		it->second.color_tex_rid = VisualServer::get_singleton()->viewport_get_texture(p_vp->get_viewport_rid(), VS::VIEWPORT_TEXTURE_BUFFER_COLOR);
	}
	it->second.active_composite = auto_composite;
	it->second.last_seen = vp_collect_counter;
	return it->second;
}

void VolumetricFog::_prune_stale_viewports() {
	for (auto it = viewport_states.begin(); it != viewport_states.end(); ) {
		if (it->second.last_seen != vp_collect_counter) {
			it->second.active_composite = false;
			it->second.history_valid = false;
			if (vp_collect_counter - it->second.last_seen > 60ULL) {
				if (gl_ready && gl_thread_id == Thread::get_caller_id()) {
					SharedGL43Context::ScopedScope gl43;
					if (gl43.is_active()) {
						if (it->second.cam_depth_fbo) glDeleteFramebuffers(1, &it->second.cam_depth_fbo);
						if (it->second.cam_depth_tex) glDeleteTextures(1, &it->second.cam_depth_tex);
						if (it->second.fog_work_a) glDeleteTextures(1, &it->second.fog_work_a);
						if (it->second.fog_work_b) glDeleteTextures(1, &it->second.fog_work_b);
						if (it->second.fog_history_tex[0]) glDeleteTextures(1, &it->second.fog_history_tex[0]);
						if (it->second.fog_history_tex[1]) glDeleteTextures(1, &it->second.fog_history_tex[1]);
						if (it->second.prev_linear_depth_tex) glDeleteTextures(1, &it->second.prev_linear_depth_tex);
					}
				}
				it = viewport_states.erase(it);
				continue;
			}
		}
		++it;
	}
}

void VolumetricFog::_rebuild_mesh_cache(const std::vector<MeshInstance *> &p_meshes, const std::vector<Ref<Mesh>> &p_extra_meshes) {
	for (auto it = mesh_cache.begin(); it != mesh_cache.end(); ) {
		if (it->second.cached_mesh_instance_id != 0 && ObjectDB::get_instance((ObjectID)it->second.cached_mesh_instance_id) == nullptr) {
			if (it->second.gl_vao || it->second.gl_vbo) {
				gl_buffers_to_delete.push_back(it->second.gl_vao);
				gl_buffers_to_delete.push_back(it->second.gl_vbo);
			}
			it = mesh_cache.erase(it);
		} else {
			++it;
		}
	}

	auto process_mesh = [&](const Ref<Mesh> &mesh) {
		if (mesh.is_null() || !mesh->get_rid().is_valid()) return;
		uint32_t mid = mesh->get_rid().get_id();
		uint64_t m_inst_id = (uint64_t)mesh->get_instance_id();
		AABB mesh_aabb = mesh->get_aabb();

		auto existing = mesh_cache.find(mid);
		if (existing != mesh_cache.end()) {
			if (existing->second.cached_mesh_instance_id == m_inst_id &&
					GL43EffectHelpers::aabb_same(existing->second.cached_mesh_aabb, mesh_aabb)) {
				return;
			}
			if (existing->second.gl_vao || existing->second.gl_vbo) {
				gl_buffers_to_delete.push_back(existing->second.gl_vao);
				gl_buffers_to_delete.push_back(existing->second.gl_vbo);
			}
			mesh_cache.erase(existing);
		}

		MeshGeom geom;
		geom.mesh_id = mid;
		geom.cached_mesh_instance_id = m_inst_id;
		geom.cached_mesh_aabb = mesh_aabb;
		bool first_aabb = true;

		int surf_count = mesh->get_surface_count();
		for (int s = 0; s < surf_count; s++) {
			if (mesh->surface_get_primitive_type(s) != Mesh::PRIMITIVE_TRIANGLES) continue;
			Array arrays = mesh->surface_get_arrays(s);
			if (arrays.size() <= Mesh::ARRAY_VERTEX) continue;

			PoolVector<Vector3> verts = arrays[Mesh::ARRAY_VERTEX];
			int vcount = verts.size();
			if (vcount < 3) continue;

			PoolVector<Vector2> uvs;
			bool has_uvs = arrays.size() > Mesh::ARRAY_TEX_UV && arrays[Mesh::ARRAY_TEX_UV].get_type() == Variant::POOL_VECTOR2_ARRAY;
			if (has_uvs) uvs = arrays[Mesh::ARRAY_TEX_UV];
			if (uvs.size() != vcount) has_uvs = false;

			PoolVector<int> bones;
			PoolVector<real_t> weights;
			bool has_skin = arrays.size() > Mesh::ARRAY_WEIGHTS
						 && arrays[Mesh::ARRAY_BONES].get_type() == Variant::POOL_INT_ARRAY
						 && arrays[Mesh::ARRAY_WEIGHTS].get_type() == Variant::POOL_REAL_ARRAY;
			int bpv = 4;
			if (has_skin) {
				bones = arrays[Mesh::ARRAY_BONES];
				weights = arrays[Mesh::ARRAY_WEIGHTS];
				if (bones.size() >= vcount * 8 && weights.size() >= vcount * 8) bpv = 8;
				else if (bones.size() < vcount * 4 || weights.size() < vcount * 4) has_skin = false;
			}
			if (has_skin) {
				geom.has_skin_weights = true;
				geom.bones_per_vert = MAX(geom.bones_per_vert, bpv);
			}

			PoolVector<int> indices;
			bool has_indices = arrays[Mesh::ARRAY_INDEX].get_type() == Variant::POOL_INT_ARRAY;
			if (has_indices) indices = arrays[Mesh::ARRAY_INDEX];

			PoolVector<Vector3>::Read vr = verts.read();
			PoolVector<Vector2>::Read uvr = uvs.read();
			PoolVector<int>::Read br = bones.read();
			PoolVector<real_t>::Read wr = weights.read();
			PoolVector<int>::Read ir = indices.read();

			uint32_t bind_base = (uint32_t)geom.bind_pos.size();
			for (int v = 0; v < vcount; v++) {
				geom.bind_pos.push_back(vr[v]);
				geom.bind_uv.push_back(has_uvs ? uvr[v] : Vector2());
				if (first_aabb) {
					geom.local_aabb = AABB(vr[v], Vector3());
					first_aabb = false;
				} else {
					geom.local_aabb.expand_to(vr[v]);
				}
				if (has_skin) {
					for (int k = 0; k < bpv; k++) {
						geom.bind_bones.push_back(br[v * bpv + k]);
						geom.bind_weights.push_back((float)wr[v * bpv + k]);
					}
				}
			}

			SurfaceRange sr;
			sr.first_vert = geom.total_verts;

			int tri_idx_count = has_indices ? indices.size() : vcount;
			for (int i = 0; i + 2 < tri_idx_count; i += 3) {
				int i0 = has_indices ? ir[i + 0] : (i + 0);
				int i1 = has_indices ? ir[i + 1] : (i + 1);
				int i2 = has_indices ? ir[i + 2] : (i + 2);
				if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= vcount || i1 >= vcount || i2 >= vcount) {
					continue;
				}
				const Vector3 &p0 = vr[i0];
				const Vector3 &p1 = vr[i1];
				const Vector3 &p2 = vr[i2];
				if ((p1 - p0).cross(p2 - p0).length_squared() < 1e-12f) continue;

				const Vector2 uv0 = has_uvs ? uvr[i0] : Vector2();
				const Vector2 uv1 = has_uvs ? uvr[i1] : Vector2();
				const Vector2 uv2 = has_uvs ? uvr[i2] : Vector2();

				geom.verts.push_back((float)p0.x); geom.verts.push_back((float)p0.y); geom.verts.push_back((float)p0.z);
				geom.verts.push_back((float)uv0.x); geom.verts.push_back((float)uv0.y);

				geom.verts.push_back((float)p1.x); geom.verts.push_back((float)p1.y); geom.verts.push_back((float)p1.z);
				geom.verts.push_back((float)uv1.x); geom.verts.push_back((float)uv1.y);

				geom.verts.push_back((float)p2.x); geom.verts.push_back((float)p2.y); geom.verts.push_back((float)p2.z);
				geom.verts.push_back((float)uv2.x); geom.verts.push_back((float)uv2.y);

				geom.tri_verts.push_back(bind_base + (uint32_t)i0);
				geom.tri_verts.push_back(bind_base + (uint32_t)i1);
				geom.tri_verts.push_back(bind_base + (uint32_t)i2);
				geom.total_verts += 3;
			}

			sr.vert_count = geom.total_verts - sr.first_vert;
			geom.surfaces.push_back(sr);
		}

		if (geom.total_verts == 0) return;

		if (geom.has_skin_weights && geom.bind_bones.size() < geom.bind_pos.size() * (size_t)geom.bones_per_vert) {
			geom.bind_bones.resize(geom.bind_pos.size() * (size_t)geom.bones_per_vert, 0);
			geom.bind_weights.resize(geom.bind_pos.size() * (size_t)geom.bones_per_vert, 0.0f);
		}

		geom.local_aabb.grow_by(0.002f);
		mesh_cache[mid] = std::move(geom);
	};

	for (MeshInstance *mi : p_meshes) {
		process_mesh(mi->get_mesh());
	}
	for (const Ref<Mesh> &m : p_extra_meshes) {
		process_mesh(m);
	}
}

bool VolumetricFog::_build_skinned_snapshot(MeshInstance *p_mi, const MeshGeom &p_geom, SkinnedSnapshot &r_snap, AABB &r_local_aabb) const {
	if (!p_geom.has_skin_weights) {
		return false;
	}

	std::vector<Vector3> posed_pos;
	if (!GL43SkinningHelper::pose_mesh_positions(p_mi, p_geom.bind_pos, p_geom.bind_bones, p_geom.bind_weights, p_geom.bones_per_vert, posed_pos, &r_local_aabb)) {
		return false;
	}

	r_snap.obj_id = (uint64_t)p_mi->get_instance_id();
	r_snap.vertex_count = (int32_t)p_geom.tri_verts.size();
	GL43SkinningHelper::pack_pos_uv_triangles(posed_pos, p_geom.bind_uv, p_geom.tri_verts, r_snap.posed_verts, nullptr);
	return true;
}

void VolumetricFog::_collect_scene_snapshot() {
	SceneTree *tree = get_tree();
	Viewport *my_vp = get_viewport();
	if (!tree || !my_vp) return;

	vp_collect_counter++;
	std::vector<std::pair<Viewport *, Camera *>> active_vps;
	GL43EffectHelpers::find_active_viewports(this, my_vp, active_vps);
	if (active_vps.empty()) {
		MutexLock lock(scene_mutex);
		_prune_stale_viewports();
		snapshot.valid = false;
		return;
	}

	bool in_editor = Engine::get_singleton()->is_editor_hint();
	Viewport *active_vp = active_vps[0].first;
	Camera *cam = active_vps[0].second;
	Size2 vp_size = active_vp->get_size();

	Transform cam_xf = cam->get_camera_transform();
	Vector3 cam_pos = cam_xf.origin;
	Vector3 cam_fwd = (-cam_xf.basis.get_axis(2)).normalized();
	uint32_t cam_cull_mask = cam->get_cull_mask();

	bool is_ortho = (cam->get_projection() == Camera::PROJECTION_ORTHOGONAL);
	float aspect = (float)(vp_size.x / vp_size.y);
	bool flip_fov = (cam->get_keep_aspect_mode() == Camera::KEEP_WIDTH);
	CameraMatrix cam_proj;
	if (is_ortho) {
		cam_proj.set_orthogonal(cam->get_size(), aspect, cam->get_znear(), cam->get_zfar(), flip_fov);
	} else {
		cam_proj.set_perspective(cam->get_fov(), aspect, cam->get_znear(), cam->get_zfar(), flip_fov);
	}
	CameraMatrix cam_view(cam_xf.affine_inverse());
	CameraMatrix cam_vp = cam_proj * cam_view;

	// Collect scene nodes
	Node *search_root = my_vp ? (Node *)my_vp : (Node *)tree->get_root();
	std::vector<MeshInstance *> meshes;
	std::vector<MultiMeshInstance *> multimeshes;
	std::vector<Ref<Mesh>> extra_meshes;
	std::vector<DirectionalLight *> dir_lights;
	std::vector<SpotLight *> spot_lights;
	std::vector<OmniLight *> omni_lights;
	std::vector<LocalFogVolume *> local_vols;
	std::vector<BakedFogVolume *> baked_vols;
	std::vector<VolumetricFogFocus *> focuses;
#ifdef MODULE_CSG_ENABLED
	struct CsgEntry {
		CSGShape *csg;
		Transform local_xf;
		Ref<Mesh> mesh;
	};
	std::vector<CsgEntry> csg_entries;
#endif

	struct SceneCollector {
		static void visit(Node *n, uint32_t cam_mask,
				std::vector<MeshInstance *> &out_m,
				std::vector<MultiMeshInstance *> &out_mm,
				std::vector<Ref<Mesh>> &out_em,
				std::vector<DirectionalLight *> &dirs,
				std::vector<SpotLight *> &spots,
				std::vector<OmniLight *> &omnis,
				std::vector<LocalFogVolume *> &locals,
				std::vector<BakedFogVolume *> &baked,
				std::vector<VolumetricFogFocus *> &focus_list
#ifdef MODULE_CSG_ENABLED
				, std::vector<CsgEntry> &out_csg
#endif
				) {
			if (MeshInstance *mi = Object::cast_to<MeshInstance>(n)) {
				if (mi->is_visible_in_tree() && mi->get_mesh().is_valid()) {
					out_m.push_back(mi);
				}
			} else if (MultiMeshInstance *mmi = Object::cast_to<MultiMeshInstance>(n)) {
				if (mmi->is_visible_in_tree() && mmi->get_multimesh().is_valid() && mmi->get_multimesh()->get_mesh().is_valid()) {
					out_mm.push_back(mmi);
					out_em.push_back(mmi->get_multimesh()->get_mesh());
				}
			}
#ifdef MODULE_CSG_ENABLED
			else if (CSGShape *csg = Object::cast_to<CSGShape>(n)) {
				if (csg->is_visible_in_tree() && csg->is_root_shape()) {
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
#endif
			else if (DirectionalLight *dl = Object::cast_to<DirectionalLight>(n)) {
				if (dl->is_visible_in_tree() && dl->get_param(Light::PARAM_ENERGY) > 0.0001f) {
					dirs.push_back(dl);
				}
			} else if (SpotLight *sl = Object::cast_to<SpotLight>(n)) {
				if (sl->is_visible_in_tree() && sl->get_param(Light::PARAM_ENERGY) > 0.0001f) {
					spots.push_back(sl);
				}
			} else if (OmniLight *ol = Object::cast_to<OmniLight>(n)) {
				if (ol->is_visible_in_tree() && ol->get_param(Light::PARAM_ENERGY) > 0.0001f) {
					omnis.push_back(ol);
				}
			} else if (LocalFogVolume *lv = Object::cast_to<LocalFogVolume>(n)) {
				if (lv->is_enabled() && lv->is_visible_in_tree() && (lv->get_cull_mask() & cam_mask)) {
					locals.push_back(lv);
				}
			} else if (BakedFogVolume *bv = Object::cast_to<BakedFogVolume>(n)) {
				if (bv->is_enabled() && bv->is_visible_in_tree() && bv->get_baked_texture().is_valid() && (bv->get_cull_mask() & cam_mask)) {
					baked.push_back(bv);
				}
			} else if (VolumetricFogFocus *vf = Object::cast_to<VolumetricFogFocus>(n)) {
				if (vf->is_enabled() && vf->is_visible_in_tree()) {
					focus_list.push_back(vf);
				}
			}

			for (int i = 0; i < n->get_child_count(); i++) {
				visit(n->get_child(i), cam_mask, out_m, out_mm, out_em, dirs, spots, omnis, locals, baked, focus_list
#ifdef MODULE_CSG_ENABLED
						, out_csg
#endif
				);
			}
		}
	};

	if (search_root) {
		SceneCollector::visit(search_root, cam_cull_mask, meshes, multimeshes, extra_meshes, dir_lights, spot_lights, omni_lights, local_vols, baked_vols, focuses
#ifdef MODULE_CSG_ENABLED
				, csg_entries
#endif
		);
	}

	// Keep lights & volumes visible across any active viewport
	std::vector<std::array<float, 24>> all_frustums(active_vps.size());
	for (size_t vi = 0; vi < active_vps.size(); vi++) {
		Viewport *v = active_vps[vi].first;
		Camera *c = active_vps[vi].second;
		CameraMatrix vview, vproj, vvp, vinv_proj, vinv_vp;
		GL43EffectHelpers::compute_camera_matrices(c, v->get_size(), vview, vproj, vvp, vinv_proj, vinv_vp);
		float fplanes[6][4];
		GL43EffectHelpers::extract_frustum_planes(vvp, fplanes);
		std::memcpy(all_frustums[vi].data(), fplanes, sizeof(fplanes));
	}

	auto any_sphere_visible = [&](const Vector3 &center, float rad) {
		for (size_t vi = 0; vi < all_frustums.size(); vi++) {
			float fplanes[6][4];
			std::memcpy(fplanes, all_frustums[vi].data(), sizeof(fplanes));
			if (GL43EffectHelpers::sphere_in_frustum(center, rad, fplanes)) return true;
		}
		return false;
	};
	auto any_aabb_visible = [&](const AABB &box) {
		for (size_t vi = 0; vi < all_frustums.size(); vi++) {
			float fplanes[6][4];
			std::memcpy(fplanes, all_frustums[vi].data(), sizeof(fplanes));
			if (GL43EffectHelpers::aabb_in_frustum(box, fplanes)) return true;
		}
		return false;
	};

	spot_lights.erase(std::remove_if(spot_lights.begin(), spot_lights.end(), [&](SpotLight *sl) {
		return !any_sphere_visible(sl->get_global_transform().origin, sl->get_param(Light::PARAM_RANGE));
	}), spot_lights.end());

	omni_lights.erase(std::remove_if(omni_lights.begin(), omni_lights.end(), [&](OmniLight *ol) {
		float r = ol->get_param(Light::PARAM_RANGE);
		if (ol->is_square_shape()) r *= 1.73205f;
		return !any_sphere_visible(ol->get_global_transform().origin, r);
	}), omni_lights.end());

	local_vols.erase(std::remove_if(local_vols.begin(), local_vols.end(), [&](LocalFogVolume *lv) {
		return !any_aabb_visible(lv->get_world_aabb());
	}), local_vols.end());

	baked_vols.erase(std::remove_if(baked_vols.begin(), baked_vols.end(), [&](BakedFogVolume *bv) {
		return !any_aabb_visible(bv->get_world_aabb());
	}), baked_vols.end());

	VolumetricFogFocus *active_focus = nullptr;
	if (!focuses.empty()) {
		std::sort(focuses.begin(), focuses.end(), [&](VolumetricFogFocus *a, VolumetricFogFocus *b) {
			if (a->get_priority() != b->get_priority()) return a->get_priority() > b->get_priority();
			return a->get_global_transform().origin.distance_squared_to(cam_pos) <
				   b->get_global_transform().origin.distance_squared_to(cam_pos);
		});
		active_focus = focuses[0];
	}

	Vector3 priority_anchor = active_focus
			? active_focus->get_global_transform().origin.linear_interpolate(cam_pos, active_focus->get_camera_bias())
			: cam_pos;

	std::sort(spot_lights.begin(), spot_lights.end(), [&](SpotLight *a, SpotLight *b) {
		float da = MAX(0.0f, a->get_global_transform().origin.distance_to(priority_anchor) - a->get_param(Light::PARAM_RANGE));
		float db = MAX(0.0f, b->get_global_transform().origin.distance_to(priority_anchor) - b->get_param(Light::PARAM_RANGE));
		return da < db;
	});
	std::sort(omni_lights.begin(), omni_lights.end(), [&](OmniLight *a, OmniLight *b) {
		float da = MAX(0.0f, a->get_global_transform().origin.distance_to(priority_anchor) - a->get_param(Light::PARAM_RANGE));
		float db = MAX(0.0f, b->get_global_transform().origin.distance_to(priority_anchor) - b->get_param(Light::PARAM_RANGE));
		return da < db;
	});
	std::sort(local_vols.begin(), local_vols.end(), [&](LocalFogVolume *a, LocalFogVolume *b) {
		if (a->get_priority() != b->get_priority()) return a->get_priority() < b->get_priority();
		return a->get_global_transform().origin.distance_squared_to(priority_anchor) <
			   b->get_global_transform().origin.distance_squared_to(priority_anchor);
	});
	std::sort(baked_vols.begin(), baked_vols.end(), [&](BakedFogVolume *a, BakedFogVolume *b) {
		return a->get_global_transform().origin.distance_squared_to(priority_anchor) <
			   b->get_global_transform().origin.distance_squared_to(priority_anchor);
	});

	bool has_any_fog = (density > 0.00001f) ||
			(emission_energy > 0.00001f && emission.to_linear().get_v() > 0.0001f) ||
			!local_vols.empty() ||
			!baked_vols.empty();

	if (!has_any_fog) {
		_detach_all_viewports();
		MutexLock lock(scene_mutex);
		snapshot.valid = false;
		return;
	}

	Ref<World> world = active_vp->find_world();
	if (world.is_null()) world = my_vp->find_world();
	Ref<Environment> env = world.is_valid() ? world->get_environment() : Ref<Environment>();
	if (env.is_null() && world.is_valid()) env = world->get_fallback_environment();

	MutexLock lock(scene_mutex);

	_rebuild_mesh_cache(meshes, extra_meshes);
	pinned_textures.clear();
	pinned_3d_textures.clear();

	snapshot.viewports.clear();
	for (size_t vi = 0; vi < active_vps.size(); vi++) {
		Viewport *vp = active_vps[vi].first;
		Camera *vcam = active_vps[vi].second;
		Size2 vsz = vp->get_size();
		int tw = MAX(16, (int)Math::round(vsz.x * resolution_scale));
		int th = MAX(16, (int)Math::round(vsz.y * resolution_scale));

		ViewportRenderState &vst = _get_or_create_vp_state(vp);
		if (vst.fog_tex_w != tw || vst.fog_tex_h != th) {
			vst.fog_tex_w = tw;
			vst.fog_tex_h = th;

			Ref<Image> fimg;
			fimg.instance();
			fimg->create(tw, th, false, Image::FORMAT_RGBAH);
			fimg->fill(Color(0.0f, 0.0f, 0.0f, 0.0f));
			vst.fog_texture->create_from_image(fimg, Texture::FLAG_FILTER);

			Ref<Image> dimg;
			dimg.instance();
			dimg->create(tw, th, false, Image::FORMAT_RF);
			dimg->fill(Color(1.0f, 1.0f, 1.0f, 1.0f));
			vst.low_depth_texture->create_from_image(dimg, 0);
		}

		if (vi == 0) {
			fog_texture = vst.fog_texture;
			low_depth_texture = vst.low_depth_texture;
		}

		Transform vxf = vcam->get_camera_transform();
		CameraMatrix v_view, v_proj, v_vp, v_inv_proj, v_inv_vp;
		GL43EffectHelpers::compute_camera_matrices(vcam, vsz, v_view, v_proj, v_vp, v_inv_proj, v_inv_vp);

		ViewportSnapshot vsnap;
		vsnap.vp_id = vp->get_viewport_rid().get_id();
		vsnap.vp_rid = vp->get_viewport_rid();
		vsnap.viewport_w = (int)vsz.x;
		vsnap.viewport_h = (int)vsz.y;
		vsnap.fog_w = tw;
		vsnap.fog_h = th;
		vsnap.cam_pos[0] = (float)vxf.origin.x;
		vsnap.cam_pos[1] = (float)vxf.origin.y;
		vsnap.cam_pos[2] = (float)vxf.origin.z;
		GL43EffectHelpers::camera_matrix_to_float16(v_view, vsnap.view_mat);
		GL43EffectHelpers::camera_matrix_to_float16(v_proj, vsnap.proj_mat);
		GL43EffectHelpers::camera_matrix_to_float16(v_vp, vsnap.vp_mat);
		GL43EffectHelpers::camera_matrix_to_float16(v_inv_proj, vsnap.inv_proj_mat);
		GL43EffectHelpers::camera_matrix_to_float16(v_inv_vp, vsnap.inv_vp_mat);
		std::memcpy(vsnap.cam_frustum_planes, all_frustums[vi].data(), sizeof(vsnap.cam_frustum_planes));

		if (use_environment_ambient && env.is_valid()) {
			Color ec = env->get_ambient_light_color().to_linear();
			float ee = env->get_ambient_light_energy();
			vsnap.amb_color[0] = ec.r * ee * ambient_light_energy;
			vsnap.amb_color[1] = ec.g * ee * ambient_light_energy;
			vsnap.amb_color[2] = ec.b * ee * ambient_light_energy;
		} else {
			Color ac = ambient_light_color.to_linear();
			vsnap.amb_color[0] = ac.r * ambient_light_energy;
			vsnap.amb_color[1] = ac.g * ambient_light_energy;
			vsnap.amb_color[2] = ac.b * ambient_light_energy;
		}

		vsnap.hdr = vp->get_hdr();
		vsnap.keep_3d_linear = vp->get_keep_3d_linear();
		if (env.is_valid()) {
			vsnap.tonemap_mode = (int)env->get_tonemapper();
			vsnap.tonemap_exposure = env->get_tonemap_exposure();
			vsnap.tonemap_white = env->get_tonemap_white();
		} else {
			vsnap.tonemap_mode = 0;
			vsnap.tonemap_exposure = 1.0f;
			vsnap.tonemap_white = 1.0f;
		}

		vsnap.effective_steps = in_editor ? MIN(steps, editor_max_steps) : steps;
		vsnap.eff_max_distance = max_distance;
		if (active_focus) {
			vsnap.focus_active = true;
			Vector3 fp = active_focus->get_global_transform().origin;
			vsnap.focus_pos[0] = (float)fp.x;
			vsnap.focus_pos[1] = (float)fp.y;
			vsnap.focus_pos[2] = (float)fp.z;
			vsnap.focus_radius = active_focus->get_radius();
			vsnap.focus_falloff = active_focus->get_focus_falloff();
			vsnap.focus_density_mult = active_focus->get_density_multiplier();
			if (active_focus->get_max_ray_distance_override() > 0.0f) {
				vsnap.eff_max_distance = active_focus->get_max_ray_distance_override();
			}
		} else {
			vsnap.focus_active = false;
		}

		snapshot.viewports.push_back(vsnap);
	}
	_prune_stale_viewports();

	// Build Depth Instances (MeshInstance + MultiMeshInstance)
	snapshot.instances.clear();
	snapshot.skinned_instances.clear();

	auto resolve_surface_ranges = [&](const MeshGeom &geom, auto get_mat_fn, std::vector<SurfaceRange> &out_surfaces) {
		out_surfaces.clear();
		for (size_t s = 0; s < geom.surfaces.size(); s++) {
			SurfaceRange sr = geom.surfaces[s];
			Ref<SpatialMaterial> smat = get_mat_fn((int)s);
			if (smat.is_valid()) {
				bool is_transparent = smat->get_feature(SpatialMaterial::FEATURE_TRANSPARENT);
				bool is_scissor = smat->get_flag(SpatialMaterial::FLAG_USE_ALPHA_SCISSOR);
				if (is_transparent && !is_scissor) {
					continue; // Transparent objects do not write depth in Godot's opaque depth pass
				}
				sr.cull_mode = (int32_t)smat->get_cull_mode();
				if (is_scissor) {
					Ref<Texture> alb_tex = smat->get_texture(SpatialMaterial::TEXTURE_ALBEDO);
					if (alb_tex.is_valid() && alb_tex->get_rid().is_valid()) {
						sr.alpha_scissor = true;
						sr.alpha_threshold = smat->get_alpha_scissor_threshold();
						sr.albedo_tex_rid = alb_tex->get_rid();
						pinned_textures.push_back(alb_tex);
					}
				}
			}
			out_surfaces.push_back(sr);
		}
	};

	for (MeshInstance *mi : meshes) {
		Ref<Mesh> mesh = mi->get_mesh();
		uint32_t mid = mesh->get_rid().get_id();
		auto it = mesh_cache.find(mid);
		if (it == mesh_cache.end()) continue;
		const MeshGeom &geom = it->second;

		GeometryInstance::ShadowCastingSetting shadow_mode = mi->get_cast_shadows_setting();
		uint32_t lm = mi->get_layer_mask();

		DepthInstance inst;
		inst.mesh_id = mid;
		inst.layer_mask = lm;
		inst.draw_in_cam = ((lm & cam_cull_mask) != 0) && (shadow_mode != GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
		inst.cast_shadows = (shadow_mode != GeometryInstance::SHADOW_CASTING_SETTING_OFF);
		inst.double_sided_shadow = (shadow_mode == GeometryInstance::SHADOW_CASTING_SETTING_DOUBLE_SIDED);

		Transform xf = mi->get_global_transform();
		GL43EffectHelpers::transform_to_float16(xf, inst.model_mat);
		inst.flip_facing = (xf.basis.determinant() < 0.0f);

		AABB local_bounds = geom.local_aabb;
		SkinnedSnapshot sk_snap;
		if (_build_skinned_snapshot(mi, geom, sk_snap, local_bounds)) {
			inst.is_skinned = true;
			inst.skinned_snap_idx = (int32_t)snapshot.skinned_instances.size();
			snapshot.skinned_instances.push_back(std::move(sk_snap));
		}
		inst.world_aabb = xf.xform(local_bounds);

		resolve_surface_ranges(geom, [&](int s) { return mi->get_active_material(s); }, inst.surfaces);
		if (inst.surfaces.empty()) {
			if (inst.is_skinned) snapshot.skinned_instances.pop_back();
			continue;
		}

		snapshot.instances.push_back(std::move(inst));
	}

	for (MultiMeshInstance *mmi : multimeshes) {
		Ref<MultiMesh> mm = mmi->get_multimesh();
		if (mm->get_transform_format() != MultiMesh::TRANSFORM_3D) continue;
		Ref<Mesh> mesh = mm->get_mesh();
		uint32_t mid = mesh->get_rid().get_id();
		auto it = mesh_cache.find(mid);
		if (it == mesh_cache.end()) continue;
		const MeshGeom &geom = it->second;

		GeometryInstance::ShadowCastingSetting shadow_mode = mmi->get_cast_shadows_setting();
		uint32_t lm = mmi->get_layer_mask();
		bool draw_in_cam = ((lm & cam_cull_mask) != 0) && (shadow_mode != GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
		bool cast_shadows = (shadow_mode != GeometryInstance::SHADOW_CASTING_SETTING_OFF);
		bool double_sided = (shadow_mode == GeometryInstance::SHADOW_CASTING_SETTING_DOUBLE_SIDED);

		std::vector<SurfaceRange> mmi_surfaces;
		Ref<Material> override_mat = mmi->get_material_override();
		resolve_surface_ranges(geom, [&](int s) {
			return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s);
		}, mmi_surfaces);
		if (mmi_surfaces.empty()) continue;

		Transform base_xf = mmi->get_global_transform();
		int count = mm->get_visible_instance_count();
		if (count < 0) count = mm->get_instance_count();

		for (int i = 0; i < count; i++) {
			Transform xf = base_xf * mm->get_instance_transform(i);
			DepthInstance inst;
			inst.mesh_id = mid;
			inst.layer_mask = lm;
			inst.draw_in_cam = draw_in_cam;
			inst.cast_shadows = cast_shadows;
			inst.double_sided_shadow = double_sided;
			inst.flip_facing = (xf.basis.determinant() < 0.0f);
			GL43EffectHelpers::transform_to_float16(xf, inst.model_mat);
			inst.world_aabb = xf.xform(geom.local_aabb);
			inst.surfaces = mmi_surfaces;
			snapshot.instances.push_back(std::move(inst));
		}
	}

#ifdef MODULE_CSG_ENABLED
	for (const CsgEntry &entry : csg_entries) {
		CSGShape *csg = entry.csg;
		Ref<Mesh> mesh = entry.mesh;
		uint32_t mid = mesh->get_rid().get_id();
		auto it = mesh_cache.find(mid);
		if (it == mesh_cache.end()) continue;
		const MeshGeom &geom = it->second;

		GeometryInstance::ShadowCastingSetting shadow_mode = csg->get_cast_shadows_setting();
		uint32_t lm = csg->get_layer_mask();

		std::vector<SurfaceRange> csg_surfaces;
		Ref<Material> override_mat = csg->get_material_override();
		resolve_surface_ranges(geom, [&](int s) {
			return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s);
		}, csg_surfaces);
		if (csg_surfaces.empty()) continue;

		Transform xf = csg->get_global_transform() * entry.local_xf;
		DepthInstance inst;
		inst.mesh_id = mid;
		inst.layer_mask = lm;
		inst.draw_in_cam = ((lm & cam_cull_mask) != 0) && (shadow_mode != GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
		inst.cast_shadows = (shadow_mode != GeometryInstance::SHADOW_CASTING_SETTING_OFF);
		inst.double_sided_shadow = (shadow_mode == GeometryInstance::SHADOW_CASTING_SETTING_DOUBLE_SIDED);
		inst.flip_facing = (xf.basis.determinant() < 0.0f);
		GL43EffectHelpers::transform_to_float16(xf, inst.model_mat);
		inst.world_aabb = xf.xform(geom.local_aabb);
		inst.surfaces = std::move(csg_surfaces);
		snapshot.instances.push_back(std::move(inst));
	}
#endif

	// Skip real-time dynamic light shadows if fog is purely baked
	bool needs_dynamic_light_shadows = (density > 0.00001f) || !local_vols.empty();

	// Directional Lights
	snapshot.dir_lights.clear();
	snapshot.dir_meta.clear();
	int dir_count = (directional_lights_enabled && needs_dynamic_light_shadows) ? MIN((int)dir_lights.size(), (int)MAX_DIR_LIGHTS) : 0;
	for (int i = 0; i < dir_count; i++) {
		DirectionalLight *dl = dir_lights[i];
		Transform lxf = dl->get_global_transform();
		Basis l_basis = lxf.basis.orthonormalized();
		Vector3 light_fwd_to_sun = l_basis.get_axis(2).normalized();

		float focus_rad = directional_shadow_distance * 0.55f;
		float focus_depth = directional_depth_range;
		Vector3 default_center = cam_pos + cam_fwd * (directional_shadow_distance * 0.45f);
		Vector3 center = default_center;
		if (active_focus) {
			center = active_focus->get_global_transform().origin.linear_interpolate(default_center, active_focus->get_camera_bias());
			focus_rad = active_focus->get_radius();
			focus_depth = active_focus->get_depth_range();
		}

		float ortho_size = MAX(focus_rad * 2.0f, 1.0f);
		float texel = ortho_size / (float)MAX(directional_depth_resolution, 16);
		Vector3 r_axis = l_basis.get_axis(0);
		Vector3 u_axis = l_basis.get_axis(1);
		float rx = Math::floor(center.dot(r_axis) / texel) * texel;
		float ry = Math::floor(center.dot(u_axis) / texel) * texel;
		float z_step = MAX(focus_depth / 4096.0f, 0.001f);
		float rz = Math::floor(center.dot(light_fwd_to_sun) / z_step) * z_step;
		Vector3 snapped_center = r_axis * rx + u_axis * ry + light_fwd_to_sun * rz;

		Vector3 eye = snapped_center + light_fwd_to_sun * (focus_depth * 0.5f);
		Transform light_cam_xf(l_basis, eye);
		float z_near = 0.05f;
		float z_far = MAX(focus_depth, 1.0f);

		CameraMatrix d_proj;
		d_proj.set_orthogonal(ortho_size, 1.0f, z_near, z_far, false);
		CameraMatrix d_view(light_cam_xf.affine_inverse());
		CameraMatrix d_vp = d_proj * d_view;

		GpuDirLight gdl = {};
		GL43EffectHelpers::camera_matrix_to_float16(d_vp, gdl.vp);
		gdl.dir_bias[0] = (float)light_fwd_to_sun.x;
		gdl.dir_bias[1] = (float)light_fwd_to_sun.y;
		gdl.dir_bias[2] = (float)light_fwd_to_sun.z;
		gdl.dir_bias[3] = directional_depth_bias;

		Color dc = dl->get_color().to_linear();
		float de = dl->get_param(Light::PARAM_ENERGY);
		gdl.color_flags[0] = dc.r * de;
		gdl.color_flags[1] = dc.g * de;
		gdl.color_flags[2] = dc.b * de;
		gdl.color_flags[3] = dl->has_shadow() ? 1.0f : 0.0f;

		DirShadowMeta meta;
		GL43EffectHelpers::camera_matrix_to_float16(d_vp, meta.vp);
		meta.cull_mask = (directional_cull_mask_override != 0) ? directional_cull_mask_override : dl->get_shadow_cull_mask();
		meta.has_shadow = dl->has_shadow();

		float half_o = ortho_size * 0.5f;
		Vector3 corners[8] = {
			snapped_center - r_axis * half_o - u_axis * half_o - light_fwd_to_sun * (focus_depth * 0.5f),
			snapped_center + r_axis * half_o - u_axis * half_o - light_fwd_to_sun * (focus_depth * 0.5f),
			snapped_center - r_axis * half_o + u_axis * half_o - light_fwd_to_sun * (focus_depth * 0.5f),
			snapped_center + r_axis * half_o + u_axis * half_o - light_fwd_to_sun * (focus_depth * 0.5f),
			snapped_center - r_axis * half_o - u_axis * half_o + light_fwd_to_sun * (focus_depth * 0.5f),
			snapped_center + r_axis * half_o - u_axis * half_o + light_fwd_to_sun * (focus_depth * 0.5f),
			snapped_center - r_axis * half_o + u_axis * half_o + light_fwd_to_sun * (focus_depth * 0.5f),
			snapped_center + r_axis * half_o + u_axis * half_o + light_fwd_to_sun * (focus_depth * 0.5f),
		};
		meta.frustum_aabb = AABB(corners[0], Vector3());
		for (int c = 1; c < 8; c++) meta.frustum_aabb.expand_to(corners[c]);

		if (i == 0 && dl->get_projector().is_valid() && !dl->is_projector_disable()) {
			Ref<Texture> proj_tex = dl->get_projector();
			if (proj_tex->get_rid().is_valid()) {
				pinned_textures.push_back(proj_tex);
				meta.proj_tex_rid = proj_tex->get_rid();
				Color pc2 = dl->get_projector_color_2().to_linear();
				Vector2 ps = dl->get_projector_uv_scale();
				Vector2 po = dl->get_projector_uv_offset();
				gdl.proj_color[0] = pc2.r * de;
				gdl.proj_color[1] = pc2.g * de;
				gdl.proj_color[2] = pc2.b * de;
				gdl.proj_color[3] = pc2.a;
				gdl.proj_uv[0] = ps.x;
				gdl.proj_uv[1] = ps.y;
				gdl.proj_uv[2] = po.x;
				gdl.proj_uv[3] = po.y;
				gdl.proj_extra[0] = dl->get_projector_rotation();
				gdl.proj_extra[1] = dl->is_projector_repeat() ? 1.0f : 0.0f;
				gdl.proj_extra[2] = dl->get_projector_lod();
				gdl.proj_extra[3] = 1.0f;
			}
		}

		snapshot.dir_lights.push_back(gdl);
		snapshot.dir_meta.push_back(meta);
	}

	// Spot Lights
	snapshot.spot_lights.clear();
	snapshot.spot_meta.clear();
	int spot_limit = (spot_lights_enabled && needs_dynamic_light_shadows) ? CLAMP(max_spot_lights, 0, (int)MAX_SPOT_LIGHTS) : 0;
	int spot_count = MIN((int)spot_lights.size(), spot_limit);
	for (int i = 0; i < spot_count; i++) {
		SpotLight *sl = spot_lights[i];
		Transform lxf = sl->get_global_transform();
		lxf.orthonormalize();
		float range = MAX(sl->get_param(Light::PARAM_RANGE), 0.25f);
		float angle_deg = CLAMP(sl->get_param(Light::PARAM_SPOT_ANGLE), 0.5f, 89.0f);
		float near_p = CLAMP(spot_near_plane, 0.01f, range * 0.25f);

		CameraMatrix s_proj;
		s_proj.set_perspective(angle_deg * 2.0f, 1.0f, near_p, range, false);
		CameraMatrix s_view(lxf.affine_inverse());
		CameraMatrix s_vp = s_proj * s_view;

		Vector3 spot_fwd = (-lxf.basis.get_axis(2)).normalized();
		Color sc = sl->get_color().to_linear();
		float se = sl->get_param(Light::PARAM_ENERGY);

		GpuSpotLight gsl = {};
		GL43EffectHelpers::camera_matrix_to_float16(s_vp, gsl.vp);
		gsl.pos_range[0] = (float)lxf.origin.x;
		gsl.pos_range[1] = (float)lxf.origin.y;
		gsl.pos_range[2] = (float)lxf.origin.z;
		gsl.pos_range[3] = range;
		gsl.dir_atten[0] = (float)spot_fwd.x;
		gsl.dir_atten[1] = (float)spot_fwd.y;
		gsl.dir_atten[2] = (float)spot_fwd.z;
		gsl.dir_atten[3] = sl->get_param(Light::PARAM_ATTENUATION);
		gsl.color_cone[0] = sc.r * se;
		gsl.color_cone[1] = sc.g * se;
		gsl.color_cone[2] = sc.b * se;
		gsl.color_cone[3] = Math::cos(Math::deg2rad(angle_deg));
		gsl.params[0] = sl->get_param(Light::PARAM_SPOT_ATTENUATION);
		gsl.params[1] = spot_depth_bias;
		gsl.params[2] = sl->is_square_shape() ? 1.0f : 0.0f;
		gsl.params[3] = sl->has_shadow() ? 1.0f : 0.0f;

		SpotShadowMeta meta;
		GL43EffectHelpers::camera_matrix_to_float16(s_vp, meta.vp);
		meta.cull_mask = (spot_cull_mask_override != 0) ? spot_cull_mask_override : sl->get_shadow_cull_mask();
		meta.has_shadow = sl->has_shadow();
		meta.pos = lxf.origin;
		meta.range = range;

		snapshot.spot_lights.push_back(gsl);
		snapshot.spot_meta.push_back(meta);
	}

	// Omni Lights
	snapshot.omni_lights.clear();
	int omni_limit = omni_lights_enabled ? CLAMP(max_omni_lights, 0, (int)MAX_OMNI_LIGHTS) : 0;
	int omni_count = MIN((int)omni_lights.size(), omni_limit);
	for (int i = 0; i < omni_count; i++) {
		OmniLight *ol = omni_lights[i];
		Vector3 wpos = ol->get_global_transform().origin;
		Vector3 vpos = cam_view.xform(wpos);
		float range = MAX(ol->get_param(Light::PARAM_RANGE), 0.1f);
		Color oc = ol->get_color().to_linear();
		float oe = ol->get_param(Light::PARAM_ENERGY);
		float flags = (ol->is_square_shape() ? 1.0f : 0.0f) + (ol->has_shadow() ? 2.0f : 0.0f);

		GpuOmniLight gol = {};
		gol.pos_range[0] = (float)wpos.x;
		gol.pos_range[1] = (float)wpos.y;
		gol.pos_range[2] = (float)wpos.z;
		gol.pos_range[3] = range;
		gol.view_atten[0] = (float)vpos.x;
		gol.view_atten[1] = (float)vpos.y;
		gol.view_atten[2] = (float)vpos.z;
		gol.view_atten[3] = ol->get_param(Light::PARAM_ATTENUATION);
		gol.color_flags[0] = oc.r * oe;
		gol.color_flags[1] = oc.g * oe;
		gol.color_flags[2] = oc.b * oe;
		gol.color_flags[3] = flags;
		snapshot.omni_lights.push_back(gol);
	}

	// LocalFogVolumes
	snapshot.local_volumes.clear();
	int local_count = MIN((int)local_vols.size(), (int)MAX_LOCAL_VOLUMES);
	for (int i = 0; i < local_count; i++) {
		LocalFogVolume *lv = local_vols[i];
		Transform inv = lv->get_global_transform().affine_inverse();
		Vector3 e = lv->get_extents();
		if (lv->get_shape() == LocalFogVolume::SHAPE_SPHERE) {
			float r = MAX(e.x, MAX(e.y, e.z));
			e = Vector3(r, r, r);
		}
		Vector3 inv_e(1.0f / MAX(e.x, 0.0001f), 1.0f / MAX(e.y, 0.0001f), 1.0f / MAX(e.z, 0.0001f));

		GpuLocalVolume glv = {};
		GL43EffectHelpers::push_affine_row(inv, inv_e, Vector3(), 0, glv.inv_xf_row0);
		GL43EffectHelpers::push_affine_row(inv, inv_e, Vector3(), 1, glv.inv_xf_row1);
		GL43EffectHelpers::push_affine_row(inv, inv_e, Vector3(), 2, glv.inv_xf_row2);

		Color la = lv->get_albedo().to_linear();
		Color le = lv->get_emission().to_linear();
		float lee = lv->get_emission_energy();
		glv.albedo_density[0] = la.r;
		glv.albedo_density[1] = la.g;
		glv.albedo_density[2] = la.b;
		glv.albedo_density[3] = lv->get_density();
		glv.emission_falloff[0] = le.r * lee;
		glv.emission_falloff[1] = le.g * lee;
		glv.emission_falloff[2] = le.b * lee;
		glv.emission_falloff[3] = lv->get_edge_falloff();
		glv.params[0] = (float)lv->get_shape();
		glv.params[1] = (float)lv->get_blend_mode();
		glv.params[2] = lv->get_noise_strength();
		glv.params[3] = lv->get_noise_scale();
		Vector3 noff = lv->get_noise_velocity() * noise_time;
		glv.noise_offset[0] = (float)noff.x;
		glv.noise_offset[1] = (float)noff.y;
		glv.noise_offset[2] = (float)noff.z;
		snapshot.local_volumes.push_back(glv);
	}

	// BakedFogVolumes
	snapshot.baked_volumes.clear();
	snapshot.baked_tex_rids.clear();
	int baked_count = MIN((int)baked_vols.size(), (int)MAX_BAKED_VOLUMES);
	for (int i = 0; i < baked_count; i++) {
		BakedFogVolume *bv = baked_vols[i];
		Transform inv = bv->get_global_transform().affine_inverse();
		Vector3 e = bv->get_extents();
		Vector3 inv_2e(1.0f / MAX(2.0f * e.x, 0.0001f), 1.0f / MAX(2.0f * e.y, 0.0001f), 1.0f / MAX(2.0f * e.z, 0.0001f));

		GpuBakedVolume gbv = {};
		GL43EffectHelpers::push_affine_row(inv, inv_2e, Vector3(0.5f, 0.5f, 0.5f), 0, gbv.inv_xf_row0);
		GL43EffectHelpers::push_affine_row(inv, inv_2e, Vector3(0.5f, 0.5f, 0.5f), 1, gbv.inv_xf_row1);
		GL43EffectHelpers::push_affine_row(inv, inv_2e, Vector3(0.5f, 0.5f, 0.5f), 2, gbv.inv_xf_row2);

		Color bt = bv->get_albedo_tint().to_linear();
		gbv.tint_dens[0] = bt.r;
		gbv.tint_dens[1] = bt.g;
		gbv.tint_dens[2] = bt.b;
		gbv.tint_dens[3] = bv->get_density_multiplier();
		gbv.params[0] = bv->get_emission_multiplier();
		gbv.params[1] = bv->get_edge_falloff();

		Ref<Texture3D> btex = bv->get_baked_texture();
		pinned_3d_textures.push_back(btex);
		snapshot.baked_tex_rids.push_back(btex->get_rid());
		snapshot.baked_volumes.push_back(gbv);
	}

	snapshot.valid = true;
}

bool VolumetricFog::_init_gl_resources() {
	if (gl_attempted) return gl_ready;
	gl_attempted = true;
	gl_thread_id = Thread::get_caller_id();

	if (!SharedGL43Context::get_singleton()->is_available()) {
		ERR_PRINT("VolumetricFog: Shared OpenGL 4.3 context not available.");
		return false;
	}

	depth_program = GL43EffectHelpers::link_vf_program(VolumetricsShaders::DEPTH_VERT_GLSL, VolumetricsShaders::DEPTH_FRAG_GLSL, "VolumetricFog", "Depth");
	raymarch_program = GL43EffectHelpers::link_compute_program(VolumetricsShaders::RAYMARCH_COMPUTE_GLSL_FULL, "VolumetricFog", "Raymarch");
	temporal_program = GL43EffectHelpers::link_compute_program(VolumetricsShaders::TEMPORAL_COMPUTE_GLSL, "VolumetricFog", "Temporal");
	denoise_program = GL43EffectHelpers::link_compute_program(VolumetricsShaders::DENOISE_COMPUTE_GLSL, "VolumetricFog", "Denoise");
	finalize_program = GL43EffectHelpers::link_compute_program(VolumetricsShaders::FINALIZE_COMPUTE_GLSL, "VolumetricFog", "Finalize");
	composite_program = GL43EffectHelpers::link_vf_program(GL43EffectHelpers::get_fullscreen_vert_glsl(), VOLUMETRICS_COMPOSITE_FRAG_GLSL, "VolumetricFog", "Composite");

	if (!depth_program || !raymarch_program || !temporal_program || !denoise_program || !finalize_program || !composite_program) {
		return false;
	}

	uint8_t white[4] = { 255, 255, 255, 255 };
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &dummy_gl_2d);
	glBindTexture(GL_TEXTURE_2D, dummy_gl_2d);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenTextures(1, &dummy_gl_3d);
	glBindTexture(GL_TEXTURE_3D, dummy_gl_3d);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, 1, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_3D, 0);

	glGenBuffers(1, &ssbo_dir_lights);
	glGenBuffers(1, &ssbo_spot_lights);
	glGenBuffers(1, &ssbo_omni_lights);
	glGenBuffers(1, &ssbo_local_vols);
	glGenBuffers(1, &ssbo_baked_vols);

	gl_ready = true;
	return true;
}

void VolumetricFog::_cleanup_gl_resources() {
	if (!gl_ready || gl_thread_id != Thread::get_caller_id()) {
		return;
	}

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	for (auto &kv : mesh_cache) {
		if (kv.second.gl_vao) glDeleteVertexArrays(1, &kv.second.gl_vao);
		if (kv.second.gl_vbo) glDeleteBuffers(1, &kv.second.gl_vbo);
	}
	mesh_cache.clear();

	for (auto &kv : skinned_gl_cache) {
		if (kv.second.gl_vao) glDeleteVertexArrays(1, &kv.second.gl_vao);
		if (kv.second.gl_vbo) glDeleteBuffers(1, &kv.second.gl_vbo);
	}
	skinned_gl_cache.clear();

	for (size_t i = 0; i + 1 < gl_buffers_to_delete.size(); i += 2) {
		if (gl_buffers_to_delete[i]) glDeleteVertexArrays(1, &gl_buffers_to_delete[i]);
		if (gl_buffers_to_delete[i + 1]) glDeleteBuffers(1, &gl_buffers_to_delete[i + 1]);
	}
	gl_buffers_to_delete.clear();

	if (depth_program) { glDeleteProgram(depth_program); depth_program = 0; }
	if (raymarch_program) { glDeleteProgram(raymarch_program); raymarch_program = 0; }
	if (temporal_program) { glDeleteProgram(temporal_program); temporal_program = 0; }
	if (denoise_program) { glDeleteProgram(denoise_program); denoise_program = 0; }
	if (finalize_program) { glDeleteProgram(finalize_program); finalize_program = 0; }
	if (composite_program) { glDeleteProgram(composite_program); composite_program = 0; }

	for (auto &kv : viewport_states) {
		ViewportRenderState &st = kv.second;
		if (st.cam_depth_fbo) { glDeleteFramebuffers(1, &st.cam_depth_fbo); st.cam_depth_fbo = 0; }
		if (st.cam_depth_tex) { glDeleteTextures(1, &st.cam_depth_tex); st.cam_depth_tex = 0; }
		if (st.fog_work_a) { glDeleteTextures(1, &st.fog_work_a); st.fog_work_a = 0; }
		if (st.fog_work_b) { glDeleteTextures(1, &st.fog_work_b); st.fog_work_b = 0; }
		if (st.fog_history_tex[0]) { glDeleteTextures(1, &st.fog_history_tex[0]); st.fog_history_tex[0] = 0; }
		if (st.fog_history_tex[1]) { glDeleteTextures(1, &st.fog_history_tex[1]); st.fog_history_tex[1] = 0; }
		if (st.prev_linear_depth_tex) { glDeleteTextures(1, &st.prev_linear_depth_tex); st.prev_linear_depth_tex = 0; }
	}

	if (dir_depth_fbo) { glDeleteFramebuffers(1, &dir_depth_fbo); dir_depth_fbo = 0; }
	if (dir_depth_array) { glDeleteTextures(1, &dir_depth_array); dir_depth_array = 0; }
	if (spot_depth_fbo) { glDeleteFramebuffers(1, &spot_depth_fbo); spot_depth_fbo = 0; }
	if (spot_depth_array) { glDeleteTextures(1, &spot_depth_array); spot_depth_array = 0; }

	if (dummy_gl_2d) { glDeleteTextures(1, &dummy_gl_2d); dummy_gl_2d = 0; }
	if (dummy_gl_3d) { glDeleteTextures(1, &dummy_gl_3d); dummy_gl_3d = 0; }

	if (ssbo_dir_lights) { glDeleteBuffers(1, &ssbo_dir_lights); ssbo_dir_lights = 0; }
	if (ssbo_spot_lights) { glDeleteBuffers(1, &ssbo_spot_lights); ssbo_spot_lights = 0; }
	if (ssbo_omni_lights) { glDeleteBuffers(1, &ssbo_omni_lights); ssbo_omni_lights = 0; }
	if (ssbo_local_vols) { glDeleteBuffers(1, &ssbo_local_vols); ssbo_local_vols = 0; }
	if (ssbo_baked_vols) { glDeleteBuffers(1, &ssbo_baked_vols); ssbo_baked_vols = 0; }

	gl_ready = false;
	gl_attempted = false;
}

void VolumetricFog::_ensure_cam_depth_size(ViewportRenderState &r_state, int p_w, int p_h) {
	if (r_state.cam_depth_fbo != 0 && r_state.cam_depth_w == p_w && r_state.cam_depth_h == p_h) return;
	r_state.cam_depth_w = p_w;
	r_state.cam_depth_h = p_h;

	if (r_state.cam_depth_fbo == 0) {
		glGenFramebuffers(1, &r_state.cam_depth_fbo);
	}

	glActiveTexture(GL_TEXTURE0);
	GL43EffectHelpers::ensure_depth24_texture(r_state.cam_depth_tex, p_w, p_h);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, r_state.cam_depth_fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, r_state.cam_depth_tex, 0);
	glDrawBuffers(0, nullptr);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void VolumetricFog::_ensure_dir_depth_array(int p_res) {
	if (dir_depth_fbo != 0 && dir_depth_res_alloc == p_res) return;
	dir_depth_res_alloc = p_res;
	dir_depth_force_refresh = true;

	if (dir_depth_fbo == 0) {
		glGenFramebuffers(1, &dir_depth_fbo);
		glGenTextures(1, &dir_depth_array);
	}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D_ARRAY, dir_depth_array);
	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT24, p_res, p_res, MAX_DIR_LIGHTS, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
}

void VolumetricFog::_ensure_spot_depth_array(int p_res) {
	if (spot_depth_fbo != 0 && spot_depth_res_alloc == p_res) return;
	spot_depth_res_alloc = p_res;
	spot_depth_force_refresh = true;

	if (spot_depth_fbo == 0) {
		glGenFramebuffers(1, &spot_depth_fbo);
		glGenTextures(1, &spot_depth_array);
	}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D_ARRAY, spot_depth_array);
	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT24, p_res, p_res, MAX_SPOT_LIGHTS, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
}

void VolumetricFog::_ensure_work_textures(ViewportRenderState &r_state, int p_w, int p_h) {
	if (r_state.work_w == p_w && r_state.work_h == p_h && r_state.fog_work_a != 0) return;
	r_state.work_w = p_w;
	r_state.work_h = p_h;
	r_state.history_valid = false;

	glActiveTexture(GL_TEXTURE0);
	GL43EffectHelpers::ensure_rgba16f_texture(r_state.fog_work_a, p_w, p_h, GL_LINEAR);
	GL43EffectHelpers::ensure_rgba16f_texture(r_state.fog_work_b, p_w, p_h, GL_LINEAR);
	GL43EffectHelpers::ensure_rgba16f_texture(r_state.fog_history_tex[0], p_w, p_h, GL_LINEAR);
	GL43EffectHelpers::ensure_rgba16f_texture(r_state.fog_history_tex[1], p_w, p_h, GL_LINEAR);

	if (r_state.prev_linear_depth_tex == 0) glGenTextures(1, &r_state.prev_linear_depth_tex);
	glBindTexture(GL_TEXTURE_2D, r_state.prev_linear_depth_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, p_w, p_h, 0, GL_RED, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void VolumetricFog::_on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx) {
	if (!enabled || !auto_composite || !is_inside_tree() || !is_visible_in_tree() || !gl_ready || composite_program == 0) {
		return;
	}
	if (Engine::get_singleton()->is_editor_hint() && !editor_preview) {
		return;
	}

	RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
	if (!rs) return;

	RID fog_rid;
	RID low_depth_rid;
	{
		MutexLock lock(scene_mutex);
		if (!snapshot.valid) return;
		for (const auto &kv : viewport_states) {
			if (kv.second.color_tex_rid == p_ctx.color_tex_rid && kv.second.active_composite && kv.second.fog_texture.is_valid() && kv.second.low_depth_texture.is_valid()) {
				fog_rid = kv.second.fog_texture->get_rid();
				low_depth_rid = kv.second.low_depth_texture->get_rid();
				break;
			}
		}
	}
	if (!fog_rid.is_valid() || !low_depth_rid.is_valid()) return;

	RasterizerStorageGLES3::Texture *godot_fog_tex = rs->texture_owner.getornull(fog_rid);
	RasterizerStorageGLES3::Texture *godot_low_depth_tex = rs->texture_owner.getornull(low_depth_rid);
	if (!godot_fog_tex || godot_fog_tex->tex_id == 0 || !godot_low_depth_tex || godot_low_depth_tex->tex_id == 0) return;

	glUseProgram(composite_program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, godot_fog_tex->tex_id);
	glUniform1i(glGetUniformLocation(composite_program, "uFogTex"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, godot_low_depth_tex->tex_id);
	glUniform1i(glGetUniformLocation(composite_program, "uLowDepthTex"), 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, p_ctx.depth_tex_id);
	glUniform1i(glGetUniformLocation(composite_program, "uDepthTex"), 2);

	glUniform1i(glGetUniformLocation(composite_program, "uBilateralUpsample"), (bilateral_upsample && resolution_scale < 0.99f) ? 1 : 0);
	glUniformMatrix4fv(glGetUniformLocation(composite_program, "uInvProj"), 1, GL_FALSE, p_ctx.inv_proj_mat);

	glEnable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	glBlendFuncSeparate(GL_ONE, GL_SRC_ALPHA, GL_ZERO, GL_ONE);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glDisable(GL_BLEND);
}

void VolumetricFog::_on_frame_pre_draw() {
	if (!hook_registered || !is_inside_tree()) return;

	bool in_editor = Engine::get_singleton()->is_editor_hint();
	if (!enabled || !is_visible_in_tree() || (in_editor && !editor_preview) || !get_world().is_valid()) {
		_detach_all_viewports();
		return;
	}

	if (in_editor) {
		_collect_scene_snapshot();
	}

	RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
	if (!rs) return;

	MutexLock lock(scene_mutex);
	if (!snapshot.valid || snapshot.viewports.empty()) return;

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	if (!_init_gl_resources()) {
		return;
	}

	for (size_t i = 0; i + 1 < gl_buffers_to_delete.size(); i += 2) {
		if (gl_buffers_to_delete[i]) glDeleteVertexArrays(1, &gl_buffers_to_delete[i]);
		if (gl_buffers_to_delete[i + 1]) glDeleteBuffers(1, &gl_buffers_to_delete[i + 1]);
	}
	gl_buffers_to_delete.clear();

	glDisable(GL_FRAMEBUFFER_SRGB);

	frame_counter++;

	// Upload new static meshes to VAO/VBO
	for (auto &kv : mesh_cache) {
		MeshGeom &g = kv.second;
		if (!g.gpu_uploaded && !g.verts.empty()) {
			glGenVertexArrays(1, &g.gl_vao);
			glGenBuffers(1, &g.gl_vbo);
			glBindVertexArray(g.gl_vao);
			glBindBuffer(GL_ARRAY_BUFFER, g.gl_vbo);
			glBufferData(GL_ARRAY_BUFFER, g.verts.size() * sizeof(float), g.verts.data(), GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
			g.gpu_uploaded = true;
		}
	}

	// Upload skinned meshes to dynamic VAO/VBO
	for (size_t s = 0; s < snapshot.skinned_instances.size(); s++) {
		const SkinnedSnapshot &sk = snapshot.skinned_instances[s];
		SkinnedMeshGL &sgl = skinned_gl_cache[sk.obj_id];
		if (sgl.gl_vao == 0) {
			glGenVertexArrays(1, &sgl.gl_vao);
			glGenBuffers(1, &sgl.gl_vbo);
			glBindVertexArray(sgl.gl_vao);
			glBindBuffer(GL_ARRAY_BUFFER, sgl.gl_vbo);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
		} else {
			glBindBuffer(GL_ARRAY_BUFFER, sgl.gl_vbo);
		}
		glBufferData(GL_ARRAY_BUFFER, sk.posed_verts.size() * sizeof(float), sk.posed_verts.data(), GL_DYNAMIC_DRAW);
		sgl.vertex_count = sk.vertex_count;
		sgl.last_frame_seen = frame_counter;
	}

	for (auto it = skinned_gl_cache.begin(); it != skinned_gl_cache.end(); ) {
		if (frame_counter - it->second.last_frame_seen > 60) {
			if (it->second.gl_vao) glDeleteVertexArrays(1, &it->second.gl_vao);
			if (it->second.gl_vbo) glDeleteBuffers(1, &it->second.gl_vbo);
			it = skinned_gl_cache.erase(it);
		} else {
			++it;
		}
	}

	// =========================================================
	// 1. SHARED LIGHT DEPTH RASTERIZATION PASSES (Directional + Spot)
	// =========================================================
	_ensure_dir_depth_array(directional_depth_resolution);
	_ensure_spot_depth_array(spot_depth_resolution);

	glUseProgram(depth_program);
	glFrontFace(GL_CW);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glDepthMask(GL_TRUE);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glClearDepth(1.0);

	GLint loc_mvp = glGetUniformLocation(depth_program, "uMVP");
	GLint loc_alb_tex = glGetUniformLocation(depth_program, "uAlbedoTex");
	GLint loc_use_scissor = glGetUniformLocation(depth_program, "uUseAlphaScissor");
	GLint loc_alpha_thresh = glGetUniformLocation(depth_program, "uAlphaThreshold");
	glUniform1i(loc_alb_tex, 0);

	auto draw_depth_instance = [&](const DepthInstance &inst, const float vp_mat[16], bool is_shadow_pass) {
		GLuint vao = 0;
		if (inst.is_skinned) {
			const SkinnedSnapshot &sk = snapshot.skinned_instances[inst.skinned_snap_idx];
			vao = skinned_gl_cache[sk.obj_id].gl_vao;
		} else {
			auto it = mesh_cache.find(inst.mesh_id);
			if (it == mesh_cache.end() || !it->second.gpu_uploaded) return;
			vao = it->second.gl_vao;
		}
		if (vao == 0) return;

		float mvp[16];
		GL43EffectHelpers::mat4_mul(vp_mat, inst.model_mat, mvp);
		glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp);
		glBindVertexArray(vao);

		for (size_t s = 0; s < inst.surfaces.size(); s++) {
			const SurfaceRange &sr = inst.surfaces[s];
			if (sr.vert_count <= 0) continue;

			if ((is_shadow_pass && inst.double_sided_shadow) || sr.cull_mode == 2) {
				glDisable(GL_CULL_FACE);
			} else {
				glEnable(GL_CULL_FACE);
				bool cull_front = (sr.cull_mode == 1);
				if (inst.flip_facing) cull_front = !cull_front;
				glCullFace(cull_front ? GL_FRONT : GL_BACK);
			}

			bool do_scissor = false;
			if (sr.alpha_scissor && sr.albedo_tex_rid.is_valid()) {
				RasterizerStorageGLES3::Texture *gt = rs->texture_owner.getornull(sr.albedo_tex_rid);
				if (gt) gt = gt->get_ptr();
				if (gt && gt->tex_id != 0) {
					glActiveTexture(GL_TEXTURE0);
					glBindTexture(GL_TEXTURE_2D, gt->tex_id);
					glUniform1i(loc_use_scissor, 1);
					glUniform1f(loc_alpha_thresh, sr.alpha_threshold);
					do_scissor = true;
				}
			}
			if (!do_scissor) {
				glUniform1i(loc_use_scissor, 0);
			}

			glDrawArrays(GL_TRIANGLES, sr.first_vert, sr.vert_count);
		}
	};

	// 1B. Directional Light Depth Passes
	dir_shadow_frame++;
	bool update_dir = dir_depth_force_refresh || (directional_update_interval <= 1) || ((dir_shadow_frame % (uint64_t)directional_update_interval) == 0);
	if (update_dir && !snapshot.dir_meta.empty()) {
		dir_depth_force_refresh = false;
		glEnable(GL_DEPTH_CLAMP);
		glBindFramebuffer(GL_FRAMEBUFFER, dir_depth_fbo);
		glViewport(0, 0, directional_depth_resolution, directional_depth_resolution);
		for (size_t d = 0; d < snapshot.dir_meta.size(); d++) {
			const DirShadowMeta &dm = snapshot.dir_meta[d];
			if (!dm.has_shadow) continue;
			glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, dir_depth_array, 0, (GLint)d);
			glDrawBuffers(0, nullptr);
			glClear(GL_DEPTH_BUFFER_BIT);
			for (size_t i = 0; i < snapshot.instances.size(); i++) {
				const DepthInstance &inst = snapshot.instances[i];
				if (!inst.cast_shadows || !(inst.layer_mask & dm.cull_mask)) continue;
				if (!inst.world_aabb.intersects(dm.frustum_aabb)) continue;
				draw_depth_instance(inst, dm.vp, true);
			}
		}
		glDisable(GL_DEPTH_CLAMP);
	}

	// 1C. Spot Light Depth Passes
	spot_shadow_frame++;
	bool update_spot = spot_depth_force_refresh || (spot_update_interval <= 1) || ((spot_shadow_frame % (uint64_t)spot_update_interval) == 0);
	if (update_spot && !snapshot.spot_meta.empty()) {
		spot_depth_force_refresh = false;
		glBindFramebuffer(GL_FRAMEBUFFER, spot_depth_fbo);
		glViewport(0, 0, spot_depth_resolution, spot_depth_resolution);
		for (size_t s = 0; s < snapshot.spot_meta.size(); s++) {
			const SpotShadowMeta &sm = snapshot.spot_meta[s];
			if (!sm.has_shadow) continue;
			glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, spot_depth_array, 0, (GLint)s);
			glDrawBuffers(0, nullptr);
			glClear(GL_DEPTH_BUFFER_BIT);
			AABB spot_box(sm.pos - Vector3(sm.range, sm.range, sm.range), Vector3(sm.range * 2.0f, sm.range * 2.0f, sm.range * 2.0f));
			for (size_t i = 0; i < snapshot.instances.size(); i++) {
				const DepthInstance &inst = snapshot.instances[i];
				if (!inst.cast_shadows || !(inst.layer_mask & sm.cull_mask)) continue;
				if (!inst.world_aabb.intersects(spot_box)) continue;
				draw_depth_instance(inst, sm.vp, true);
			}
		}
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

	// =========================================================
	// 2. UPLOAD SSBOs FOR COMPUTE SHADERS
	// =========================================================
	GL43EffectHelpers::upload_ssbo(ssbo_dir_lights, snapshot.dir_lights.data(), snapshot.dir_lights.size() * sizeof(GpuDirLight), sizeof(GpuDirLight));
	GL43EffectHelpers::upload_ssbo(ssbo_spot_lights, snapshot.spot_lights.data(), snapshot.spot_lights.size() * sizeof(GpuSpotLight), sizeof(GpuSpotLight));
	GL43EffectHelpers::upload_ssbo(ssbo_local_vols, snapshot.local_volumes.data(), snapshot.local_volumes.size() * sizeof(GpuLocalVolume), sizeof(GpuLocalVolume));
	GL43EffectHelpers::upload_ssbo(ssbo_baked_vols, snapshot.baked_volumes.data(), snapshot.baked_volumes.size() * sizeof(GpuBakedVolume), sizeof(GpuBakedVolume));

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_dir_lights);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_spot_lights);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo_omni_lights);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo_local_vols);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssbo_baked_vols);

	// =========================================================
	// 3. PER-VIEWPORT CAMERA DEPTH, RAYMARCH, TEMPORAL, DENOISE & FINALIZE
	// =========================================================
	for (size_t vi = 0; vi < snapshot.viewports.size(); vi++) {
		const ViewportSnapshot &vsnap = snapshot.viewports[vi];
		auto st_it = viewport_states.find(vsnap.vp_id);
		if (st_it == viewport_states.end()) continue;
		ViewportRenderState &vst = st_it->second;

		RasterizerStorageGLES3::Texture *godot_fog_tex = rs->texture_owner.getornull(vst.fog_texture->get_rid());
		RasterizerStorageGLES3::Texture *godot_depth_tex = rs->texture_owner.getornull(vst.low_depth_texture->get_rid());
		if (!godot_fog_tex || godot_fog_tex->tex_id == 0 || !godot_depth_tex || godot_depth_tex->tex_id == 0) continue;
		if (godot_fog_tex->width != vsnap.fog_w || godot_fog_tex->height != vsnap.fog_h) continue;
		if (godot_depth_tex->width != vsnap.fog_w || godot_depth_tex->height != vsnap.fog_h) continue;

		// Transform omni lights into this viewport's camera view space
		std::vector<GpuOmniLight> vp_omnis = snapshot.omni_lights;
		for (size_t oi = 0; oi < vp_omnis.size(); oi++) {
			const float *p = vp_omnis[oi].pos_range;
			const float *vm = vsnap.view_mat;
			vp_omnis[oi].view_atten[0] = vm[0] * p[0] + vm[4] * p[1] + vm[8] * p[2] + vm[12];
			vp_omnis[oi].view_atten[1] = vm[1] * p[0] + vm[5] * p[1] + vm[9] * p[2] + vm[13];
			vp_omnis[oi].view_atten[2] = vm[2] * p[0] + vm[6] * p[1] + vm[10] * p[2] + vm[14];
		}
		GL43EffectHelpers::upload_ssbo(ssbo_omni_lights, vp_omnis.data(), vp_omnis.size() * sizeof(GpuOmniLight), sizeof(GpuOmniLight));

		// 3A. Camera Depth Pass for this viewport
		_ensure_cam_depth_size(vst, vsnap.fog_w, vsnap.fog_h);
		glUseProgram(depth_program);
		glFrontFace(GL_CW);
		glDisable(GL_SCISSOR_TEST);
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDepthMask(GL_TRUE);
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
		glClearDepth(1.0);
		glBindFramebuffer(GL_FRAMEBUFFER, vst.cam_depth_fbo);
		glViewport(0, 0, vsnap.fog_w, vsnap.fog_h);
		glClear(GL_DEPTH_BUFFER_BIT);
		for (size_t i = 0; i < snapshot.instances.size(); i++) {
			const DepthInstance &inst = snapshot.instances[i];
			if (!inst.draw_in_cam) continue;
			if (!GL43EffectHelpers::aabb_in_frustum(inst.world_aabb, vsnap.cam_frustum_planes)) continue;
			draw_depth_instance(inst, vsnap.vp_mat, false);
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

		// 3B. Raymarching Compute Pass
		_ensure_work_textures(vst, vsnap.fog_w, vsnap.fog_h);

		GLuint gx = (GLuint)((vsnap.fog_w + 7) / 8);
		GLuint gy = (GLuint)((vsnap.fog_h + 7) / 8);

		SharedGL43Context *gl43_ctx = SharedGL43Context::get_singleton();

		glUseProgram(raymarch_program);
		gl43_ctx->bind_image_texture(0, vst.fog_work_a, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
		gl43_ctx->bind_image_texture(1, godot_depth_tex->tex_id, 0, false, 0, GL_WRITE_ONLY, GL_R32F);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, vst.cam_depth_tex);
		glUniform1i(glGetUniformLocation(raymarch_program, "uCamDepthTex"), 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D_ARRAY, dir_depth_array);
		glUniform1i(glGetUniformLocation(raymarch_program, "uDirDepthArray"), 1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D_ARRAY, spot_depth_array);
		glUniform1i(glGetUniformLocation(raymarch_program, "uSpotDepthArray"), 2);

		int dir0_has_proj = 0;
		int dir0_proj_to_lin = 0;
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, dummy_gl_2d);
		if (!snapshot.dir_meta.empty() && snapshot.dir_meta[0].proj_tex_rid.is_valid()) {
			RasterizerStorageGLES3::Texture *pt = rs->texture_owner.getornull(snapshot.dir_meta[0].proj_tex_rid);
			if (pt) pt = pt->get_ptr();
			if (pt && pt->tex_id != 0) {
				dir0_has_proj = 1;
				glBindTexture(GL_TEXTURE_2D, pt->tex_id);
				GLint int_fmt = 0;
				glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &int_fmt);
				bool is_hdr_fmt = (int_fmt == 0x881A || int_fmt == 0x881B || int_fmt == 0x8814 || int_fmt == 0x8815);
				dir0_proj_to_lin = (!GL43EffectHelpers::is_srgb_gl_format((int32_t)int_fmt) && !is_hdr_fmt) ? 1 : 0;
			}
		}
		glUniform1i(glGetUniformLocation(raymarch_program, "uDirProj0"), 3);
		glUniform1i(glGetUniformLocation(raymarch_program, "uDir0HasProj"), dir0_has_proj);
		glUniform1i(glGetUniformLocation(raymarch_program, "uDir0ProjToLinear"), dir0_proj_to_lin);

		for (int b = 0; b < MAX_BAKED_VOLUMES; b++) {
			glActiveTexture(GL_TEXTURE4 + b);
			GLuint tex3d = dummy_gl_3d;
			if (b < (int)snapshot.baked_tex_rids.size() && snapshot.baked_tex_rids[b].is_valid()) {
				RasterizerStorageGLES3::Texture *bt = rs->texture_owner.getornull(snapshot.baked_tex_rids[b]);
				if (bt) bt = bt->get_ptr();
				if (bt && bt->tex_id != 0) {
					tex3d = bt->tex_id;
				}
			}
			glBindTexture(GL_TEXTURE_3D, tex3d);
			char uname[16];
			snprintf(uname, sizeof(uname), "uBakedTex%d", b);
			glUniform1i(glGetUniformLocation(raymarch_program, uname), 4 + b);
		}

		float cam_mat[16];
		GL43EffectHelpers::mat4_mul(vsnap.inv_vp_mat, vsnap.proj_mat, cam_mat);
		glUniformMatrix4fv(glGetUniformLocation(raymarch_program, "uProj"), 1, GL_FALSE, vsnap.proj_mat);
		glUniformMatrix4fv(glGetUniformLocation(raymarch_program, "uInvProj"), 1, GL_FALSE, vsnap.inv_proj_mat);
		glUniformMatrix4fv(glGetUniformLocation(raymarch_program, "uCamMatrix"), 1, GL_FALSE, cam_mat);

		glUniform1i(glGetUniformLocation(raymarch_program, "uSteps"), vsnap.effective_steps);
		glUniform1f(glGetUniformLocation(raymarch_program, "uMaxDistance"), vsnap.eff_max_distance);
		glUniform1f(glGetUniformLocation(raymarch_program, "uDistanceFade"), distance_fade);
		glUniform1i(glGetUniformLocation(raymarch_program, "uStepDistribution"), (int)step_distribution);
		glUniform1f(glGetUniformLocation(raymarch_program, "uJitter"), temporal_reprojection ? jitter : (jitter * 0.4f));
		float temp_jit = temporal_reprojection ? Math::fmod((float)frame_counter * 0.61803398875f, 1.0f) : 0.0f;
		glUniform1f(glGetUniformLocation(raymarch_program, "uTemporalJitter"), temp_jit);
		glUniform1f(glGetUniformLocation(raymarch_program, "uSkyAffect"), sky_affect);

		Color alb_lin = albedo.to_linear();
		Color emi_lin = emission.to_linear();
		glUniform1f(glGetUniformLocation(raymarch_program, "uGlobalDensity"), density);
		glUniform3f(glGetUniformLocation(raymarch_program, "uGlobalAlbedo"), alb_lin.r, alb_lin.g, alb_lin.b);
		glUniform3f(glGetUniformLocation(raymarch_program, "uGlobalEmission"), emi_lin.r * emission_energy, emi_lin.g * emission_energy, emi_lin.b * emission_energy);
		glUniform3f(glGetUniformLocation(raymarch_program, "uAmbientColor"), vsnap.amb_color[0], vsnap.amb_color[1], vsnap.amb_color[2]);
		glUniform1f(glGetUniformLocation(raymarch_program, "uAnisotropy"), anisotropy);
		glUniform1f(glGetUniformLocation(raymarch_program, "uLightFogRatio"), light_fog_ratio);

		glUniform1i(glGetUniformLocation(raymarch_program, "uHeightFogEnabled"), height_fog_enabled ? 1 : 0);
		glUniform1f(glGetUniformLocation(raymarch_program, "uHeightFogBase"), height_fog_base_height);
		glUniform1f(glGetUniformLocation(raymarch_program, "uHeightFogFalloff"), height_fog_falloff);

		glUniform1i(glGetUniformLocation(raymarch_program, "uNoiseEnabled"), noise_enabled ? 1 : 0);
		glUniform1f(glGetUniformLocation(raymarch_program, "uNoiseScale"), noise_scale);
		glUniform1f(glGetUniformLocation(raymarch_program, "uNoiseStrength"), noise_strength);
		Vector3 noff = noise_wind * noise_time;
		glUniform3f(glGetUniformLocation(raymarch_program, "uNoiseOffset"), (float)noff.x, (float)noff.y, (float)noff.z);

		glUniform1i(glGetUniformLocation(raymarch_program, "uFocusActive"), vsnap.focus_active ? 1 : 0);
		glUniform3f(glGetUniformLocation(raymarch_program, "uFocusPos"), vsnap.focus_pos[0], vsnap.focus_pos[1], vsnap.focus_pos[2]);
		glUniform1f(glGetUniformLocation(raymarch_program, "uFocusRadius"), vsnap.focus_radius);
		glUniform1f(glGetUniformLocation(raymarch_program, "uFocusFalloff"), vsnap.focus_falloff);
		glUniform1f(glGetUniformLocation(raymarch_program, "uFocusDensityMult"), vsnap.focus_density_mult);

		glUniform1i(glGetUniformLocation(raymarch_program, "uDirCount"), (int)snapshot.dir_lights.size());
		glUniform1i(glGetUniformLocation(raymarch_program, "uDirRes"), directional_depth_resolution);
		glUniform1i(glGetUniformLocation(raymarch_program, "uSpotCount"), (int)snapshot.spot_lights.size());
		glUniform1f(glGetUniformLocation(raymarch_program, "uSpotNearPlane"), spot_near_plane);
		glUniform1i(glGetUniformLocation(raymarch_program, "uSpotRes"), spot_depth_resolution);

		glUniform1i(glGetUniformLocation(raymarch_program, "uOmniCount"), (int)vp_omnis.size());
		glUniform1i(glGetUniformLocation(raymarch_program, "uOmniSSShadows"), omni_screen_space_shadows ? 1 : 0);
		glUniform1i(glGetUniformLocation(raymarch_program, "uOmniSSSteps"), omni_ss_shadow_steps);
		glUniform1f(glGetUniformLocation(raymarch_program, "uOmniSSThickness"), omni_ss_shadow_thickness);
		glUniform1f(glGetUniformLocation(raymarch_program, "uOmniSSSoftness"), omni_ss_shadow_softness);
		glUniform1f(glGetUniformLocation(raymarch_program, "uOmniSSStrength"), omni_ss_shadow_strength);

		glUniform1i(glGetUniformLocation(raymarch_program, "uLocalCount"), (int)snapshot.local_volumes.size());
		glUniform1i(glGetUniformLocation(raymarch_program, "uBakedCount"), (int)snapshot.baked_volumes.size());

		gl43_ctx->dispatch_compute(gx, gy, 1);
		gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

		GLuint cur_fog_tex = vst.fog_work_a;

		// 3C. Temporal Reprojection Compute Pass
		if (temporal_reprojection) {
			int write_idx = vst.history_idx;
			int read_idx = 1 - vst.history_idx;

			glUseProgram(temporal_program);
			gl43_ctx->bind_image_texture(0, vst.fog_history_tex[write_idx], 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
			gl43_ctx->bind_image_texture(1, vst.prev_linear_depth_tex, 0, false, 0, GL_WRITE_ONLY, GL_R32F);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, cur_fog_tex);
			glUniform1i(glGetUniformLocation(temporal_program, "uCurrentFog"), 0);

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, vst.history_valid ? vst.fog_history_tex[read_idx] : cur_fog_tex);
			glUniform1i(glGetUniformLocation(temporal_program, "uHistoryFog"), 1);

			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, vst.cam_depth_tex);
			glUniform1i(glGetUniformLocation(temporal_program, "uCamDepthTex"), 2);

			glActiveTexture(GL_TEXTURE3);
			glBindTexture(GL_TEXTURE_2D, godot_depth_tex->tex_id);
			glUniform1i(glGetUniformLocation(temporal_program, "uCurLinearDepth"), 3);

			glActiveTexture(GL_TEXTURE4);
			glBindTexture(GL_TEXTURE_2D, vst.prev_linear_depth_tex);
			glUniform1i(glGetUniformLocation(temporal_program, "uPrevLinearDepth"), 4);

			glUniformMatrix4fv(glGetUniformLocation(temporal_program, "uInvViewProj"), 1, GL_FALSE, vsnap.inv_vp_mat);
			glUniformMatrix4fv(glGetUniformLocation(temporal_program, "uPrevViewProj"), 1, GL_FALSE, vst.prev_vp_mat);
			glUniformMatrix4fv(glGetUniformLocation(temporal_program, "uPrevView"), 1, GL_FALSE, vst.prev_view_mat);
			glUniform1i(glGetUniformLocation(temporal_program, "uHistoryValid"), vst.history_valid ? 1 : 0);
			glUniform1f(glGetUniformLocation(temporal_program, "uBlend"), temporal_blend);

			gl43_ctx->dispatch_compute(gx, gy, 1);
			gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

			cur_fog_tex = vst.fog_history_tex[write_idx];
			std::memcpy(vst.prev_vp_mat, vsnap.vp_mat, sizeof(vst.prev_vp_mat));
			std::memcpy(vst.prev_view_mat, vsnap.view_mat, sizeof(vst.prev_view_mat));
			vst.history_valid = true;
			vst.history_idx = 1 - vst.history_idx;
		} else {
			vst.history_valid = false;
		}

		// 3D. Edge-Aware Spatial Denoise
		if (spatial_denoise_passes > 0) {
			glUseProgram(denoise_program);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, godot_depth_tex->tex_id);
			glUniform1i(glGetUniformLocation(denoise_program, "uLinearDepth"), 1);

			for (int p = 0; p < spatial_denoise_passes; p++) {
				GLuint dst_tex = (cur_fog_tex == vst.fog_work_b) ? vst.fog_work_a : vst.fog_work_b;
				gl43_ctx->bind_image_texture(0, dst_tex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, cur_fog_tex);
				glUniform1i(glGetUniformLocation(denoise_program, "uSrcFog"), 0);
				glUniform1i(glGetUniformLocation(denoise_program, "uStepSize"), 1 << p);

				gl43_ctx->dispatch_compute(gx, gy, 1);
				gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
				cur_fog_tex = dst_tex;
			}
		}

		// 3E. Finalize Pass (output pure linear HDR; Godot _post_process handles tonemapping & bloom)
		glUseProgram(finalize_program);
		gl43_ctx->bind_image_texture(0, godot_fog_tex->tex_id, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, cur_fog_tex);
		glUniform1i(glGetUniformLocation(finalize_program, "uSrcFog"), 0);
		glUniform1i(glGetUniformLocation(finalize_program, "uTonemapMode"), 0);
		glUniform1f(glGetUniformLocation(finalize_program, "uExposure"), 1.0f);
		glUniform1f(glGetUniformLocation(finalize_program, "uWhite"), 1.0f);
		glUniform1i(glGetUniformLocation(finalize_program, "uHdr"), vsnap.hdr ? 1 : 0);
		glUniform1i(glGetUniformLocation(finalize_program, "uKeep3DLinear"), 1);

		gl43_ctx->dispatch_compute(gx, gy, 1);
		gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
	}

	for (int i = 0; i < 5; i++) {
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)i, 0);
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	for (int i = 0; i < 8; i++) {
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
		glBindTexture(GL_TEXTURE_3D, 0);
	}
	glActiveTexture(GL_TEXTURE0);
	glBindVertexArray(0);
	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void VolumetricFog::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	if (!enabled) {
		_detach_all_viewports();
	}
}

void VolumetricFog::set_editor_preview(bool p_preview) {
	editor_preview = p_preview;
	if (!editor_preview && Engine::get_singleton()->is_editor_hint()) {
		_detach_all_viewports();
	}
}

void VolumetricFog::set_auto_composite(bool p_auto) {
	auto_composite = p_auto;
	if (!auto_composite) {
		_detach_all_viewports();
	}
}

void VolumetricFog::set_density(float p_density) {
	density = MAX(p_density, 0.0f);
}

void VolumetricFog::set_emission_energy(float p_energy) {
	emission_energy = MAX(p_energy, 0.0f);
}

void VolumetricFog::set_anisotropy(float p_anisotropy) {
	anisotropy = CLAMP(p_anisotropy, -0.9f, 0.9f);
}

void VolumetricFog::set_light_fog_ratio(float p_ratio) {
	light_fog_ratio = CLAMP(p_ratio, -10.0f, 10.0f);
}

void VolumetricFog::set_use_environment_ambient(bool p_use) {
	use_environment_ambient = p_use;
	_change_notify();
}

void VolumetricFog::set_ambient_light_energy(float p_energy) {
	ambient_light_energy = MAX(p_energy, 0.0f);
}

void VolumetricFog::set_sky_affect(float p_affect) {
	sky_affect = CLAMP(p_affect, 0.0f, 1.0f);
}

void VolumetricFog::set_height_fog_enabled(bool p_enable) {
	height_fog_enabled = p_enable;
	_change_notify();
}

void VolumetricFog::set_height_fog_falloff(float p_falloff) {
	height_fog_falloff = MAX(p_falloff, 0.0f);
}

void VolumetricFog::set_noise_enabled(bool p_enable) {
	noise_enabled = p_enable;
	_change_notify();
}

void VolumetricFog::set_noise_scale(float p_scale) {
	noise_scale = MAX(p_scale, 0.001f);
}

void VolumetricFog::set_noise_strength(float p_strength) {
	noise_strength = CLAMP(p_strength, 0.0f, 1.0f);
}

void VolumetricFog::set_steps(int p_steps) {
	steps = CLAMP(p_steps, 4, 128);
}

void VolumetricFog::set_editor_max_steps(int p_steps) {
	editor_max_steps = CLAMP(p_steps, 4, 64);
}

void VolumetricFog::set_max_distance(float p_distance) {
	max_distance = MAX(p_distance, 1.0f);
}

void VolumetricFog::set_distance_fade(float p_fade) {
	distance_fade = CLAMP(p_fade, 0.0f, 1.0f);
}

void VolumetricFog::set_jitter(float p_jitter) {
	jitter = CLAMP(p_jitter, 0.0f, 1.0f);
}

void VolumetricFog::set_resolution_scale(float p_scale) {
	float clamped = CLAMP(p_scale, 0.25f, 1.0f);
	if (Math::abs(clamped - resolution_scale) > 0.001f) {
		resolution_scale = clamped;
		for (auto &kv : viewport_states) {
			kv.second.history_valid = false;
		}
		_change_notify();
	}
}

void VolumetricFog::set_temporal_reprojection(bool p_enable) {
	temporal_reprojection = p_enable;
	for (auto &kv : viewport_states) {
		kv.second.history_valid = false;
	}
	_change_notify();
}

void VolumetricFog::set_temporal_blend(float p_blend) {
	temporal_blend = CLAMP(p_blend, 0.02f, 1.0f);
}

void VolumetricFog::set_spatial_denoise_passes(int p_passes) {
	spatial_denoise_passes = CLAMP(p_passes, 0, 5);
}

void VolumetricFog::set_directional_depth_resolution(int p_res) {
	directional_depth_resolution = CLAMP(p_res, 64, 4096);
}

void VolumetricFog::set_directional_shadow_distance(float p_dist) {
	directional_shadow_distance = MAX(p_dist, 2.0f);
}

void VolumetricFog::set_directional_depth_range(float p_range) {
	directional_depth_range = MAX(p_range, 5.0f);
}

void VolumetricFog::set_directional_depth_bias(float p_bias) {
	directional_depth_bias = CLAMP(p_bias, 0.0f, 0.1f);
}

void VolumetricFog::set_directional_update_interval(int p_interval) {
	directional_update_interval = CLAMP(p_interval, 1, 10);
}

void VolumetricFog::set_max_spot_lights(int p_max) {
	max_spot_lights = CLAMP(p_max, 0, (int)MAX_SPOT_LIGHTS);
}

void VolumetricFog::set_spot_depth_resolution(int p_res) {
	spot_depth_resolution = CLAMP(p_res, 64, 2048);
}

void VolumetricFog::set_spot_depth_bias(float p_bias) {
	spot_depth_bias = CLAMP(p_bias, 0.0f, 2.0f);
}

void VolumetricFog::set_spot_near_plane(float p_near) {
	spot_near_plane = CLAMP(p_near, 0.01f, 2.0f);
}

void VolumetricFog::set_spot_update_interval(int p_interval) {
	spot_update_interval = CLAMP(p_interval, 1, 10);
}

void VolumetricFog::set_max_omni_lights(int p_max) {
	max_omni_lights = CLAMP(p_max, 0, (int)MAX_OMNI_LIGHTS);
}

void VolumetricFog::set_omni_ss_shadow_steps(int p_steps) {
	omni_ss_shadow_steps = CLAMP(p_steps, 1, 16);
}

void VolumetricFog::set_omni_ss_shadow_thickness(float p_thickness) {
	omni_ss_shadow_thickness = MAX(p_thickness, 0.01f);
}

void VolumetricFog::set_omni_ss_shadow_softness(float p_softness) {
	omni_ss_shadow_softness = MAX(p_softness, 0.005f);
}

void VolumetricFog::set_omni_ss_shadow_strength(float p_strength) {
	omni_ss_shadow_strength = CLAMP(p_strength, 0.0f, 1.0f);
}
