#include "rtgi.h"
#include "rtgi_shaders.h"

#include "core/engine.h"
#include "core/math/math_funcs.h"
#include "core/project_settings.h"
#include "drivers/gles3/rasterizer_storage_gles3.h"
#include "drivers/gles3/shared_gl43_context.h"
#include "modules/modules_enabled.gen.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/multimesh_instance.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/resources/environment.h"
#include "scene/resources/sky.h"
#include "scene/resources/world.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual_server.h"

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

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
#ifndef GL_R32F
#define GL_R32F 0x822E
#endif
#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif
#ifndef GL_TEXTURE_3D
#define GL_TEXTURE_3D 0x806F
#endif

RTGI::RTGI() {
	gi_texture.instance();
	gi_albedo_texture.instance();
	gi_volume_texture.instance();
}

RTGI::~RTGI() {
	_free_vs_resources();
	_cleanup_gl_resources();
}

void RTGI::_bind_methods() {
	// Core
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &RTGI::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &RTGI::is_enabled);
	ClassDB::bind_method(D_METHOD("set_editor_preview", "preview"), &RTGI::set_editor_preview);
	ClassDB::bind_method(D_METHOD("get_editor_preview"), &RTGI::get_editor_preview);
	ClassDB::bind_method(D_METHOD("set_auto_composite", "auto_composite"), &RTGI::set_auto_composite);
	ClassDB::bind_method(D_METHOD("get_auto_composite"), &RTGI::get_auto_composite);
	ClassDB::bind_method(D_METHOD("set_gi_intensity", "intensity"), &RTGI::set_gi_intensity);
	ClassDB::bind_method(D_METHOD("get_gi_intensity"), &RTGI::get_gi_intensity);
	ClassDB::bind_method(D_METHOD("set_gi_cull_mask", "mask"), &RTGI::set_gi_cull_mask);
	ClassDB::bind_method(D_METHOD("get_gi_cull_mask"), &RTGI::get_gi_cull_mask);
	ClassDB::bind_method(D_METHOD("set_gi_viewing_cull_mask", "mask"), &RTGI::set_gi_viewing_cull_mask);
	ClassDB::bind_method(D_METHOD("get_gi_viewing_cull_mask"), &RTGI::get_gi_viewing_cull_mask);

	// Resolution & Denoising
	ClassDB::bind_method(D_METHOD("set_resolution_scale", "scale"), &RTGI::set_resolution_scale);
	ClassDB::bind_method(D_METHOD("get_resolution_scale"), &RTGI::get_resolution_scale);
	ClassDB::bind_method(D_METHOD("set_bilateral_upsample", "enable"), &RTGI::set_bilateral_upsample);
	ClassDB::bind_method(D_METHOD("get_bilateral_upsample"), &RTGI::get_bilateral_upsample);
	ClassDB::bind_method(D_METHOD("set_checkerboard_tracing", "enable"), &RTGI::set_checkerboard_tracing);
	ClassDB::bind_method(D_METHOD("get_checkerboard_tracing"), &RTGI::get_checkerboard_tracing);
	ClassDB::bind_method(D_METHOD("set_temporal_accumulation", "enable"), &RTGI::set_temporal_accumulation);
	ClassDB::bind_method(D_METHOD("get_temporal_accumulation"), &RTGI::get_temporal_accumulation);
	ClassDB::bind_method(D_METHOD("set_temporal_blend", "blend"), &RTGI::set_temporal_blend);
	ClassDB::bind_method(D_METHOD("get_temporal_blend"), &RTGI::get_temporal_blend);
	ClassDB::bind_method(D_METHOD("set_temporal_noise", "enable"), &RTGI::set_temporal_noise);
	ClassDB::bind_method(D_METHOD("get_temporal_noise"), &RTGI::get_temporal_noise);
	ClassDB::bind_method(D_METHOD("set_spatial_denoise_passes", "passes"), &RTGI::set_spatial_denoise_passes);
	ClassDB::bind_method(D_METHOD("get_spatial_denoise_passes"), &RTGI::get_spatial_denoise_passes);
	ClassDB::bind_method(D_METHOD("set_firefly_clamp", "clamp"), &RTGI::set_firefly_clamp);
	ClassDB::bind_method(D_METHOD("get_firefly_clamp"), &RTGI::get_firefly_clamp);

	// Tracing & Bounces
	ClassDB::bind_method(D_METHOD("set_rays_per_pixel", "rays"), &RTGI::set_rays_per_pixel);
	ClassDB::bind_method(D_METHOD("get_rays_per_pixel"), &RTGI::get_rays_per_pixel);
	ClassDB::bind_method(D_METHOD("set_max_bounces", "bounces"), &RTGI::set_max_bounces);
	ClassDB::bind_method(D_METHOD("get_max_bounces"), &RTGI::get_max_bounces);
	ClassDB::bind_method(D_METHOD("set_max_ray_distance", "distance"), &RTGI::set_max_ray_distance);
	ClassDB::bind_method(D_METHOD("get_max_ray_distance"), &RTGI::get_max_ray_distance);
	ClassDB::bind_method(D_METHOD("set_bounce_distance_falloff", "falloff"), &RTGI::set_bounce_distance_falloff);
	ClassDB::bind_method(D_METHOD("get_bounce_distance_falloff"), &RTGI::get_bounce_distance_falloff);
	ClassDB::bind_method(D_METHOD("set_bounce_energy_boost", "boost"), &RTGI::set_bounce_energy_boost);
	ClassDB::bind_method(D_METHOD("get_bounce_energy_boost"), &RTGI::get_bounce_energy_boost);
	ClassDB::bind_method(D_METHOD("set_bounce_throughput_threshold", "threshold"), &RTGI::set_bounce_throughput_threshold);
	ClassDB::bind_method(D_METHOD("get_bounce_throughput_threshold"), &RTGI::get_bounce_throughput_threshold);
	ClassDB::bind_method(D_METHOD("set_emissive_intensity", "intensity"), &RTGI::set_emissive_intensity);
	ClassDB::bind_method(D_METHOD("get_emissive_intensity"), &RTGI::get_emissive_intensity);
	ClassDB::bind_method(D_METHOD("set_bounce_local_lights", "enable"), &RTGI::set_bounce_local_lights);
	ClassDB::bind_method(D_METHOD("get_bounce_local_lights"), &RTGI::get_bounce_local_lights);
	ClassDB::bind_method(D_METHOD("set_normal_maps_in_bounces", "enable"), &RTGI::set_normal_maps_in_bounces);
	ClassDB::bind_method(D_METHOD("get_normal_maps_in_bounces"), &RTGI::get_normal_maps_in_bounces);
	ClassDB::bind_method(D_METHOD("set_respect_material_cull_mode", "enable"), &RTGI::set_respect_material_cull_mode);
	ClassDB::bind_method(D_METHOD("get_respect_material_cull_mode"), &RTGI::get_respect_material_cull_mode);
	ClassDB::bind_method(D_METHOD("set_skinned_meshes_in_gi", "enable"), &RTGI::set_skinned_meshes_in_gi);
	ClassDB::bind_method(D_METHOD("get_skinned_meshes_in_gi"), &RTGI::get_skinned_meshes_in_gi);
	ClassDB::bind_method(D_METHOD("set_ray_origin_bias", "bias"), &RTGI::set_ray_origin_bias);
	ClassDB::bind_method(D_METHOD("get_ray_origin_bias"), &RTGI::get_ray_origin_bias);

	// Sky & Irradiance
	ClassDB::bind_method(D_METHOD("set_sky_indirect_enabled", "enable"), &RTGI::set_sky_indirect_enabled);
	ClassDB::bind_method(D_METHOD("is_sky_indirect_enabled"), &RTGI::is_sky_indirect_enabled);
	ClassDB::bind_method(D_METHOD("set_sky_indirect_energy", "energy"), &RTGI::set_sky_indirect_energy);
	ClassDB::bind_method(D_METHOD("get_sky_indirect_energy"), &RTGI::get_sky_indirect_energy);
	ClassDB::bind_method(D_METHOD("set_custom_sky_mode", "mode"), &RTGI::set_custom_sky_mode);
	ClassDB::bind_method(D_METHOD("get_custom_sky_mode"), &RTGI::get_custom_sky_mode);
	ClassDB::bind_method(D_METHOD("set_custom_sky", "sky"), &RTGI::set_custom_sky);
	ClassDB::bind_method(D_METHOD("get_custom_sky"), &RTGI::get_custom_sky);
	ClassDB::bind_method(D_METHOD("set_custom_sky_color", "color"), &RTGI::set_custom_sky_color);
	ClassDB::bind_method(D_METHOD("get_custom_sky_color"), &RTGI::get_custom_sky_color);
	ClassDB::bind_method(D_METHOD("set_custom_sky_energy", "energy"), &RTGI::set_custom_sky_energy);
	ClassDB::bind_method(D_METHOD("get_custom_sky_energy"), &RTGI::get_custom_sky_energy);
	ClassDB::bind_method(D_METHOD("set_custom_sky_rotation", "rotation_degrees"), &RTGI::set_custom_sky_rotation);
	ClassDB::bind_method(D_METHOD("get_custom_sky_rotation"), &RTGI::get_custom_sky_rotation);
	ClassDB::bind_method(D_METHOD("set_ambient_bounce_approximation", "strength"), &RTGI::set_ambient_bounce_approximation);
	ClassDB::bind_method(D_METHOD("get_ambient_bounce_approximation"), &RTGI::get_ambient_bounce_approximation);

	// 3D Irradiance Volume
	ClassDB::bind_method(D_METHOD("set_volume_enabled", "enable"), &RTGI::set_volume_enabled);
	ClassDB::bind_method(D_METHOD("is_volume_enabled"), &RTGI::is_volume_enabled);
	ClassDB::bind_method(D_METHOD("set_volume_extents", "extents"), &RTGI::set_volume_extents);
	ClassDB::bind_method(D_METHOD("get_volume_extents"), &RTGI::get_volume_extents);
	ClassDB::bind_method(D_METHOD("set_volume_resolution", "resolution"), &RTGI::set_volume_resolution);
	ClassDB::bind_method(D_METHOD("get_volume_resolution"), &RTGI::get_volume_resolution);
	ClassDB::bind_method(D_METHOD("set_volume_rays_per_voxel", "rays"), &RTGI::set_volume_rays_per_voxel);
	ClassDB::bind_method(D_METHOD("get_volume_rays_per_voxel"), &RTGI::get_volume_rays_per_voxel);
	ClassDB::bind_method(D_METHOD("set_volume_ray_distance", "distance"), &RTGI::set_volume_ray_distance);
	ClassDB::bind_method(D_METHOD("get_volume_ray_distance"), &RTGI::get_volume_ray_distance);
	ClassDB::bind_method(D_METHOD("set_volume_feedback", "feedback"), &RTGI::set_volume_feedback);
	ClassDB::bind_method(D_METHOD("get_volume_feedback"), &RTGI::get_volume_feedback);
	ClassDB::bind_method(D_METHOD("set_volume_update_interval", "interval"), &RTGI::set_volume_update_interval);
	ClassDB::bind_method(D_METHOD("get_volume_update_interval"), &RTGI::get_volume_update_interval);

	// Material Alpha
	ClassDB::bind_method(D_METHOD("set_alpha_consideration", "enable"), &RTGI::set_alpha_consideration);
	ClassDB::bind_method(D_METHOD("get_alpha_consideration"), &RTGI::get_alpha_consideration);
	ClassDB::bind_method(D_METHOD("set_alpha_ray_transparency", "enable"), &RTGI::set_alpha_ray_transparency);
	ClassDB::bind_method(D_METHOD("get_alpha_ray_transparency"), &RTGI::get_alpha_ray_transparency);
	ClassDB::bind_method(D_METHOD("set_alpha_max_steps", "steps"), &RTGI::set_alpha_max_steps);
	ClassDB::bind_method(D_METHOD("get_alpha_max_steps"), &RTGI::get_alpha_max_steps);
	ClassDB::bind_method(D_METHOD("set_alpha_shadow_transmission", "enable"), &RTGI::set_alpha_shadow_transmission);
	ClassDB::bind_method(D_METHOD("get_alpha_shadow_transmission"), &RTGI::get_alpha_shadow_transmission);

	// Mesh Simplification
	ClassDB::bind_method(D_METHOD("set_mesh_simplification_enabled", "enable"), &RTGI::set_mesh_simplification_enabled);
	ClassDB::bind_method(D_METHOD("is_mesh_simplification_enabled"), &RTGI::is_mesh_simplification_enabled);
	ClassDB::bind_method(D_METHOD("set_mesh_simplification_mode", "mode"), &RTGI::set_mesh_simplification_mode);
	ClassDB::bind_method(D_METHOD("get_mesh_simplification_mode"), &RTGI::get_mesh_simplification_mode);
	ClassDB::bind_method(D_METHOD("set_mesh_simplification_ratio", "ratio"), &RTGI::set_mesh_simplification_ratio);
	ClassDB::bind_method(D_METHOD("get_mesh_simplification_ratio"), &RTGI::get_mesh_simplification_ratio);
	ClassDB::bind_method(D_METHOD("set_mesh_simplification_min_triangles", "min_triangles"), &RTGI::set_mesh_simplification_min_triangles);
	ClassDB::bind_method(D_METHOD("get_mesh_simplification_min_triangles"), &RTGI::get_mesh_simplification_min_triangles);
	ClassDB::bind_method(D_METHOD("set_mesh_simplification_max_error", "max_error"), &RTGI::set_mesh_simplification_max_error);
	ClassDB::bind_method(D_METHOD("get_mesh_simplification_max_error"), &RTGI::get_mesh_simplification_max_error);
	ClassDB::bind_method(D_METHOD("set_mesh_simplification_preserve_seams", "preserve"), &RTGI::set_mesh_simplification_preserve_seams);
	ClassDB::bind_method(D_METHOD("get_mesh_simplification_preserve_seams"), &RTGI::get_mesh_simplification_preserve_seams);
	ClassDB::bind_method(D_METHOD("set_mesh_simplification_skinned", "enable"), &RTGI::set_mesh_simplification_skinned);
	ClassDB::bind_method(D_METHOD("get_mesh_simplification_skinned"), &RTGI::get_mesh_simplification_skinned);
	ClassDB::bind_method(D_METHOD("set_mesh_simplification_gbuffer", "enable"), &RTGI::set_mesh_simplification_gbuffer);
	ClassDB::bind_method(D_METHOD("get_mesh_simplification_gbuffer"), &RTGI::get_mesh_simplification_gbuffer);

	// Shadows
	ClassDB::bind_method(D_METHOD("set_trace_shadows", "enable"), &RTGI::set_trace_shadows);
	ClassDB::bind_method(D_METHOD("get_trace_shadows"), &RTGI::get_trace_shadows);
	ClassDB::bind_method(D_METHOD("set_shadow_bounces", "bounces"), &RTGI::set_shadow_bounces);
	ClassDB::bind_method(D_METHOD("get_shadow_bounces"), &RTGI::get_shadow_bounces);
	ClassDB::bind_method(D_METHOD("set_shadow_max_distance", "distance"), &RTGI::set_shadow_max_distance);
	ClassDB::bind_method(D_METHOD("get_shadow_max_distance"), &RTGI::get_shadow_max_distance);
	ClassDB::bind_method(D_METHOD("set_shadow_ray_bias", "bias"), &RTGI::set_shadow_ray_bias);
	ClassDB::bind_method(D_METHOD("get_shadow_ray_bias"), &RTGI::get_shadow_ray_bias);

	// Scalability & Culling
	ClassDB::bind_method(D_METHOD("set_update_interval", "interval"), &RTGI::set_update_interval);
	ClassDB::bind_method(D_METHOD("get_update_interval"), &RTGI::get_update_interval);
	ClassDB::bind_method(D_METHOD("set_blas_builds_per_frame", "budget"), &RTGI::set_blas_builds_per_frame);
	ClassDB::bind_method(D_METHOD("get_blas_builds_per_frame"), &RTGI::get_blas_builds_per_frame);
	ClassDB::bind_method(D_METHOD("set_max_tlas_instances", "max_instances"), &RTGI::set_max_tlas_instances);
	ClassDB::bind_method(D_METHOD("get_max_tlas_instances"), &RTGI::get_max_tlas_instances);
	ClassDB::bind_method(D_METHOD("set_frustum_cull_gbuffer", "enable"), &RTGI::set_frustum_cull_gbuffer);
	ClassDB::bind_method(D_METHOD("get_frustum_cull_gbuffer"), &RTGI::get_frustum_cull_gbuffer);
	ClassDB::bind_method(D_METHOD("set_texture_array_resolution", "resolution"), &RTGI::set_texture_array_resolution);
	ClassDB::bind_method(D_METHOD("get_texture_array_resolution"), &RTGI::get_texture_array_resolution);
	ClassDB::bind_method(D_METHOD("set_max_texture_layers", "layers"), &RTGI::set_max_texture_layers);
	ClassDB::bind_method(D_METHOD("get_max_texture_layers"), &RTGI::get_max_texture_layers);
	ClassDB::bind_method(D_METHOD("set_texture_lod_bias", "bias"), &RTGI::set_texture_lod_bias);
	ClassDB::bind_method(D_METHOD("get_texture_lod_bias"), &RTGI::get_texture_lod_bias);
	ClassDB::bind_method(D_METHOD("set_distance_lod_strength", "strength"), &RTGI::set_distance_lod_strength);
	ClassDB::bind_method(D_METHOD("get_distance_lod_strength"), &RTGI::get_distance_lod_strength);
	ClassDB::bind_method(D_METHOD("set_mesh_cull_distance", "distance"), &RTGI::set_mesh_cull_distance);
	ClassDB::bind_method(D_METHOD("get_mesh_cull_distance"), &RTGI::get_mesh_cull_distance);
	ClassDB::bind_method(D_METHOD("set_mesh_cull_min_angle", "angle"), &RTGI::set_mesh_cull_min_angle);
	ClassDB::bind_method(D_METHOD("get_mesh_cull_min_angle"), &RTGI::get_mesh_cull_min_angle);
	ClassDB::bind_method(D_METHOD("set_light_cull_distance", "distance"), &RTGI::set_light_cull_distance);
	ClassDB::bind_method(D_METHOD("get_light_cull_distance"), &RTGI::get_light_cull_distance);
	ClassDB::bind_method(D_METHOD("set_debug_mode", "mode"), &RTGI::set_debug_mode);
	ClassDB::bind_method(D_METHOD("get_debug_mode"), &RTGI::get_debug_mode);

	ClassDB::bind_method(D_METHOD("apply_quality_preset", "preset"), &RTGI::apply_quality_preset);
	ClassDB::bind_method(D_METHOD("mark_geometry_dirty"), &RTGI::mark_geometry_dirty);
	ClassDB::bind_method(D_METHOD("reset_history"), &RTGI::reset_history);
	ClassDB::bind_method(D_METHOD("get_gi_texture"), &RTGI::get_gi_texture);
	ClassDB::bind_method(D_METHOD("get_gi_albedo_texture"), &RTGI::get_gi_albedo_texture);
	ClassDB::bind_method(D_METHOD("get_gi_volume_texture"), &RTGI::get_gi_volume_texture);
	ClassDB::bind_method(D_METHOD("_on_frame_pre_draw"), &RTGI::_on_frame_pre_draw);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_preview"), "set_editor_preview", "get_editor_preview");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_composite"), "set_auto_composite", "get_auto_composite");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "gi_intensity", PROPERTY_HINT_RANGE, "0.0,8.0,0.05,or_greater"), "set_gi_intensity", "get_gi_intensity");

	ADD_GROUP("Resolution & Denoising", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "resolution_scale", PROPERTY_HINT_RANGE, "0.15,1.0,0.05"), "set_resolution_scale", "get_resolution_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bilateral_upsample"), "set_bilateral_upsample", "get_bilateral_upsample");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "checkerboard_tracing"), "set_checkerboard_tracing", "get_checkerboard_tracing");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "temporal_accumulation"), "set_temporal_accumulation", "get_temporal_accumulation");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "temporal_blend", PROPERTY_HINT_RANGE, "0.02,1.0,0.02"), "set_temporal_blend", "get_temporal_blend");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "temporal_noise"), "set_temporal_noise", "get_temporal_noise");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spatial_denoise_passes", PROPERTY_HINT_RANGE, "0,5,1"), "set_spatial_denoise_passes", "get_spatial_denoise_passes");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "firefly_clamp", PROPERTY_HINT_RANGE, "0.0,50.0,0.5"), "set_firefly_clamp", "get_firefly_clamp");

	ADD_GROUP("Tracing & Bounces", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rays_per_pixel", PROPERTY_HINT_RANGE, "1,16,1"), "set_rays_per_pixel", "get_rays_per_pixel");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_bounces", PROPERTY_HINT_RANGE, "1,4,1"), "set_max_bounces", "get_max_bounces");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_ray_distance", PROPERTY_HINT_RANGE, "2.0,500.0,1.0"), "set_max_ray_distance", "get_max_ray_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce_distance_falloff", PROPERTY_HINT_RANGE, "0.1,1.0,0.05"), "set_bounce_distance_falloff", "get_bounce_distance_falloff");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce_energy_boost", PROPERTY_HINT_RANGE, "0.0,4.0,0.05"), "set_bounce_energy_boost", "get_bounce_energy_boost");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bounce_throughput_threshold", PROPERTY_HINT_RANGE, "0.0,0.25,0.005"), "set_bounce_throughput_threshold", "get_bounce_throughput_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "emissive_intensity", PROPERTY_HINT_RANGE, "0.0,8.0,0.05,or_greater"), "set_emissive_intensity", "get_emissive_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bounce_local_lights"), "set_bounce_local_lights", "get_bounce_local_lights");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "normal_maps_in_bounces"), "set_normal_maps_in_bounces", "get_normal_maps_in_bounces");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "respect_material_cull_mode"), "set_respect_material_cull_mode", "get_respect_material_cull_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "skinned_meshes_in_gi"), "set_skinned_meshes_in_gi", "get_skinned_meshes_in_gi");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ray_origin_bias", PROPERTY_HINT_RANGE, "0.0005,0.05,0.0005"), "set_ray_origin_bias", "get_ray_origin_bias");

	ADD_GROUP("Sky & Irradiance", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sky_indirect_enabled"), "set_sky_indirect_enabled", "is_sky_indirect_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "sky_indirect_energy", PROPERTY_HINT_RANGE, "0.0,8.0,0.05"), "set_sky_indirect_energy", "get_sky_indirect_energy");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "custom_sky_mode", PROPERTY_HINT_ENUM, "Environment,Custom Sky,Color"), "set_custom_sky_mode", "get_custom_sky_mode");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "custom_sky", PROPERTY_HINT_RESOURCE_TYPE, "Sky"), "set_custom_sky", "get_custom_sky");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "custom_sky_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_custom_sky_color", "get_custom_sky_color");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "custom_sky_energy", PROPERTY_HINT_RANGE, "0.0,16.0,0.05"), "set_custom_sky_energy", "get_custom_sky_energy");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "custom_sky_rotation"), "set_custom_sky_rotation", "get_custom_sky_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ambient_bounce_approximation", PROPERTY_HINT_RANGE, "0.0,1.0,0.05"), "set_ambient_bounce_approximation", "get_ambient_bounce_approximation");

	ADD_GROUP("3D Irradiance Volume", "volume_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "volume_enabled"), "set_volume_enabled", "is_volume_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "volume_extents"), "set_volume_extents", "get_volume_extents");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "volume_resolution"), "set_volume_resolution", "get_volume_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "volume_rays_per_voxel", PROPERTY_HINT_RANGE, "1,32,1"), "set_volume_rays_per_voxel", "get_volume_rays_per_voxel");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "volume_ray_distance", PROPERTY_HINT_RANGE, "2.0,200.0,1.0"), "set_volume_ray_distance", "get_volume_ray_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "volume_feedback", PROPERTY_HINT_RANGE, "0.0,1.0,0.02"), "set_volume_feedback", "get_volume_feedback");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "volume_update_interval", PROPERTY_HINT_RANGE, "1,10,1"), "set_volume_update_interval", "get_volume_update_interval");

	ADD_GROUP("Material Alpha", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alpha_consideration"), "set_alpha_consideration", "get_alpha_consideration");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alpha_ray_transparency"), "set_alpha_ray_transparency", "get_alpha_ray_transparency");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "alpha_max_steps", PROPERTY_HINT_RANGE, "1,16,1"), "set_alpha_max_steps", "get_alpha_max_steps");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alpha_shadow_transmission"), "set_alpha_shadow_transmission", "get_alpha_shadow_transmission");

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

	ADD_GROUP("Scalability & Culling", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "update_interval", PROPERTY_HINT_RANGE, "1,10,1"), "set_update_interval", "get_update_interval");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "blas_builds_per_frame", PROPERTY_HINT_RANGE, "0,64,1"), "set_blas_builds_per_frame", "get_blas_builds_per_frame");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tlas_instances", PROPERTY_HINT_RANGE, "0,8192,16"), "set_max_tlas_instances", "get_max_tlas_instances");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "frustum_cull_gbuffer"), "set_frustum_cull_gbuffer", "get_frustum_cull_gbuffer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_array_resolution", PROPERTY_HINT_ENUM, "128px:128,256px:256,512px:512,1024px:1024"), "set_texture_array_resolution", "get_texture_array_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_texture_layers", PROPERTY_HINT_RANGE, "8,256,8"), "set_max_texture_layers", "get_max_texture_layers");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "gi_cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_gi_cull_mask", "get_gi_cull_mask");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "gi_viewing_cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_gi_viewing_cull_mask", "get_gi_viewing_cull_mask");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "texture_lod_bias", PROPERTY_HINT_RANGE, "-2.0,6.0,0.1"), "set_texture_lod_bias", "get_texture_lod_bias");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "distance_lod_strength", PROPERTY_HINT_RANGE, "0.0,4.0,0.1"), "set_distance_lod_strength", "get_distance_lod_strength");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "mesh_cull_distance", PROPERTY_HINT_RANGE, "5.0,500.0,5.0"), "set_mesh_cull_distance", "get_mesh_cull_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "mesh_cull_min_angle", PROPERTY_HINT_RANGE, "0.0,0.1,0.002"), "set_mesh_cull_min_angle", "get_mesh_cull_min_angle");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "light_cull_distance", PROPERTY_HINT_RANGE, "10.0,500.0,5.0"), "set_light_cull_distance", "get_light_cull_distance");

	ADD_GROUP("Debug", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_mode", PROPERTY_HINT_ENUM, "None,Irradiance,GI Only,Receiver Albedo,3D Volume"), "set_debug_mode", "get_debug_mode");

	BIND_ENUM_CONSTANT(PRESET_LOW);
	BIND_ENUM_CONSTANT(PRESET_MEDIUM);
	BIND_ENUM_CONSTANT(PRESET_HIGH);
	BIND_ENUM_CONSTANT(PRESET_ULTRA);

	BIND_ENUM_CONSTANT(SKY_MODE_ENVIRONMENT);
	BIND_ENUM_CONSTANT(SKY_MODE_CUSTOM_SKY);
	BIND_ENUM_CONSTANT(SKY_MODE_COLOR);

	BIND_ENUM_CONSTANT(SIMPLIFY_QEM);
	BIND_ENUM_CONSTANT(SIMPLIFY_VERTEX_CLUSTERING);

	BIND_ENUM_CONSTANT(TEX_RES_128);
	BIND_ENUM_CONSTANT(TEX_RES_256);
	BIND_ENUM_CONSTANT(TEX_RES_512);
	BIND_ENUM_CONSTANT(TEX_RES_1024);

	BIND_ENUM_CONSTANT(DEBUG_NONE);
	BIND_ENUM_CONSTANT(DEBUG_IRRADIANCE);
	BIND_ENUM_CONSTANT(DEBUG_GI_ONLY);
	BIND_ENUM_CONSTANT(DEBUG_ALBEDO);
	BIND_ENUM_CONSTANT(DEBUG_VOLUME_3D);
}

void RTGI::_validate_property(PropertyInfo &property) const {
	if (property.name == "sky_indirect_energy" && !sky_indirect_enabled) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	} else if (property.name == "custom_sky" && custom_sky_mode != SKY_MODE_CUSTOM_SKY) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	} else if (property.name == "custom_sky_color" && custom_sky_mode != SKY_MODE_COLOR) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	} else if ((property.name == "custom_sky_energy" || property.name == "custom_sky_rotation") && custom_sky_mode == SKY_MODE_ENVIRONMENT) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	} else if (property.name == "temporal_blend" && !temporal_accumulation) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	} else if (!volume_enabled &&
			   (property.name == "volume_extents" ||
				property.name == "volume_resolution" ||
				property.name == "volume_rays_per_voxel" ||
				property.name == "volume_ray_distance" ||
				property.name == "volume_feedback" ||
				property.name == "volume_update_interval")) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	} else if (!alpha_consideration &&
			   (property.name == "alpha_ray_transparency" ||
				property.name == "alpha_max_steps" ||
				property.name == "alpha_shadow_transmission")) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	} else if (!trace_shadows &&
			   (property.name == "shadow_bounces" ||
				property.name == "shadow_max_distance" ||
				property.name == "shadow_ray_bias")) {
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

void RTGI::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_create_vs_resources();
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
			_free_vs_resources();
		} break;

		case NOTIFICATION_PROCESS: {
			if (!enabled || !is_inside_tree()) {
				_detach_all_viewports();
				MutexLock lock(scene_mutex);
				snapshot.valid = false;
				return;
			}
			if (!Engine::get_singleton()->is_editor_hint()) {
				process_frame_counter++;
				if (update_interval > 1 && (process_frame_counter % (uint64_t)update_interval) != 0) {
					return;
				}
				_collect_scene_snapshot();
			}
		} break;
	}
}

GL43Raytracer::SimplificationConfig RTGI::_get_simplification_config() const {
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

void RTGI::_create_vs_resources() {
	if (vs_resources_created || !VisualServer::get_singleton()) {
		return;
	}
	composite_shader.instance();
	composite_shader->set_code(RTGIShaders::COMPOSITE_CANVAS_SHADER);
	vs_resources_created = true;
}

void RTGI::_free_vs_resources() {
	if (!vs_resources_created) {
		return;
	}
	for (auto &kv : viewport_states) {
		kv.second.overlay.destroy();
		kv.second.history_valid = false;
	}
	viewport_states.clear();
	composite_shader.unref();
	vs_resources_created = false;
}

void RTGI::_detach_all_viewports() {
	for (auto &kv : viewport_states) {
		kv.second.overlay.detach();
		kv.second.history_valid = false;
	}
}

RTGI::ViewportRenderState &RTGI::_get_or_create_vp_state(Viewport *p_vp) {
	uint32_t key = p_vp->get_viewport_rid().get_id();
	auto it = viewport_states.find(key);
	if (it == viewport_states.end()) {
		ViewportRenderState st;
		st.gi_texture.instance();
		st.gi_albedo_texture.instance();
		st.gi_normal_texture.instance();
		it = viewport_states.emplace(key, std::move(st)).first;
	}
	it->second.overlay.ensure_attached(p_vp, composite_shader, -129, -64, auto_composite);
	it->second.last_seen = vp_collect_counter;
	return it->second;
}

void RTGI::_prune_stale_viewports() {
	for (auto it = viewport_states.begin(); it != viewport_states.end(); ) {
		if (it->second.last_seen != vp_collect_counter) {
			it->second.overlay.detach();
			it->second.history_valid = false;
			if (vp_collect_counter - it->second.last_seen > 60ULL) {
				it->second.overlay.destroy();
				if (gl_ready && gl_thread_id == Thread::get_caller_id()) {
					SharedGL43Context::ScopedScope gl43;
					if (gl43.is_active()) {
						if (it->second.gbuf_fbo) glDeleteFramebuffers(1, &it->second.gbuf_fbo);
						if (it->second.gbuf_depth) glDeleteTextures(1, &it->second.gbuf_depth);
						if (it->second.low_normal_mask) glDeleteTextures(1, &it->second.low_normal_mask);
						if (it->second.low_depth) glDeleteTextures(1, &it->second.low_depth);
						if (it->second.gi_work_a) glDeleteTextures(1, &it->second.gi_work_a);
						if (it->second.gi_work_b) glDeleteTextures(1, &it->second.gi_work_b);
						if (it->second.gi_history_tex[0]) glDeleteTextures(1, &it->second.gi_history_tex[0]);
						if (it->second.gi_history_tex[1]) glDeleteTextures(1, &it->second.gi_history_tex[1]);
					}
				}
				it = viewport_states.erase(it);
				continue;
			}
		}
		++it;
	}
}

void RTGI::_collect_scene_snapshot() {
	SceneTree *tree = get_tree();
	Viewport *my_vp = get_viewport();
	if (!tree || !my_vp) return;

	vp_collect_counter++;
	std::vector<std::pair<Viewport *, Camera *>> active_vps;
	GL43EffectHelpers::find_active_viewports(this, my_vp, active_vps);
	if (active_vps.empty()) {
		_prune_stale_viewports();
		MutexLock lock(scene_mutex);
		snapshot.valid = false;
		return;
	}

	Viewport *primary_vp = active_vps[0].first;
	Camera *primary_cam = active_vps[0].second;
	Transform primary_cam_xf = primary_cam->get_camera_transform();
	Vector3 primary_cam_pos = primary_cam_xf.origin;
	uint32_t primary_cam_mask = primary_cam->get_cull_mask();
	uint32_t recv_mask = gi_viewing_cull_mask & primary_cam_mask;

	std::vector<std::array<float, 24>> all_frustums(active_vps.size());
	for (size_t vi = 0; vi < active_vps.size(); vi++) {
		Viewport *vp = active_vps[vi].first;
		Camera *vcam = active_vps[vi].second;
		CameraMatrix vview, vproj, vvp, vinv_proj, vinv_vp;
		GL43EffectHelpers::compute_camera_matrices(vcam, vp->get_size(), vview, vproj, vvp, vinv_proj, vinv_vp);
		float fplanes[6][4];
		GL43EffectHelpers::extract_frustum_planes(vvp, fplanes);
		std::memcpy(all_frustums[vi].data(), fplanes, sizeof(fplanes));
	}

	Node *search_root = my_vp ? (Node *)my_vp : (Node *)tree->get_root();
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

	struct SceneCollector {
		static void visit(Node *n, uint32_t gi_m, uint32_t recv_m,
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
					if ((lm & gi_m) != 0 || (lm & recv_m) != 0) {
						out_m.push_back(mi);
					}
				}
			} else if (MultiMeshInstance *mmi = Object::cast_to<MultiMeshInstance>(n)) {
				if (mmi->is_visible_in_tree() && mmi->get_multimesh().is_valid() && mmi->get_multimesh()->get_mesh().is_valid()) {
					uint32_t lm = mmi->get_layer_mask();
					if ((lm & gi_m) != 0 || (lm & recv_m) != 0) {
						out_mm.push_back(mmi);
						out_em.push_back(mmi->get_multimesh()->get_mesh());
					}
				}
			}
#ifdef MODULE_CSG_ENABLED
			else if (CSGShape *csg = Object::cast_to<CSGShape>(n)) {
				if (csg->is_visible_in_tree() && csg->is_root_shape()) {
					uint32_t lm = csg->get_layer_mask();
					if ((lm & gi_m) != 0 || (lm & recv_m) != 0) {
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
				if (l->is_visible_in_tree() && !l->is_negative() && l->get_param(Light::PARAM_ENERGY) > 0.0001f) {
					out_l.push_back(l);
				}
			}
			for (int i = 0; i < n->get_child_count(); i++) {
				visit(n->get_child(i), gi_m, recv_m, out_m, out_mm, out_em, out_l
#ifdef MODULE_CSG_ENABLED
						, out_csg
#endif
				);
			}
		}
	};

	if (search_root) {
		SceneCollector::visit(search_root, gi_cull_mask, recv_mask, meshes, multimeshes, extra_meshes, scene_lights
#ifdef MODULE_CSG_ENABLED
				, csg_entries
#endif
		);
	}

	int vx = CLAMP((int)Math::round(volume_resolution.x), 4, 128);
	int vy = CLAMP((int)Math::round(volume_resolution.y), 4, 128);
	int vz = CLAMP((int)Math::round(volume_resolution.z), 4, 128);
	if (volume_enabled && (alloc_vol_x != vx || alloc_vol_y != vy || alloc_vol_z != vz)) {
		alloc_vol_x = vx;
		alloc_vol_y = vy;
		alloc_vol_z = vz;
		gi_volume_texture->create(vx, vy, vz, Image::FORMAT_RGBAH, Texture::FLAG_FILTER);
		for (int z = 0; z < vz; z++) {
			Ref<Image> slice;
			slice.instance();
			slice->create(vx, vy, false, Image::FORMAT_RGBAH);
			slice->fill(Color(0.0f, 0.0f, 0.0f, 0.0f));
			gi_volume_texture->set_layer_data(slice, z);
		}
	}

	MutexLock lock(scene_mutex);

	raytracer.rebuild_blas_cache(meshes, extra_meshes, _get_simplification_config(), blas_builds_per_frame);

	snapshot.viewports.clear();
	for (size_t vi = 0; vi < active_vps.size(); vi++) {
		Viewport *vp = active_vps[vi].first;
		Camera *vcam = active_vps[vi].second;
		Size2 vsz = vp->get_size();
		int fw = MAX(16, (int)vsz.x);
		int fh = MAX(16, (int)vsz.y);
		int tw = MAX(16, (int)Math::round(vsz.x * resolution_scale));
		int th = MAX(16, (int)Math::round(vsz.y * resolution_scale));

		ViewportRenderState &vst = _get_or_create_vp_state(vp);
		if (vst.tex_w != tw || vst.tex_h != th) {
			vst.tex_w = tw;
			vst.tex_h = th;
			vst.history_valid = false;

			Ref<Image> init_img;
			init_img.instance();
			init_img->create(tw, th, false, Image::FORMAT_RGBAH);
			init_img->fill(Color(0.0f, 0.0f, 0.0f, -1.0f));
			vst.gi_texture->create_from_image(init_img, Texture::FLAG_FILTER);
		}
		if (vst.full_tex_w != fw || vst.full_tex_h != fh) {
			vst.full_tex_w = fw;
			vst.full_tex_h = fh;

			Ref<Image> alb_img;
			alb_img.instance();
			alb_img->create(fw, fh, false, Image::FORMAT_RGBAH);
			alb_img->fill(Color(0.0f, 0.0f, 0.0f, 0.0f));
			vst.gi_albedo_texture->create_from_image(alb_img, 0);

			Ref<Image> norm_img;
			norm_img.instance();
			norm_img->create(fw, fh, false, Image::FORMAT_RGBAH);
			norm_img->fill(Color(0.0f, 0.0f, 0.0f, 0.0f));
			vst.gi_normal_texture->create_from_image(norm_img, 0);
		}

		if (vi == 0) {
			gi_texture = vst.gi_texture;
			gi_albedo_texture = vst.gi_albedo_texture;
		}

		Transform vxf = vcam->get_camera_transform();
		CameraMatrix v_view, v_proj, v_vp, v_inv_proj, v_inv_vp;
		GL43EffectHelpers::compute_camera_matrices(vcam, vsz, v_view, v_proj, v_vp, v_inv_proj, v_inv_vp);

		Ref<World> vp_world = vp->find_world();
		if (vp_world.is_null()) vp_world = my_vp->find_world();
		Ref<Environment> vp_env = vp_world.is_valid() ? vp_world->get_environment() : Ref<Environment>();
		if (vp_env.is_null() && vp_world.is_valid()) vp_env = vp_world->get_fallback_environment();

		int tm_mode = 0;
		float tm_exp = 1.0f;
		float tm_white = 1.0f;
		if (vp_env.is_valid()) {
			tm_mode = (int)vp_env->get_tonemapper();
			tm_exp = vp_env->get_tonemap_exposure();
			tm_white = vp_env->get_tonemap_white();
		}

		if (vst.overlay.material.is_valid()) {
			vst.overlay.material->set_shader_param("u_gi_tex", vst.gi_texture);
			vst.overlay.material->set_shader_param("u_albedo_tex", vst.gi_albedo_texture);
			vst.overlay.material->set_shader_param("u_full_normal_tex", vst.gi_normal_texture);
			vst.overlay.material->set_shader_param("u_intensity", gi_intensity);
			vst.overlay.material->set_shader_param("u_bilateral_upsample", (bilateral_upsample && resolution_scale < 0.99f) ? 1.0f : 0.0f);
			vst.overlay.material->set_shader_param("u_keep_3d_linear", vp->get_keep_3d_linear() ? 1.0f : 0.0f);
			vst.overlay.material->set_shader_param("u_vflip", vp->get_vflip() ? 1.0f : 0.0f);
			vst.overlay.material->set_shader_param("u_tonemap_mode", tm_mode);
			vst.overlay.material->set_shader_param("u_tonemap_exposure", tm_exp);
			vst.overlay.material->set_shader_param("u_tonemap_white", tm_white);
			vst.overlay.material->set_shader_param("u_hdr", vp->get_hdr() ? 1 : 0);
			vst.overlay.material->set_shader_param("u_debug_mode", (int)debug_mode);
			vst.overlay.material->set_shader_param("u_inv_proj_0", Plane(v_inv_proj.matrix[0][0], v_inv_proj.matrix[0][1], v_inv_proj.matrix[0][2], v_inv_proj.matrix[0][3]));
			vst.overlay.material->set_shader_param("u_inv_proj_1", Plane(v_inv_proj.matrix[1][0], v_inv_proj.matrix[1][1], v_inv_proj.matrix[1][2], v_inv_proj.matrix[1][3]));
			vst.overlay.material->set_shader_param("u_inv_proj_2", Plane(v_inv_proj.matrix[2][0], v_inv_proj.matrix[2][1], v_inv_proj.matrix[2][2], v_inv_proj.matrix[2][3]));
			vst.overlay.material->set_shader_param("u_inv_proj_3", Plane(v_inv_proj.matrix[3][0], v_inv_proj.matrix[3][1], v_inv_proj.matrix[3][2], v_inv_proj.matrix[3][3]));
		}

		ViewportSnapshot vsnap;
		vsnap.vp_id = vp->get_viewport_rid().get_id();
		vsnap.vp_rid = vp->get_viewport_rid();
		vsnap.viewport_w = (int)vsz.x;
		vsnap.viewport_h = (int)vsz.y;
		vsnap.gi_w = tw;
		vsnap.gi_h = th;
		vsnap.cam_pos[0] = (float)vxf.origin.x;
		vsnap.cam_pos[1] = (float)vxf.origin.y;
		vsnap.cam_pos[2] = (float)vxf.origin.z;
		GL43EffectHelpers::camera_matrix_to_float16(v_vp, vsnap.view_proj_mat);
		GL43EffectHelpers::camera_matrix_to_float16(v_inv_vp, vsnap.inv_view_proj);
		GL43EffectHelpers::camera_matrix_to_float16(v_inv_proj, vsnap.inv_proj_mat);
		std::memcpy(vsnap.cam_frustum_planes, all_frustums[vi].data(), sizeof(vsnap.cam_frustum_planes));
		vsnap.tonemap_mode = tm_mode;
		vsnap.tonemap_exposure = tm_exp;
		vsnap.tonemap_white = tm_white;
		vsnap.hdr = vp->get_hdr();
		vsnap.keep_3d_linear = vp->get_keep_3d_linear();
		vsnap.vflip = vp->get_vflip();
		snapshot.viewports.push_back(vsnap);
	}
	_prune_stale_viewports();

	Vector3 voxel_size(
			(2.0f * volume_extents.x) / (float)vx,
			(2.0f * volume_extents.y) / (float)vy,
			(2.0f * volume_extents.z) / (float)vz);
	Vector3 snapped_center(
			Math::floor(primary_cam_pos.x / voxel_size.x) * voxel_size.x,
			Math::floor(primary_cam_pos.y / voxel_size.y) * voxel_size.y,
			Math::floor(primary_cam_pos.z / voxel_size.z) * voxel_size.z);
	Vector3 vol_min = snapped_center - volume_extents;
	snapshot.vol_res[0] = vx;
	snapshot.vol_res[1] = vy;
	snapshot.vol_res[2] = vz;
	snapshot.vol_min[0] = (float)vol_min.x;
	snapshot.vol_min[1] = (float)vol_min.y;
	snapshot.vol_min[2] = (float)vol_min.z;
	snapshot.vol_size[0] = (float)(volume_extents.x * 2.0f);
	snapshot.vol_size[1] = (float)(volume_extents.y * 2.0f);
	snapshot.vol_size[2] = (float)(volume_extents.z * 2.0f);
	snapshot.vol_voxel_size = MAX(voxel_size.x, MAX(voxel_size.y, voxel_size.z));

	// =========================================================
	// SKY RESOLUTION
	// =========================================================
	Ref<World> world = primary_vp->find_world();
	if (world.is_null()) world = my_vp->find_world();
	Ref<Environment> env = world.is_valid() ? world->get_environment() : Ref<Environment>();
	if (env.is_null() && world.is_valid()) env = world->get_fallback_environment();

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
	// LIGHTS
	// =========================================================
	snapshot.lights.clear();
	const float pi = 3.14159265359f;
	const float max_local_light_dist = light_cull_distance * (0.5f + 0.5f * resolution_scale);
	const float max_local_shadow_dist = shadow_max_distance * (0.5f + 0.5f * resolution_scale);
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
			float dist_to_cam = MAX(0.0f, (float)lx.origin.distance_to(primary_cam_pos) - range);
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

	// =========================================================
	// INSTANCES & MATERIALS
	// =========================================================
	raytracer.begin_frame_instances();
	snapshot.materials.clear();

	auto push_surface_materials = [&](const Ref<Mesh> &mesh, auto get_mat_fn) {
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

				int32_t mflags = CLAMP((int32_t)smat->get_cull_mode(), 0, 2);
				if (smat->get_flag(SpatialMaterial::FLAG_DISABLE_AMBIENT_LIGHT)) mflags |= 4;
				if (smat->get_flag(SpatialMaterial::FLAG_UNSHADED)) mflags |= 8;
				gm.extra_tex[3] = mflags;
			}
			snapshot.materials.push_back(gm);
		}
	};

	for (MeshInstance *mi : meshes) {
		Ref<Mesh> mesh = mi->get_mesh();
		uint32_t mid = mesh->get_rid().get_id();
		const GL43Raytracer::MeshBLAS *blas = raytracer.get_blas(mid);
		if (!blas || (!skinned_meshes_in_gi && blas->has_skin_weights)) {
			continue;
		}

		uint32_t layer_mask = mi->get_layer_mask();
		bool in_gi_mask = (layer_mask & gi_cull_mask) != 0;
		bool in_view_mask = (layer_mask & gi_viewing_cull_mask) != 0;
		bool cam_visible = (layer_mask & primary_cam_mask) != 0;
		bool shadows_only = (mi->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
		uint32_t uflags = (!in_view_mask || shadows_only) ? INST_FLAG_SKIP_RECEIVER : 0;

		if (raytracer.add_mesh_instance(
				mi,
				(int32_t)snapshot.materials.size(),
				primary_cam_pos,
				mesh_cull_distance,
				mesh_cull_min_angle,
				resolution_scale,
				skinned_meshes_in_gi,
				in_gi_mask,
				cam_visible,
				uflags)) {
			push_surface_materials(mesh, [&](int s) { return mi->get_active_material(s); });
		}
	}

	for (MultiMeshInstance *mmi : multimeshes) {
		Ref<MultiMesh> mm = mmi->get_multimesh();
		Ref<Mesh> mesh = mm->get_mesh();
		uint32_t mid = mesh->get_rid().get_id();
		const GL43Raytracer::MeshBLAS *blas = raytracer.get_blas(mid);
		if (!blas || (!skinned_meshes_in_gi && blas->has_skin_weights)) {
			continue;
		}

		uint32_t layer_mask = mmi->get_layer_mask();
		bool in_gi_mask = (layer_mask & gi_cull_mask) != 0;
		bool in_view_mask = (layer_mask & gi_viewing_cull_mask) != 0;
		bool cam_visible = (layer_mask & primary_cam_mask) != 0;
		bool shadows_only = (mmi->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
		uint32_t uflags = (!in_view_mask || shadows_only) ? INST_FLAG_SKIP_RECEIVER : 0;

		if (raytracer.add_multimesh_instance(
				mmi,
				(int32_t)snapshot.materials.size(),
				primary_cam_pos,
				mesh_cull_distance,
				mesh_cull_min_angle,
				resolution_scale,
				in_gi_mask,
				cam_visible,
				uflags)) {
			Ref<Material> override_mat = mmi->get_material_override();
			push_surface_materials(mesh, [&](int s) {
				return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s);
			});
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
		bool in_gi_mask = (layer_mask & gi_cull_mask) != 0;
		bool in_view_mask = (layer_mask & gi_viewing_cull_mask) != 0;
		bool cam_visible = (layer_mask & primary_cam_mask) != 0;
		bool shadows_only = (csg->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
		uint32_t uflags = (!in_view_mask || shadows_only) ? INST_FLAG_SKIP_RECEIVER : 0;

		if (raytracer.add_custom_mesh_instance(
				mesh,
				csg->get_global_transform() * entry.local_xf,
				layer_mask,
				csg->get_cast_shadows_setting(),
				(int32_t)snapshot.materials.size(),
				primary_cam_pos,
				mesh_cull_distance,
				mesh_cull_min_angle,
				resolution_scale,
				in_gi_mask,
				cam_visible,
				uflags)) {
			Ref<Material> override_mat = csg->get_material_override();
			push_surface_materials(mesh, [&](int s) {
				return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s);
			});
		}
	}
#endif

	raytracer.build_tlas(max_tlas_instances);
	snapshot.valid = !raytracer.is_empty();
}

bool RTGI::_init_gl_resources() {
	if (gl_attempted) return gl_ready;
	gl_attempted = true;
	gl_thread_id = Thread::get_caller_id();

	if (!SharedGL43Context::get_singleton()->is_available()) {
		ERR_PRINT("RTGI: Shared OpenGL 4.3 context not available.");
		return false;
	}

	if (!raytracer.init_gl() || !tex_pool.init_gl("RTGI")) {
		return false;
	}

	gbuf_program = GL43EffectHelpers::link_vf_program(RTGIShaders::GBUF_VERT_GLSL, RTGIShaders::GBUF_FRAG_GLSL, "RTGI", "GBuffer");
	downsample_gbuf_program = GL43EffectHelpers::link_compute_program(RTGIShaders::DOWNSAMPLE_GBUF_COMPUTE_GLSL, "RTGI", "DownsampleGBuf");
	sh_program = GL43EffectHelpers::link_compute_program(RTGIShaders::SH_COMPUTE_GLSL, "RTGI", "SkySH");

	String vol_src = String("#version 430 core\n") + GL43Raytracer::get_glsl_common() + RTGIShaders::GI_TRACE_COMMON_GLSL + RTGIShaders::VOLUME_COMPUTE_GLSL_BODY;
	String rt_src = String(RTGIShaders::RT_COMPUTE_HEADER_GLSL) + GL43Raytracer::get_glsl_common() + RTGIShaders::GI_TRACE_COMMON_GLSL + RTGIShaders::RT_COMPUTE_BODY_GLSL;

	volume_program = GL43EffectHelpers::link_compute_program(vol_src.utf8().get_data(), "RTGI", "Volume3D");
	volume_filter_program = GL43EffectHelpers::link_compute_program(RTGIShaders::VOLUME_FILTER_COMPUTE_GLSL, "RTGI", "Volume3DFilter");
	rt_program = GL43EffectHelpers::link_compute_program(rt_src.utf8().get_data(), "RTGI", "RaytraceGI");
	reconstruct_program = GL43EffectHelpers::link_compute_program(RTGIShaders::RECONSTRUCT_COMPUTE_GLSL, "RTGI", "Reconstruct");
	denoise_program = GL43EffectHelpers::link_compute_program(RTGIShaders::DENOISE_COMPUTE_GLSL, "RTGI", "Denoise");

	if (!gbuf_program || !downsample_gbuf_program || !sh_program || !volume_program || !volume_filter_program || !rt_program || !reconstruct_program || !denoise_program) {
		return false;
	}

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

	glGenTextures(1, &dummy_gl_3d);
	glBindTexture(GL_TEXTURE_3D, dummy_gl_3d);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, 1, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_3D, 0);

	glGenBuffers(1, &ssbo_materials);
	glGenBuffers(1, &ssbo_lights);
	glGenBuffers(1, &ssbo_sky_sh);

	float zero_sh[9 * 4] = {};
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_sky_sh);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero_sh), zero_sh, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	gl_ready = true;
	return true;
}

void RTGI::_cleanup_gl_resources() {
	if (!gl_ready || gl_thread_id != Thread::get_caller_id()) {
		return;
	}

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	raytracer.cleanup_gl();
	tex_pool.cleanup_gl();

	for (auto &kv : viewport_states) {
		ViewportRenderState &st = kv.second;
		if (st.gbuf_fbo) { glDeleteFramebuffers(1, &st.gbuf_fbo); st.gbuf_fbo = 0; }
		if (st.gbuf_depth) { glDeleteTextures(1, &st.gbuf_depth); st.gbuf_depth = 0; }
		if (st.low_normal_mask) { glDeleteTextures(1, &st.low_normal_mask); st.low_normal_mask = 0; }
		if (st.low_depth) { glDeleteTextures(1, &st.low_depth); st.low_depth = 0; }
		if (st.gi_work_a) { glDeleteTextures(1, &st.gi_work_a); st.gi_work_a = 0; }
		if (st.gi_work_b) { glDeleteTextures(1, &st.gi_work_b); st.gi_work_b = 0; }
		if (st.gi_history_tex[0]) { glDeleteTextures(1, &st.gi_history_tex[0]); st.gi_history_tex[0] = 0; }
		if (st.gi_history_tex[1]) { glDeleteTextures(1, &st.gi_history_tex[1]); st.gi_history_tex[1] = 0; }
	}

	if (gbuf_program) { glDeleteProgram(gbuf_program); gbuf_program = 0; }
	if (downsample_gbuf_program) { glDeleteProgram(downsample_gbuf_program); downsample_gbuf_program = 0; }
	if (sh_program) { glDeleteProgram(sh_program); sh_program = 0; }
	if (volume_program) { glDeleteProgram(volume_program); volume_program = 0; }
	if (volume_filter_program) { glDeleteProgram(volume_filter_program); volume_filter_program = 0; }
	if (rt_program) { glDeleteProgram(rt_program); rt_program = 0; }
	if (reconstruct_program) { glDeleteProgram(reconstruct_program); reconstruct_program = 0; }
	if (denoise_program) { glDeleteProgram(denoise_program); denoise_program = 0; }

	if (vol_hist_tex[0]) { glDeleteTextures(1, &vol_hist_tex[0]); vol_hist_tex[0] = 0; }
	if (vol_hist_tex[1]) { glDeleteTextures(1, &vol_hist_tex[1]); vol_hist_tex[1] = 0; }
	vol_hist_idx = 0;
	vol_hist_valid = false;
	alloc_gl_vol_x = 0;
	alloc_gl_vol_y = 0;
	alloc_gl_vol_z = 0;
	has_rendered_vol = false;

	if (dummy_white_tex) { glDeleteTextures(1, &dummy_white_tex); dummy_white_tex = 0; }
	if (dummy_gl_3d) { glDeleteTextures(1, &dummy_gl_3d); dummy_gl_3d = 0; }
	if (pano_sampler_mip) { glDeleteSamplers(1, &pano_sampler_mip); pano_sampler_mip = 0; }
	if (pano_sampler_nomip) { glDeleteSamplers(1, &pano_sampler_nomip); pano_sampler_nomip = 0; }

	if (ssbo_materials) { glDeleteBuffers(1, &ssbo_materials); ssbo_materials = 0; }
	if (ssbo_lights) { glDeleteBuffers(1, &ssbo_lights); ssbo_lights = 0; }
	if (ssbo_sky_sh) { glDeleteBuffers(1, &ssbo_sky_sh); ssbo_sky_sh = 0; }

	gl_ready = false;
	gl_attempted = false;
}

void RTGI::_ensure_gbuffer_size(ViewportRenderState &r_state, int p_w, int p_h) {
	if (r_state.gbuf_fbo != 0 && r_state.gbuf_w == p_w && r_state.gbuf_h == p_h) return;
	r_state.gbuf_w = p_w;
	r_state.gbuf_h = p_h;

	if (r_state.gbuf_fbo == 0) {
		glGenFramebuffers(1, &r_state.gbuf_fbo);
	}

	glActiveTexture(GL_TEXTURE0);
	GL43EffectHelpers::ensure_depth24_texture(r_state.gbuf_depth, p_w, p_h);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, r_state.gbuf_fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, r_state.gbuf_depth, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RTGI::_ensure_low_gbuf_size(ViewportRenderState &r_state, int p_w, int p_h) {
	if (r_state.low_normal_mask != 0 && r_state.low_depth != 0 && r_state.low_w == p_w && r_state.low_h == p_h) return;
	r_state.low_w = p_w;
	r_state.low_h = p_h;

	glActiveTexture(GL_TEXTURE0);
	GL43EffectHelpers::ensure_rgba16f_texture(r_state.low_normal_mask, p_w, p_h, GL_NEAREST);
	if (r_state.low_depth == 0) {
		glGenTextures(1, &r_state.low_depth);
	}
	glBindTexture(GL_TEXTURE_2D, r_state.low_depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, p_w, p_h, 0, GL_RED, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void RTGI::_ensure_vol_raw_size(int p_x, int p_y, int p_z) {
	if (vol_hist_tex[0] != 0 && vol_hist_tex[1] != 0 && alloc_gl_vol_x == p_x && alloc_gl_vol_y == p_y && alloc_gl_vol_z == p_z) {
		return;
	}
	alloc_gl_vol_x = p_x;
	alloc_gl_vol_y = p_y;
	alloc_gl_vol_z = p_z;
	vol_hist_valid = false;

	glActiveTexture(GL_TEXTURE0);
	for (int i = 0; i < 2; i++) {
		if (vol_hist_tex[i] == 0) {
			glGenTextures(1, &vol_hist_tex[i]);
		}
		glBindTexture(GL_TEXTURE_3D, vol_hist_tex[i]);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, p_x, p_y, p_z, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	}
	glBindTexture(GL_TEXTURE_3D, 0);
}

void RTGI::_ensure_work_textures(ViewportRenderState &r_state, int p_w, int p_h, bool p_need_history) {
	bool size_changed = (r_state.work_w != p_w || r_state.work_h != p_h);
	if (size_changed) {
		r_state.work_w = p_w;
		r_state.work_h = p_h;
		r_state.history_valid = false;
	}

	glActiveTexture(GL_TEXTURE0);
	if (size_changed || r_state.gi_work_a == 0) GL43EffectHelpers::ensure_rgba16f_texture(r_state.gi_work_a, p_w, p_h, GL_LINEAR);
	if (size_changed || r_state.gi_work_b == 0) GL43EffectHelpers::ensure_rgba16f_texture(r_state.gi_work_b, p_w, p_h, GL_LINEAR);
	if (p_need_history && (size_changed || r_state.gi_history_tex[0] == 0 || r_state.gi_history_tex[1] == 0)) {
		GL43EffectHelpers::ensure_rgba16f_texture(r_state.gi_history_tex[0], p_w, p_h, GL_LINEAR);
		GL43EffectHelpers::ensure_rgba16f_texture(r_state.gi_history_tex[1], p_w, p_h, GL_LINEAR);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
}

void RTGI::_on_frame_pre_draw() {
	if (!vs_resources_created || !is_inside_tree()) return;

	bool in_editor = Engine::get_singleton()->is_editor_hint();
	if (!enabled || (in_editor && !editor_preview) || !get_viewport()) {
		_detach_all_viewports();
		return;
	}

	if (in_editor) {
		process_frame_counter++;
		if (update_interval <= 1 || (process_frame_counter % (uint64_t)update_interval) == 0) {
			_collect_scene_snapshot();
		}
	}

	if (update_interval > 1 && (process_frame_counter % (uint64_t)update_interval) != 0) {
		return;
	}

	RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
	if (!rs) return;

	MutexLock lock(scene_mutex);
	if (!snapshot.valid || snapshot.viewports.empty()) return;

	RasterizerStorageGLES3::Texture *godot_vol_tex = nullptr;
	if (volume_enabled) {
		godot_vol_tex = rs->texture_owner.getornull(gi_volume_texture->get_rid());
		if (godot_vol_tex && (godot_vol_tex->tex_id == 0 || godot_vol_tex->width != snapshot.vol_res[0] || godot_vol_tex->height != snapshot.vol_res[1] || godot_vol_tex->depth != snapshot.vol_res[2])) {
			godot_vol_tex = nullptr;
		}
	}

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) return;

	if (!_init_gl_resources()) return;

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
	// 1. PANORAMA BINDING & L2 SKY SH IRRADIANCE UPDATE
	// =========================================================
	int use_pano = 0;
	int pano_to_lin = 0;
	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, dummy_white_tex);
	glBindSampler(4, pano_sampler_nomip);

	if (snapshot.use_panorama && snapshot.panorama_rid.is_valid()) {
		RasterizerStorageGLES3::Texture *pt = rs->texture_owner.getornull(snapshot.panorama_rid);
		if (pt) pt = pt->get_ptr();
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

	SharedGL43Context *gl43_ctx = SharedGL43Context::get_singleton();

	// =========================================================
	// 2. REAL-TIME 3D IRRADIANCE VOLUME PASS (Infinite-Bounce Cache)
	// =========================================================
	if (volume_enabled && godot_vol_tex) {
		bool vol_res_changed = (alloc_gl_vol_x != snapshot.vol_res[0] || alloc_gl_vol_y != snapshot.vol_res[1] || alloc_gl_vol_z != snapshot.vol_res[2]);
		_ensure_vol_raw_size(snapshot.vol_res[0], snapshot.vol_res[1], snapshot.vol_res[2]);
		volume_frame_counter++;
		if (!has_rendered_vol || vol_res_changed || volume_frame_counter == 1 || volume_update_interval <= 1 || (volume_frame_counter % (uint64_t)volume_update_interval) == 0) {
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_3D, godot_vol_tex->tex_id);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
			glBindTexture(GL_TEXTURE_3D, 0);

			int write_vol = vol_hist_idx;
			int read_vol = 1 - vol_hist_idx;

			glUseProgram(volume_program);
			gl43_ctx->bind_image_texture(0, vol_hist_tex[write_vol], 0, true, 0, GL_WRITE_ONLY, GL_RGBA16F);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D_ARRAY, tex_pool.get_tex_array());
			glUniform1i(glGetUniformLocation(volume_program, "uTexArray"), 0);

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_3D, vol_hist_tex[read_vol]);
			glUniform1i(glGetUniformLocation(volume_program, "uPrevHistVolume"), 1);

			glUniform3f(glGetUniformLocation(volume_program, "uVolMin"), snapshot.vol_min[0], snapshot.vol_min[1], snapshot.vol_min[2]);
			glUniform3f(glGetUniformLocation(volume_program, "uPrevVolMin"), prev_vol_min[0], prev_vol_min[1], prev_vol_min[2]);
			glUniform3f(glGetUniformLocation(volume_program, "uVolSize"), snapshot.vol_size[0], snapshot.vol_size[1], snapshot.vol_size[2]);
			glUniform1f(glGetUniformLocation(volume_program, "uVolRayDist"), volume_ray_distance);
			glUniform1i(glGetUniformLocation(volume_program, "uVolSamples"), volume_rays_per_voxel);
			glUniform1f(glGetUniformLocation(volume_program, "uRayBias"), ray_origin_bias);
			glUniform1f(glGetUniformLocation(volume_program, "uShadowBias"), shadow_ray_bias);
			glUniform1i(glGetUniformLocation(volume_program, "uHasTLAS"), has_tlas ? 1 : 0);
			glUniform1i(glGetUniformLocation(volume_program, "uNumLights"), (int)snapshot.lights.size());
			glUniform1i(glGetUniformLocation(volume_program, "uRespectCullMode"), respect_material_cull_mode ? 1 : 0);
			glUniform1i(glGetUniformLocation(volume_program, "uTraceShadows"), (trace_shadows && shadow_bounces > 0) ? 1 : 0);
			glUniform1i(glGetUniformLocation(volume_program, "uSkyIndirectEnabled"), sky_indirect_enabled ? 1 : 0);
			glUniform1f(glGetUniformLocation(volume_program, "uSkyIndirectEnergy"), sky_indirect_energy);
			glUniformMatrix3fv(glGetUniformLocation(volume_program, "uSkyInvBasis"), 1, GL_FALSE, snapshot.sky_inv_basis);
			glUniform1f(glGetUniformLocation(volume_program, "uEmissiveIntensity"), emissive_intensity);
			glUniform1f(glGetUniformLocation(volume_program, "uVolumeFeedback"), volume_feedback);
			glUniform1f(glGetUniformLocation(volume_program, "uFireflyClamp"), firefly_clamp);
			glUniform1i(glGetUniformLocation(volume_program, "uVolHistValid"), vol_hist_valid ? 1 : 0);
			glUniform1i(glGetUniformLocation(volume_program, "uTemporalNoise"), temporal_noise ? 1 : 0);
			glUniform1i(glGetUniformLocation(volume_program, "uVolFrame"), (int)(vol_update_count++ & 0x7FFFFFFF));

			GLuint vgx = (GLuint)((snapshot.vol_res[0] + 3) / 4);
			GLuint vgy = (GLuint)((snapshot.vol_res[1] + 3) / 4);
			GLuint vgz = (GLuint)((snapshot.vol_res[2] + 3) / 4);
			gl43_ctx->dispatch_compute(vgx, vgy, vgz);
			gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

			glUseProgram(volume_filter_program);
			gl43_ctx->bind_image_texture(0, godot_vol_tex->tex_id, 0, true, 0, GL_WRITE_ONLY, GL_RGBA16F);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_3D, vol_hist_tex[write_vol]);
			glUniform1i(glGetUniformLocation(volume_filter_program, "uRawGIVolume"), 0);
			gl43_ctx->dispatch_compute(vgx, vgy, vgz);
			gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

			std::memcpy(prev_vol_min, snapshot.vol_min, sizeof(prev_vol_min));
			std::memcpy(rendered_vol_min, snapshot.vol_min, sizeof(rendered_vol_min));
			std::memcpy(rendered_vol_size, snapshot.vol_size, sizeof(rendered_vol_size));
			rendered_vol_voxel_size = snapshot.vol_voxel_size;
			vol_hist_idx = read_vol;
			vol_hist_valid = true;
			has_rendered_vol = true;
		}
	} else {
		vol_hist_valid = false;
		has_rendered_vol = false;
	}

	// =========================================================
	// 3. PER-VIEWPORT G-BUFFER, PATH TRACE, RECONSTRUCT & DENOISE
	// =========================================================
	bool need_reconstruct = checkerboard_tracing || temporal_accumulation;
	bool need_denoise = (spatial_denoise_passes > 0);
	bool need_post = need_reconstruct || need_denoise;

	const std::vector<GL43Raytracer::InstanceEntry> &insts = raytracer.get_instances();
	float lod_bias = texture_lod_bias + std::max(0.0f, -std::log2(std::max(resolution_scale, 0.125f)));
	float eff_shadow_dist = (trace_shadows && shadow_bounces > 0) ? (shadow_max_distance * (0.35f + 0.65f * resolution_scale)) : 0.0f;
	float eff_ray_dist = max_ray_distance * (0.5f + 0.5f * resolution_scale);

	const float *active_vol_min = has_rendered_vol ? rendered_vol_min : snapshot.vol_min;
	const float *active_vol_size = has_rendered_vol ? rendered_vol_size : snapshot.vol_size;
	float active_voxel_size = has_rendered_vol ? rendered_vol_voxel_size : snapshot.vol_voxel_size;

	for (size_t vi = 0; vi < snapshot.viewports.size(); vi++) {
		const ViewportSnapshot &vsnap = snapshot.viewports[vi];
		auto st_it = viewport_states.find(vsnap.vp_id);
		if (st_it == viewport_states.end()) continue;
		ViewportRenderState &vst = st_it->second;

		RasterizerStorageGLES3::Texture *godot_gi_tex = rs->texture_owner.getornull(vst.gi_texture->get_rid());
		RasterizerStorageGLES3::Texture *godot_alb_tex = rs->texture_owner.getornull(vst.gi_albedo_texture->get_rid());
		RasterizerStorageGLES3::Texture *godot_norm_tex = rs->texture_owner.getornull(vst.gi_normal_texture->get_rid());
		if (!godot_gi_tex || godot_gi_tex->tex_id == 0 || !godot_alb_tex || godot_alb_tex->tex_id == 0 || !godot_norm_tex || godot_norm_tex->tex_id == 0) continue;
		if (godot_gi_tex->width != vsnap.gi_w || godot_gi_tex->height != vsnap.gi_h) continue;
		if (godot_alb_tex->width != vsnap.viewport_w || godot_alb_tex->height != vsnap.viewport_h) continue;
		if (godot_norm_tex->width != vsnap.viewport_w || godot_norm_tex->height != vsnap.viewport_h) continue;

		// 3A. Full-Resolution G-Buffer Pass (NormalMask + Receiver Albedo*(1-Metallic))
		_ensure_gbuffer_size(vst, vsnap.viewport_w, vsnap.viewport_h);
		glBindFramebuffer(GL_FRAMEBUFFER, vst.gbuf_fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, godot_norm_tex->tex_id, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, godot_alb_tex->tex_id, 0);
		GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
		glDrawBuffers(2, bufs);

		glViewport(0, 0, vsnap.viewport_w, vsnap.viewport_h);
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
		glUniformMatrix4fv(glGetUniformLocation(gbuf_program, "uViewProj"), 1, GL_FALSE, vsnap.view_proj_mat);
		glUniform1i(glGetUniformLocation(gbuf_program, "uRespectCullMode"), respect_material_cull_mode ? 1 : 0);
		glUniform1i(glGetUniformLocation(gbuf_program, "uAlphaConsideration"), alpha_consideration ? 1 : 0);

		GLint loc_model = glGetUniformLocation(gbuf_program, "uModel");
		GLint loc_mat_off = glGetUniformLocation(gbuf_program, "uMatOffset");
		GLint loc_skip = glGetUniformLocation(gbuf_program, "uSkipGI");
		GLint loc_flip = glGetUniformLocation(gbuf_program, "uFlipFacing");

		for (size_t i = 0; i < insts.size(); i++) {
			const GL43Raytracer::InstanceEntry &inst = insts[i];
			if (!inst.draw_in_gbuf) continue;
			if (frustum_cull_gbuffer && !GL43EffectHelpers::aabb_in_frustum(inst.world_aabb, vsnap.cam_frustum_planes)) continue;
			uint32_t vao = 0;
			int32_t vcount = 0;
			if (!raytracer.get_instance_draw_data(i, vao, vcount)) continue;

			glUniformMatrix4fv(loc_model, 1, GL_FALSE, inst.model_mat);
			glUniform1i(loc_mat_off, inst.mat_offset);
			glUniform1i(loc_skip, (inst.user_flags & INST_FLAG_SKIP_RECEIVER) ? 1 : 0);
			glUniform1i(loc_flip, inst.flip_facing ? 1 : 0);
			glBindVertexArray((GLuint)vao);
			glDrawArrays(GL_TRIANGLES, 0, vcount);
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, godot_alb_tex->tex_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, godot_norm_tex->tex_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, 0);

		// Downsample Normal & Depth to Trace Resolution
		GLuint gx = (GLuint)((vsnap.gi_w + 7) / 8);
		GLuint gy = (GLuint)((vsnap.gi_h + 7) / 8);
		_ensure_low_gbuf_size(vst, vsnap.gi_w, vsnap.gi_h);

		glUseProgram(downsample_gbuf_program);
		gl43_ctx->bind_image_texture(0, vst.low_normal_mask, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
		gl43_ctx->bind_image_texture(1, vst.low_depth, 0, false, 0, GL_WRITE_ONLY, GL_R32F);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, godot_norm_tex->tex_id);
		glUniform1i(glGetUniformLocation(downsample_gbuf_program, "uFullNormalMask"), 0);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, vst.gbuf_depth);
		glUniform1i(glGetUniformLocation(downsample_gbuf_program, "uFullDepth"), 1);
		gl43_ctx->dispatch_compute(gx, gy, 1);
		gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

		// 3B. Diffuse GI Ray-Tracing Compute Pass
		if (need_post) {
			_ensure_work_textures(vst, vsnap.gi_w, vsnap.gi_h, temporal_accumulation);
		}

		GLuint rt_target_tex = need_post ? vst.gi_work_a : godot_gi_tex->tex_id;

		glUseProgram(rt_program);
		gl43_ctx->bind_image_texture(0, rt_target_tex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, vst.low_normal_mask);
		glUniform1i(glGetUniformLocation(rt_program, "uGNormalMask"), 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, vst.low_depth);
		glUniform1i(glGetUniformLocation(rt_program, "uGDepth"), 1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D_ARRAY, tex_pool.get_tex_array());
		glUniform1i(glGetUniformLocation(rt_program, "uTexArray"), 2);

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_3D, (volume_enabled && godot_vol_tex) ? godot_vol_tex->tex_id : dummy_gl_3d);
		glUniform1i(glGetUniformLocation(rt_program, "uGIVolumeTex"), 3);

		glUniform1i(glGetUniformLocation(rt_program, "uPanoramaTex"), 4);
		glUniform1i(glGetUniformLocation(rt_program, "uUsePanorama"), use_pano);
		glUniform1i(glGetUniformLocation(rt_program, "uPanoramaToLinear"), pano_to_lin);
		glUniform1f(glGetUniformLocation(rt_program, "uSkyEnergy"), snapshot.sky_energy);
		glUniformMatrix3fv(glGetUniformLocation(rt_program, "uSkyInvBasis"), 1, GL_FALSE, snapshot.sky_inv_basis);

		glUniformMatrix4fv(glGetUniformLocation(rt_program, "uInvViewProj"), 1, GL_FALSE, vsnap.inv_view_proj);
		glUniform3f(glGetUniformLocation(rt_program, "uCamPos"), vsnap.cam_pos[0], vsnap.cam_pos[1], vsnap.cam_pos[2]);
		glUniform3f(glGetUniformLocation(rt_program, "uSkyTop"), snapshot.sky_top[0], snapshot.sky_top[1], snapshot.sky_top[2]);
		glUniform3f(glGetUniformLocation(rt_program, "uSkyHorizon"), snapshot.sky_horizon[0], snapshot.sky_horizon[1], snapshot.sky_horizon[2]);
		glUniform3f(glGetUniformLocation(rt_program, "uSkyGroundHorizon"), snapshot.sky_ground_horizon[0], snapshot.sky_ground_horizon[1], snapshot.sky_ground_horizon[2]);
		glUniform3f(glGetUniformLocation(rt_program, "uSkyGround"), snapshot.sky_ground[0], snapshot.sky_ground[1], snapshot.sky_ground[2]);
		glUniform1f(glGetUniformLocation(rt_program, "uSkyCurve"), snapshot.sky_curve);
		glUniform1f(glGetUniformLocation(rt_program, "uGroundCurve"), snapshot.ground_curve);
		glUniform3f(glGetUniformLocation(rt_program, "uSkySunDir"), snapshot.sky_sun_dir[0], snapshot.sky_sun_dir[1], snapshot.sky_sun_dir[2]);
		glUniform3f(glGetUniformLocation(rt_program, "uSkySunColor"), snapshot.sky_sun_color[0], snapshot.sky_sun_color[1], snapshot.sky_sun_color[2]);
		glUniform1f(glGetUniformLocation(rt_program, "uSkySunAngleMin"), snapshot.sky_sun_angle_min);
		glUniform1f(glGetUniformLocation(rt_program, "uSkySunAngleMax"), snapshot.sky_sun_angle_max);
		glUniform1f(glGetUniformLocation(rt_program, "uSkySunCurve"), snapshot.sky_sun_curve);
		glUniform1f(glGetUniformLocation(rt_program, "uSkySunEnergy"), snapshot.sky_sun_energy);

		glUniform1i(glGetUniformLocation(rt_program, "uHasTLAS"), has_tlas ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uNumLights"), (int)snapshot.lights.size());
		glUniform1i(glGetUniformLocation(rt_program, "uRaysPerPixel"), rays_per_pixel);
		glUniform1i(glGetUniformLocation(rt_program, "uMaxBounces"), max_bounces);
		glUniform1f(glGetUniformLocation(rt_program, "uMaxRayDistance"), eff_ray_dist);
		glUniform1f(glGetUniformLocation(rt_program, "uBounceDistFalloff"), bounce_distance_falloff);
		glUniform1f(glGetUniformLocation(rt_program, "uBounceEnergyBoost"), bounce_energy_boost);
		glUniform1f(glGetUniformLocation(rt_program, "uBounceThroughputCutoff"), bounce_throughput_threshold);
		glUniform1f(glGetUniformLocation(rt_program, "uEmissiveIntensity"), emissive_intensity);
		glUniform1f(glGetUniformLocation(rt_program, "uAmbientBounceApprox"), ambient_bounce_approximation);
		glUniform1i(glGetUniformLocation(rt_program, "uSkyIndirectEnabled"), sky_indirect_enabled ? 1 : 0);
		glUniform1f(glGetUniformLocation(rt_program, "uSkyIndirectEnergy"), sky_indirect_energy);
		glUniform1i(glGetUniformLocation(rt_program, "uBounceLocalLights"), bounce_local_lights ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uNormalMapsInBounces"), normal_maps_in_bounces ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uRespectCullMode"), respect_material_cull_mode ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uTraceShadows"), trace_shadows ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uShadowBounces"), trace_shadows ? shadow_bounces : 0);
		glUniform1f(glGetUniformLocation(rt_program, "uShadowMaxDist"), eff_shadow_dist);
		glUniform1f(glGetUniformLocation(rt_program, "uRayBias"), ray_origin_bias);
		glUniform1f(glGetUniformLocation(rt_program, "uShadowBias"), shadow_ray_bias);
		glUniform1f(glGetUniformLocation(rt_program, "uLodBias"), lod_bias);
		glUniform1f(glGetUniformLocation(rt_program, "uDistLodStrength"), distance_lod_strength);
		glUniform1f(glGetUniformLocation(rt_program, "uFireflyClamp"), firefly_clamp);

		glUniform1i(glGetUniformLocation(rt_program, "uAlphaConsideration"), alpha_consideration ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uAlphaRayTransparency"), alpha_ray_transparency ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uAlphaMaxSteps"), alpha_max_steps);
		glUniform1i(glGetUniformLocation(rt_program, "uAlphaShadowTransmission"), alpha_shadow_transmission ? 1 : 0);

		glUniform1i(glGetUniformLocation(rt_program, "uVolumeEnabled"), (volume_enabled && godot_vol_tex) ? 1 : 0);
		glUniform3f(glGetUniformLocation(rt_program, "uVolMin"), active_vol_min[0], active_vol_min[1], active_vol_min[2]);
		glUniform3f(glGetUniformLocation(rt_program, "uVolSize"), active_vol_size[0], active_vol_size[1], active_vol_size[2]);
		glUniform1f(glGetUniformLocation(rt_program, "uVolVoxelSize"), active_voxel_size);
		glUniform1f(glGetUniformLocation(rt_program, "uVolumeFeedback"), volume_feedback);

		glUniform1i(glGetUniformLocation(rt_program, "uCheckerboard"), checkerboard_tracing ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uTemporalNoise"), temporal_noise ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uFrameIdx"), (int)(frame_counter & 0x7FFFFFFF));
		glUniform1i(glGetUniformLocation(rt_program, "uDebugMode"), (int)debug_mode);

		gl43_ctx->dispatch_compute(gx, gy, 1);
		gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

		// 3C. Checkerboard / Temporal Reconstruction + Spatial Denoise
		if (need_post) {
			GLuint cur_tex = vst.gi_work_a;

			if (need_reconstruct) {
				int next_hist = 1 - vst.history_idx;
				GLuint out_hist = temporal_accumulation ? vst.gi_history_tex[next_hist] : vst.gi_work_b;
				GLuint in_hist = (temporal_accumulation && vst.history_valid) ? vst.gi_history_tex[vst.history_idx] : cur_tex;

				glUseProgram(reconstruct_program);
				gl43_ctx->bind_image_texture(0, out_hist, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
				gl43_ctx->bind_image_texture(1, need_denoise ? vst.gi_work_b : godot_gi_tex->tex_id, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, cur_tex);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uRawTex"), 0);

				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D, in_hist);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uHistoryTex"), 1);

				glActiveTexture(GL_TEXTURE2);
				glBindTexture(GL_TEXTURE_2D, vst.low_normal_mask);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uGNormalMask"), 2);

				glActiveTexture(GL_TEXTURE3);
				glBindTexture(GL_TEXTURE_2D, vst.low_depth);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uGDepth"), 3);

				glUniformMatrix4fv(glGetUniformLocation(reconstruct_program, "uInvViewProj"), 1, GL_FALSE, vsnap.inv_view_proj);
				glUniformMatrix4fv(glGetUniformLocation(reconstruct_program, "uPrevViewProj"), 1, GL_FALSE, vst.prev_view_proj);
				glUniform3f(glGetUniformLocation(reconstruct_program, "uCamPos"), vsnap.cam_pos[0], vsnap.cam_pos[1], vsnap.cam_pos[2]);
				glUniform3f(glGetUniformLocation(reconstruct_program, "uPrevCamPos"), vst.prev_cam_pos[0], vst.prev_cam_pos[1], vst.prev_cam_pos[2]);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uCheckerboard"), checkerboard_tracing ? 1 : 0);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uTemporalAccum"), temporal_accumulation ? 1 : 0);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uHistoryValid"), vst.history_valid ? 1 : 0);
				glUniform1f(glGetUniformLocation(reconstruct_program, "uTemporalBlend"), temporal_blend);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uWriteFinal"), need_denoise ? 0 : 1);

				gl43_ctx->dispatch_compute(gx, gy, 1);
				gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

				if (temporal_accumulation) {
					vst.history_idx = next_hist;
					vst.history_valid = true;
					std::memcpy(vst.prev_view_proj, vsnap.view_proj_mat, sizeof(vst.prev_view_proj));
					std::memcpy(vst.prev_cam_pos, vsnap.cam_pos, sizeof(vst.prev_cam_pos));
				}
				cur_tex = out_hist;
			}

			if (need_denoise) {
				glUseProgram(denoise_program);
				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D, vst.low_normal_mask);
				glUniform1i(glGetUniformLocation(denoise_program, "uGNormalMask"), 1);

				glActiveTexture(GL_TEXTURE2);
				glBindTexture(GL_TEXTURE_2D, vst.low_depth);
				glUniform1i(glGetUniformLocation(denoise_program, "uGDepth"), 2);

				glUniformMatrix4fv(glGetUniformLocation(denoise_program, "uInvViewProj"), 1, GL_FALSE, vsnap.inv_view_proj);
				glUniform3f(glGetUniformLocation(denoise_program, "uCamPos"), vsnap.cam_pos[0], vsnap.cam_pos[1], vsnap.cam_pos[2]);

				for (int p = 0; p < spatial_denoise_passes; p++) {
					bool is_last = (p + 1 == spatial_denoise_passes);
					GLuint dst_tex = is_last ? godot_gi_tex->tex_id : ((cur_tex == vst.gi_work_a) ? vst.gi_work_b : vst.gi_work_a);
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
		glBindTexture(GL_TEXTURE_2D, godot_gi_tex->tex_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	for (int i = 0; i < 8; i++) {
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)i, 0);
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindSampler(4, 0);
	for (int i = 0; i < 5; i++) {
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

// =========================================================
// PROPERTY SETTERS & PRESETS
// =========================================================

void RTGI::mark_geometry_dirty() {
	bvh_dirty = true;
	raytracer.invalidate_blas_cache();
}

void RTGI::reset_history() {
	for (auto &kv : viewport_states) {
		kv.second.history_valid = false;
	}
	vol_hist_valid = false;
}

void RTGI::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	if (!enabled) {
		_detach_all_viewports();
		reset_history();
	}
}

void RTGI::set_editor_preview(bool p_preview) {
	editor_preview = p_preview;
	if (!editor_preview && Engine::get_singleton()->is_editor_hint()) {
		_detach_all_viewports();
		reset_history();
	}
}

void RTGI::set_auto_composite(bool p_auto) {
	auto_composite = p_auto;
	if (!auto_composite) {
		_detach_all_viewports();
	}
}

void RTGI::set_gi_intensity(float p_intensity) {
	gi_intensity = MAX(p_intensity, 0.0f);
}

void RTGI::set_resolution_scale(float p_scale) {
	float clamped = CLAMP(p_scale, 0.15f, 1.0f);
	if (std::abs(clamped - resolution_scale) > 0.001f) {
		resolution_scale = clamped;
		reset_history();
	}
}

void RTGI::set_temporal_accumulation(bool p_enable) {
	if (temporal_accumulation != p_enable) {
		temporal_accumulation = p_enable;
		reset_history();
		_change_notify();
	}
}

void RTGI::set_temporal_blend(float p_blend) {
	temporal_blend = CLAMP(p_blend, 0.02f, 1.0f);
}

void RTGI::set_spatial_denoise_passes(int p_passes) {
	spatial_denoise_passes = CLAMP(p_passes, 0, 5);
}

void RTGI::set_firefly_clamp(float p_clamp) {
	firefly_clamp = CLAMP(p_clamp, 0.0f, 50.0f);
}

void RTGI::set_rays_per_pixel(int p_rays) {
	rays_per_pixel = CLAMP(p_rays, 1, 16);
}

void RTGI::set_max_bounces(int p_bounces) {
	max_bounces = CLAMP(p_bounces, 1, 4);
}

void RTGI::set_max_ray_distance(float p_dist) {
	max_ray_distance = MAX(p_dist, 2.0f);
}

void RTGI::set_bounce_distance_falloff(float p_falloff) {
	bounce_distance_falloff = CLAMP(p_falloff, 0.1f, 1.0f);
}

void RTGI::set_bounce_energy_boost(float p_boost) {
	bounce_energy_boost = CLAMP(p_boost, 0.0f, 4.0f);
}

void RTGI::set_bounce_throughput_threshold(float p_thresh) {
	bounce_throughput_threshold = CLAMP(p_thresh, 0.0f, 0.25f);
}

void RTGI::set_emissive_intensity(float p_intensity) {
	emissive_intensity = MAX(p_intensity, 0.0f);
}

void RTGI::set_ray_origin_bias(float p_bias) {
	ray_origin_bias = CLAMP(p_bias, 0.0005f, 0.05f);
}

void RTGI::set_sky_indirect_enabled(bool p_enable) {
	sky_indirect_enabled = p_enable;
	_change_notify();
}

void RTGI::set_sky_indirect_energy(float p_energy) {
	sky_indirect_energy = CLAMP(p_energy, 0.0f, 8.0f);
}

void RTGI::set_custom_sky_mode(CustomSkyMode p_mode) {
	custom_sky_mode = p_mode;
	sky_sh_dirty = true;
	_change_notify();
}

void RTGI::set_custom_sky(const Ref<Sky> &p_sky) {
	custom_sky = p_sky;
	sky_sh_dirty = true;
}

void RTGI::set_custom_sky_color(const Color &p_color) {
	custom_sky_color = p_color;
	sky_sh_dirty = true;
}

void RTGI::set_custom_sky_energy(float p_energy) {
	custom_sky_energy = CLAMP(p_energy, 0.0f, 16.0f);
	sky_sh_dirty = true;
}

void RTGI::set_ambient_bounce_approximation(float p_approx) {
	ambient_bounce_approximation = CLAMP(p_approx, 0.0f, 1.0f);
}

void RTGI::set_volume_enabled(bool p_enable) {
	volume_enabled = p_enable;
	vol_hist_valid = false;
	_change_notify();
}

void RTGI::set_volume_extents(const Vector3 &p_extents) {
	volume_extents = Vector3(MAX(p_extents.x, 1.0f), MAX(p_extents.y, 1.0f), MAX(p_extents.z, 1.0f));
	vol_hist_valid = false;
}

void RTGI::set_volume_resolution(const Vector3 &p_res) {
	volume_resolution = Vector3(
			CLAMP((int)Math::round(p_res.x), 4, 128),
			CLAMP((int)Math::round(p_res.y), 4, 128),
			CLAMP((int)Math::round(p_res.z), 4, 128));
	vol_hist_valid = false;
}

void RTGI::set_volume_rays_per_voxel(int p_rays) {
	volume_rays_per_voxel = CLAMP(p_rays, 1, 32);
}

void RTGI::set_volume_ray_distance(float p_dist) {
	volume_ray_distance = MAX(p_dist, 2.0f);
}

void RTGI::set_volume_feedback(float p_feedback) {
	volume_feedback = CLAMP(p_feedback, 0.0f, 1.0f);
}

void RTGI::set_volume_update_interval(int p_interval) {
	volume_update_interval = CLAMP(p_interval, 1, 10);
}

void RTGI::set_alpha_consideration(bool p_enable) {
	alpha_consideration = p_enable;
	_change_notify();
}

void RTGI::set_alpha_max_steps(int p_steps) {
	alpha_max_steps = CLAMP(p_steps, 1, 16);
}

void RTGI::set_mesh_simplification_enabled(bool p_enable) {
	mesh_simplification_enabled = p_enable;
	mark_geometry_dirty();
	_change_notify();
}

void RTGI::set_mesh_simplification_mode(SimplificationMode p_mode) {
	mesh_simplification_mode = p_mode;
	mark_geometry_dirty();
}

void RTGI::set_mesh_simplification_ratio(float p_ratio) {
	mesh_simplification_ratio = CLAMP(p_ratio, 0.05f, 1.0f);
	mark_geometry_dirty();
}

void RTGI::set_mesh_simplification_min_triangles(int p_min_tris) {
	mesh_simplification_min_triangles = MAX(p_min_tris, 0);
	mark_geometry_dirty();
}

void RTGI::set_mesh_simplification_max_error(float p_max_error) {
	mesh_simplification_max_error = CLAMP(p_max_error, 0.0f, 1.0f);
	mark_geometry_dirty();
}

void RTGI::set_mesh_simplification_preserve_seams(bool p_preserve) {
	mesh_simplification_preserve_seams = p_preserve;
	mark_geometry_dirty();
}

void RTGI::set_mesh_simplification_skinned(bool p_enable) {
	mesh_simplification_skinned = p_enable;
	mark_geometry_dirty();
}

void RTGI::set_mesh_simplification_gbuffer(bool p_enable) {
	mesh_simplification_gbuffer = p_enable;
	mark_geometry_dirty();
}

void RTGI::set_trace_shadows(bool p_enable) {
	trace_shadows = p_enable;
	_change_notify();
}

void RTGI::set_shadow_bounces(int p_bounces) {
	shadow_bounces = CLAMP(p_bounces, 0, 4);
}

void RTGI::set_shadow_max_distance(float p_dist) {
	shadow_max_distance = MAX(p_dist, 0.0f);
}

void RTGI::set_shadow_ray_bias(float p_bias) {
	shadow_ray_bias = CLAMP(p_bias, 0.0005f, 0.05f);
}

void RTGI::set_update_interval(int p_interval) {
	update_interval = CLAMP(p_interval, 1, 10);
}

void RTGI::set_blas_builds_per_frame(int p_budget) {
	blas_builds_per_frame = CLAMP(p_budget, 0, 64);
}

void RTGI::set_max_tlas_instances(int p_max) {
	max_tlas_instances = CLAMP(p_max, 0, 8192);
}

void RTGI::set_texture_array_resolution(TextureResolution p_res) {
	if (texture_array_resolution != p_res) {
		texture_array_resolution = p_res;
		tex_pool.mark_reset_needed();
	}
}

void RTGI::set_max_texture_layers(int p_layers) {
	int clamped = CLAMP(p_layers, 8, 256);
	if (max_texture_layers != clamped) {
		max_texture_layers = clamped;
		tex_pool.mark_reset_needed();
	}
}

void RTGI::set_texture_lod_bias(float p_bias) {
	texture_lod_bias = CLAMP(p_bias, -2.0f, 6.0f);
}

void RTGI::set_distance_lod_strength(float p_strength) {
	distance_lod_strength = CLAMP(p_strength, 0.0f, 4.0f);
}

void RTGI::set_mesh_cull_distance(float p_dist) {
	mesh_cull_distance = MAX(p_dist, 1.0f);
}

void RTGI::set_mesh_cull_min_angle(float p_angle) {
	mesh_cull_min_angle = CLAMP(p_angle, 0.0f, 0.2f);
}

void RTGI::set_light_cull_distance(float p_dist) {
	light_cull_distance = MAX(p_dist, 1.0f);
}

void RTGI::apply_quality_preset(QualityPreset p_preset) {
	switch (p_preset) {
		case PRESET_LOW: {
			resolution_scale = 0.25f;
			bilateral_upsample = false;
			checkerboard_tracing = true;
			temporal_accumulation = true;
			temporal_blend = 0.2f;
			spatial_denoise_passes = 2;
			rays_per_pixel = 1;
			max_bounces = 1;
			max_ray_distance = 35.0f;
			bounce_local_lights = false;
			normal_maps_in_bounces = false;
			skinned_meshes_in_gi = false;
			volume_enabled = true;
			volume_resolution = Vector3(16, 8, 16);
			volume_rays_per_voxel = 4;
			volume_update_interval = 4;
			alpha_ray_transparency = false;
			alpha_shadow_transmission = false;
			mesh_simplification_enabled = true;
			mesh_simplification_mode = SIMPLIFY_VERTEX_CLUSTERING;
			mesh_simplification_ratio = 0.35f;
			trace_shadows = true;
			shadow_bounces = 1;
			shadow_max_distance = 40.0f;
			texture_array_resolution = TEX_RES_128;
			max_texture_layers = 32;
			blas_builds_per_frame = 4;
			mesh_cull_distance = 25.0f;
			mesh_cull_min_angle = 0.025f;
			light_cull_distance = 60.0f;
		} break;
		case PRESET_MEDIUM: {
			resolution_scale = 0.5f;
			bilateral_upsample = true;
			checkerboard_tracing = false;
			temporal_accumulation = true;
			temporal_blend = 0.14f;
			spatial_denoise_passes = 3;
			rays_per_pixel = 2;
			max_bounces = 2;
			max_ray_distance = 60.0f;
			bounce_local_lights = true;
			normal_maps_in_bounces = true;
			skinned_meshes_in_gi = true;
			volume_enabled = true;
			volume_resolution = Vector3(24, 12, 24);
			volume_rays_per_voxel = 6;
			volume_update_interval = 2;
			alpha_ray_transparency = true;
			alpha_max_steps = 4;
			alpha_shadow_transmission = true;
			mesh_simplification_enabled = true;
			mesh_simplification_mode = SIMPLIFY_QEM;
			mesh_simplification_ratio = 0.65f;
			trace_shadows = true;
			shadow_bounces = 2;
			shadow_max_distance = 80.0f;
			texture_array_resolution = TEX_RES_256;
			max_texture_layers = 64;
			blas_builds_per_frame = 8;
			mesh_cull_distance = 45.0f;
			mesh_cull_min_angle = 0.012f;
			light_cull_distance = 120.0f;
		} break;
		case PRESET_HIGH: {
			resolution_scale = 0.75f;
			bilateral_upsample = true;
			checkerboard_tracing = false;
			temporal_accumulation = true;
			temporal_blend = 0.1f;
			spatial_denoise_passes = 3;
			rays_per_pixel = 4;
			max_bounces = 2;
			max_ray_distance = 100.0f;
			bounce_local_lights = true;
			normal_maps_in_bounces = true;
			skinned_meshes_in_gi = true;
			volume_enabled = true;
			volume_resolution = Vector3(32, 16, 32);
			volume_rays_per_voxel = 8;
			volume_update_interval = 1;
			alpha_ray_transparency = true;
			alpha_max_steps = 6;
			alpha_shadow_transmission = true;
			mesh_simplification_enabled = false;
			trace_shadows = true;
			shadow_bounces = 2;
			shadow_max_distance = 140.0f;
			texture_array_resolution = TEX_RES_512;
			max_texture_layers = 96;
			blas_builds_per_frame = 0;
			mesh_cull_distance = 80.0f;
			mesh_cull_min_angle = 0.008f;
			light_cull_distance = 180.0f;
		} break;
		case PRESET_ULTRA: {
			resolution_scale = 1.0f;
			bilateral_upsample = true;
			checkerboard_tracing = false;
			temporal_accumulation = true;
			temporal_blend = 0.08f;
			spatial_denoise_passes = 4;
			rays_per_pixel = 8;
			max_bounces = 3;
			max_ray_distance = 200.0f;
			bounce_local_lights = true;
			normal_maps_in_bounces = true;
			skinned_meshes_in_gi = true;
			volume_enabled = true;
			volume_resolution = Vector3(48, 24, 48);
			volume_rays_per_voxel = 12;
			volume_update_interval = 1;
			alpha_ray_transparency = true;
			alpha_max_steps = 8;
			alpha_shadow_transmission = true;
			mesh_simplification_enabled = false;
			trace_shadows = true;
			shadow_bounces = 3;
			shadow_max_distance = 250.0f;
			texture_array_resolution = TEX_RES_512;
			max_texture_layers = 128;
			blas_builds_per_frame = 0;
			mesh_cull_distance = 150.0f;
			mesh_cull_min_angle = 0.0f;
			light_cull_distance = 300.0f;
		} break;
	}
	reset_history();
	tex_pool.mark_reset_needed();
	mark_geometry_dirty();
	_change_notify();
}
