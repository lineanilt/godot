#ifndef VOLUMETRIC_FOG_H
#define VOLUMETRIC_FOG_H

#include "core/math/aabb.h"
#include "core/math/camera_matrix.h"
#include "core/math/transform.h"
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "drivers/gles3/gl43_effect_helpers.h"
#include "scene/3d/spatial.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/texture.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

class Camera;
class Viewport;
class DirectionalLight;
class SpotLight;
class OmniLight;
class LocalFogVolume;
class VolumetricFogFocus;
class BakedFogVolume;
class MeshInstance;

class VolumetricFog : public Spatial {
	GDCLASS(VolumetricFog, Spatial);

public:
	enum StepDistribution {
		STEP_LINEAR = 0,
		STEP_QUADRATIC = 1,
	};

	enum {
		MAX_DIR_LIGHTS = 2,
		MAX_SPOT_LIGHTS = 6,
		MAX_OMNI_LIGHTS = 16,
		MAX_LOCAL_VOLUMES = 16,
		MAX_BAKED_VOLUMES = 4,
	};

	// Std430-aligned GPU SSBO structs (all multiples of 16 bytes)
	struct GpuDirLight {
		float vp[16];
		float dir_bias[4];    // xyz = dir toward light, w = bias
		float color_flags[4]; // rgb = linear color * energy, w = has_shadow (0/1)
		float proj_color[4];  // rgb = projector_color_2 * energy, a = unused
		float proj_uv[4];     // xy = scale, zw = offset
		float proj_extra[4];  // x = rotation, y = repeat, z = lod, w = has_proj (0/1)
	};

	struct GpuSpotLight {
		float vp[16];
		float pos_range[4];   // xyz = pos, w = range
		float dir_atten[4];   // xyz = forward, w = atten
		float color_cone[4];  // rgb = linear color * energy, w = cos(angle)
		float params[4];      // x = spot_atten, y = bias, z = is_square, w = has_shadow
	};

	struct GpuOmniLight {
		float pos_range[4];   // xyz = world pos, w = range
		float view_atten[4];  // xyz = view pos, w = atten
		float color_flags[4]; // rgb = linear color * energy, w = bitmask (1=square, 2=shadow)
	};

	struct GpuLocalVolume {
		float inv_xf_row0[4];
		float inv_xf_row1[4];
		float inv_xf_row2[4];
		float albedo_density[4];   // rgb, density
		float emission_falloff[4]; // rgb, edge_falloff
		float params[4];           // shape, blend_mode, noise_strength, noise_scale
		float noise_offset[4];     // xyz, 0
	};

	struct GpuBakedVolume {
		float inv_xf_row0[4];
		float inv_xf_row1[4];
		float inv_xf_row2[4];
		float tint_dens[4]; // rgb, density_mult
		float params[4];    // emission_mult, edge_falloff, 0, 0
	};

private:
	struct SurfaceRange {
		int32_t first_vert = 0;
		int32_t vert_count = 0;
		int32_t cull_mode = 0; // 0=back, 1=front, 2=disabled
		bool alpha_scissor = false;
		float alpha_threshold = 0.5f;
		RID albedo_tex_rid;
	};

	struct MeshGeom {
		uint32_t mesh_id = 0;
		uint64_t cached_mesh_instance_id = 0;
		AABB local_aabb;
		AABB cached_mesh_aabb;
		std::vector<float> verts; // 5 floats per vertex: x, y, z, u, v
		std::vector<SurfaceRange> surfaces;
		int32_t total_verts = 0;

		bool has_skin_weights = false;
		int bones_per_vert = 4;
		std::vector<Vector3> bind_pos;
		std::vector<Vector2> bind_uv;
		std::vector<int32_t> bind_bones;
		std::vector<float> bind_weights;
		std::vector<uint32_t> tri_verts;

		uint32_t gl_vao = 0;
		uint32_t gl_vbo = 0;
		bool gpu_uploaded = false;
	};

	struct SkinnedMeshGL {
		uint32_t gl_vao = 0;
		uint32_t gl_vbo = 0;
		int32_t vertex_count = 0;
		uint64_t last_frame_seen = 0;
	};

	struct SkinnedSnapshot {
		uint64_t obj_id = 0;
		std::vector<float> posed_verts;
		int32_t vertex_count = 0;
	};

	struct DepthInstance {
		uint32_t mesh_id = 0;
		bool is_skinned = false;
		int32_t skinned_snap_idx = -1;
		float model_mat[16] = {};
		AABB world_aabb;
		uint32_t layer_mask = 0xFFFFFFFF;
		bool draw_in_cam = true;
		bool cast_shadows = true;
		bool double_sided_shadow = false;
		bool flip_facing = false;
		std::vector<SurfaceRange> surfaces;
	};

	struct DirShadowMeta {
		float vp[16] = {};
		uint32_t cull_mask = 0xFFFFFFFF;
		bool has_shadow = false;
		AABB frustum_aabb;
		RID proj_tex_rid;
	};

	struct SpotShadowMeta {
		float vp[16] = {};
		uint32_t cull_mask = 0xFFFFFFFF;
		bool has_shadow = false;
		Vector3 pos;
		float range = 10.0f;
	};

	struct ViewportSnapshot {
		uint32_t vp_id = 0;
		RID vp_rid;
		int viewport_w = 0;
		int viewport_h = 0;
		int fog_w = 0;
		int fog_h = 0;

		float cam_pos[3] = { 0, 0, 0 };
		float view_mat[16] = {};
		float proj_mat[16] = {};
		float vp_mat[16] = {};
		float inv_proj_mat[16] = {};
		float inv_vp_mat[16] = {};
		float cam_frustum_planes[6][4] = {};

		float amb_color[3] = { 0.05f, 0.05f, 0.06f };
		int tonemap_mode = 0;
		float tonemap_exposure = 1.0f;
		float tonemap_white = 1.0f;
		bool hdr = true;
		bool keep_3d_linear = false;

		int effective_steps = 24;
		float eff_max_distance = 120.0f;
		bool focus_active = false;
		float focus_pos[3] = { 0, 0, 0 };
		float focus_radius = 30.0f;
		float focus_falloff = 0.25f;
		float focus_density_mult = 1.0f;
	};

	struct ViewportRenderState {
		RID color_tex_rid;
		bool active_composite = false;
		Ref<ImageTexture> fog_texture;
		Ref<ImageTexture> low_depth_texture;
		int fog_tex_w = 0;
		int fog_tex_h = 0;

		uint32_t cam_depth_fbo = 0;
		uint32_t cam_depth_tex = 0;
		int cam_depth_w = 0;
		int cam_depth_h = 0;

		uint32_t fog_work_a = 0;
		uint32_t fog_work_b = 0;
		uint32_t fog_history_tex[2] = { 0, 0 };
		uint32_t prev_linear_depth_tex = 0;
		int work_w = 0;
		int work_h = 0;

		int history_idx = 0;
		bool history_valid = false;
		float prev_view_mat[16] = {};
		float prev_vp_mat[16] = {};
		uint64_t last_seen = 0;
	};

	struct SceneSnapshot {
		bool valid = false;
		std::vector<ViewportSnapshot> viewports;

		std::vector<DepthInstance> instances;
		std::vector<SkinnedSnapshot> skinned_instances;

		std::vector<GpuDirLight> dir_lights;
		std::vector<DirShadowMeta> dir_meta;

		std::vector<GpuSpotLight> spot_lights;
		std::vector<SpotShadowMeta> spot_meta;

		std::vector<GpuOmniLight> omni_lights;
		std::vector<GpuLocalVolume> local_volumes;
		std::vector<GpuBakedVolume> baked_volumes;
		std::vector<RID> baked_tex_rids;
	};

	// General
	bool enabled = true;
	bool editor_preview = true;
	bool auto_composite = true;
	float density = 0.02f;
	Color albedo = Color(0.9f, 0.92f, 0.95f, 1.0f);
	Color emission = Color(0.0f, 0.0f, 0.0f, 1.0f);
	float emission_energy = 0.0f;
	float anisotropy = 0.5f;
	float light_fog_ratio = 0.0f;
	bool use_environment_ambient = true;
	Color ambient_light_color = Color(0.15f, 0.18f, 0.22f, 1.0f);
	float ambient_light_energy = 0.2f;
	float sky_affect = 1.0f;

	// Height Fog
	bool height_fog_enabled = false;
	float height_fog_base_height = 0.0f;
	float height_fog_falloff = 0.2f;

	// Noise
	bool noise_enabled = false;
	float noise_scale = 0.1f;
	float noise_strength = 0.5f;
	Vector3 noise_wind = Vector3(0.4f, 0.05f, 0.2f);

	// Raymarching & Quality
	int steps = 24;
	int editor_max_steps = 16;
	float max_distance = 120.0f;
	float distance_fade = 0.2f;
	StepDistribution step_distribution = STEP_QUADRATIC;
	float jitter = 0.5f;
	float resolution_scale = 0.5f;
	bool temporal_reprojection = true;
	float temporal_blend = 0.18f;
	int spatial_denoise_passes = 1;
	bool bilateral_upsample = true;

	// Directional Lights (Depth-Texture Based)
	bool directional_lights_enabled = true;
	int directional_depth_resolution = 512;
	float directional_shadow_distance = 80.0f;
	float directional_depth_range = 200.0f;
	float directional_depth_bias = 0.002f;
	int directional_update_interval = 1;
	uint32_t directional_cull_mask_override = 0;

	// Spot Lights (Depth-Texture Based)
	bool spot_lights_enabled = true;
	int max_spot_lights = 4;
	int spot_depth_resolution = 256;
	float spot_depth_bias = 0.08f;
	float spot_near_plane = 0.1f;
	int spot_update_interval = 1;
	uint32_t spot_cull_mask_override = 0;

	// Omni Lights (Screen-Space)
	bool omni_lights_enabled = true;
	int max_omni_lights = 16;
	bool omni_screen_space_shadows = true;
	int omni_ss_shadow_steps = 4;
	float omni_ss_shadow_thickness = 0.5f;
	float omni_ss_shadow_softness = 0.25f;
	float omni_ss_shadow_strength = 0.9f;

	// Exported Textures & Per-Viewport State
	Ref<ImageTexture> fog_texture;
	Ref<ImageTexture> low_depth_texture;
	std::unordered_map<uint32_t, ViewportRenderState> viewport_states;
	uint64_t vp_collect_counter = 0;
	bool hook_registered = false;

	// Thread-safe Snapshot & Mesh Geometry Cache
	Mutex scene_mutex;
	SceneSnapshot snapshot;
	std::unordered_map<uint32_t, MeshGeom> mesh_cache;
	std::unordered_map<uint64_t, SkinnedMeshGL> skinned_gl_cache;
	std::vector<uint32_t> gl_buffers_to_delete;
	std::vector<Ref<Texture>> pinned_textures;
	std::vector<Ref<Texture3D>> pinned_3d_textures;

	uint64_t frame_counter = 0;
	uint64_t dir_shadow_frame = 0;
	uint64_t spot_shadow_frame = 0;
	float noise_time = 0.0f;

	// Raw OpenGL 4.3 Resources
	bool gl_attempted = false;
	bool gl_ready = false;
	Thread::ID gl_thread_id = 0;

	uint32_t depth_program = 0;
	uint32_t raymarch_program = 0;
	uint32_t temporal_program = 0;
	uint32_t denoise_program = 0;
	uint32_t finalize_program = 0;
	uint32_t composite_program = 0;

	uint32_t dir_depth_fbo = 0;
	uint32_t dir_depth_array = 0;
	int dir_depth_res_alloc = 0;
	bool dir_depth_force_refresh = true;

	uint32_t spot_depth_fbo = 0;
	uint32_t spot_depth_array = 0;
	int spot_depth_res_alloc = 0;
	bool spot_depth_force_refresh = true;

	uint32_t dummy_gl_2d = 0;
	uint32_t dummy_gl_3d = 0;

	uint32_t ssbo_dir_lights = 0;
	uint32_t ssbo_spot_lights = 0;
	uint32_t ssbo_omni_lights = 0;
	uint32_t ssbo_local_vols = 0;
	uint32_t ssbo_baked_vols = 0;

	void _register_scene_hook();
	void _unregister_scene_hook();
	void _detach_all_viewports();
	ViewportRenderState &_get_or_create_vp_state(Viewport *p_vp);
	void _prune_stale_viewports();
	void _on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx);

	bool _init_gl_resources();
	void _cleanup_gl_resources();
	void _ensure_cam_depth_size(ViewportRenderState &r_state, int p_w, int p_h);
	void _ensure_dir_depth_array(int p_res);
	void _ensure_spot_depth_array(int p_res);
	void _ensure_work_textures(ViewportRenderState &r_state, int p_w, int p_h);

	void _rebuild_mesh_cache(const std::vector<MeshInstance *> &p_meshes, const std::vector<Ref<Mesh>> &p_extra_meshes);
	bool _build_skinned_snapshot(MeshInstance *p_mi, const MeshGeom &p_geom, SkinnedSnapshot &r_snap, AABB &r_local_aabb) const;
	void _collect_scene_snapshot();

	void _on_frame_pre_draw();

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &property) const;

public:
	VolumetricFog();
	~VolumetricFog();

	// General
	void set_enabled(bool p_enabled);
	bool is_enabled() const { return enabled; }

	void set_editor_preview(bool p_preview);
	bool get_editor_preview() const { return editor_preview; }

	void set_auto_composite(bool p_auto);
	bool get_auto_composite() const { return auto_composite; }

	void set_density(float p_density);
	float get_density() const { return density; }

	void set_albedo(const Color &p_albedo) { albedo = p_albedo; }
	Color get_albedo() const { return albedo; }

	void set_emission(const Color &p_emission) { emission = p_emission; }
	Color get_emission() const { return emission; }

	void set_emission_energy(float p_energy);
	float get_emission_energy() const { return emission_energy; }

	void set_anisotropy(float p_anisotropy);
	float get_anisotropy() const { return anisotropy; }

	void set_light_fog_ratio(float p_ratio);
	float get_light_fog_ratio() const { return light_fog_ratio; }

	void set_use_environment_ambient(bool p_use);
	bool get_use_environment_ambient() const { return use_environment_ambient; }

	void set_ambient_light_color(const Color &p_color) { ambient_light_color = p_color; }
	Color get_ambient_light_color() const { return ambient_light_color; }

	void set_ambient_light_energy(float p_energy);
	float get_ambient_light_energy() const { return ambient_light_energy; }

	void set_sky_affect(float p_affect);
	float get_sky_affect() const { return sky_affect; }

	// Height Fog
	void set_height_fog_enabled(bool p_enable);
	bool is_height_fog_enabled() const { return height_fog_enabled; }

	void set_height_fog_base_height(float p_height) { height_fog_base_height = p_height; }
	float get_height_fog_base_height() const { return height_fog_base_height; }

	void set_height_fog_falloff(float p_falloff);
	float get_height_fog_falloff() const { return height_fog_falloff; }

	// Noise
	void set_noise_enabled(bool p_enable);
	bool is_noise_enabled() const { return noise_enabled; }

	void set_noise_scale(float p_scale);
	float get_noise_scale() const { return noise_scale; }

	void set_noise_strength(float p_strength);
	float get_noise_strength() const { return noise_strength; }

	void set_noise_wind(const Vector3 &p_wind) { noise_wind = p_wind; }
	Vector3 get_noise_wind() const { return noise_wind; }

	// Raymarching & Quality
	void set_steps(int p_steps);
	int get_steps() const { return steps; }

	void set_editor_max_steps(int p_steps);
	int get_editor_max_steps() const { return editor_max_steps; }

	void set_max_distance(float p_distance);
	float get_max_distance() const { return max_distance; }

	void set_distance_fade(float p_fade);
	float get_distance_fade() const { return distance_fade; }

	void set_step_distribution(StepDistribution p_dist) { step_distribution = p_dist; }
	StepDistribution get_step_distribution() const { return step_distribution; }

	void set_jitter(float p_jitter);
	float get_jitter() const { return jitter; }

	void set_resolution_scale(float p_scale);
	float get_resolution_scale() const { return resolution_scale; }

	void set_temporal_reprojection(bool p_enable);
	bool get_temporal_reprojection() const { return temporal_reprojection; }

	void set_temporal_blend(float p_blend);
	float get_temporal_blend() const { return temporal_blend; }

	void set_spatial_denoise_passes(int p_passes);
	int get_spatial_denoise_passes() const { return spatial_denoise_passes; }

	void set_bilateral_upsample(bool p_enable) { bilateral_upsample = p_enable; }
	bool get_bilateral_upsample() const { return bilateral_upsample; }

	// Directional Lights
	void set_directional_lights_enabled(bool p_enable) { directional_lights_enabled = p_enable; }
	bool are_directional_lights_enabled() const { return directional_lights_enabled; }

	void set_directional_depth_resolution(int p_res);
	int get_directional_depth_resolution() const { return directional_depth_resolution; }

	void set_directional_shadow_distance(float p_dist);
	float get_directional_shadow_distance() const { return directional_shadow_distance; }

	void set_directional_depth_range(float p_range);
	float get_directional_depth_range() const { return directional_depth_range; }

	void set_directional_depth_bias(float p_bias);
	float get_directional_depth_bias() const { return directional_depth_bias; }

	void set_directional_update_interval(int p_interval);
	int get_directional_update_interval() const { return directional_update_interval; }

	void set_directional_cull_mask_override(uint32_t p_mask) { directional_cull_mask_override = p_mask; }
	uint32_t get_directional_cull_mask_override() const { return directional_cull_mask_override; }

	// Spot Lights
	void set_spot_lights_enabled(bool p_enable) { spot_lights_enabled = p_enable; }
	bool are_spot_lights_enabled() const { return spot_lights_enabled; }

	void set_max_spot_lights(int p_max);
	int get_max_spot_lights() const { return max_spot_lights; }

	void set_spot_depth_resolution(int p_res);
	int get_spot_depth_resolution() const { return spot_depth_resolution; }

	void set_spot_depth_bias(float p_bias);
	float get_spot_depth_bias() const { return spot_depth_bias; }

	void set_spot_near_plane(float p_near);
	float get_spot_near_plane() const { return spot_near_plane; }

	void set_spot_update_interval(int p_interval);
	int get_spot_update_interval() const { return spot_update_interval; }

	void set_spot_cull_mask_override(uint32_t p_mask) { spot_cull_mask_override = p_mask; }
	uint32_t get_spot_cull_mask_override() const { return spot_cull_mask_override; }

	// Omni Lights (Screen-Space)
	void set_omni_lights_enabled(bool p_enable) { omni_lights_enabled = p_enable; }
	bool are_omni_lights_enabled() const { return omni_lights_enabled; }

	void set_max_omni_lights(int p_max);
	int get_max_omni_lights() const { return max_omni_lights; }

	void set_omni_screen_space_shadows(bool p_enable) { omni_screen_space_shadows = p_enable; }
	bool get_omni_screen_space_shadows() const { return omni_screen_space_shadows; }

	void set_omni_ss_shadow_steps(int p_steps);
	int get_omni_ss_shadow_steps() const { return omni_ss_shadow_steps; }

	void set_omni_ss_shadow_thickness(float p_thickness);
	float get_omni_ss_shadow_thickness() const { return omni_ss_shadow_thickness; }

	void set_omni_ss_shadow_softness(float p_softness);
	float get_omni_ss_shadow_softness() const { return omni_ss_shadow_softness; }

	void set_omni_ss_shadow_strength(float p_strength);
	float get_omni_ss_shadow_strength() const { return omni_ss_shadow_strength; }

	Ref<Texture> get_fog_texture() const { return fog_texture; }
};

VARIANT_ENUM_CAST(VolumetricFog::StepDistribution);

#endif // VOLUMETRIC_FOG_H
