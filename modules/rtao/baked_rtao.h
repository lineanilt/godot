#ifndef BAKED_RTAO_H
#define BAKED_RTAO_H

#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "drivers/gles3/gl43_bvh_helper.h"
#include "drivers/gles3/gl43_effect_helpers.h"
#include "drivers/gles3/gl43_raytracer.h"
#include "scene/3d/spatial.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/texture.h"

#include <unordered_map>
#include <vector>

class Camera;
class MeshInstance;
class Viewport;

class BakedRTAO : public Spatial {
	GDCLASS(BakedRTAO, Spatial);

public:
	enum BlendMode {
		BLEND_MULTIPLY = 0,
		BLEND_MIN = 1,
		BLEND_OVERRIDE = 2,
	};

	enum DebugMode {
		DEBUG_NONE = 0,
		DEBUG_AO = 1,
		DEBUG_BENT_NORMAL = 2,
	};

private:
	bool enabled = true;
	bool editor_preview = true;
	bool auto_composite = true;
	Vector3 extents = Vector3(8.0f, 4.0f, 8.0f);
	Vector3 resolution = Vector3(32, 16, 32);
	Ref<Texture3D> baked_texture;
	PoolByteArray baked_data;
	Vector3 baked_data_resolution = Vector3(0, 0, 0);

	void _rebuild_texture_from_baked_data();

	float intensity = 1.0f;
	Color ao_color = Color(0.0f, 0.0f, 0.0f, 1.0f);
	float edge_falloff = 0.15f;
	float direct_light_preservation = 0.5f;
	BlendMode blend_mode = BLEND_MULTIPLY;
	uint32_t cull_mask = 0xFFFFFFFF;
	uint32_t viewing_cull_mask = 0xFFFFFFFF;
	DebugMode debug_mode = DEBUG_NONE;

	// Bake settings
	int bake_samples = 32;
	float bake_radius = 3.0f;
	float bake_ray_bias = 0.05f;
	float bake_power = 1.0f;
	bool bake_local_volumes = true;
	bool bake_directional_gi = true;
	float bake_gi_direction_strength = 1.0f;
	int bake_denoise_passes = 2;
	bool bake_filter = true;
	uint32_t shadow_cull_mask = 0xFFFFFFFF;

	// Standalone Lightweight VisualServer Compositor (Used when RTAO is not active)
	struct MaskMaterial {
		float uv_scale_offset[4];
		float params[4];  // x = alpha_base, y = alpha_scissor
		int32_t flags[4]; // x = albedo_layer, y = alpha_mode, z = cull_mode
	};

	struct MaskViewportTarget {
		uint32_t vp_id = 0;
		int width = 0;
		int height = 0;
		uint32_t cam_cull_mask = 0xFFFFFFFF;
		uint32_t viewing_mask = 0xFFFFFFFF;
		float vp_mat[16] = {};
		float frustum_planes[6][4] = {};
		RID mask_tex_rid;
	};

	struct MaskSnapshot {
		bool valid = false;
		std::vector<MaskViewportTarget> viewports;
		std::vector<MaskMaterial> materials;
	};

	struct ViewportCompositeState {
		RID color_tex_rid;
		bool active_composite = false;
		Ref<ImageTexture> mask_texture;
		int mask_w = 0;
		int mask_h = 0;
		uint32_t mask_fbo = 0;
		uint32_t mask_depth_tex = 0;
		int alloc_depth_w = 0;
		int alloc_depth_h = 0;
		uint64_t last_seen = 0;

		bool use_viewing_mask = false;
		float inv_xform[16] = {};
		float cam_matrix[16] = {};
		float voxel_size = 0.5f;
		int num_lights = 0;
		float ambient_lum = 0.15f;
		bool use_physical_atten = false;
		float light_pos_range[32 * 4] = {};
		float light_dir_type[32 * 4] = {};
		float light_color_atten[32 * 4] = {};
		float light_spot_params[32 * 4] = {};
	};
	std::unordered_map<uint32_t, ViewportCompositeState> viewport_states;
	uint64_t vp_collect_counter = 0;
	bool hook_registered = false;

	Mutex mask_mutex;
	MaskSnapshot mask_snapshot;
	GL43Raytracer mask_raytracer;
	GL43TextureArrayPool mask_tex_pool;
	bool gl_attempted = false;
	bool gl_ready = false;
	Thread::ID gl_thread_id = 0;
	uint32_t mask_program = 0;
	uint32_t composite_program = 0;
	uint32_t mask_ssbo_materials = 0;

	void _register_scene_hook();
	void _unregister_scene_hook();
	void _detach_all_viewports();
	ViewportCompositeState &_get_or_create_vp_state(Viewport *p_vp);
	void _prune_stale_viewports();
	bool _is_rtao_active() const;
	void _collect_mask_snapshot(const std::vector<std::pair<Viewport *, Camera *>> &p_active_vps);
	bool _init_gl_mask_resources();
	void _cleanup_gl_mask_resources();
	void _ensure_mask_fbo_depth(ViewportCompositeState &r_state, int p_w, int p_h);
	void _render_standalone_viewing_mask();
	void _update_standalone_composite();
	void _on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx);
	void _on_frame_pre_draw();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	BakedRTAO();
	~BakedRTAO();

	void set_enabled(bool p_enabled);
	bool is_enabled() const { return enabled; }

	void set_editor_preview(bool p_preview);
	bool get_editor_preview() const { return editor_preview; }

	void set_auto_composite(bool p_auto);
	bool get_auto_composite() const { return auto_composite; }

	void set_extents(const Vector3 &p_extents);
	Vector3 get_extents() const { return extents; }

	void set_resolution(const Vector3 &p_res);
	Vector3 get_resolution() const { return resolution; }

	void set_baked_texture(const Ref<Texture3D> &p_tex);
	Ref<Texture3D> get_baked_texture() const { return baked_texture; }

	void set_baked_data(const PoolByteArray &p_data);
	PoolByteArray get_baked_data() const { return baked_data; }

	void set_baked_data_resolution(const Vector3 &p_res);
	Vector3 get_baked_data_resolution() const { return baked_data_resolution; }

	void set_intensity(float p_intensity);
	float get_intensity() const { return intensity; }

	void set_ao_color(const Color &p_color) { ao_color = p_color; }
	Color get_ao_color() const { return ao_color; }

	void set_edge_falloff(float p_falloff);
	float get_edge_falloff() const { return edge_falloff; }

	void set_direct_light_preservation(float p_val);
	float get_direct_light_preservation() const { return direct_light_preservation; }

	void set_blend_mode(BlendMode p_mode) { blend_mode = p_mode; }
	BlendMode get_blend_mode() const { return blend_mode; }

	void set_cull_mask(uint32_t p_mask) { cull_mask = p_mask; }
	uint32_t get_cull_mask() const { return cull_mask; }

	void set_viewing_cull_mask(uint32_t p_mask) { viewing_cull_mask = p_mask; }
	uint32_t get_viewing_cull_mask() const { return viewing_cull_mask; }

	void set_debug_mode(DebugMode p_mode) { debug_mode = p_mode; }
	DebugMode get_debug_mode() const { return debug_mode; }

	void set_bake_samples(int p_samples);
	int get_bake_samples() const { return bake_samples; }

	void set_bake_radius(float p_radius);
	float get_bake_radius() const { return bake_radius; }

	void set_bake_ray_bias(float p_bias);
	float get_bake_ray_bias() const { return bake_ray_bias; }

	void set_bake_power(float p_power);
	float get_bake_power() const { return bake_power; }

	void set_bake_local_volumes(bool p_enable) { bake_local_volumes = p_enable; }
	bool get_bake_local_volumes() const { return bake_local_volumes; }

	void set_bake_directional_gi(bool p_enable) { bake_directional_gi = p_enable; }
	bool get_bake_directional_gi() const { return bake_directional_gi; }

	void set_bake_gi_direction_strength(float p_strength);
	float get_bake_gi_direction_strength() const { return bake_gi_direction_strength; }

	void set_bake_denoise_passes(int p_passes);
	int get_bake_denoise_passes() const { return bake_denoise_passes; }

	void set_bake_filter(bool p_filter) { bake_filter = p_filter; }
	bool get_bake_filter() const { return bake_filter; }

	void set_shadow_cull_mask(uint32_t p_mask) { shadow_cull_mask = p_mask; }
	uint32_t get_shadow_cull_mask() const { return shadow_cull_mask; }

	AABB get_world_aabb() const;
	bool bake();
};

VARIANT_ENUM_CAST(BakedRTAO::BlendMode);
VARIANT_ENUM_CAST(BakedRTAO::DebugMode);

#endif // BAKED_RTAO_H
