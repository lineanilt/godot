#ifndef MOTION_BLUR_H
#define MOTION_BLUR_H

#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "drivers/gles3/gl43_effect_helpers.h"
#include "drivers/gles3/gl43_velocity_handler.h"
#include "scene/main/node.h"
#include "scene/resources/material.h"
#include "scene/resources/texture.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

class Camera;
class Viewport;

class MotionBlur : public Node {
	GDCLASS(MotionBlur, Node);

public:
	enum TileSize {
		TILE_8 = 8,
		TILE_16 = 16,
		TILE_24 = 24,
		TILE_32 = 32,
	};

	enum DebugMode {
		DEBUG_NONE = 0,
		DEBUG_VELOCITY = 1,
		DEBUG_NEIGHBOR_MAX = 2,
		DEBUG_SPEED = 3,
	};

private:
	struct ViewportRenderState {
		RID color_tex_rid;
		bool active_composite = false;
		int viewport_w = 0;
		int viewport_h = 0;
		Ref<ImageTexture> velocity_texture;
		Ref<ImageTexture> neighbor_max_texture;
		int tex_vel_w = 0;
		int tex_vel_h = 0;
		int tex_tile_w = 0;
		int tex_tile_h = 0;

		uint32_t tile_max_tex = 0;
		int alloc_tile_w = 0;
		int alloc_tile_h = 0;

		uint64_t last_seen = 0;
	};

	// Properties
	bool enabled = true;
	bool editor_preview = true;
	bool auto_composite = true;
	float intensity = 1.0f;
	int samples = 15;
	float max_velocity_px = 64.0f;
	float min_velocity_px = 0.5f;
	TileSize tile_size = TILE_16;
	float velocity_scale = 1.0f;
	float jitter = 1.0f;
	float shutter_offset = 0.0f;
	float depth_softness = 0.15f;
	bool camera_motion = true;
	bool object_motion = true;
	bool skinned_meshes = true;
	bool alpha_scissor_support = true;
	bool fps_compensation = false;
	float target_fps = 60.0f;
	uint32_t cull_mask = 0xFFFFFFFF;
	DebugMode debug_mode = DEBUG_NONE;

	// Exported Textures & Per-Viewport State
	Ref<ImageTexture> velocity_texture;
	Ref<ImageTexture> neighbor_max_texture;
	std::unordered_map<uint32_t, ViewportRenderState> viewport_states;
	uint64_t vp_collect_counter = 0;
	bool hook_registered = false;

	// Snapshot & History State
	Mutex scene_mutex;
	GL43VelocityHandler velocity_handler;
	uint64_t last_tick_usec = 0;

	// Raw OpenGL 4.3 Resources
	bool gl_attempted = false;
	bool gl_ready = false;
	Thread::ID gl_thread_id = 0;

	uint32_t tile_max_program = 0;
	uint32_t neighbor_max_program = 0;
	uint32_t composite_program = 0;

	void _register_scene_hook();
	void _unregister_scene_hook();
	void _detach_all_viewports();
	ViewportRenderState &_get_or_create_vp_state(Viewport *p_vp);
	void _prune_stale_viewports();
	void _on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx);

	bool _init_gl_resources();
	void _cleanup_gl_resources();
	void _ensure_tile_tex_size(ViewportRenderState &r_state, int p_w, int p_h);

	void _collect_scene_snapshot();
	void _on_frame_pre_draw();

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &property) const;

public:
	MotionBlur();
	~MotionBlur();

	void set_enabled(bool p_enabled);
	bool is_enabled() const { return enabled; }

	void set_editor_preview(bool p_preview);
	bool get_editor_preview() const { return editor_preview; }

	void set_auto_composite(bool p_auto);
	bool get_auto_composite() const { return auto_composite; }

	void set_intensity(float p_intensity);
	float get_intensity() const { return intensity; }

	void set_samples(int p_samples);
	int get_samples() const { return samples; }

	void set_max_velocity_px(float p_max_px);
	float get_max_velocity_px() const { return max_velocity_px; }

	void set_min_velocity_px(float p_min_px);
	float get_min_velocity_px() const { return min_velocity_px; }

	void set_tile_size(TileSize p_size);
	TileSize get_tile_size() const { return tile_size; }

	void set_velocity_scale(float p_scale);
	float get_velocity_scale() const { return velocity_scale; }

	void set_jitter(float p_jitter);
	float get_jitter() const { return jitter; }

	void set_shutter_offset(float p_offset);
	float get_shutter_offset() const { return shutter_offset; }

	void set_depth_softness(float p_softness);
	float get_depth_softness() const { return depth_softness; }

	void set_camera_motion(bool p_enable) { camera_motion = p_enable; }
	bool get_camera_motion() const { return camera_motion; }

	void set_object_motion(bool p_enable) { object_motion = p_enable; }
	bool get_object_motion() const { return object_motion; }

	void set_skinned_meshes(bool p_enable) { skinned_meshes = p_enable; }
	bool get_skinned_meshes() const { return skinned_meshes; }

	void set_alpha_scissor_support(bool p_enable) { alpha_scissor_support = p_enable; }
	bool get_alpha_scissor_support() const { return alpha_scissor_support; }

	void set_fps_compensation(bool p_enable);
	bool get_fps_compensation() const { return fps_compensation; }

	void set_target_fps(float p_fps);
	float get_target_fps() const { return target_fps; }

	void set_cull_mask(uint32_t p_mask) { cull_mask = p_mask; }
	uint32_t get_cull_mask() const { return cull_mask; }

	void set_debug_mode(DebugMode p_mode) { debug_mode = p_mode; }
	DebugMode get_debug_mode() const { return debug_mode; }

	void reset_history();
	Ref<Texture> get_velocity_texture() const { return velocity_texture; }
	Ref<Texture> get_neighbor_max_texture() const { return neighbor_max_texture; }
};

VARIANT_ENUM_CAST(MotionBlur::TileSize);
VARIANT_ENUM_CAST(MotionBlur::DebugMode);

#endif // MOTION_BLUR_H
