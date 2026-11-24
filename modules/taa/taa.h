#ifndef TAA_H
#define TAA_H

#include "core/math/transform.h"
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "drivers/gles3/gl43_effect_helpers.h"
#include "drivers/gles3/gl43_velocity_handler.h"
#include "scene/main/node.h"
#include "scene/resources/material.h"
#include "scene/resources/texture.h"

#include <cstdint>
#include <unordered_map>

class Camera;
class Viewport;

class TAA : public Node {
	GDCLASS(TAA, Node);

public:
	enum JitterPattern {
		JITTER_HALTON_8 = 0,
		JITTER_HALTON_16 = 1,
		JITTER_R2 = 2,
	};

	enum DebugMode {
		DEBUG_NONE = 0,
		DEBUG_VELOCITY = 1,
		DEBUG_JITTER = 2,
		DEBUG_CLIP_DIFF = 3,
	};

private:
	struct ViewportRenderState {
		RID color_tex_rid;
		bool active_composite = false;
		Vector2 jitter_px;
		Ref<ImageTexture> velocity_texture;
		Ref<ImageTexture> history_texture;
		int tex_w = 0;
		int tex_h = 0;

		RID jittered_cam_rid;
		bool jittered_was_ortho = false;
		float jittered_fov = 70.0f;
		float jittered_size = 1.0f;
		float jittered_near = 0.05f;
		float jittered_far = 100.0f;
		Transform jittered_base_xf;

		bool history_valid = false;
		uint64_t last_seen = 0;
	};

	// Properties
	bool enabled = true;
	bool editor_preview = true;
	JitterPattern jitter_pattern = JITTER_HALTON_16;
	float jitter_scale = 1.0f;
	float history_blend = 0.9f;
	float motion_rejection = 0.35f;
	bool variance_clipping = true;
	float variance_gamma = 1.25f;
	bool catmull_rom_history = true;
	float sharpen_intensity = 0.35f;
	bool alpha_scissor_support = true;
	uint32_t cull_mask = 0xFFFFFFFF;
	DebugMode debug_mode = DEBUG_NONE;

	// Exported Textures & Per-Viewport State
	Ref<ImageTexture> velocity_texture;
	Ref<ImageTexture> history_texture;
	std::unordered_map<uint32_t, ViewportRenderState> viewport_states;
	uint64_t vp_collect_counter = 0;
	bool hook_registered = false;

	uint64_t jitter_index = 0;

	// Snapshot & History State
	Mutex scene_mutex;
	GL43VelocityHandler velocity_handler;

	// Raw OpenGL 4.3 Resources
	bool gl_attempted = false;
	bool gl_ready = false;
	Thread::ID gl_thread_id = 0;

	uint32_t composite_program = 0;

	void _register_scene_hook();
	void _unregister_scene_hook();
	void _detach_all_viewports();
	ViewportRenderState &_get_or_create_vp_state(Viewport *p_vp);
	void _prune_stale_viewports();
	void _on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx);

	void _apply_camera_jitter(ViewportRenderState &r_state, Camera *p_cam, const Size2 &p_vp_size, const Vector2 &p_jitter_px);
	void _restore_camera_jitter(ViewportRenderState &r_state);
	void _restore_all_camera_jitters();
	Vector2 _compute_jitter_px(uint64_t p_frame_idx) const;

	bool _init_gl_resources();
	void _cleanup_gl_resources();

	void _collect_scene_snapshot();
	void _on_frame_pre_draw();
	void _on_frame_post_draw();

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &property) const;

public:
	TAA();
	~TAA();

	void set_enabled(bool p_enabled);
	bool is_enabled() const { return enabled; }

	void set_editor_preview(bool p_preview);
	bool get_editor_preview() const { return editor_preview; }

	void set_jitter_pattern(JitterPattern p_pattern) { jitter_pattern = p_pattern; }
	JitterPattern get_jitter_pattern() const { return jitter_pattern; }

	void set_jitter_scale(float p_scale);
	float get_jitter_scale() const { return jitter_scale; }

	void set_history_blend(float p_blend);
	float get_history_blend() const { return history_blend; }

	void set_motion_rejection(float p_rejection);
	float get_motion_rejection() const { return motion_rejection; }

	void set_variance_clipping(bool p_enable);
	bool get_variance_clipping() const { return variance_clipping; }

	void set_variance_gamma(float p_gamma);
	float get_variance_gamma() const { return variance_gamma; }

	void set_catmull_rom_history(bool p_enable) { catmull_rom_history = p_enable; }
	bool get_catmull_rom_history() const { return catmull_rom_history; }

	void set_sharpen_intensity(float p_intensity);
	float get_sharpen_intensity() const { return sharpen_intensity; }

	void set_alpha_scissor_support(bool p_enable) { alpha_scissor_support = p_enable; }
	bool get_alpha_scissor_support() const { return alpha_scissor_support; }

	void set_cull_mask(uint32_t p_mask) { cull_mask = p_mask; }
	uint32_t get_cull_mask() const { return cull_mask; }

	void set_debug_mode(DebugMode p_mode) { debug_mode = p_mode; }
	DebugMode get_debug_mode() const { return debug_mode; }

	void reset_history();
	Ref<Texture> get_velocity_texture() const { return velocity_texture; }
	Ref<Texture> get_history_texture() const { return history_texture; }
};

VARIANT_ENUM_CAST(TAA::JitterPattern);
VARIANT_ENUM_CAST(TAA::DebugMode);

#endif // TAA_H
