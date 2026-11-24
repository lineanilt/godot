#ifndef SHARED_GL43_CONTEXT_H
#define SHARED_GL43_CONTEXT_H

#include "core/os/thread.h"

#include <cstdint>

class SharedGL43Context {
public:
	struct ScopedScope {
		bool active = false;

		ScopedScope() {
			active = SharedGL43Context::get_singleton()->begin();
		}

		~ScopedScope() {
			if (active) {
				SharedGL43Context::get_singleton()->end();
			}
		}

		bool is_active() const { return active; }
	};

private:
	bool attempted = false;
	bool available = false;
	Thread::ID gl_thread_id = 0;
	int nest_depth = 0;

#if defined(WINDOWS_ENABLED)
	void *gl43_rc = nullptr;
	void *saved_dc = nullptr;
	void *saved_rc = nullptr;
#elif defined(X11_ENABLED)
	void *x_display = nullptr;
	void *gl43_ctx = nullptr;
	unsigned long pbuffer = 0;
	bool use_pbuffer = false;
	void *saved_display = nullptr;
	unsigned long saved_draw = 0;
	unsigned long saved_read = 0;
	void *saved_ctx = nullptr;
	void *lib_gl = nullptr;
	void *lib_x11 = nullptr;
#endif

	void *fn_dispatch_compute = nullptr;
	void *fn_memory_barrier = nullptr;
	void *fn_bind_image_texture = nullptr;
	void *fn_fence_sync = nullptr;
	void *fn_wait_sync = nullptr;
	void *fn_delete_sync = nullptr;

	bool _load_gl43_procs();
	void *_sync_before_switch();
	void _sync_after_switch(void *p_sync);

public:
	static SharedGL43Context *get_singleton();

	bool initialize();
	void shutdown();
	bool is_available() const { return available; }

	bool begin();
	void end();

	void dispatch_compute(uint32_t p_groups_x, uint32_t p_groups_y, uint32_t p_groups_z) const;
	void memory_barrier(uint32_t p_barriers) const;
	void bind_image_texture(uint32_t p_unit, uint32_t p_texture, int32_t p_level, bool p_layered, int32_t p_layer, uint32_t p_access, uint32_t p_format) const;
};

#endif // SHARED_GL43_CONTEXT_H
