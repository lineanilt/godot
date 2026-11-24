#include "shared_gl43_context.h"

#include "core/print_string.h"
#include "drivers/gles3/rasterizer_storage_gles3.h"

#if defined(WINDOWS_ENABLED)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#ifndef WGL_CONTEXT_MAJOR_VERSION_ARB
#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#endif
#ifndef WGL_CONTEXT_MINOR_VERSION_ARB
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#endif
#ifndef WGL_CONTEXT_PROFILE_MASK_ARB
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126
#endif
#ifndef WGL_CONTEXT_CORE_PROFILE_BIT_ARB
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
#endif

typedef HGLRC(WINAPI *PFN_wglCreateContextAttribsARB)(HDC hDC, HGLRC hShareContext, const int *attribList);
#elif defined(X11_ENABLED)
#include <dlfcn.h>

#define GLX_SCREEN_EXT 0x800C
#define GLX_DRAWABLE_TYPE_EXT 0x8010
#define GLX_RENDER_TYPE_EXT 0x8011
#define GLX_FBCONFIG_ID_EXT 0x8013
#define GLX_WINDOW_BIT_EXT 0x00000001
#define GLX_PBUFFER_BIT_EXT 0x00000004
#define GLX_RGBA_BIT_EXT 0x00000001
#define GLX_PBUFFER_HEIGHT_EXT 0x8040
#define GLX_PBUFFER_WIDTH_EXT 0x8041
#define GLX_CONTEXT_MAJOR_VERSION_ARB_EXT 0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB_EXT 0x2092
#define GLX_CONTEXT_PROFILE_MASK_ARB_EXT 0x9126
#define GLX_CONTEXT_CORE_PROFILE_BIT_ARB_EXT 0x00000001

typedef void *(*PFN_glXGetCurrentDisplay)(void);
typedef void *(*PFN_glXGetCurrentContext)(void);
typedef unsigned long (*PFN_glXGetCurrentDrawable)(void);
typedef unsigned long (*PFN_glXGetCurrentReadDrawable)(void);
typedef int (*PFN_glXQueryContext)(void *dpy, void *ctx, int attribute, int *value);
typedef void **(*PFN_glXChooseFBConfig)(void *dpy, int screen, const int *attrib_list, int *nelements);
typedef unsigned long (*PFN_glXCreatePbuffer)(void *dpy, void *config, const int *attrib_list);
typedef void (*PFN_glXDestroyPbuffer)(void *dpy, unsigned long pbuf);
typedef void *(*PFN_glXCreateContextAttribsARB)(void *dpy, void *config, void *share_context, int direct, const int *attrib_list);
typedef int (*PFN_glXMakeContextCurrent)(void *dpy, unsigned long draw, unsigned long read, void *ctx);
typedef void (*PFN_glXDestroyContext)(void *dpy, void *ctx);
typedef void (*(*PFN_glXGetProcAddressARB)(const unsigned char *procName))(void);
typedef int (*PFN_XFree)(void *data);
typedef int (*PFN_XSync)(void *dpy, int discard);
typedef int (*XErrorHandlerFn)(void *, void *);
typedef XErrorHandlerFn (*PFN_XSetErrorHandler)(XErrorHandlerFn handler);

static bool s_x11_ctx_error = false;
static int s_x11_ctx_error_handler(void *, void *) {
	s_x11_ctx_error = true;
	return 0;
}
#endif

#ifndef APIENTRY
#define APIENTRY
#endif

#ifndef GL_SYNC_GPU_COMMANDS_COMPLETE
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#endif
#ifndef GL_TIMEOUT_IGNORED
#define GL_TIMEOUT_IGNORED 0xFFFFFFFFFFFFFFFFULL
#endif

typedef void(APIENTRY *PFN_glDispatchCompute)(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
typedef void(APIENTRY *PFN_glMemoryBarrier)(GLbitfield barriers);
typedef void(APIENTRY *PFN_glBindImageTexture)(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format);
typedef void *(APIENTRY *PFN_glFenceSync)(GLenum condition, GLbitfield flags);
typedef void(APIENTRY *PFN_glWaitSync)(void *sync, GLbitfield flags, uint64_t timeout);
typedef void(APIENTRY *PFN_glDeleteSync)(void *sync);

SharedGL43Context *SharedGL43Context::get_singleton() {
	static SharedGL43Context instance;
	return &instance;
}

bool SharedGL43Context::_load_gl43_procs() {
#if defined(WINDOWS_ENABLED)
	static HMODULE ogl32 = GetModuleHandleA("opengl32.dll");
	if (!ogl32) {
		ogl32 = LoadLibraryA("opengl32.dll");
	}
	auto get_proc = [&](const char *p_name) -> void * {
		void *p = (void *)wglGetProcAddress(p_name);
		if (!p && ogl32) {
			p = (void *)GetProcAddress(ogl32, p_name);
		}
		return p;
	};
#elif defined(X11_ENABLED)
	PFN_glXGetProcAddressARB glx_get_proc = (PFN_glXGetProcAddressARB)dlsym(lib_gl, "glXGetProcAddressARB");
	if (!glx_get_proc) {
		glx_get_proc = (PFN_glXGetProcAddressARB)dlsym(lib_gl, "glXGetProcAddress");
	}
	auto get_proc = [&](const char *p_name) -> void * {
		void *p = nullptr;
		if (glx_get_proc) {
			p = (void *)glx_get_proc((const unsigned char *)p_name);
		}
		if (!p && lib_gl) {
			p = dlsym(lib_gl, p_name);
		}
		return p;
	};
#else
	auto get_proc = [&](const char *) -> void * {
		return nullptr;
	};
#endif

	fn_dispatch_compute = get_proc("glDispatchCompute");
	fn_memory_barrier = get_proc("glMemoryBarrier");
	fn_bind_image_texture = get_proc("glBindImageTexture");
	fn_fence_sync = get_proc("glFenceSync");
	fn_wait_sync = get_proc("glWaitSync");
	fn_delete_sync = get_proc("glDeleteSync");

	return fn_dispatch_compute && fn_memory_barrier && fn_bind_image_texture;
}

void *SharedGL43Context::_sync_before_switch() {
	void *sync_obj = nullptr;
	if (fn_fence_sync && fn_wait_sync && fn_delete_sync) {
		sync_obj = ((PFN_glFenceSync)fn_fence_sync)(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
	}
	glFlush();
	return sync_obj;
}

void SharedGL43Context::_sync_after_switch(void *p_sync) {
	if (p_sync && fn_wait_sync && fn_delete_sync) {
		((PFN_glWaitSync)fn_wait_sync)(p_sync, 0, GL_TIMEOUT_IGNORED);
		((PFN_glDeleteSync)fn_delete_sync)(p_sync);
	}
}

bool SharedGL43Context::initialize() {
	if (attempted) {
		return available;
	}
	attempted = true;
	gl_thread_id = Thread::get_caller_id();

#if defined(WINDOWS_ENABLED)
	HDC hdc = wglGetCurrentDC();
	HGLRC main_rc = wglGetCurrentContext();
	if (!hdc || !main_rc) {
		ERR_PRINT("SharedGL43Context: No active WGL context to share with.");
		return false;
	}

	PFN_wglCreateContextAttribsARB create_attribs = (PFN_wglCreateContextAttribsARB)wglGetProcAddress("wglCreateContextAttribsARB");
	if (!create_attribs) {
		ERR_PRINT("SharedGL43Context: wglCreateContextAttribsARB unavailable.");
		return false;
	}

	const int attribs[] = {
		WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
		WGL_CONTEXT_MINOR_VERSION_ARB, 3,
		WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
		0
	};

	HGLRC ctx43 = create_attribs(hdc, main_rc, attribs);
	if (!ctx43) {
		ERR_PRINT("SharedGL43Context: Failed to create shared OpenGL 4.3 Core WGL context.");
		return false;
	}

	if (!wglMakeCurrent(hdc, ctx43)) {
		wglDeleteContext(ctx43);
		ERR_PRINT("SharedGL43Context: Failed to activate shared OpenGL 4.3 WGL context.");
		return false;
	}

	bool procs_ok = _load_gl43_procs();
	wglMakeCurrent(hdc, main_rc);

	if (!procs_ok) {
		wglDeleteContext(ctx43);
		ERR_PRINT("SharedGL43Context: OpenGL 4.3 compute entry points not found.");
		return false;
	}

	gl43_rc = (void *)ctx43;
	available = true;
	print_verbose("SharedGL43Context: Created shared OpenGL 4.3 Core context (WGL).");
	return true;

#elif defined(X11_ENABLED)
	lib_gl = dlopen("libGL.so.1", RTLD_LAZY | RTLD_LOCAL);
	if (!lib_gl) lib_gl = dlopen("libGL.so", RTLD_LAZY | RTLD_LOCAL);
	if (!lib_gl) lib_gl = dlopen("libGLX.so.0", RTLD_LAZY | RTLD_LOCAL);
	if (!lib_gl) {
		ERR_PRINT("SharedGL43Context: Unable to open libGL.");
		return false;
	}

	lib_x11 = dlopen("libX11.so.6", RTLD_LAZY | RTLD_LOCAL);
	if (!lib_x11) lib_x11 = dlopen("libX11.so", RTLD_LAZY | RTLD_LOCAL);

	PFN_glXGetCurrentDisplay fn_get_dpy = (PFN_glXGetCurrentDisplay)dlsym(lib_gl, "glXGetCurrentDisplay");
	PFN_glXGetCurrentContext fn_get_ctx = (PFN_glXGetCurrentContext)dlsym(lib_gl, "glXGetCurrentContext");
	PFN_glXGetCurrentDrawable fn_get_draw = (PFN_glXGetCurrentDrawable)dlsym(lib_gl, "glXGetCurrentDrawable");
	PFN_glXGetCurrentReadDrawable fn_get_read = (PFN_glXGetCurrentReadDrawable)dlsym(lib_gl, "glXGetCurrentReadDrawable");
	PFN_glXQueryContext fn_query_ctx = (PFN_glXQueryContext)dlsym(lib_gl, "glXQueryContext");
	PFN_glXChooseFBConfig fn_choose_fbc = (PFN_glXChooseFBConfig)dlsym(lib_gl, "glXChooseFBConfig");
	PFN_glXCreatePbuffer fn_create_pbuf = (PFN_glXCreatePbuffer)dlsym(lib_gl, "glXCreatePbuffer");
	PFN_glXDestroyPbuffer fn_destroy_pbuf = (PFN_glXDestroyPbuffer)dlsym(lib_gl, "glXDestroyPbuffer");
	PFN_glXMakeContextCurrent fn_make_current = (PFN_glXMakeContextCurrent)dlsym(lib_gl, "glXMakeContextCurrent");
	PFN_glXDestroyContext fn_destroy_ctx = (PFN_glXDestroyContext)dlsym(lib_gl, "glXDestroyContext");
	PFN_glXGetProcAddressARB fn_get_proc = (PFN_glXGetProcAddressARB)dlsym(lib_gl, "glXGetProcAddressARB");
	if (!fn_get_proc) fn_get_proc = (PFN_glXGetProcAddressARB)dlsym(lib_gl, "glXGetProcAddress");

	PFN_XFree fn_xfree = (PFN_XFree)(lib_x11 ? dlsym(lib_x11, "XFree") : dlsym(RTLD_DEFAULT, "XFree"));
	PFN_XSync fn_xsync = (PFN_XSync)(lib_x11 ? dlsym(lib_x11, "XSync") : dlsym(RTLD_DEFAULT, "XSync"));
	PFN_XSetErrorHandler fn_set_err = (PFN_XSetErrorHandler)(lib_x11 ? dlsym(lib_x11, "XSetErrorHandler") : dlsym(RTLD_DEFAULT, "XSetErrorHandler"));

	if (!fn_get_dpy || !fn_get_ctx || !fn_get_draw || !fn_choose_fbc || !fn_make_current || !fn_destroy_ctx || !fn_get_proc) {
		ERR_PRINT("SharedGL43Context: Required GLX symbols missing.");
		return false;
	}

	PFN_glXCreateContextAttribsARB fn_create_attribs = (PFN_glXCreateContextAttribsARB)fn_get_proc((const unsigned char *)"glXCreateContextAttribsARB");
	if (!fn_create_attribs) {
		ERR_PRINT("SharedGL43Context: glXCreateContextAttribsARB unavailable.");
		return false;
	}

	void *dpy = fn_get_dpy();
	void *main_ctx = fn_get_ctx();
	unsigned long draw_drawable = fn_get_draw();
	unsigned long read_drawable = fn_get_read ? fn_get_read() : draw_drawable;
	if (!dpy || !main_ctx || !draw_drawable) {
		ERR_PRINT("SharedGL43Context: No active GLX context to share with.");
		return false;
	}

	int screen = 0;
	int fbconfig_id = 0;
	if (fn_query_ctx) {
		fn_query_ctx(dpy, main_ctx, GLX_SCREEN_EXT, &screen);
		if (fn_query_ctx(dpy, main_ctx, GLX_FBCONFIG_ID_EXT, &fbconfig_id) != 0) {
			fbconfig_id = 0;
		}
	}

	void **configs = nullptr;
	int n_configs = 0;
	if (fbconfig_id != 0) {
		const int fb_id_attribs[] = {
			GLX_FBCONFIG_ID_EXT, fbconfig_id,
			0
		};
		configs = fn_choose_fbc(dpy, screen, fb_id_attribs, &n_configs);
	}
	if (!configs || n_configs == 0) {
		const int fb_fallback_attribs[] = {
			GLX_RENDER_TYPE_EXT, GLX_RGBA_BIT_EXT,
			GLX_DRAWABLE_TYPE_EXT, GLX_WINDOW_BIT_EXT | GLX_PBUFFER_BIT_EXT,
			0
		};
		configs = fn_choose_fbc(dpy, screen, fb_fallback_attribs, &n_configs);
	}
	if (!configs || n_configs == 0) {
		ERR_PRINT("SharedGL43Context: Failed to find matching GLXFBConfig.");
		return false;
	}

	void *chosen_cfg = configs[0];
	if (fn_xfree) {
		fn_xfree(configs);
	}

	const int ctx_attribs[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB_EXT, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB_EXT, 3,
		GLX_CONTEXT_PROFILE_MASK_ARB_EXT, GLX_CONTEXT_CORE_PROFILE_BIT_ARB_EXT,
		0
	};

	s_x11_ctx_error = false;
	XErrorHandlerFn prev_handler = fn_set_err ? fn_set_err(s_x11_ctx_error_handler) : nullptr;
	void *ctx43 = fn_create_attribs(dpy, chosen_cfg, main_ctx, 1, ctx_attribs);
	if (fn_xsync) {
		fn_xsync(dpy, 0);
	}
	if (fn_set_err) {
		fn_set_err(prev_handler);
	}

	if (!ctx43 || s_x11_ctx_error) {
		if (ctx43) {
			fn_destroy_ctx(dpy, ctx43);
		}
		ERR_PRINT("SharedGL43Context: Failed to create shared OpenGL 4.3 Core GLX context.");
		return false;
	}

	unsigned long test_draw = draw_drawable;
	unsigned long test_read = read_drawable;
	bool bound = (fn_make_current(dpy, test_draw, test_read, ctx43) != 0);
	if (!bound && fn_create_pbuf) {
		const int pbuf_attribs[] = {
			GLX_PBUFFER_WIDTH_EXT, 1,
			GLX_PBUFFER_HEIGHT_EXT, 1,
			0
		};
		pbuffer = fn_create_pbuf(dpy, chosen_cfg, pbuf_attribs);
		if (pbuffer) {
			use_pbuffer = true;
			bound = (fn_make_current(dpy, pbuffer, pbuffer, ctx43) != 0);
		}
	}

	if (!bound) {
		fn_make_current(dpy, draw_drawable, read_drawable, main_ctx);
		if (pbuffer && fn_destroy_pbuf) {
			fn_destroy_pbuf(dpy, pbuffer);
			pbuffer = 0;
		}
		fn_destroy_ctx(dpy, ctx43);
		ERR_PRINT("SharedGL43Context: Failed to activate shared OpenGL 4.3 GLX context.");
		return false;
	}

	bool procs_ok = _load_gl43_procs();
	fn_make_current(dpy, draw_drawable, read_drawable, main_ctx);

	if (!procs_ok) {
		if (pbuffer && fn_destroy_pbuf) {
			fn_destroy_pbuf(dpy, pbuffer);
			pbuffer = 0;
		}
		fn_destroy_ctx(dpy, ctx43);
		ERR_PRINT("SharedGL43Context: OpenGL 4.3 compute entry points not found.");
		return false;
	}

	x_display = dpy;
	gl43_ctx = ctx43;
	available = true;
	print_verbose("SharedGL43Context: Created shared OpenGL 4.3 Core context (GLX).");
	return true;
#else
	ERR_PRINT("SharedGL43Context: Not supported on this platform.");
	return false;
#endif
}

void SharedGL43Context::shutdown() {
	if (!available) {
		return;
	}
#if defined(WINDOWS_ENABLED)
	if (gl43_rc) {
		wglDeleteContext((HGLRC)gl43_rc);
		gl43_rc = nullptr;
	}
#elif defined(X11_ENABLED)
	if (lib_gl && x_display) {
		PFN_glXDestroyContext fn_destroy_ctx = (PFN_glXDestroyContext)dlsym(lib_gl, "glXDestroyContext");
		PFN_glXDestroyPbuffer fn_destroy_pbuf = (PFN_glXDestroyPbuffer)dlsym(lib_gl, "glXDestroyPbuffer");
		if (gl43_ctx && fn_destroy_ctx) {
			fn_destroy_ctx(x_display, gl43_ctx);
			gl43_ctx = nullptr;
		}
		if (pbuffer && fn_destroy_pbuf) {
			fn_destroy_pbuf(x_display, pbuffer);
			pbuffer = 0;
		}
	}
	if (lib_gl) {
		dlclose(lib_gl);
		lib_gl = nullptr;
	}
	if (lib_x11) {
		dlclose(lib_x11);
		lib_x11 = nullptr;
	}
#endif
	available = false;
	attempted = false;
	nest_depth = 0;
}

bool SharedGL43Context::begin() {
	if (!attempted) {
		initialize();
	}
	if (!available || Thread::get_caller_id() != gl_thread_id) {
		return false;
	}

	if (nest_depth == 0) {
#if defined(WINDOWS_ENABLED)
		HDC cur_dc = wglGetCurrentDC();
		HGLRC cur_rc = wglGetCurrentContext();
		if (!cur_dc || !cur_rc) {
			return false;
		}
		saved_dc = (void *)cur_dc;
		saved_rc = (void *)cur_rc;

		if (cur_rc != (HGLRC)gl43_rc) {
			void *sync_obj = _sync_before_switch();
			if (!wglMakeCurrent(cur_dc, (HGLRC)gl43_rc)) {
				return false;
			}
			_sync_after_switch(sync_obj);
		}
#elif defined(X11_ENABLED)
		PFN_glXGetCurrentDisplay fn_get_dpy = (PFN_glXGetCurrentDisplay)dlsym(lib_gl, "glXGetCurrentDisplay");
		PFN_glXGetCurrentContext fn_get_ctx = (PFN_glXGetCurrentContext)dlsym(lib_gl, "glXGetCurrentContext");
		PFN_glXGetCurrentDrawable fn_get_draw = (PFN_glXGetCurrentDrawable)dlsym(lib_gl, "glXGetCurrentDrawable");
		PFN_glXGetCurrentReadDrawable fn_get_read = (PFN_glXGetCurrentReadDrawable)dlsym(lib_gl, "glXGetCurrentReadDrawable");
		PFN_glXMakeContextCurrent fn_make_current = (PFN_glXMakeContextCurrent)dlsym(lib_gl, "glXMakeContextCurrent");

		void *cur_dpy = fn_get_dpy ? fn_get_dpy() : x_display;
		void *cur_ctx = fn_get_ctx ? fn_get_ctx() : nullptr;
		unsigned long cur_draw = fn_get_draw ? fn_get_draw() : 0;
		unsigned long cur_read = fn_get_read ? fn_get_read() : cur_draw;
		if (!cur_dpy || !cur_ctx || !fn_make_current) {
			return false;
		}

		saved_display = cur_dpy;
		saved_draw = cur_draw;
		saved_read = cur_read;
		saved_ctx = cur_ctx;

		if (cur_ctx != gl43_ctx) {
			void *sync_obj = _sync_before_switch();
			unsigned long target_draw = use_pbuffer ? pbuffer : cur_draw;
			unsigned long target_read = use_pbuffer ? pbuffer : cur_read;
			if (!fn_make_current(x_display, target_draw, target_read, gl43_ctx)) {
				return false;
			}
			_sync_after_switch(sync_obj);
		}
#endif
	}

	nest_depth++;
	return true;
}

void SharedGL43Context::end() {
	if (nest_depth <= 0) {
		return;
	}
	nest_depth--;
	if (nest_depth > 0) {
		return;
	}

#if defined(WINDOWS_ENABLED)
	if (saved_rc && saved_rc != gl43_rc) {
		void *sync_obj = _sync_before_switch();
		wglMakeCurrent((HDC)saved_dc, (HGLRC)saved_rc);
		_sync_after_switch(sync_obj);
	}
	saved_dc = nullptr;
	saved_rc = nullptr;
#elif defined(X11_ENABLED)
	if (saved_ctx && saved_ctx != gl43_ctx) {
		PFN_glXMakeContextCurrent fn_make_current = (PFN_glXMakeContextCurrent)dlsym(lib_gl, "glXMakeContextCurrent");
		if (fn_make_current) {
			void *sync_obj = _sync_before_switch();
			fn_make_current(saved_display ? saved_display : x_display, saved_draw, saved_read, saved_ctx);
			_sync_after_switch(sync_obj);
		}
	}
	saved_display = nullptr;
	saved_draw = 0;
	saved_read = 0;
	saved_ctx = nullptr;
#endif
}

void SharedGL43Context::dispatch_compute(uint32_t p_groups_x, uint32_t p_groups_y, uint32_t p_groups_z) const {
	if (fn_dispatch_compute) {
		((PFN_glDispatchCompute)fn_dispatch_compute)(p_groups_x, p_groups_y, p_groups_z);
	}
}

void SharedGL43Context::memory_barrier(uint32_t p_barriers) const {
	if (fn_memory_barrier) {
		((PFN_glMemoryBarrier)fn_memory_barrier)(p_barriers);
	}
}

void SharedGL43Context::bind_image_texture(uint32_t p_unit, uint32_t p_texture, int32_t p_level, bool p_layered, int32_t p_layer, uint32_t p_access, uint32_t p_format) const {
	if (fn_bind_image_texture) {
		((PFN_glBindImageTexture)fn_bind_image_texture)(p_unit, p_texture, p_level, p_layered ? GL_TRUE : GL_FALSE, p_layer, p_access, p_format);
	}
}
