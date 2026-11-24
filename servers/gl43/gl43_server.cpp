#include "gl43_server.h"

#include "core/math/camera_matrix.h"
#include "core/os/os.h"
#include "core/print_string.h"
#include "core/resource.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual_server.h"

#include <cstring>

#if defined(WINDOWS_ENABLED) || defined(X11_ENABLED)
#define GL43_BACKEND_SUPPORTED 1

#include "drivers/gles3/rasterizer_storage_gles3.h"
#include "drivers/gles3/shared_gl43_context.h"

#if defined(WINDOWS_ENABLED)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(X11_ENABLED)
#include <dlfcn.h>
typedef void (*(*PFN_glXGetProcAddressARB_GL43)(const unsigned char *procName))(void);
#endif

#ifndef APIENTRY
#define APIENTRY
#endif

#ifndef GL_COMPUTE_SHADER
#define GL_COMPUTE_SHADER 0x91B9
#endif
#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_UNIFORM_BUFFER
#define GL_UNIFORM_BUFFER 0x8A11
#endif
#ifndef GL_DISPATCH_INDIRECT_BUFFER
#define GL_DISPATCH_INDIRECT_BUFFER 0x90EE
#endif
#ifndef GL_COPY_READ_BUFFER
#define GL_COPY_READ_BUFFER 0x8F36
#endif
#ifndef GL_COPY_WRITE_BUFFER
#define GL_COPY_WRITE_BUFFER 0x8F37
#endif
#ifndef GL_STATIC_DRAW
#define GL_STATIC_DRAW 0x88E4
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif
#ifndef GL_STREAM_DRAW
#define GL_STREAM_DRAW 0x88E0
#endif
#ifndef GL_READ_ONLY
#define GL_READ_ONLY 0x88B8
#endif
#ifndef GL_WRITE_ONLY
#define GL_WRITE_ONLY 0x88B9
#endif
#ifndef GL_READ_WRITE
#define GL_READ_WRITE 0x88BA
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
#ifndef GL_BUFFER_UPDATE_BARRIER_BIT
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x00000200
#endif
#ifndef GL_UNIFORM_BARRIER_BIT
#define GL_UNIFORM_BARRIER_BIT 0x00000004
#endif
#ifndef GL_ALL_BARRIER_BITS
#define GL_ALL_BARRIER_BITS 0xFFFFFFFFu
#endif
#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif
#ifndef GL_TEXTURE_3D
#define GL_TEXTURE_3D 0x806F
#endif
#ifndef GL_TEXTURE_CUBE_MAP
#define GL_TEXTURE_CUBE_MAP 0x8513
#endif
#ifndef GL_TEXTURE_WRAP_R
#define GL_TEXTURE_WRAP_R 0x8072
#endif
#ifndef GL_TEXTURE_INTERNAL_FORMAT
#define GL_TEXTURE_INTERNAL_FORMAT 0x1003
#endif
#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif
#ifndef GL_SRGB8_ALPHA8
#define GL_SRGB8_ALPHA8 0x8C43
#endif
#ifndef GL_RGBA16F
#define GL_RGBA16F 0x881A
#endif
#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif
#ifndef GL_RG8
#define GL_RG8 0x822B
#endif
#ifndef GL_RG16F
#define GL_RG16F 0x822F
#endif
#ifndef GL_RG32F
#define GL_RG32F 0x8230
#endif
#ifndef GL_R8
#define GL_R8 0x8229
#endif
#ifndef GL_R16F
#define GL_R16F 0x822D
#endif
#ifndef GL_R32F
#define GL_R32F 0x822E
#endif
#ifndef GL_R32I
#define GL_R32I 0x8235
#endif
#ifndef GL_R32UI
#define GL_R32UI 0x8236
#endif
#ifndef GL_RGBA16UI
#define GL_RGBA16UI 0x8D76
#endif
#ifndef GL_RGBA32UI
#define GL_RGBA32UI 0x8D70
#endif
#ifndef GL_RED_INTEGER
#define GL_RED_INTEGER 0x8D94
#endif
#ifndef GL_RGBA_INTEGER
#define GL_RGBA_INTEGER 0x8D99
#endif
#ifndef GL_RG
#define GL_RG 0x8227
#endif
#ifndef GL_HALF_FLOAT
#define GL_HALF_FLOAT 0x140B
#endif
#ifndef GL_MAX_COMPUTE_WORK_GROUP_COUNT
#define GL_MAX_COMPUTE_WORK_GROUP_COUNT 0x91BE
#endif
#ifndef GL_MAX_COMPUTE_WORK_GROUP_SIZE
#define GL_MAX_COMPUTE_WORK_GROUP_SIZE 0x91BF
#endif
#ifndef GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS
#define GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 0x90EB
#endif
#ifndef GL_MAX_COMPUTE_SHARED_MEMORY_SIZE
#define GL_MAX_COMPUTE_SHARED_MEMORY_SIZE 0x8262
#endif
#ifndef GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS
#define GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS 0x90DD
#endif
#ifndef GL_MAX_IMAGE_UNITS
#define GL_MAX_IMAGE_UNITS 0x8F38
#endif
#ifndef GL_FRAMEBUFFER_SRGB
#define GL_FRAMEBUFFER_SRGB 0x8DB9
#endif
#ifndef GL_RGB8
#define GL_RGB8 0x8051
#endif
#ifndef GL_SRGB8
#define GL_SRGB8 0x8C41
#endif
#ifndef GL_SRGB_ALPHA
#define GL_SRGB_ALPHA 0x8C42
#endif

typedef void(APIENTRY *PFN_glDispatchComputeIndirect_GL43)(GLintptr indirect);
typedef void(APIENTRY *PFN_glClearBufferData_GL43)(GLenum target, GLenum internalformat, GLenum format, GLenum type, const void *data);
typedef void(APIENTRY *PFN_glClearTexImage_GL43)(GLuint texture, GLint level, GLenum format, GLenum type, const void *data);
typedef void(APIENTRY *PFN_glCopyImageSubData_GL43)(GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth);

namespace {

static GLenum to_gl_usage(GL43Server::BufferUsage p_usage) {
	switch (p_usage) {
		case GL43Server::USAGE_STATIC:
			return GL_STATIC_DRAW;
		case GL43Server::USAGE_STREAM:
			return GL_STREAM_DRAW;
		default:
			return GL_DYNAMIC_DRAW;
	}
}

static GLenum to_gl_access(GL43Server::ImageAccess p_access) {
	switch (p_access) {
		case GL43Server::ACCESS_READ_ONLY:
			return GL_READ_ONLY;
		case GL43Server::ACCESS_WRITE_ONLY:
			return GL_WRITE_ONLY;
		default:
			return GL_READ_WRITE;
	}
}

static void to_gl_format(GL43Server::ImageFormat p_fmt, GLenum &r_internal, GLenum &r_format, GLenum &r_type) {
	switch (p_fmt) {
		case GL43Server::FORMAT_RGBA8:
			r_internal = GL_RGBA8;
			r_format = GL_RGBA;
			r_type = GL_UNSIGNED_BYTE;
			break;
		case GL43Server::FORMAT_RGBA32F:
			r_internal = GL_RGBA32F;
			r_format = GL_RGBA;
			r_type = GL_FLOAT;
			break;
		case GL43Server::FORMAT_RG8:
			r_internal = GL_RG8;
			r_format = GL_RG;
			r_type = GL_UNSIGNED_BYTE;
			break;
		case GL43Server::FORMAT_RG16F:
			r_internal = GL_RG16F;
			r_format = GL_RG;
			r_type = GL_HALF_FLOAT;
			break;
		case GL43Server::FORMAT_RG32F:
			r_internal = GL_RG32F;
			r_format = GL_RG;
			r_type = GL_FLOAT;
			break;
		case GL43Server::FORMAT_R8:
			r_internal = GL_R8;
			r_format = GL_RED;
			r_type = GL_UNSIGNED_BYTE;
			break;
		case GL43Server::FORMAT_R16F:
			r_internal = GL_R16F;
			r_format = GL_RED;
			r_type = GL_HALF_FLOAT;
			break;
		case GL43Server::FORMAT_R32F:
			r_internal = GL_R32F;
			r_format = GL_RED;
			r_type = GL_FLOAT;
			break;
		case GL43Server::FORMAT_R32I:
			r_internal = GL_R32I;
			r_format = GL_RED_INTEGER;
			r_type = GL_INT;
			break;
		case GL43Server::FORMAT_R32UI:
			r_internal = GL_R32UI;
			r_format = GL_RED_INTEGER;
			r_type = GL_UNSIGNED_INT;
			break;
		case GL43Server::FORMAT_RGBA16UI:
			r_internal = GL_RGBA16UI;
			r_format = GL_RGBA_INTEGER;
			r_type = GL_UNSIGNED_SHORT;
			break;
		case GL43Server::FORMAT_RGBA32UI:
			r_internal = GL_RGBA32UI;
			r_format = GL_RGBA_INTEGER;
			r_type = GL_UNSIGNED_INT;
			break;
		case GL43Server::FORMAT_RGBA16F:
		default:
			r_internal = GL_RGBA16F;
			r_format = GL_RGBA;
			r_type = GL_HALF_FLOAT;
			break;
	}
}

static GLenum to_gl_wrap(GL43Server::WrapMode p_wrap) {
	switch (p_wrap) {
		case GL43Server::WRAP_REPEAT:
			return GL_REPEAT;
		case GL43Server::WRAP_MIRROR:
			return GL_MIRRORED_REPEAT;
		default:
			return GL_CLAMP_TO_EDGE;
	}
}

static GLbitfield to_gl_barriers(int p_barriers) {
	if (p_barriers == GL43Server::BARRIER_ALL) {
		return GL_ALL_BARRIER_BITS;
	}
	GLbitfield bits = 0;
	if (p_barriers & GL43Server::BARRIER_IMAGE) {
		bits |= GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
	}
	if (p_barriers & GL43Server::BARRIER_SSBO) {
		bits |= GL_SHADER_STORAGE_BARRIER_BIT;
	}
	if (p_barriers & GL43Server::BARRIER_TEXTURE) {
		bits |= GL_TEXTURE_FETCH_BARRIER_BIT;
	}
	if (p_barriers & GL43Server::BARRIER_BUFFER_UPDATE) {
		bits |= GL_BUFFER_UPDATE_BARRIER_BIT;
	}
	if (p_barriers & GL43Server::BARRIER_UBO) {
		bits |= GL_UNIFORM_BARRIER_BIT;
	}
	return bits ? bits : GL_ALL_BARRIER_BITS;
}

static bool is_integer_internal_format(GLenum p_fmt) {
	switch (p_fmt) {
		case GL_R32I:
		case GL_R32UI:
		case GL_RGBA16UI:
		case GL_RGBA32UI:
			return true;
		default:
			return false;
	}
}

static bool is_signed_integer_internal_format(GLenum p_fmt) {
	return p_fmt == GL_R32I;
}

static GLenum sanitize_image_format(GLenum p_fmt) {
	switch (p_fmt) {
		case GL_RGBA:
		case GL_SRGB_ALPHA:
		case GL_SRGB8_ALPHA8:
			return GL_RGBA8;
		case GL_RGB:
		case GL_RGB8:
		case GL_SRGB8:
			ERR_PRINT_ONCE("GL43Server: 3-component RGB textures are unsupported by OpenGL imageLoad/imageStore; use an RGBA format.");
			return 0;
		default:
			return p_fmt;
	}
}

} // namespace
#endif // GL43_BACKEND_SUPPORTED

GL43Server *GL43Server::singleton = nullptr;

GL43Server::GL43Server() {
	singleton = this;
}

GL43Server::~GL43Server() {
	if (vs_connected && VisualServer::get_singleton() && VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
		VisualServer::get_singleton()->disconnect("frame_pre_draw", this, "_on_frame_pre_draw");
		vs_connected = false;
	}
	if (!gl_finished) {
		finish_gl();
	}
	if (singleton == this) {
		singleton = nullptr;
	}
}

void GL43Server::finish_gl() {
	if (gl_finished) {
		return;
	}

#ifdef GL43_BACKEND_SUPPORTED
	bool can_delete_gl = SharedGL43Context::get_singleton() && SharedGL43Context::get_singleton()->is_available() && _is_on_gl_thread();
	if (can_delete_gl) {
		SharedGL43Context::ScopedScope gl43;
		if (gl43.is_active()) {
			MutexLock lock(data_mutex);

			List<RID> srids;
			shader_owner.get_owned_list(&srids);
			for (List<RID>::Element *E = srids.front(); E; E = E->next()) {
				ShaderEntry *s = shader_owner.getornull(E->get());
				if (s && s->program) {
					glDeleteProgram((GLuint)s->program);
					s->program = 0;
				}
			}

			List<RID> brids;
			buffer_owner.get_owned_list(&brids);
			for (List<RID>::Element *E = brids.front(); E; E = E->next()) {
				BufferEntry *b = buffer_owner.getornull(E->get());
				if (b && b->buffer) {
					GLuint buf = (GLuint)b->buffer;
					glDeleteBuffers(1, &buf);
					b->buffer = 0;
				}
			}

			List<RID> trids;
			texture_owner.get_owned_list(&trids);
			for (List<RID>::Element *E = trids.front(); E; E = E->next()) {
				TextureEntry *t = texture_owner.getornull(E->get());
				if (t && t->tex_id) {
					GLuint tex = (GLuint)t->tex_id;
					glDeleteTextures(1, &tex);
					t->tex_id = 0;
				}
			}

			if (helper_fbo) {
				GLuint fbo = (GLuint)helper_fbo;
				glDeleteFramebuffers(1, &fbo);
				helper_fbo = 0;
			}
		}
	}
#endif

	{
		MutexLock lock(data_mutex);

		List<RID> srids;
		shader_owner.get_owned_list(&srids);
		for (List<RID>::Element *E = srids.front(); E; E = E->next()) {
			ShaderEntry *s = shader_owner.getornull(E->get());
			shader_owner.free(E->get());
			if (s) {
				memdelete(s);
			}
		}

		List<RID> brids;
		buffer_owner.get_owned_list(&brids);
		for (List<RID>::Element *E = brids.front(); E; E = E->next()) {
			BufferEntry *b = buffer_owner.getornull(E->get());
			buffer_owner.free(E->get());
			if (b) {
				memdelete(b);
			}
		}

		List<RID> trids;
		texture_owner.get_owned_list(&trids);
		for (List<RID>::Element *E = trids.front(); E; E = E->next()) {
			TextureEntry *t = texture_owner.getornull(E->get());
			texture_owner.free(E->get());
			if (t) {
				memdelete(t);
			}
		}

		bound_shader = RID();
		helper_fbo = 0;
	}

	{
		MutexLock qlock(queue_mutex);
		pending_commands.clear();
	}

	gl_finished = true;
}

bool GL43Server::_is_on_gl_thread() const {
	if (OS::get_singleton()->get_render_thread_mode() != OS::RENDER_SEPARATE_THREAD) {
		return Thread::get_caller_id() == Thread::get_main_id();
	}
	if (gl_thread_known) {
		return Thread::get_caller_id() == gl_thread_id;
	}
	return false;
}

void GL43Server::_ensure_vs_connected() {
	MutexLock lock(queue_mutex);
	if (vs_connected) {
		return;
	}
	if (VisualServer::get_singleton()) {
		if (!VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
			VisualServer::get_singleton()->connect("frame_pre_draw", this, "_on_frame_pre_draw");
		}
		vs_connected = true;
	}
}

void GL43Server::_enqueue_or_run(std::function<void()> p_fn) {
	if (!is_available()) {
		return;
	}
	if (_is_on_gl_thread()) {
		_flush_pending_commands();
		p_fn();
		return;
	}
	_ensure_vs_connected();
	MutexLock lock(queue_mutex);
	pending_commands.push_back(std::move(p_fn));
}

void GL43Server::_flush_pending_commands() {
	std::vector<std::function<void()>> cmds;
	{
		MutexLock lock(queue_mutex);
		if (pending_commands.empty()) {
			return;
		}
		cmds.swap(pending_commands);
	}
	for (size_t i = 0; i < cmds.size(); i++) {
		cmds[i]();
	}
}

void GL43Server::_on_frame_pre_draw() {
	if (!gl_thread_known) {
		gl_thread_id = Thread::get_caller_id();
		gl_thread_known = true;
	}
	if (!limits_cached) {
		get_limits();
	}
	_flush_pending_commands();
}

void GL43Server::_load_extra_gl43_procs() {
	if (procs_loaded) {
		return;
	}
	procs_loaded = true;
#ifdef GL43_BACKEND_SUPPORTED
#if defined(WINDOWS_ENABLED)
	auto get_proc = [](const char *p_name) -> void * {
		return (void *)wglGetProcAddress(p_name);
	};
#elif defined(X11_ENABLED)
	static void *s_lib_gl = dlopen("libGL.so.1", RTLD_LAZY | RTLD_LOCAL);
	if (!s_lib_gl) {
		s_lib_gl = dlopen("libGL.so", RTLD_LAZY | RTLD_LOCAL);
	}
	static PFN_glXGetProcAddressARB_GL43 s_glx_get_proc = s_lib_gl ? (PFN_glXGetProcAddressARB_GL43)dlsym(s_lib_gl, "glXGetProcAddressARB") : nullptr;
	if (!s_glx_get_proc && s_lib_gl) {
		s_glx_get_proc = (PFN_glXGetProcAddressARB_GL43)dlsym(s_lib_gl, "glXGetProcAddress");
	}
	auto get_proc = [](const char *p_name) -> void * {
		void *p = s_glx_get_proc ? (void *)s_glx_get_proc((const unsigned char *)p_name) : nullptr;
		if (!p && s_lib_gl) {
			p = dlsym(s_lib_gl, p_name);
		}
		return p;
	};
#endif
	fn_dispatch_compute_indirect = get_proc("glDispatchComputeIndirect");
	fn_clear_buffer_data = get_proc("glClearBufferData");
	fn_clear_tex_image = get_proc("glClearTexImage");
	fn_copy_image_sub_data = get_proc("glCopyImageSubData");
#endif
}

bool GL43Server::is_available() const {
#ifdef GL43_BACKEND_SUPPORTED
	if (gl_finished) {
		return false;
	}
	if (!SharedGL43Context::get_singleton() || !SharedGL43Context::get_singleton()->is_available()) {
		return false;
	}
	return dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage) != nullptr;
#else
	return false;
#endif
}

String GL43Server::get_last_error() const {
	MutexLock lock(data_mutex);
	return last_error;
}

Dictionary GL43Server::get_limits() const {
	{
		MutexLock lock(data_mutex);
		if (limits_cached) {
			return cached_limits.duplicate();
		}
	}

	Dictionary d;
#ifdef GL43_BACKEND_SUPPORTED
	if (!is_available() || !_is_on_gl_thread()) {
		return d;
	}
	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return d;
	}

	GLint wg_count[3] = { 0, 0, 0 };
	GLint wg_size[3] = { 0, 0, 0 };
	GLint wg_invoc = 0;
	GLint shared_mem = 0;
	GLint ssbo_bindings = 0;
	GLint image_units = 0;

	for (int i = 0; i < 3; i++) {
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i, &wg_count[i]);
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &wg_size[i]);
	}
	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &wg_invoc);
	glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &shared_mem);
	glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &ssbo_bindings);
	glGetIntegerv(GL_MAX_IMAGE_UNITS, &image_units);

	d["max_work_group_count"] = Vector3((float)wg_count[0], (float)wg_count[1], (float)wg_count[2]);
	d["max_work_group_size"] = Vector3((float)wg_size[0], (float)wg_size[1], (float)wg_size[2]);
	d["max_work_group_invocations"] = (int)wg_invoc;
	d["max_shared_memory_bytes"] = (int)shared_mem;
	d["max_ssbo_bindings"] = (int)ssbo_bindings;
	d["max_image_units"] = (int)image_units;

	MutexLock lock(data_mutex);
	cached_limits = d.duplicate();
	limits_cached = true;
#endif
	return d;
}

bool GL43Server::begin() {
	if (!is_available()) {
		return false;
	}
	if (_is_on_gl_thread()) {
		_flush_pending_commands();
#ifdef GL43_BACKEND_SUPPORTED
		if (SharedGL43Context::get_singleton()->begin()) {
			begin_depth++;
			return true;
		}
#endif
		return false;
	}
	_enqueue_or_run([this]() {
#ifdef GL43_BACKEND_SUPPORTED
		if (SharedGL43Context::get_singleton()->begin()) {
			begin_depth++;
		}
#endif
	});
	return true;
}

void GL43Server::end() {
	if (!is_available()) {
		return;
	}
	_enqueue_or_run([this]() {
#ifdef GL43_BACKEND_SUPPORTED
		if (begin_depth > 0) {
			begin_depth--;
			SharedGL43Context::get_singleton()->end();
		}
#endif
	});
}

void GL43Server::flush() {
	_enqueue_or_run([]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (gl43.is_active()) {
			glFlush();
		}
#endif
	});
}

void GL43Server::finish() {
	_enqueue_or_run([]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (gl43.is_active()) {
			glFinish();
		}
#endif
	});
}

// =========================================================
// COMPUTE SHADERS
// =========================================================

void GL43Server::_compile_shader_gl(RID p_rid, const String &p_code) {
#ifdef GL43_BACKEND_SUPPORTED
	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	String trimmed = p_code.strip_edges(true, false);
	String final_src = trimmed.begins_with("#version") ? p_code : (String("#version 430 core\n") + p_code);
	CharString utf8 = final_src.utf8();
	const char *src_ptr = utf8.get_data();

	GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(cs, 1, &src_ptr, nullptr);
	glCompileShader(cs);

	GLint compile_ok = 0;
	glGetShaderiv(cs, GL_COMPILE_STATUS, &compile_ok);
	if (!compile_ok) {
		char log_buf[4096];
		glGetShaderInfoLog(cs, sizeof(log_buf), nullptr, log_buf);
		glDeleteShader(cs);

		String err_msg = String("GL43Server compute shader compile error:\n") + String::utf8(log_buf);
		ERR_PRINT(err_msg);

		MutexLock lock(data_mutex);
		last_error = err_msg;
		ShaderEntry *entry = shader_owner.getornull(p_rid);
		if (entry) {
			entry->log = err_msg;
			entry->valid = false;
		}
		return;
	}

	GLuint prog = glCreateProgram();
	glAttachShader(prog, cs);
	glLinkProgram(prog);
	glDeleteShader(cs);

	GLint link_ok = 0;
	glGetProgramiv(prog, GL_LINK_STATUS, &link_ok);
	if (!link_ok) {
		char log_buf[4096];
		glGetProgramInfoLog(prog, sizeof(log_buf), nullptr, log_buf);
		glDeleteProgram(prog);

		String err_msg = String("GL43Server compute shader link error:\n") + String::utf8(log_buf);
		ERR_PRINT(err_msg);

		MutexLock lock(data_mutex);
		last_error = err_msg;
		ShaderEntry *entry = shader_owner.getornull(p_rid);
		if (entry) {
			entry->log = err_msg;
			entry->valid = false;
		}
		return;
	}

	uint32_t old_prog = 0;
	{
		MutexLock lock(data_mutex);
		last_error = String();
		ShaderEntry *entry = shader_owner.getornull(p_rid);
		if (!entry) {
			glDeleteProgram(prog);
			return;
		}
		old_prog = entry->program;
		entry->program = (uint32_t)prog;
		entry->valid = true;
		entry->log = String();
		entry->uniform_cache.clear();
		if (bound_shader == p_rid) {
			glUseProgram(prog);
		}
	}

	if (old_prog) {
		glDeleteProgram((GLuint)old_prog);
	}
#endif
}

RID GL43Server::create_compute_shader(const String &p_glsl_code) {
	ERR_FAIL_COND_V_MSG(!is_available(), RID(), "GL43Server is not available on this renderer/platform.");

	ShaderEntry *entry = memnew(ShaderEntry);
	RID rid;
	{
		MutexLock lock(data_mutex);
		rid = shader_owner.make_rid(entry);
	}

	_enqueue_or_run([this, rid, p_glsl_code]() {
		_compile_shader_gl(rid, p_glsl_code);
	});
	return rid;
}

bool GL43Server::reload_compute_shader(RID p_shader, const String &p_glsl_code) {
	ERR_FAIL_COND_V(!is_available(), false);
	{
		MutexLock lock(data_mutex);
		ERR_FAIL_COND_V(!shader_owner.owns(p_shader), false);
	}
	_enqueue_or_run([this, p_shader, p_glsl_code]() {
		_compile_shader_gl(p_shader, p_glsl_code);
	});
	if (_is_on_gl_thread()) {
		MutexLock lock(data_mutex);
		ShaderEntry *entry = shader_owner.getornull(p_shader);
		return entry ? entry->valid : false;
	}
	return true;
}

bool GL43Server::is_shader_valid(RID p_shader) const {
	if (!p_shader.is_valid()) {
		return false;
	}
	MutexLock lock(data_mutex);
	const ShaderEntry *entry = shader_owner.owns(p_shader) ? shader_owner.getornull(p_shader) : nullptr;
	return entry ? entry->valid : false;
}

String GL43Server::get_shader_log(RID p_shader) const {
	if (!p_shader.is_valid()) {
		return String();
	}
	MutexLock lock(data_mutex);
	const ShaderEntry *entry = shader_owner.owns(p_shader) ? shader_owner.getornull(p_shader) : nullptr;
	return entry ? entry->log : String();
}

void GL43Server::bind_compute_shader(RID p_shader) {
	_enqueue_or_run([this, p_shader]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		if (!p_shader.is_valid()) {
			bound_shader = RID();
			glUseProgram(0);
			return;
		}
		uint32_t prog = 0;
		{
			MutexLock lock(data_mutex);
			ShaderEntry *entry = shader_owner.owns(p_shader) ? shader_owner.getornull(p_shader) : nullptr;
			if (!entry || !entry->valid || entry->program == 0) {
				bound_shader = RID();
				glUseProgram(0);
				return;
			}
			prog = entry->program;
			bound_shader = p_shader;
		}
		glUseProgram((GLuint)prog);
#endif
	});
}

void GL43Server::unbind_compute_shader() {
	bind_compute_shader(RID());
}

// =========================================================
// UNIFORMS
// =========================================================

int32_t GL43Server::_get_bound_uniform_loc(const String &p_name) {
#ifdef GL43_BACKEND_SUPPORTED
	MutexLock lock(data_mutex);
	if (!bound_shader.is_valid() || !shader_owner.owns(bound_shader)) {
		return -1;
	}
	ShaderEntry *entry = shader_owner.getornull(bound_shader);
	if (!entry || !entry->valid || entry->program == 0) {
		return -1;
	}
	CharString utf8 = p_name.utf8();
	std::string key(utf8.get_data());
	auto it = entry->uniform_cache.find(key);
	if (it != entry->uniform_cache.end()) {
		return it->second;
	}
	GLint loc = glGetUniformLocation((GLuint)entry->program, key.c_str());
	entry->uniform_cache[key] = (int32_t)loc;
	return (int32_t)loc;
#else
	return -1;
#endif
}

void GL43Server::set_uniform(const String &p_name, const Variant &p_value) {
	_enqueue_or_run([this, p_name, p_value]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		GLint loc = (GLint)_get_bound_uniform_loc(p_name);
		if (loc < 0) {
			return;
		}

		switch (p_value.get_type()) {
			case Variant::BOOL: {
				glUniform1i(loc, bool(p_value) ? 1 : 0);
			} break;
			case Variant::INT: {
				glUniform1i(loc, (GLint)(int)p_value);
			} break;
			case Variant::REAL: {
				glUniform1f(loc, (GLfloat)(double)p_value);
			} break;
			case Variant::VECTOR2: {
				Vector2 v = p_value;
				glUniform2f(loc, (GLfloat)v.x, (GLfloat)v.y);
			} break;
			case Variant::VECTOR3: {
				Vector3 v = p_value;
				glUniform3f(loc, (GLfloat)v.x, (GLfloat)v.y, (GLfloat)v.z);
			} break;
			case Variant::COLOR: {
				Color c = p_value;
				glUniform4f(loc, c.r, c.g, c.b, c.a);
			} break;
			case Variant::PLANE: {
				Plane p = p_value;
				glUniform4f(loc, (GLfloat)p.normal.x, (GLfloat)p.normal.y, (GLfloat)p.normal.z, (GLfloat)p.d);
			} break;
			case Variant::QUAT: {
				Quat q = p_value;
				glUniform4f(loc, (GLfloat)q.x, (GLfloat)q.y, (GLfloat)q.z, (GLfloat)q.w);
			} break;
			case Variant::RECT2: {
				Rect2 r = p_value;
				glUniform4f(loc, (GLfloat)r.position.x, (GLfloat)r.position.y, (GLfloat)r.size.x, (GLfloat)r.size.y);
			} break;
			case Variant::BASIS: {
				Basis b = p_value;
				float m[9] = {
					(float)b.elements[0][0], (float)b.elements[1][0], (float)b.elements[2][0],
					(float)b.elements[0][1], (float)b.elements[1][1], (float)b.elements[2][1],
					(float)b.elements[0][2], (float)b.elements[1][2], (float)b.elements[2][2]
				};
				glUniformMatrix3fv(loc, 1, GL_FALSE, m);
			} break;
			case Variant::TRANSFORM2D: {
				Transform2D t = p_value;
				float m[9] = {
					(float)t.elements[0][0], (float)t.elements[0][1], 0.0f,
					(float)t.elements[1][0], (float)t.elements[1][1], 0.0f,
					(float)t.elements[2][0], (float)t.elements[2][1], 1.0f
				};
				glUniformMatrix3fv(loc, 1, GL_FALSE, m);
			} break;
			case Variant::TRANSFORM: {
				Transform t = p_value;
				float m[16] = {
					(float)t.basis.elements[0][0], (float)t.basis.elements[1][0], (float)t.basis.elements[2][0], 0.0f,
					(float)t.basis.elements[0][1], (float)t.basis.elements[1][1], (float)t.basis.elements[2][1], 0.0f,
					(float)t.basis.elements[0][2], (float)t.basis.elements[1][2], (float)t.basis.elements[2][2], 0.0f,
					(float)t.origin.x, (float)t.origin.y, (float)t.origin.z, 1.0f
				};
				glUniformMatrix4fv(loc, 1, GL_FALSE, m);
			} break;
			case Variant::POOL_INT_ARRAY: {
				PoolIntArray arr = p_value;
				if (arr.size() > 0) {
					PoolIntArray::Read r = arr.read();
					glUniform1iv(loc, arr.size(), r.ptr());
				}
			} break;
			case Variant::POOL_REAL_ARRAY: {
				PoolRealArray arr = p_value;
				if (arr.size() > 0) {
					std::vector<float> tmp((size_t)arr.size());
					PoolRealArray::Read r = arr.read();
					for (int i = 0; i < arr.size(); i++) {
						tmp[i] = (float)r[i];
					}
					glUniform1fv(loc, arr.size(), tmp.data());
				}
			} break;
			case Variant::POOL_VECTOR2_ARRAY: {
				PoolVector2Array arr = p_value;
				if (arr.size() > 0) {
					std::vector<float> tmp((size_t)arr.size() * 2);
					PoolVector2Array::Read r = arr.read();
					for (int i = 0; i < arr.size(); i++) {
						tmp[i * 2 + 0] = (float)r[i].x;
						tmp[i * 2 + 1] = (float)r[i].y;
					}
					glUniform2fv(loc, arr.size(), tmp.data());
				}
			} break;
			case Variant::POOL_VECTOR3_ARRAY: {
				PoolVector3Array arr = p_value;
				if (arr.size() > 0) {
					std::vector<float> tmp((size_t)arr.size() * 3);
					PoolVector3Array::Read r = arr.read();
					for (int i = 0; i < arr.size(); i++) {
						tmp[i * 3 + 0] = (float)r[i].x;
						tmp[i * 3 + 1] = (float)r[i].y;
						tmp[i * 3 + 2] = (float)r[i].z;
					}
					glUniform3fv(loc, arr.size(), tmp.data());
				}
			} break;
			case Variant::POOL_COLOR_ARRAY: {
				PoolColorArray arr = p_value;
				if (arr.size() > 0) {
					std::vector<float> tmp((size_t)arr.size() * 4);
					PoolColorArray::Read r = arr.read();
					for (int i = 0; i < arr.size(); i++) {
						tmp[i * 4 + 0] = r[i].r;
						tmp[i * 4 + 1] = r[i].g;
						tmp[i * 4 + 2] = r[i].b;
						tmp[i * 4 + 3] = r[i].a;
					}
					glUniform4fv(loc, arr.size(), tmp.data());
				}
			} break;
			default: {
				ERR_PRINT(vformat("GL43Server.set_uniform: Unsupported Variant type %s for uniform '%s'.", Variant::get_type_name(p_value.get_type()), p_name));
			} break;
		}
#endif
	});
}

void GL43Server::set_uniform_color(const String &p_name, const Color &p_color, bool p_to_linear) {
	Color c = p_to_linear ? p_color.to_linear() : p_color;
	set_uniform(p_name, c);
}

static void _gl43_upload_camera_matrix(GLint p_loc, const CameraMatrix &p_cm) {
#ifdef GL43_BACKEND_SUPPORTED
	float m[16];
	for (int c = 0; c < 4; c++) {
		for (int r = 0; r < 4; r++) {
			m[c * 4 + r] = (float)p_cm.matrix[c][r];
		}
	}
	glUniformMatrix4fv(p_loc, 1, GL_FALSE, m);
#endif
}

void GL43Server::set_uniform_projection(const String &p_name, float p_fov_degrees, float p_aspect, float p_z_near, float p_z_far) {
	_enqueue_or_run([this, p_name, p_fov_degrees, p_aspect, p_z_near, p_z_far]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		GLint loc = (GLint)_get_bound_uniform_loc(p_name);
		if (loc < 0) {
			return;
		}
		CameraMatrix cm;
		cm.set_perspective(p_fov_degrees, MAX(p_aspect, 0.0001f), MAX(p_z_near, 0.0001f), MAX(p_z_far, p_z_near + 0.01f), false);
		_gl43_upload_camera_matrix(loc, cm);
#endif
	});
}

void GL43Server::set_uniform_inv_projection(const String &p_name, float p_fov_degrees, float p_aspect, float p_z_near, float p_z_far) {
	_enqueue_or_run([this, p_name, p_fov_degrees, p_aspect, p_z_near, p_z_far]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		GLint loc = (GLint)_get_bound_uniform_loc(p_name);
		if (loc < 0) {
			return;
		}
		CameraMatrix cm;
		cm.set_perspective(p_fov_degrees, MAX(p_aspect, 0.0001f), MAX(p_z_near, 0.0001f), MAX(p_z_far, p_z_near + 0.01f), false);
		_gl43_upload_camera_matrix(loc, cm.inverse());
#endif
	});
}

void GL43Server::set_uniform_ortho(const String &p_name, float p_size, float p_aspect, float p_z_near, float p_z_far) {
	_enqueue_or_run([this, p_name, p_size, p_aspect, p_z_near, p_z_far]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		GLint loc = (GLint)_get_bound_uniform_loc(p_name);
		if (loc < 0) {
			return;
		}
		CameraMatrix cm;
		cm.set_orthogonal(p_size, MAX(p_aspect, 0.0001f), p_z_near, p_z_far, false);
		_gl43_upload_camera_matrix(loc, cm);
#endif
	});
}

void GL43Server::set_uniform_inv_ortho(const String &p_name, float p_size, float p_aspect, float p_z_near, float p_z_far) {
	_enqueue_or_run([this, p_name, p_size, p_aspect, p_z_near, p_z_far]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		GLint loc = (GLint)_get_bound_uniform_loc(p_name);
		if (loc < 0) {
			return;
		}
		CameraMatrix cm;
		cm.set_orthogonal(p_size, MAX(p_aspect, 0.0001f), p_z_near, p_z_far, false);
		_gl43_upload_camera_matrix(loc, cm.inverse());
#endif
	});
}

// =========================================================
// SSBOs & UBOs
// =========================================================

bool GL43Server::_variant_to_bytes(const Variant &p_data, std::vector<uint8_t> &r_bytes) {
	r_bytes.clear();
	if (p_data.get_type() == Variant::NIL) {
		return true;
	}

	auto push_float = [&](float f) {
		const uint8_t *p = (const uint8_t *)&f;
		r_bytes.insert(r_bytes.end(), p, p + 4);
	};
	auto push_int = [&](int32_t v) {
		const uint8_t *p = (const uint8_t *)&v;
		r_bytes.insert(r_bytes.end(), p, p + 4);
	};

	switch (p_data.get_type()) {
		case Variant::POOL_BYTE_ARRAY: {
			PoolByteArray arr = p_data;
			r_bytes.resize((size_t)arr.size());
			if (arr.size() > 0) {
				PoolByteArray::Read r = arr.read();
				std::memcpy(r_bytes.data(), r.ptr(), (size_t)arr.size());
			}
			return true;
		}
		case Variant::POOL_INT_ARRAY: {
			PoolIntArray arr = p_data;
			r_bytes.resize((size_t)arr.size() * 4);
			if (arr.size() > 0) {
				PoolIntArray::Read r = arr.read();
				std::memcpy(r_bytes.data(), r.ptr(), (size_t)arr.size() * 4);
			}
			return true;
		}
		case Variant::POOL_REAL_ARRAY: {
			PoolRealArray arr = p_data;
			r_bytes.reserve((size_t)arr.size() * 4);
			PoolRealArray::Read r = arr.read();
			for (int i = 0; i < arr.size(); i++) {
				push_float((float)r[i]);
			}
			return true;
		}
		case Variant::POOL_VECTOR2_ARRAY: {
			PoolVector2Array arr = p_data;
			r_bytes.reserve((size_t)arr.size() * 8);
			PoolVector2Array::Read r = arr.read();
			for (int i = 0; i < arr.size(); i++) {
				push_float((float)r[i].x);
				push_float((float)r[i].y);
			}
			return true;
		}
		case Variant::POOL_VECTOR3_ARRAY: {
			PoolVector3Array arr = p_data;
			r_bytes.reserve((size_t)arr.size() * 12);
			PoolVector3Array::Read r = arr.read();
			for (int i = 0; i < arr.size(); i++) {
				push_float((float)r[i].x);
				push_float((float)r[i].y);
				push_float((float)r[i].z);
			}
			return true;
		}
		case Variant::POOL_COLOR_ARRAY: {
			PoolColorArray arr = p_data;
			r_bytes.reserve((size_t)arr.size() * 16);
			PoolColorArray::Read r = arr.read();
			for (int i = 0; i < arr.size(); i++) {
				push_float(r[i].r);
				push_float(r[i].g);
				push_float(r[i].b);
				push_float(r[i].a);
			}
			return true;
		}
		case Variant::ARRAY: {
			Array arr = p_data;
			for (int i = 0; i < arr.size(); i++) {
				const Variant &el = arr[i];
				switch (el.get_type()) {
					case Variant::BOOL:
						push_int(bool(el) ? 1 : 0);
						break;
					case Variant::INT:
						push_int((int32_t)(int)el);
						break;
					case Variant::REAL:
						push_float((float)(double)el);
						break;
					case Variant::VECTOR2: {
						Vector2 v = el;
						push_float((float)v.x);
						push_float((float)v.y);
					} break;
					case Variant::VECTOR3: {
						Vector3 v = el;
						push_float((float)v.x);
						push_float((float)v.y);
						push_float((float)v.z);
					} break;
					case Variant::COLOR: {
						Color c = el;
						push_float(c.r);
						push_float(c.g);
						push_float(c.b);
						push_float(c.a);
					} break;
					case Variant::PLANE: {
						Plane p = el;
						push_float((float)p.normal.x);
						push_float((float)p.normal.y);
						push_float((float)p.normal.z);
						push_float((float)p.d);
					} break;
					case Variant::QUAT: {
						Quat q = el;
						push_float((float)q.x);
						push_float((float)q.y);
						push_float((float)q.z);
						push_float((float)q.w);
					} break;
					case Variant::RECT2: {
						Rect2 rc = el;
						push_float((float)rc.position.x);
						push_float((float)rc.position.y);
						push_float((float)rc.size.x);
						push_float((float)rc.size.y);
					} break;
					case Variant::TRANSFORM: {
						Transform t = el;
						for (int c = 0; c < 3; c++) {
							push_float((float)t.basis.elements[0][c]);
							push_float((float)t.basis.elements[1][c]);
							push_float((float)t.basis.elements[2][c]);
							push_float(0.0f);
						}
						push_float((float)t.origin.x);
						push_float((float)t.origin.y);
						push_float((float)t.origin.z);
						push_float(1.0f);
					} break;
					default:
						return false;
				}
			}
			return true;
		}
		default:
			return false;
	}
}

void GL43Server::_create_buffer_gl(RID p_rid, uint32_t p_target, int p_size_bytes, std::vector<uint8_t> p_bytes, BufferUsage p_usage) {
#ifdef GL43_BACKEND_SUPPORTED
	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	GLuint buf = 0;
	glGenBuffers(1, &buf);
	GLenum gl_usage = to_gl_usage(p_usage);

	int alloc_size = MAX(p_size_bytes, (int)p_bytes.size());
	glBindBuffer((GLenum)p_target, buf);
	if (!p_bytes.empty()) {
		if (alloc_size == (int)p_bytes.size()) {
			glBufferData((GLenum)p_target, (GLsizeiptr)alloc_size, p_bytes.data(), gl_usage);
		} else {
			std::vector<uint8_t> padded((size_t)alloc_size, 0);
			std::memcpy(padded.data(), p_bytes.data(), p_bytes.size());
			glBufferData((GLenum)p_target, (GLsizeiptr)alloc_size, padded.data(), gl_usage);
		}
	} else if (alloc_size > 0) {
		std::vector<uint8_t> zeros((size_t)alloc_size, 0);
		glBufferData((GLenum)p_target, (GLsizeiptr)alloc_size, zeros.data(), gl_usage);
	} else {
		uint32_t dummy = 0;
		alloc_size = 4;
		glBufferData((GLenum)p_target, 4, &dummy, gl_usage);
	}
	glBindBuffer((GLenum)p_target, 0);

	MutexLock lock(data_mutex);
	BufferEntry *entry = buffer_owner.getornull(p_rid);
	if (!entry) {
		glDeleteBuffers(1, &buf);
		return;
	}
	entry->buffer = (uint32_t)buf;
	entry->target = p_target;
	entry->size_bytes = alloc_size;
	entry->usage = (uint32_t)gl_usage;
#endif
}

void GL43Server::_upload_buffer_gl(RID p_rid, std::vector<uint8_t> p_bytes, BufferUsage p_usage) {
#ifdef GL43_BACKEND_SUPPORTED
	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	uint32_t buf = 0;
	uint32_t target = GL_SHADER_STORAGE_BUFFER;
	int32_t cur_size = 0;
	uint32_t cur_usage = 0;
	{
		MutexLock lock(data_mutex);
		BufferEntry *entry = buffer_owner.getornull(p_rid);
		if (!entry || entry->buffer == 0) {
			return;
		}
		buf = entry->buffer;
		target = entry->target;
		cur_size = entry->size_bytes;
		cur_usage = entry->usage;
	}

	GLenum new_usage = to_gl_usage(p_usage);
	int new_size = (int)p_bytes.size();
	glBindBuffer((GLenum)target, (GLuint)buf);
	if (new_size == 0) {
		uint32_t dummy = 0;
		glBufferData((GLenum)target, 4, &dummy, new_usage);
		new_size = 4;
	} else if (new_size == cur_size && new_usage == (GLenum)cur_usage) {
		glBufferSubData((GLenum)target, 0, (GLsizeiptr)new_size, p_bytes.data());
	} else {
		glBufferData((GLenum)target, (GLsizeiptr)new_size, p_bytes.data(), new_usage);
	}
	glBindBuffer((GLenum)target, 0);

	MutexLock lock(data_mutex);
	BufferEntry *entry = buffer_owner.getornull(p_rid);
	if (entry) {
		entry->size_bytes = new_size;
		entry->usage = (uint32_t)new_usage;
	}
#endif
}

void GL43Server::_upload_buffer_subdata_gl(RID p_rid, int p_offset, std::vector<uint8_t> p_bytes) {
#ifdef GL43_BACKEND_SUPPORTED
	if (p_bytes.empty() || p_offset < 0) {
		return;
	}
	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	uint32_t buf = 0;
	uint32_t target = GL_SHADER_STORAGE_BUFFER;
	int32_t cur_size = 0;
	{
		MutexLock lock(data_mutex);
		BufferEntry *entry = buffer_owner.getornull(p_rid);
		if (!entry || entry->buffer == 0) {
			return;
		}
		buf = entry->buffer;
		target = entry->target;
		cur_size = entry->size_bytes;
	}

	if (p_offset + (int)p_bytes.size() > cur_size) {
		ERR_PRINT("GL43Server.upload_ssbo_subdata: Write exceeds allocated SSBO size.");
		return;
	}

	glBindBuffer((GLenum)target, (GLuint)buf);
	glBufferSubData((GLenum)target, (GLintptr)p_offset, (GLsizeiptr)p_bytes.size(), p_bytes.data());
	glBindBuffer((GLenum)target, 0);
#endif
}

RID GL43Server::create_ssbo(int p_size_bytes, const Variant &p_initial_data, BufferUsage p_usage) {
	ERR_FAIL_COND_V_MSG(!is_available(), RID(), "GL43Server is not available.");

	std::vector<uint8_t> bytes;
	if (!_variant_to_bytes(p_initial_data, bytes)) {
		ERR_PRINT("GL43Server.create_ssbo: Unsupported initial_data type.");
		return RID();
	}

	BufferEntry *entry = memnew(BufferEntry);
#ifdef GL43_BACKEND_SUPPORTED
	entry->target = GL_SHADER_STORAGE_BUFFER;
	entry->usage = (uint32_t)to_gl_usage(p_usage);
#endif
	entry->size_bytes = MAX(p_size_bytes, (int)bytes.size());

	RID rid;
	{
		MutexLock lock(data_mutex);
		rid = buffer_owner.make_rid(entry);
	}

	_enqueue_or_run([this, rid, p_size_bytes, bytes = std::move(bytes), p_usage]() mutable {
#ifdef GL43_BACKEND_SUPPORTED
		_create_buffer_gl(rid, GL_SHADER_STORAGE_BUFFER, p_size_bytes, std::move(bytes), p_usage);
#endif
	});
	return rid;
}

void GL43Server::upload_ssbo(RID p_ssbo, const Variant &p_data, BufferUsage p_usage) {
	ERR_FAIL_COND(!is_available());
	std::vector<uint8_t> bytes;
	if (!_variant_to_bytes(p_data, bytes)) {
		ERR_PRINT("GL43Server.upload_ssbo: Unsupported data type.");
		return;
	}
	_enqueue_or_run([this, p_ssbo, bytes = std::move(bytes), p_usage]() mutable {
		_upload_buffer_gl(p_ssbo, std::move(bytes), p_usage);
	});
}

void GL43Server::upload_ssbo_subdata(RID p_ssbo, int p_byte_offset, const Variant &p_data) {
	ERR_FAIL_COND(!is_available());
	ERR_FAIL_COND(p_byte_offset < 0);
	std::vector<uint8_t> bytes;
	if (!_variant_to_bytes(p_data, bytes)) {
		ERR_PRINT("GL43Server.upload_ssbo_subdata: Unsupported data type.");
		return;
	}
	_enqueue_or_run([this, p_ssbo, p_byte_offset, bytes = std::move(bytes)]() mutable {
		_upload_buffer_subdata_gl(p_ssbo, p_byte_offset, std::move(bytes));
	});
}

void GL43Server::clear_ssbo(RID p_ssbo, uint32_t p_clear_value) {
	_enqueue_or_run([this, p_ssbo, p_clear_value]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		_load_extra_gl43_procs();

		uint32_t buf = 0;
		uint32_t target = GL_SHADER_STORAGE_BUFFER;
		int32_t sz = 0;
		{
			MutexLock lock(data_mutex);
			BufferEntry *entry = buffer_owner.getornull(p_ssbo);
			if (!entry || entry->buffer == 0) {
				return;
			}
			buf = entry->buffer;
			target = entry->target;
			sz = entry->size_bytes;
		}

		glBindBuffer((GLenum)target, (GLuint)buf);
		if (fn_clear_buffer_data && (sz % 4 == 0)) {
			((PFN_glClearBufferData_GL43)fn_clear_buffer_data)((GLenum)target, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &p_clear_value);
		} else {
			std::vector<uint8_t> fill((size_t)sz, 0);
			if (p_clear_value != 0) {
				for (int i = 0; i + 3 < sz; i += 4) {
					std::memcpy(&fill[i], &p_clear_value, 4);
				}
			}
			glBufferSubData((GLenum)target, 0, (GLsizeiptr)sz, fill.data());
		}
		glBindBuffer((GLenum)target, 0);
#endif
	});
}

void GL43Server::copy_ssbo(RID p_src_ssbo, RID p_dst_ssbo, int p_byte_size, int p_src_offset, int p_dst_offset) {
	ERR_FAIL_COND(p_src_offset < 0 || p_dst_offset < 0);
	_enqueue_or_run([this, p_src_ssbo, p_dst_ssbo, p_byte_size, p_src_offset, p_dst_offset]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}

		uint32_t src_buf = 0, dst_buf = 0;
		int32_t src_sz = 0, dst_sz = 0;
		{
			MutexLock lock(data_mutex);
			BufferEntry *src = buffer_owner.getornull(p_src_ssbo);
			BufferEntry *dst = buffer_owner.getornull(p_dst_ssbo);
			if (!src || !dst || src->buffer == 0 || dst->buffer == 0) {
				return;
			}
			src_buf = src->buffer;
			dst_buf = dst->buffer;
			src_sz = src->size_bytes;
			dst_sz = dst->size_bytes;
		}

		int copy_sz = (p_byte_size < 0) ? MIN(src_sz - p_src_offset, dst_sz - p_dst_offset) : p_byte_size;
		if (copy_sz <= 0 || p_src_offset + copy_sz > src_sz || p_dst_offset + copy_sz > dst_sz) {
			ERR_PRINT("GL43Server.copy_ssbo: Invalid copy range.");
			return;
		}

		glBindBuffer(GL_COPY_READ_BUFFER, (GLuint)src_buf);
		glBindBuffer(GL_COPY_WRITE_BUFFER, (GLuint)dst_buf);
		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, (GLintptr)p_src_offset, (GLintptr)p_dst_offset, (GLsizeiptr)copy_sz);
		glBindBuffer(GL_COPY_READ_BUFFER, 0);
		glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
#endif
	});
}

void GL43Server::bind_ssbo(RID p_ssbo, int p_binding_index) {
	ERR_FAIL_COND(p_binding_index < 0);
	_enqueue_or_run([this, p_ssbo, p_binding_index]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		if (!p_ssbo.is_valid()) {
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)p_binding_index, 0);
			return;
		}
		uint32_t buf = 0;
		{
			MutexLock lock(data_mutex);
			BufferEntry *entry = buffer_owner.owns(p_ssbo) ? buffer_owner.getornull(p_ssbo) : nullptr;
			if (entry) {
				buf = entry->buffer;
			}
		}
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)p_binding_index, (GLuint)buf);
#endif
	});
}

void GL43Server::unbind_ssbo(int p_binding_index) {
	bind_ssbo(RID(), p_binding_index);
}

int GL43Server::get_ssbo_size(RID p_ssbo) const {
	if (!p_ssbo.is_valid()) {
		return 0;
	}
	MutexLock lock(data_mutex);
	const BufferEntry *entry = buffer_owner.owns(p_ssbo) ? buffer_owner.getornull(p_ssbo) : nullptr;
	return entry ? entry->size_bytes : 0;
}

PoolByteArray GL43Server::read_ssbo(RID p_ssbo, int p_byte_size, int p_byte_offset) {
	PoolByteArray out;
#ifdef GL43_BACKEND_SUPPORTED
	ERR_FAIL_COND_V_MSG(!is_available(), out, "GL43Server is not available.");
	ERR_FAIL_COND_V_MSG(!_is_on_gl_thread(), out, "GL43Server.read_ssbo must be called on the render thread (use Single-Threaded mode or VisualServer.frame_post_draw).");
	ERR_FAIL_COND_V(p_byte_offset < 0, out);

	_flush_pending_commands();

	uint32_t buf = 0;
	uint32_t target = GL_SHADER_STORAGE_BUFFER;
	int32_t total_sz = 0;
	{
		MutexLock lock(data_mutex);
		BufferEntry *entry = buffer_owner.owns(p_ssbo) ? buffer_owner.getornull(p_ssbo) : nullptr;
		ERR_FAIL_COND_V(!entry || entry->buffer == 0, out);
		buf = entry->buffer;
		target = entry->target;
		total_sz = entry->size_bytes;
	}

	int read_sz = (p_byte_size < 0) ? (total_sz - p_byte_offset) : p_byte_size;
	ERR_FAIL_COND_V(read_sz <= 0 || p_byte_offset + read_sz > total_sz, out);

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return out;
	}

	out.resize(read_sz);
	{
		PoolByteArray::Write w = out.write();
		glBindBuffer((GLenum)target, (GLuint)buf);
		glGetBufferSubData((GLenum)target, (GLintptr)p_byte_offset, (GLsizeiptr)read_sz, w.ptr());
		glBindBuffer((GLenum)target, 0);
	}
#endif
	return out;
}

PoolRealArray GL43Server::read_ssbo_floats(RID p_ssbo, int p_float_count, int p_byte_offset) {
	PoolRealArray out;
	int byte_size = (p_float_count < 0) ? -1 : (p_float_count * 4);
	PoolByteArray bytes = read_ssbo(p_ssbo, byte_size, p_byte_offset);
	int count = bytes.size() / 4;
	if (count <= 0) {
		return out;
	}
	out.resize(count);
	PoolByteArray::Read r = bytes.read();
	PoolRealArray::Write w = out.write();
	const float *fptr = (const float *)r.ptr();
	for (int i = 0; i < count; i++) {
		w[i] = fptr[i];
	}
	return out;
}

PoolIntArray GL43Server::read_ssbo_ints(RID p_ssbo, int p_int_count, int p_byte_offset) {
	PoolIntArray out;
	int byte_size = (p_int_count < 0) ? -1 : (p_int_count * 4);
	PoolByteArray bytes = read_ssbo(p_ssbo, byte_size, p_byte_offset);
	int count = bytes.size() / 4;
	if (count <= 0) {
		return out;
	}
	out.resize(count);
	PoolByteArray::Read r = bytes.read();
	PoolIntArray::Write w = out.write();
	std::memcpy(w.ptr(), r.ptr(), (size_t)count * 4);
	return out;
}

RID GL43Server::create_ubo(int p_size_bytes, const Variant &p_initial_data, BufferUsage p_usage) {
	ERR_FAIL_COND_V_MSG(!is_available(), RID(), "GL43Server is not available.");

	std::vector<uint8_t> bytes;
	if (!_variant_to_bytes(p_initial_data, bytes)) {
		ERR_PRINT("GL43Server.create_ubo: Unsupported initial_data type.");
		return RID();
	}

	BufferEntry *entry = memnew(BufferEntry);
#ifdef GL43_BACKEND_SUPPORTED
	entry->target = GL_UNIFORM_BUFFER;
	entry->usage = (uint32_t)to_gl_usage(p_usage);
#endif
	entry->size_bytes = MAX(p_size_bytes, (int)bytes.size());

	RID rid;
	{
		MutexLock lock(data_mutex);
		rid = buffer_owner.make_rid(entry);
	}

	_enqueue_or_run([this, rid, p_size_bytes, bytes = std::move(bytes), p_usage]() mutable {
#ifdef GL43_BACKEND_SUPPORTED
		_create_buffer_gl(rid, GL_UNIFORM_BUFFER, p_size_bytes, std::move(bytes), p_usage);
#endif
	});
	return rid;
}

void GL43Server::upload_ubo(RID p_ubo, const Variant &p_data, BufferUsage p_usage) {
	upload_ssbo(p_ubo, p_data, p_usage);
}

void GL43Server::bind_ubo(RID p_ubo, int p_binding_index) {
	ERR_FAIL_COND(p_binding_index < 0);
	_enqueue_or_run([this, p_ubo, p_binding_index]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		if (!p_ubo.is_valid()) {
			glBindBufferBase(GL_UNIFORM_BUFFER, (GLuint)p_binding_index, 0);
			return;
		}
		uint32_t buf = 0;
		{
			MutexLock lock(data_mutex);
			BufferEntry *entry = buffer_owner.owns(p_ubo) ? buffer_owner.getornull(p_ubo) : nullptr;
			if (entry) {
				buf = entry->buffer;
			}
		}
		glBindBufferBase(GL_UNIFORM_BUFFER, (GLuint)p_binding_index, (GLuint)buf);
#endif
	});
}

void GL43Server::unbind_ubo(int p_binding_index) {
	bind_ubo(RID(), p_binding_index);
}

PoolByteArray GL43Server::pack_vec3_as_vec4(const PoolVector3Array &p_vectors, float p_w) const {
	PoolByteArray out;
	int count = p_vectors.size();
	if (count <= 0) {
		return out;
	}
	out.resize(count * 16);
	PoolVector3Array::Read r = p_vectors.read();
	PoolByteArray::Write w = out.write();
	float *dst = (float *)w.ptr();
	for (int i = 0; i < count; i++) {
		dst[i * 4 + 0] = (float)r[i].x;
		dst[i * 4 + 1] = (float)r[i].y;
		dst[i * 4 + 2] = (float)r[i].z;
		dst[i * 4 + 3] = p_w;
	}
	return out;
}

// =========================================================
// TEXTURES & IMAGE LOAD/STORE
// =========================================================

bool GL43Server::_resolve_texture(const Variant &p_tex_or_rid, uint32_t &r_tex_id, uint32_t &r_target, uint32_t &r_internal_fmt, int &r_w, int &r_h, int &r_d) const {
#ifdef GL43_BACKEND_SUPPORTED
	RID rid;
	if (p_tex_or_rid.get_type() == Variant::_RID) {
		rid = p_tex_or_rid;
	} else if (p_tex_or_rid.get_type() == Variant::OBJECT) {
		Resource *res = Object::cast_to<Resource>(p_tex_or_rid);
		if (res) {
			rid = res->get_rid();
		}
	}
	if (!rid.is_valid()) {
		return false;
	}

	{
		MutexLock lock(data_mutex);
		if (texture_owner.owns(rid)) {
			const TextureEntry *own = texture_owner.getornull(rid);
			if (!own || own->tex_id == 0) {
				return false;
			}
			r_tex_id = own->tex_id;
			r_target = own->target;
			r_internal_fmt = own->internal_format;
			r_w = own->width;
			r_h = own->height;
			r_d = own->depth;
			return true;
		}
	}

	RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
	if (!rs || !rs->texture_owner.owns(rid)) {
		return false;
	}
	RasterizerStorageGLES3::Texture *gt = rs->texture_owner.getornull(rid);
	if (gt) {
		gt = gt->get_ptr();
	}
	if (!gt || gt->tex_id == 0) {
		return false;
	}

	r_tex_id = gt->tex_id;
	r_target = gt->target ? gt->target : GL_TEXTURE_2D;
	r_internal_fmt = gt->gl_internal_format_cache;
	r_w = gt->width;
	r_h = gt->height;
	r_d = MAX(gt->depth, 1);
	return true;
#else
	return false;
#endif
}

void GL43Server::_create_texture_gl(RID p_rid, uint32_t p_target, int p_w, int p_h, int p_d, ImageFormat p_fmt, FilterMode p_filter, WrapMode p_wrap, bool p_mipmaps) {
#ifdef GL43_BACKEND_SUPPORTED
	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	GLenum int_fmt, fmt, type;
	to_gl_format(p_fmt, int_fmt, fmt, type);
	GLenum wrap = to_gl_wrap(p_wrap);

	GLuint tex = 0;
	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture((GLenum)p_target, tex);

	if (p_target == GL_TEXTURE_2D) {
		glTexImage2D(GL_TEXTURE_2D, 0, int_fmt, p_w, p_h, 0, fmt, type, nullptr);
	} else {
		glTexImage3D((GLenum)p_target, 0, int_fmt, p_w, p_h, p_d, 0, fmt, type, nullptr);
	}

	bool is_int = is_integer_internal_format(int_fmt);
	bool use_mips = p_mipmaps && !is_int;

	GLenum min_filter = GL_LINEAR;
	GLenum mag_filter = GL_LINEAR;
	if (is_int || p_filter == FILTER_NEAREST) {
		min_filter = GL_NEAREST;
		mag_filter = GL_NEAREST;
	} else if (p_filter == FILTER_LINEAR_MIPMAP || use_mips) {
		min_filter = GL_LINEAR_MIPMAP_LINEAR;
		mag_filter = GL_LINEAR;
	}

	glTexParameteri((GLenum)p_target, GL_TEXTURE_MIN_FILTER, min_filter);
	glTexParameteri((GLenum)p_target, GL_TEXTURE_MAG_FILTER, mag_filter);
	glTexParameteri((GLenum)p_target, GL_TEXTURE_WRAP_S, wrap);
	glTexParameteri((GLenum)p_target, GL_TEXTURE_WRAP_T, wrap);
	if (p_target == GL_TEXTURE_3D) {
		glTexParameteri((GLenum)p_target, GL_TEXTURE_WRAP_R, wrap);
	}
	if (use_mips && (p_target == GL_TEXTURE_2D || p_target == GL_TEXTURE_2D_ARRAY)) {
		glGenerateMipmap((GLenum)p_target);
	}
	glBindTexture((GLenum)p_target, 0);

	MutexLock lock(data_mutex);
	TextureEntry *entry = texture_owner.getornull(p_rid);
	if (!entry) {
		glDeleteTextures(1, &tex);
		return;
	}
	entry->tex_id = (uint32_t)tex;
	entry->internal_format = (uint32_t)int_fmt;
	entry->format = (uint32_t)fmt;
	entry->type = (uint32_t)type;
	entry->mipmaps = use_mips;
#endif
}

RID GL43Server::create_texture_2d(int p_width, int p_height, ImageFormat p_format, FilterMode p_filter, WrapMode p_wrap) {
	ERR_FAIL_COND_V_MSG(!is_available(), RID(), "GL43Server is not available.");
	ERR_FAIL_COND_V(p_width <= 0 || p_height <= 0, RID());

	bool mips = (p_filter == FILTER_LINEAR_MIPMAP);
	TextureEntry *entry = memnew(TextureEntry);
#ifdef GL43_BACKEND_SUPPORTED
	entry->target = GL_TEXTURE_2D;
#endif
	entry->width = p_width;
	entry->height = p_height;
	entry->depth = 1;
	entry->mipmaps = mips;

	RID rid;
	{
		MutexLock lock(data_mutex);
		rid = texture_owner.make_rid(entry);
	}
	_enqueue_or_run([this, rid, p_width, p_height, p_format, p_filter, p_wrap, mips]() {
#ifdef GL43_BACKEND_SUPPORTED
		_create_texture_gl(rid, GL_TEXTURE_2D, p_width, p_height, 1, p_format, p_filter, p_wrap, mips);
#endif
	});
	return rid;
}

RID GL43Server::create_texture_3d(int p_width, int p_height, int p_depth, ImageFormat p_format, FilterMode p_filter) {
	ERR_FAIL_COND_V_MSG(!is_available(), RID(), "GL43Server is not available.");
	ERR_FAIL_COND_V(p_width <= 0 || p_height <= 0 || p_depth <= 0, RID());

	TextureEntry *entry = memnew(TextureEntry);
#ifdef GL43_BACKEND_SUPPORTED
	entry->target = GL_TEXTURE_3D;
#endif
	entry->width = p_width;
	entry->height = p_height;
	entry->depth = p_depth;
	entry->mipmaps = false;

	RID rid;
	{
		MutexLock lock(data_mutex);
		rid = texture_owner.make_rid(entry);
	}
	_enqueue_or_run([this, rid, p_width, p_height, p_depth, p_format, p_filter]() {
#ifdef GL43_BACKEND_SUPPORTED
		_create_texture_gl(rid, GL_TEXTURE_3D, p_width, p_height, p_depth, p_format, p_filter, WRAP_CLAMP, false);
#endif
	});
	return rid;
}

RID GL43Server::create_texture_2d_array(int p_width, int p_height, int p_layers, ImageFormat p_format, FilterMode p_filter) {
	ERR_FAIL_COND_V_MSG(!is_available(), RID(), "GL43Server is not available.");
	ERR_FAIL_COND_V(p_width <= 0 || p_height <= 0 || p_layers <= 0, RID());

	bool mips = (p_filter == FILTER_LINEAR_MIPMAP);
	TextureEntry *entry = memnew(TextureEntry);
#ifdef GL43_BACKEND_SUPPORTED
	entry->target = GL_TEXTURE_2D_ARRAY;
#endif
	entry->width = p_width;
	entry->height = p_height;
	entry->depth = p_layers;
	entry->mipmaps = mips;

	RID rid;
	{
		MutexLock lock(data_mutex);
		rid = texture_owner.make_rid(entry);
	}
	_enqueue_or_run([this, rid, p_width, p_height, p_layers, p_format, p_filter, mips]() {
#ifdef GL43_BACKEND_SUPPORTED
		_create_texture_gl(rid, GL_TEXTURE_2D_ARRAY, p_width, p_height, p_layers, p_format, p_filter, WRAP_CLAMP, mips);
#endif
	});
	return rid;
}

void GL43Server::set_texture_sampling(const Variant &p_texture, FilterMode p_filter, WrapMode p_wrap) {
	_enqueue_or_run([this, p_texture, p_filter, p_wrap]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		uint32_t tex_id = 0, target = GL_TEXTURE_2D, int_fmt = 0;
		int w = 0, h = 0, d = 1;
		if (!_resolve_texture(p_texture, tex_id, target, int_fmt, w, h, d)) {
			return;
		}

		bool is_int = is_integer_internal_format((GLenum)int_fmt);
		GLenum min_filter = GL_LINEAR;
		GLenum mag_filter = GL_LINEAR;
		if (is_int || p_filter == FILTER_NEAREST) {
			min_filter = GL_NEAREST;
			mag_filter = GL_NEAREST;
		} else if (p_filter == FILTER_LINEAR_MIPMAP) {
			min_filter = GL_LINEAR_MIPMAP_LINEAR;
			mag_filter = GL_LINEAR;
		}
		GLenum wrap = to_gl_wrap(p_wrap);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture((GLenum)target, (GLuint)tex_id);
		glTexParameteri((GLenum)target, GL_TEXTURE_MIN_FILTER, min_filter);
		glTexParameteri((GLenum)target, GL_TEXTURE_MAG_FILTER, mag_filter);
		glTexParameteri((GLenum)target, GL_TEXTURE_WRAP_S, wrap);
		glTexParameteri((GLenum)target, GL_TEXTURE_WRAP_T, wrap);
		if (target == GL_TEXTURE_3D) {
			glTexParameteri((GLenum)target, GL_TEXTURE_WRAP_R, wrap);
		}
		glBindTexture((GLenum)target, 0);
#endif
	});
}

void GL43Server::resize_texture_2d(RID p_texture, int p_width, int p_height) {
	ERR_FAIL_COND(p_width <= 0 || p_height <= 0);
	_enqueue_or_run([this, p_texture, p_width, p_height]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		uint32_t tex = 0, target = 0, int_fmt = 0, fmt = 0, type = 0;
		bool mips = false;
		{
			MutexLock lock(data_mutex);
			TextureEntry *entry = (p_texture.is_valid() && texture_owner.owns(p_texture)) ? texture_owner.getornull(p_texture) : nullptr;
			if (!entry || entry->tex_id == 0 || entry->target != GL_TEXTURE_2D) {
				return;
			}
			entry->width = p_width;
			entry->height = p_height;
			tex = entry->tex_id;
			target = entry->target;
			int_fmt = entry->internal_format;
			fmt = entry->format;
			type = entry->type;
			mips = entry->mipmaps;
		}
		glActiveTexture(GL_TEXTURE0);
		glBindTexture((GLenum)target, (GLuint)tex);
		glTexImage2D((GLenum)target, 0, (GLint)int_fmt, p_width, p_height, 0, (GLenum)fmt, (GLenum)type, nullptr);
		if (mips) {
			glGenerateMipmap((GLenum)target);
		}
		glBindTexture((GLenum)target, 0);
#endif
	});
}

void GL43Server::upload_texture_2d(const Variant &p_texture, const Ref<Image> &p_image, int p_mip_level) {
	ERR_FAIL_COND(p_image.is_null() || p_image->empty());
	ERR_FAIL_COND(p_mip_level < 0);

	Ref<Image> src_copy = p_image->duplicate();
	_enqueue_or_run([this, p_texture, src_copy, p_mip_level]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		uint32_t tex_id = 0, target = 0, int_fmt = 0;
		int w = 0, h = 0, d = 1;
		if (!_resolve_texture(p_texture, tex_id, target, int_fmt, w, h, d) || target != GL_TEXTURE_2D) {
			return;
		}

		Ref<Image> src = src_copy->duplicate();
		if (src->is_compressed()) {
			src->decompress();
		}

		int mip_w = MAX(1, w >> p_mip_level);
		int mip_h = MAX(1, h >> p_mip_level);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, (GLuint)tex_id);

		if (is_integer_internal_format((GLenum)int_fmt)) {
			if (src->get_format() != Image::FORMAT_RGBAF) {
				src->convert(Image::FORMAT_RGBAF);
			}
			if (src->get_width() != mip_w || src->get_height() != mip_h) {
				src->resize(mip_w, mip_h, Image::INTERPOLATE_NEAREST);
			}
			PoolByteArray::Read r = src->get_data().read();
			const float *fptr = (const float *)r.ptr();
			size_t count = (size_t)mip_w * (size_t)mip_h * 4;
			if (is_signed_integer_internal_format((GLenum)int_fmt)) {
				std::vector<int32_t> ints(count);
				for (size_t i = 0; i < count; i++) {
					ints[i] = (int32_t)fptr[i];
				}
				glTexSubImage2D(GL_TEXTURE_2D, p_mip_level, 0, 0, mip_w, mip_h, GL_RGBA_INTEGER, GL_INT, ints.data());
			} else {
				std::vector<uint32_t> uints(count);
				for (size_t i = 0; i < count; i++) {
					uints[i] = (uint32_t)MAX(0.0f, fptr[i]);
				}
				glTexSubImage2D(GL_TEXTURE_2D, p_mip_level, 0, 0, mip_w, mip_h, GL_RGBA_INTEGER, GL_UNSIGNED_INT, uints.data());
			}
		} else {
			bool use_float = (src->get_format() == Image::FORMAT_RGBAF || src->get_format() == Image::FORMAT_RGBAH ||
					src->get_format() == Image::FORMAT_RGBF || src->get_format() == Image::FORMAT_RGBH ||
					src->get_format() == Image::FORMAT_RF || src->get_format() == Image::FORMAT_RH ||
					src->get_format() == Image::FORMAT_RGF || src->get_format() == Image::FORMAT_RGH);
			if (use_float) {
				if (src->get_format() != Image::FORMAT_RGBAF) {
					src->convert(Image::FORMAT_RGBAF);
				}
			} else {
				if (src->get_format() != Image::FORMAT_RGBA8) {
					src->convert(Image::FORMAT_RGBA8);
				}
			}

			if (src->get_width() != mip_w || src->get_height() != mip_h) {
				src->resize(mip_w, mip_h, Image::INTERPOLATE_BILINEAR);
			}

			PoolByteArray::Read r = src->get_data().read();
			glTexSubImage2D(GL_TEXTURE_2D, p_mip_level, 0, 0, mip_w, mip_h, GL_RGBA, use_float ? GL_FLOAT : GL_UNSIGNED_BYTE, r.ptr());
		}

		glBindTexture(GL_TEXTURE_2D, 0);
#endif
	});
}

void GL43Server::bind_texture(const Variant &p_texture, int p_unit, const String &p_uniform_name) {
	ERR_FAIL_COND(p_unit < 0);
	_enqueue_or_run([this, p_texture, p_unit, p_uniform_name]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		if (p_texture.get_type() == Variant::NIL) {
			glActiveTexture(GL_TEXTURE0 + p_unit);
			glBindTexture(GL_TEXTURE_2D, 0);
			return;
		}

		uint32_t tex_id = 0, target = GL_TEXTURE_2D, int_fmt = 0;
		int w = 0, h = 0, d = 1;
		if (!_resolve_texture(p_texture, tex_id, target, int_fmt, w, h, d)) {
			return;
		}

		glActiveTexture(GL_TEXTURE0 + p_unit);
		glBindTexture((GLenum)target, (GLuint)tex_id);

		if (!p_uniform_name.empty()) {
			GLint loc = (GLint)_get_bound_uniform_loc(p_uniform_name);
			if (loc >= 0) {
				glUniform1i(loc, p_unit);
			}
		}
#endif
	});
}

void GL43Server::unbind_texture(int p_unit) {
	ERR_FAIL_COND(p_unit < 0);
	_enqueue_or_run([p_unit]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		glActiveTexture(GL_TEXTURE0 + p_unit);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindTexture(GL_TEXTURE_3D, 0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
#endif
	});
}

void GL43Server::bind_image_texture(const Variant &p_texture, int p_unit, ImageAccess p_access, ImageFormat p_format, int p_mip_level) {
	ERR_FAIL_COND(p_unit < 0);
	_enqueue_or_run([this, p_texture, p_unit, p_access, p_format, p_mip_level]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		if (p_texture.get_type() == Variant::NIL) {
			SharedGL43Context::get_singleton()->bind_image_texture((uint32_t)p_unit, 0, p_mip_level, false, 0, to_gl_access(p_access), GL_RGBA8);
			return;
		}

		uint32_t tex_id = 0, target = GL_TEXTURE_2D, int_fmt = 0;
		int w = 0, h = 0, d = 1;
		if (!_resolve_texture(p_texture, tex_id, target, int_fmt, w, h, d)) {
			return;
		}

		GLenum img_fmt = GL_RGBA16F;
		if (p_format != FORMAT_AUTO) {
			GLenum dummy_f, dummy_t;
			to_gl_format(p_format, img_fmt, dummy_f, dummy_t);
		} else {
			if (int_fmt == 0) {
				GLint queried = 0;
				glActiveTexture(GL_TEXTURE0);
				glBindTexture((GLenum)target, (GLuint)tex_id);
				glGetTexLevelParameteriv((GLenum)target, p_mip_level, GL_TEXTURE_INTERNAL_FORMAT, &queried);
				glBindTexture((GLenum)target, 0);
				int_fmt = (uint32_t)queried;
			}
			img_fmt = int_fmt ? sanitize_image_format((GLenum)int_fmt) : GL_RGBA8;
			if (img_fmt == 0) {
				return;
			}
		}

		bool layered = (target == GL_TEXTURE_3D || target == GL_TEXTURE_2D_ARRAY);
		SharedGL43Context::get_singleton()->bind_image_texture((uint32_t)p_unit, tex_id, p_mip_level, layered, 0, to_gl_access(p_access), img_fmt);
#endif
	});
}

void GL43Server::bind_image_texture_layer(const Variant &p_texture, int p_unit, ImageAccess p_access, int p_layer, ImageFormat p_format) {
	ERR_FAIL_COND(p_unit < 0 || p_layer < 0);
	_enqueue_or_run([this, p_texture, p_unit, p_access, p_layer, p_format]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		if (p_texture.get_type() == Variant::NIL) {
			SharedGL43Context::get_singleton()->bind_image_texture((uint32_t)p_unit, 0, 0, false, 0, to_gl_access(p_access), GL_RGBA8);
			return;
		}

		uint32_t tex_id = 0, target = GL_TEXTURE_2D, int_fmt = 0;
		int w = 0, h = 0, d = 1;
		if (!_resolve_texture(p_texture, tex_id, target, int_fmt, w, h, d)) {
			return;
		}

		GLenum img_fmt = GL_RGBA16F;
		if (p_format != FORMAT_AUTO) {
			GLenum dummy_f, dummy_t;
			to_gl_format(p_format, img_fmt, dummy_f, dummy_t);
		} else {
			if (int_fmt == 0) {
				GLint queried = 0;
				glActiveTexture(GL_TEXTURE0);
				glBindTexture((GLenum)target, (GLuint)tex_id);
				glGetTexLevelParameteriv((GLenum)target, 0, GL_TEXTURE_INTERNAL_FORMAT, &queried);
				glBindTexture((GLenum)target, 0);
				int_fmt = (uint32_t)queried;
			}
			img_fmt = int_fmt ? sanitize_image_format((GLenum)int_fmt) : GL_RGBA8;
			if (img_fmt == 0) {
				return;
			}
		}

		SharedGL43Context::get_singleton()->bind_image_texture((uint32_t)p_unit, tex_id, 0, false, p_layer, to_gl_access(p_access), img_fmt);
#endif
	});
}

void GL43Server::unbind_image_texture(int p_unit) {
	bind_image_texture(Variant(), p_unit);
}

void GL43Server::generate_mipmaps(const Variant &p_texture) {
	_enqueue_or_run([this, p_texture]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		uint32_t tex_id = 0, target = GL_TEXTURE_2D, int_fmt = 0;
		int w = 0, h = 0, d = 1;
		if (!_resolve_texture(p_texture, tex_id, target, int_fmt, w, h, d)) {
			return;
		}
		glActiveTexture(GL_TEXTURE0);
		glBindTexture((GLenum)target, (GLuint)tex_id);
		glGenerateMipmap((GLenum)target);
		glBindTexture((GLenum)target, 0);
#endif
	});
}

void GL43Server::clear_texture(const Variant &p_texture, const Color &p_color) {
	_enqueue_or_run([this, p_texture, p_color]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		_load_extra_gl43_procs();

		uint32_t tex_id = 0, target = GL_TEXTURE_2D, int_fmt = 0;
		int w = 0, h = 0, d = 1;
		if (!_resolve_texture(p_texture, tex_id, target, int_fmt, w, h, d)) {
			return;
		}

		bool is_int = is_integer_internal_format((GLenum)int_fmt);
		bool is_signed = is_signed_integer_internal_format((GLenum)int_fmt);
		float rgba[4] = { p_color.r, p_color.g, p_color.b, p_color.a };
		int32_t irgba[4] = { (int32_t)p_color.r, (int32_t)p_color.g, (int32_t)p_color.b, (int32_t)p_color.a };
		uint32_t urgba[4] = { (uint32_t)MAX(0.0f, p_color.r), (uint32_t)MAX(0.0f, p_color.g), (uint32_t)MAX(0.0f, p_color.b), (uint32_t)MAX(0.0f, p_color.a) };

		if (fn_clear_tex_image) {
			if (is_signed) {
				((PFN_glClearTexImage_GL43)fn_clear_tex_image)((GLuint)tex_id, 0, GL_RGBA_INTEGER, GL_INT, irgba);
			} else if (is_int) {
				((PFN_glClearTexImage_GL43)fn_clear_tex_image)((GLuint)tex_id, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, urgba);
			} else {
				((PFN_glClearTexImage_GL43)fn_clear_tex_image)((GLuint)tex_id, 0, GL_RGBA, GL_FLOAT, rgba);
			}
			return;
		}

		if (helper_fbo == 0) {
			GLuint fbo = 0;
			glGenFramebuffers(1, &fbo);
			helper_fbo = (uint32_t)fbo;
		}

		GLint prev_fbo = 0;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
		glDisable(GL_SCISSOR_TEST);
		glDisable(GL_FRAMEBUFFER_SRGB);
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glClearColor(rgba[0], rgba[1], rgba[2], rgba[3]);
		glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)helper_fbo);

		auto do_clear = [&]() {
			if (is_signed) {
				glClearBufferiv(GL_COLOR, 0, irgba);
			} else if (is_int) {
				glClearBufferuiv(GL_COLOR, 0, urgba);
			} else {
				glClear(GL_COLOR_BUFFER_BIT);
			}
		};

		if (target == GL_TEXTURE_2D) {
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, (GLuint)tex_id, 0);
			GLenum buf = GL_COLOR_ATTACHMENT0;
			glDrawBuffers(1, &buf);
			do_clear();
		} else {
			GLenum buf = GL_COLOR_ATTACHMENT0;
			glDrawBuffers(1, &buf);
			for (int layer = 0; layer < d; layer++) {
				glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, (GLuint)tex_id, 0, layer);
				do_clear();
			}
		}

		glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
#endif
	});
}

void GL43Server::copy_texture(const Variant &p_src_texture, const Variant &p_dst_texture, int p_src_mip, int p_dst_mip) {
	ERR_FAIL_COND(p_src_mip < 0 || p_dst_mip < 0);
	_enqueue_or_run([this, p_src_texture, p_dst_texture, p_src_mip, p_dst_mip]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		_load_extra_gl43_procs();

		uint32_t src_id = 0, src_t = 0, src_fmt = 0;
		int sw = 0, sh = 0, sd = 1;
		uint32_t dst_id = 0, dst_t = 0, dst_fmt = 0;
		int dw = 0, dh = 0, dd = 1;

		if (!_resolve_texture(p_src_texture, src_id, src_t, src_fmt, sw, sh, sd)) {
			return;
		}
		if (!_resolve_texture(p_dst_texture, dst_id, dst_t, dst_fmt, dw, dh, dd)) {
			return;
		}

		int copy_w = MIN(MAX(1, sw >> p_src_mip), MAX(1, dw >> p_dst_mip));
		int copy_h = MIN(MAX(1, sh >> p_src_mip), MAX(1, dh >> p_dst_mip));
		int src_d_mip = (src_t == GL_TEXTURE_3D) ? MAX(1, sd >> p_src_mip) : sd;
		int dst_d_mip = (dst_t == GL_TEXTURE_3D) ? MAX(1, dd >> p_dst_mip) : dd;
		int copy_d = MIN(src_d_mip, dst_d_mip);

		if (fn_copy_image_sub_data) {
			((PFN_glCopyImageSubData_GL43)fn_copy_image_sub_data)(
					(GLuint)src_id, (GLenum)src_t, p_src_mip, 0, 0, 0,
					(GLuint)dst_id, (GLenum)dst_t, p_dst_mip, 0, 0, 0,
					copy_w, copy_h, copy_d);
		}
#endif
	});
}

Ref<Image> GL43Server::read_texture_2d(const Variant &p_texture, int p_mip_level) {
	Ref<Image> out;
#ifdef GL43_BACKEND_SUPPORTED
	ERR_FAIL_COND_V_MSG(!is_available(), out, "GL43Server is not available.");
	ERR_FAIL_COND_V_MSG(!_is_on_gl_thread(), out, "GL43Server.read_texture_2d must be called on the render thread.");
	ERR_FAIL_COND_V(p_mip_level < 0, out);

	_flush_pending_commands();

	uint32_t tex_id = 0, target = GL_TEXTURE_2D, int_fmt = 0;
	int w = 0, h = 0, d = 1;
	if (!_resolve_texture(p_texture, tex_id, target, int_fmt, w, h, d) || target != GL_TEXTURE_2D) {
		return out;
	}

	int mip_w = MAX(1, w >> p_mip_level);
	int mip_h = MAX(1, h >> p_mip_level);

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return out;
	}

	bool is_int = is_integer_internal_format((GLenum)int_fmt);
	bool use_rgba8 = !is_int && (int_fmt == GL_RGBA8 || int_fmt == GL_SRGB8_ALPHA8);
	Image::Format img_fmt = use_rgba8 ? Image::FORMAT_RGBA8 : Image::FORMAT_RGBAF;
	int byte_size = mip_w * mip_h * (use_rgba8 ? 4 : 16);

	PoolByteArray data;
	data.resize(byte_size);
	{
		PoolByteArray::Write wr = data.write();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, (GLuint)tex_id);
		if (is_int) {
			size_t count = (size_t)mip_w * (size_t)mip_h * 4;
			float *dst = (float *)wr.ptr();
			if (is_signed_integer_internal_format((GLenum)int_fmt)) {
				std::vector<int32_t> ints(count);
				glGetTexImage(GL_TEXTURE_2D, p_mip_level, GL_RGBA_INTEGER, GL_INT, ints.data());
				for (size_t i = 0; i < count; i++) {
					dst[i] = (float)ints[i];
				}
			} else {
				std::vector<uint32_t> uints(count);
				glGetTexImage(GL_TEXTURE_2D, p_mip_level, GL_RGBA_INTEGER, GL_UNSIGNED_INT, uints.data());
				for (size_t i = 0; i < count; i++) {
					dst[i] = (float)uints[i];
				}
			}
		} else {
			glGetTexImage(GL_TEXTURE_2D, p_mip_level, GL_RGBA, use_rgba8 ? GL_UNSIGNED_BYTE : GL_FLOAT, wr.ptr());
		}
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	out.instance();
	out->create(mip_w, mip_h, false, img_fmt, data);
#endif
	return out;
}

Vector3 GL43Server::get_texture_size(const Variant &p_texture) const {
	uint32_t tex_id = 0, target = 0, int_fmt = 0;
	int w = 0, h = 0, d = 1;
	if (_resolve_texture(p_texture, tex_id, target, int_fmt, w, h, d)) {
		return Vector3((float)w, (float)h, (float)d);
	}
	return Vector3();
}

// =========================================================
// COMPUTE DISPATCH & BARRIERS
// =========================================================

void GL43Server::dispatch_compute(int p_groups_x, int p_groups_y, int p_groups_z) {
	ERR_FAIL_COND(p_groups_x <= 0 || p_groups_y <= 0 || p_groups_z <= 0);
	_enqueue_or_run([p_groups_x, p_groups_y, p_groups_z]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		SharedGL43Context::get_singleton()->dispatch_compute((uint32_t)p_groups_x, (uint32_t)p_groups_y, (uint32_t)p_groups_z);
#endif
	});
}

void GL43Server::dispatch_compute_for_size(int p_width, int p_height, int p_depth, const Vector3 &p_local_size) {
	int lx = MAX(1, (int)p_local_size.x);
	int ly = MAX(1, (int)p_local_size.y);
	int lz = MAX(1, (int)p_local_size.z);
	ERR_FAIL_COND(p_width <= 0 || p_height <= 0 || p_depth <= 0);
	int gx = (p_width + lx - 1) / lx;
	int gy = (p_height + ly - 1) / ly;
	int gz = (p_depth + lz - 1) / lz;
	dispatch_compute(gx, gy, gz);
}

void GL43Server::dispatch_compute_indirect(RID p_indirect_ssbo, int p_byte_offset) {
	ERR_FAIL_COND(p_byte_offset < 0);
	_enqueue_or_run([this, p_indirect_ssbo, p_byte_offset]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		_load_extra_gl43_procs();
		if (!fn_dispatch_compute_indirect) {
			ERR_PRINT("GL43Server.dispatch_compute_indirect: glDispatchComputeIndirect unavailable.");
			return;
		}
		uint32_t buf = 0;
		{
			MutexLock lock(data_mutex);
			BufferEntry *entry = (p_indirect_ssbo.is_valid() && buffer_owner.owns(p_indirect_ssbo)) ? buffer_owner.getornull(p_indirect_ssbo) : nullptr;
			if (!entry || entry->buffer == 0) {
				return;
			}
			buf = entry->buffer;
		}
		glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, (GLuint)buf);
		((PFN_glDispatchComputeIndirect_GL43)fn_dispatch_compute_indirect)((GLintptr)p_byte_offset);
		glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
#endif
	});
}

void GL43Server::memory_barrier(int p_barriers) {
	_enqueue_or_run([p_barriers]() {
#ifdef GL43_BACKEND_SUPPORTED
		SharedGL43Context::ScopedScope gl43;
		if (!gl43.is_active()) {
			return;
		}
		SharedGL43Context::get_singleton()->memory_barrier((uint32_t)to_gl_barriers(p_barriers));
#endif
	});
}

// =========================================================
// RESOURCE CLEANUP
// =========================================================

void GL43Server::_free_rid_gl(RID p_rid) {
#ifdef GL43_BACKEND_SUPPORTED
	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	MutexLock lock(data_mutex);
	if (shader_owner.owns(p_rid)) {
		ShaderEntry *s = shader_owner.getornull(p_rid);
		if (s) {
			if (bound_shader == p_rid) {
				bound_shader = RID();
				glUseProgram(0);
			}
			if (s->program) {
				glDeleteProgram((GLuint)s->program);
			}
			shader_owner.free(p_rid);
			memdelete(s);
		}
		return;
	}

	if (buffer_owner.owns(p_rid)) {
		BufferEntry *b = buffer_owner.getornull(p_rid);
		if (b) {
			if (b->buffer) {
				GLuint buf = (GLuint)b->buffer;
				glDeleteBuffers(1, &buf);
			}
			buffer_owner.free(p_rid);
			memdelete(b);
		}
		return;
	}

	if (texture_owner.owns(p_rid)) {
		TextureEntry *t = texture_owner.getornull(p_rid);
		if (t) {
			if (t->tex_id) {
				GLuint tex = (GLuint)t->tex_id;
				glDeleteTextures(1, &tex);
			}
			texture_owner.free(p_rid);
			memdelete(t);
		}
		return;
	}
#endif
}

void GL43Server::free_rid(RID p_rid) {
	if (!p_rid.is_valid()) {
		return;
	}
	_enqueue_or_run([this, p_rid]() {
		_free_rid_gl(p_rid);
	});
}

bool GL43Server::is_rid_valid(RID p_rid) const {
	if (!p_rid.is_valid()) {
		return false;
	}
	MutexLock lock(data_mutex);
	return shader_owner.owns(p_rid) || buffer_owner.owns(p_rid) || texture_owner.owns(p_rid);
}

// =========================================================
// CLASSDB BINDINGS
// =========================================================

void GL43Server::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_available"), &GL43Server::is_available);
	ClassDB::bind_method(D_METHOD("get_last_error"), &GL43Server::get_last_error);
	ClassDB::bind_method(D_METHOD("get_limits"), &GL43Server::get_limits);
	ClassDB::bind_method(D_METHOD("begin"), &GL43Server::begin);
	ClassDB::bind_method(D_METHOD("end"), &GL43Server::end);
	ClassDB::bind_method(D_METHOD("flush"), &GL43Server::flush);
	ClassDB::bind_method(D_METHOD("finish"), &GL43Server::finish);

	ClassDB::bind_method(D_METHOD("create_compute_shader", "glsl_code"), &GL43Server::create_compute_shader);
	ClassDB::bind_method(D_METHOD("reload_compute_shader", "shader", "glsl_code"), &GL43Server::reload_compute_shader);
	ClassDB::bind_method(D_METHOD("is_shader_valid", "shader"), &GL43Server::is_shader_valid);
	ClassDB::bind_method(D_METHOD("get_shader_log", "shader"), &GL43Server::get_shader_log);
	ClassDB::bind_method(D_METHOD("bind_compute_shader", "shader"), &GL43Server::bind_compute_shader);
	ClassDB::bind_method(D_METHOD("unbind_compute_shader"), &GL43Server::unbind_compute_shader);

	ClassDB::bind_method(D_METHOD("set_uniform", "name", "value"), &GL43Server::set_uniform);
	ClassDB::bind_method(D_METHOD("set_uniform_color", "name", "color", "to_linear"), &GL43Server::set_uniform_color, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_uniform_projection", "name", "fov_degrees", "aspect", "z_near", "z_far"), &GL43Server::set_uniform_projection);
	ClassDB::bind_method(D_METHOD("set_uniform_inv_projection", "name", "fov_degrees", "aspect", "z_near", "z_far"), &GL43Server::set_uniform_inv_projection);
	ClassDB::bind_method(D_METHOD("set_uniform_ortho", "name", "size", "aspect", "z_near", "z_far"), &GL43Server::set_uniform_ortho);
	ClassDB::bind_method(D_METHOD("set_uniform_inv_ortho", "name", "size", "aspect", "z_near", "z_far"), &GL43Server::set_uniform_inv_ortho);

	ClassDB::bind_method(D_METHOD("create_ssbo", "size_bytes", "initial_data", "usage"), &GL43Server::create_ssbo, DEFVAL(0), DEFVAL(Variant()), DEFVAL(USAGE_DYNAMIC));
	ClassDB::bind_method(D_METHOD("upload_ssbo", "ssbo", "data", "usage"), &GL43Server::upload_ssbo, DEFVAL(USAGE_DYNAMIC));
	ClassDB::bind_method(D_METHOD("upload_ssbo_subdata", "ssbo", "byte_offset", "data"), &GL43Server::upload_ssbo_subdata);
	ClassDB::bind_method(D_METHOD("clear_ssbo", "ssbo", "clear_value"), &GL43Server::clear_ssbo, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("copy_ssbo", "src_ssbo", "dst_ssbo", "byte_size", "src_offset", "dst_offset"), &GL43Server::copy_ssbo, DEFVAL(-1), DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("bind_ssbo", "ssbo", "binding_index"), &GL43Server::bind_ssbo);
	ClassDB::bind_method(D_METHOD("unbind_ssbo", "binding_index"), &GL43Server::unbind_ssbo);
	ClassDB::bind_method(D_METHOD("get_ssbo_size", "ssbo"), &GL43Server::get_ssbo_size);
	ClassDB::bind_method(D_METHOD("read_ssbo", "ssbo", "byte_size", "byte_offset"), &GL43Server::read_ssbo, DEFVAL(-1), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("read_ssbo_floats", "ssbo", "float_count", "byte_offset"), &GL43Server::read_ssbo_floats, DEFVAL(-1), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("read_ssbo_ints", "ssbo", "int_count", "byte_offset"), &GL43Server::read_ssbo_ints, DEFVAL(-1), DEFVAL(0));

	ClassDB::bind_method(D_METHOD("create_ubo", "size_bytes", "initial_data", "usage"), &GL43Server::create_ubo, DEFVAL(0), DEFVAL(Variant()), DEFVAL(USAGE_DYNAMIC));
	ClassDB::bind_method(D_METHOD("upload_ubo", "ubo", "data", "usage"), &GL43Server::upload_ubo, DEFVAL(USAGE_DYNAMIC));
	ClassDB::bind_method(D_METHOD("bind_ubo", "ubo", "binding_index"), &GL43Server::bind_ubo);
	ClassDB::bind_method(D_METHOD("unbind_ubo", "binding_index"), &GL43Server::unbind_ubo);
	ClassDB::bind_method(D_METHOD("pack_vec3_as_vec4", "vectors", "w"), &GL43Server::pack_vec3_as_vec4, DEFVAL(1.0f));

	ClassDB::bind_method(D_METHOD("create_texture_2d", "width", "height", "format", "filter", "wrap"), &GL43Server::create_texture_2d, DEFVAL(FORMAT_RGBA16F), DEFVAL(FILTER_LINEAR), DEFVAL(WRAP_CLAMP));
	ClassDB::bind_method(D_METHOD("create_texture_3d", "width", "height", "depth", "format", "filter"), &GL43Server::create_texture_3d, DEFVAL(FORMAT_RGBA16F), DEFVAL(FILTER_LINEAR));
	ClassDB::bind_method(D_METHOD("create_texture_2d_array", "width", "height", "layers", "format", "filter"), &GL43Server::create_texture_2d_array, DEFVAL(FORMAT_RGBA16F), DEFVAL(FILTER_LINEAR));
	ClassDB::bind_method(D_METHOD("set_texture_sampling", "texture", "filter", "wrap"), &GL43Server::set_texture_sampling, DEFVAL(FILTER_LINEAR), DEFVAL(WRAP_CLAMP));
	ClassDB::bind_method(D_METHOD("resize_texture_2d", "texture", "width", "height"), &GL43Server::resize_texture_2d);
	ClassDB::bind_method(D_METHOD("upload_texture_2d", "texture", "image", "mip_level"), &GL43Server::upload_texture_2d, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("bind_texture", "texture", "unit", "uniform_name"), &GL43Server::bind_texture, DEFVAL(0), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("unbind_texture", "unit"), &GL43Server::unbind_texture, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("bind_image_texture", "texture", "unit", "access", "format", "mip_level"), &GL43Server::bind_image_texture, DEFVAL(0), DEFVAL(ACCESS_READ_WRITE), DEFVAL(FORMAT_AUTO), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("bind_image_texture_layer", "texture", "unit", "access", "layer", "format"), &GL43Server::bind_image_texture_layer, DEFVAL(0), DEFVAL(ACCESS_READ_WRITE), DEFVAL(0), DEFVAL(FORMAT_AUTO));
	ClassDB::bind_method(D_METHOD("unbind_image_texture", "unit"), &GL43Server::unbind_image_texture, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("generate_mipmaps", "texture"), &GL43Server::generate_mipmaps);
	ClassDB::bind_method(D_METHOD("clear_texture", "texture", "color"), &GL43Server::clear_texture, DEFVAL(Color(0, 0, 0, 0)));
	ClassDB::bind_method(D_METHOD("copy_texture", "src_texture", "dst_texture", "src_mip", "dst_mip"), &GL43Server::copy_texture, DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("read_texture_2d", "texture", "mip_level"), &GL43Server::read_texture_2d, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_texture_size", "texture"), &GL43Server::get_texture_size);

	ClassDB::bind_method(D_METHOD("dispatch_compute", "groups_x", "groups_y", "groups_z"), &GL43Server::dispatch_compute, DEFVAL(1), DEFVAL(1));
	ClassDB::bind_method(D_METHOD("dispatch_compute_for_size", "width", "height", "depth", "local_size"), &GL43Server::dispatch_compute_for_size, DEFVAL(1), DEFVAL(Vector3(8, 8, 1)));
	ClassDB::bind_method(D_METHOD("dispatch_compute_indirect", "indirect_ssbo", "byte_offset"), &GL43Server::dispatch_compute_indirect, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("memory_barrier", "barriers"), &GL43Server::memory_barrier, DEFVAL(BARRIER_ALL));

	ClassDB::bind_method(D_METHOD("free_rid", "rid"), &GL43Server::free_rid);
	ClassDB::bind_method(D_METHOD("is_rid_valid", "rid"), &GL43Server::is_rid_valid);
	ClassDB::bind_method(D_METHOD("_on_frame_pre_draw"), &GL43Server::_on_frame_pre_draw);

	BIND_ENUM_CONSTANT(USAGE_DYNAMIC);
	BIND_ENUM_CONSTANT(USAGE_STATIC);
	BIND_ENUM_CONSTANT(USAGE_STREAM);

	BIND_ENUM_CONSTANT(ACCESS_READ_ONLY);
	BIND_ENUM_CONSTANT(ACCESS_WRITE_ONLY);
	BIND_ENUM_CONSTANT(ACCESS_READ_WRITE);

	BIND_ENUM_CONSTANT(FORMAT_AUTO);
	BIND_ENUM_CONSTANT(FORMAT_RGBA8);
	BIND_ENUM_CONSTANT(FORMAT_RGBA16F);
	BIND_ENUM_CONSTANT(FORMAT_RGBA32F);
	BIND_ENUM_CONSTANT(FORMAT_RG8);
	BIND_ENUM_CONSTANT(FORMAT_RG16F);
	BIND_ENUM_CONSTANT(FORMAT_RG32F);
	BIND_ENUM_CONSTANT(FORMAT_R8);
	BIND_ENUM_CONSTANT(FORMAT_R16F);
	BIND_ENUM_CONSTANT(FORMAT_R32F);
	BIND_ENUM_CONSTANT(FORMAT_R32I);
	BIND_ENUM_CONSTANT(FORMAT_R32UI);
	BIND_ENUM_CONSTANT(FORMAT_RGBA16UI);
	BIND_ENUM_CONSTANT(FORMAT_RGBA32UI);

	BIND_ENUM_CONSTANT(FILTER_NEAREST);
	BIND_ENUM_CONSTANT(FILTER_LINEAR);
	BIND_ENUM_CONSTANT(FILTER_LINEAR_MIPMAP);

	BIND_ENUM_CONSTANT(WRAP_CLAMP);
	BIND_ENUM_CONSTANT(WRAP_REPEAT);
	BIND_ENUM_CONSTANT(WRAP_MIRROR);

	BIND_ENUM_CONSTANT(BARRIER_IMAGE);
	BIND_ENUM_CONSTANT(BARRIER_SSBO);
	BIND_ENUM_CONSTANT(BARRIER_TEXTURE);
	BIND_ENUM_CONSTANT(BARRIER_BUFFER_UPDATE);
	BIND_ENUM_CONSTANT(BARRIER_UBO);
	BIND_ENUM_CONSTANT(BARRIER_ALL);
}
