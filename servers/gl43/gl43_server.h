#ifndef GL43_SERVER_H
#define GL43_SERVER_H

#include "core/image.h"
#include "core/math/transform.h"
#include "core/math/transform_2d.h"
#include "core/object.h"
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "core/rid.h"
#include "core/variant.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

class GL43Server : public Object {
	GDCLASS(GL43Server, Object);

public:
	enum BufferUsage {
		USAGE_DYNAMIC = 0,
		USAGE_STATIC = 1,
		USAGE_STREAM = 2,
	};

	enum ImageAccess {
		ACCESS_READ_ONLY = 0,
		ACCESS_WRITE_ONLY = 1,
		ACCESS_READ_WRITE = 2,
	};

	enum ImageFormat {
		FORMAT_AUTO = 0,
		FORMAT_RGBA8 = 1,
		FORMAT_RGBA16F = 2,
		FORMAT_RGBA32F = 3,
		FORMAT_RG8 = 4,
		FORMAT_RG16F = 5,
		FORMAT_RG32F = 6,
		FORMAT_R8 = 7,
		FORMAT_R16F = 8,
		FORMAT_R32F = 9,
		FORMAT_R32I = 10,
		FORMAT_R32UI = 11,
		FORMAT_RGBA16UI = 12,
		FORMAT_RGBA32UI = 13,
	};

	enum FilterMode {
		FILTER_NEAREST = 0,
		FILTER_LINEAR = 1,
		FILTER_LINEAR_MIPMAP = 2,
	};

	enum WrapMode {
		WRAP_CLAMP = 0,
		WRAP_REPEAT = 1,
		WRAP_MIRROR = 2,
	};

	enum BarrierFlags {
		BARRIER_IMAGE = 1 << 0,
		BARRIER_SSBO = 1 << 1,
		BARRIER_TEXTURE = 1 << 2,
		BARRIER_BUFFER_UPDATE = 1 << 3,
		BARRIER_UBO = 1 << 4,
		BARRIER_ALL = 0x7FFFFFFF,
	};

private:
	static GL43Server *singleton;

	struct ShaderEntry : public RID_Data {
		uint32_t program = 0;
		bool valid = false;
		String log;
		std::unordered_map<std::string, int32_t> uniform_cache;
	};

	struct BufferEntry : public RID_Data {
		uint32_t buffer = 0;
		uint32_t target = 0; // GL_SHADER_STORAGE_BUFFER or GL_UNIFORM_BUFFER
		int32_t size_bytes = 0;
		uint32_t usage = 0;
	};

	struct TextureEntry : public RID_Data {
		uint32_t tex_id = 0;
		uint32_t target = 0; // GL_TEXTURE_2D, GL_TEXTURE_3D, GL_TEXTURE_2D_ARRAY
		uint32_t internal_format = 0;
		uint32_t format = 0;
		uint32_t type = 0;
		int32_t width = 0;
		int32_t height = 0;
		int32_t depth = 1;
		bool mipmaps = false;
	};

	mutable Mutex data_mutex;
	mutable Mutex queue_mutex;

	mutable RID_Owner<ShaderEntry> shader_owner;
	mutable RID_Owner<BufferEntry> buffer_owner;
	mutable RID_Owner<TextureEntry> texture_owner;

	std::vector<std::function<void()>> pending_commands;
	bool vs_connected = false;
	bool gl_finished = false;
	bool gl_thread_known = false;
	Thread::ID gl_thread_id = 0;
	int begin_depth = 0;

	RID bound_shader;
	String last_error;
	uint32_t helper_fbo = 0;
	mutable Dictionary cached_limits;
	mutable bool limits_cached = false;

	bool procs_loaded = false;
	void *fn_dispatch_compute_indirect = nullptr;
	void *fn_clear_buffer_data = nullptr;
	void *fn_clear_tex_image = nullptr;
	void *fn_copy_image_sub_data = nullptr;

	bool _is_on_gl_thread() const;
	void _ensure_vs_connected();
	void _enqueue_or_run(std::function<void()> p_fn);
	void _flush_pending_commands();
	void _on_frame_pre_draw();
	void _load_extra_gl43_procs();

	int32_t _get_bound_uniform_loc(const String &p_name);
	bool _resolve_texture(const Variant &p_tex_or_rid, uint32_t &r_tex_id, uint32_t &r_target, uint32_t &r_internal_fmt, int &r_w, int &r_h, int &r_d) const;
	static bool _variant_to_bytes(const Variant &p_data, std::vector<uint8_t> &r_bytes);

	void _compile_shader_gl(RID p_rid, const String &p_code);
	void _create_buffer_gl(RID p_rid, uint32_t p_target, int p_size_bytes, std::vector<uint8_t> p_bytes, BufferUsage p_usage);
	void _upload_buffer_gl(RID p_rid, std::vector<uint8_t> p_bytes, BufferUsage p_usage);
	void _upload_buffer_subdata_gl(RID p_rid, int p_offset, std::vector<uint8_t> p_bytes);
	void _create_texture_gl(RID p_rid, uint32_t p_target, int p_w, int p_h, int p_d, ImageFormat p_fmt, FilterMode p_filter, WrapMode p_wrap, bool p_mipmaps);
	void _free_rid_gl(RID p_rid);

protected:
	static void _bind_methods();

public:
	static GL43Server *get_singleton() { return singleton; }

	GL43Server();
	~GL43Server();

	void finish_gl();

	// Status & Batching
	bool is_available() const;
	String get_last_error() const;
	Dictionary get_limits() const;
	bool begin();
	void end();
	void flush();
	void finish();

	// Compute Shaders
	RID create_compute_shader(const String &p_glsl_code);
	bool reload_compute_shader(RID p_shader, const String &p_glsl_code);
	bool is_shader_valid(RID p_shader) const;
	String get_shader_log(RID p_shader) const;
	void bind_compute_shader(RID p_shader);
	void unbind_compute_shader();

	// Uniforms (for currently bound compute shader)
	void set_uniform(const String &p_name, const Variant &p_value);
	void set_uniform_color(const String &p_name, const Color &p_color, bool p_to_linear = false);
	void set_uniform_projection(const String &p_name, float p_fov_degrees, float p_aspect, float p_z_near, float p_z_far);
	void set_uniform_inv_projection(const String &p_name, float p_fov_degrees, float p_aspect, float p_z_near, float p_z_far);
	void set_uniform_ortho(const String &p_name, float p_size, float p_aspect, float p_z_near, float p_z_far);
	void set_uniform_inv_ortho(const String &p_name, float p_size, float p_aspect, float p_z_near, float p_z_far);

	// SSBOs & UBOs
	RID create_ssbo(int p_size_bytes = 0, const Variant &p_initial_data = Variant(), BufferUsage p_usage = USAGE_DYNAMIC);
	void upload_ssbo(RID p_ssbo, const Variant &p_data, BufferUsage p_usage = USAGE_DYNAMIC);
	void upload_ssbo_subdata(RID p_ssbo, int p_byte_offset, const Variant &p_data);
	void clear_ssbo(RID p_ssbo, uint32_t p_clear_value = 0);
	void copy_ssbo(RID p_src_ssbo, RID p_dst_ssbo, int p_byte_size = -1, int p_src_offset = 0, int p_dst_offset = 0);
	void bind_ssbo(RID p_ssbo, int p_binding_index);
	void unbind_ssbo(int p_binding_index);
	int get_ssbo_size(RID p_ssbo) const;
	PoolByteArray read_ssbo(RID p_ssbo, int p_byte_size = -1, int p_byte_offset = 0);
	PoolRealArray read_ssbo_floats(RID p_ssbo, int p_float_count = -1, int p_byte_offset = 0);
	PoolIntArray read_ssbo_ints(RID p_ssbo, int p_int_count = -1, int p_byte_offset = 0);

	RID create_ubo(int p_size_bytes = 0, const Variant &p_initial_data = Variant(), BufferUsage p_usage = USAGE_DYNAMIC);
	void upload_ubo(RID p_ubo, const Variant &p_data, BufferUsage p_usage = USAGE_DYNAMIC);
	void bind_ubo(RID p_ubo, int p_binding_index);
	void unbind_ubo(int p_binding_index);

	// Helper to align Vector3 arrays to std430 vec4 stride (16 bytes per element)
	PoolByteArray pack_vec3_as_vec4(const PoolVector3Array &p_vectors, float p_w = 1.0f) const;

	// Textures & Image Load/Store
	RID create_texture_2d(int p_width, int p_height, ImageFormat p_format = FORMAT_RGBA16F, FilterMode p_filter = FILTER_LINEAR, WrapMode p_wrap = WRAP_CLAMP);
	RID create_texture_3d(int p_width, int p_height, int p_depth, ImageFormat p_format = FORMAT_RGBA16F, FilterMode p_filter = FILTER_LINEAR);
	RID create_texture_2d_array(int p_width, int p_height, int p_layers, ImageFormat p_format = FORMAT_RGBA16F, FilterMode p_filter = FILTER_LINEAR);
	void set_texture_sampling(const Variant &p_texture, FilterMode p_filter = FILTER_LINEAR, WrapMode p_wrap = WRAP_CLAMP);
	void resize_texture_2d(RID p_texture, int p_width, int p_height);
	void upload_texture_2d(const Variant &p_texture, const Ref<Image> &p_image, int p_mip_level = 0);
	void bind_texture(const Variant &p_texture, int p_unit = 0, const String &p_uniform_name = String());
	void unbind_texture(int p_unit = 0);
	void bind_image_texture(const Variant &p_texture, int p_unit = 0, ImageAccess p_access = ACCESS_READ_WRITE, ImageFormat p_format = FORMAT_AUTO, int p_mip_level = 0);
	void bind_image_texture_layer(const Variant &p_texture, int p_unit = 0, ImageAccess p_access = ACCESS_READ_WRITE, int p_layer = 0, ImageFormat p_format = FORMAT_AUTO);
	void unbind_image_texture(int p_unit = 0);
	void generate_mipmaps(const Variant &p_texture);
	void clear_texture(const Variant &p_texture, const Color &p_color = Color(0, 0, 0, 0));
	void copy_texture(const Variant &p_src_texture, const Variant &p_dst_texture, int p_src_mip = 0, int p_dst_mip = 0);
	Ref<Image> read_texture_2d(const Variant &p_texture, int p_mip_level = 0);
	Vector3 get_texture_size(const Variant &p_texture) const;

	// Compute Dispatch & Barriers
	void dispatch_compute(int p_groups_x, int p_groups_y = 1, int p_groups_z = 1);
	void dispatch_compute_for_size(int p_width, int p_height, int p_depth = 1, const Vector3 &p_local_size = Vector3(8, 8, 1));
	void dispatch_compute_indirect(RID p_indirect_ssbo, int p_byte_offset = 0);
	void memory_barrier(int p_barriers = BARRIER_ALL);

	// Resource Cleanup
	void free_rid(RID p_rid);
	bool is_rid_valid(RID p_rid) const;
};

VARIANT_ENUM_CAST(GL43Server::BufferUsage);
VARIANT_ENUM_CAST(GL43Server::ImageAccess);
VARIANT_ENUM_CAST(GL43Server::ImageFormat);
VARIANT_ENUM_CAST(GL43Server::FilterMode);
VARIANT_ENUM_CAST(GL43Server::WrapMode);
VARIANT_ENUM_CAST(GL43Server::BarrierFlags);

#endif // GL43_SERVER_H
