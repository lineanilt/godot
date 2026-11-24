#ifndef GL43_BVH_HELPER_H
#define GL43_BVH_HELPER_H

#include "core/math/aabb.h"
#include "scene/resources/texture.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

class RasterizerStorageGLES3;

struct GL43BVHNode {
	float min_x, min_y, min_z;
	int32_t left_or_first;
	float max_x, max_y, max_z;
	int32_t count;
};

namespace GL43BVHHelper {

void build_bvh(const std::vector<AABB> &p_aabbs, std::vector<int32_t> &r_indices, std::vector<GL43BVHNode> &r_nodes);
void refit_bvh_bottom_up(std::vector<GL43BVHNode> &r_nodes, const void *p_tris, size_t p_tri_stride_bytes);

} // namespace GL43BVHHelper

class GL43TextureArrayPool {
public:
	struct PendingBlit {
		RID tex_rid;
		int32_t layer = -1;
		bool is_srgb = false;
	};

private:
	uint32_t tex_array = 0;
	bool tex_array_allocated = false;
	int allocated_res = 0;
	int allocated_layers = 0;
	bool reset_needed = false;

	uint32_t blit_fbo = 0;
	uint32_t blit_program = 0;
	uint32_t empty_vao = 0;

	std::unordered_map<uint64_t, int32_t> layer_map;
	std::vector<Ref<Texture>> pinned_textures;
	std::vector<PendingBlit> pending_blits;

public:
	bool init_gl(const char *p_tag);
	void cleanup_gl();

	void mark_reset_needed() { reset_needed = true; }
	int32_t register_layer(const Ref<Texture> &p_tex, bool p_is_srgb, int p_max_layers);
	void flush_blits(RasterizerStorageGLES3 *p_rs, int p_resolution, int p_max_layers, bool p_hdr_format);

	uint32_t get_tex_array() const { return tex_array; }
};

#endif // GL43_BVH_HELPER_H
