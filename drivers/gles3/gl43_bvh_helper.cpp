#include "gl43_bvh_helper.h"
#include "gl43_effect_helpers.h"

#include "drivers/gles3/rasterizer_storage_gles3.h"

#include <algorithm>
#include <cstring>

#ifndef GL_RGBA16F
#define GL_RGBA16F 0x881A
#endif
#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif

namespace {

static const char *GL43_TEX_BLIT_VERT_GLSL = R"GLSL(
#version 430 core
out vec2 vUv;
void main() {
	vec2 pos = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
	vUv = pos * 0.5 + 0.5;
	gl_Position = vec4(pos, 0.0, 1.0);
}
)GLSL";

static const char *GL43_TEX_BLIT_FRAG_GLSL = R"GLSL(
#version 430 core
in vec2 vUv;
out vec4 FragColor;
uniform sampler2D uSrcTex;
uniform int uToLinear;
void main() {
	vec4 c = texture(uSrcTex, vUv);
	if (uToLinear == 1) {
		c.rgb = pow(max(c.rgb, vec3(0.0)), vec3(2.2));
	}
	FragColor = c;
}
)GLSL";

} // namespace

namespace GL43BVHHelper {

void build_bvh(const std::vector<AABB> &p_aabbs, std::vector<int32_t> &r_indices, std::vector<GL43BVHNode> &r_nodes) {
	r_nodes.clear();
	r_indices.clear();
	int n = (int)p_aabbs.size();
	if (n == 0) {
		return;
	}

	r_indices.resize(n);
	for (int i = 0; i < n; i++) {
		r_indices[i] = i;
	}

	struct BuildHelper {
		static int build(const std::vector<AABB> &boxes, std::vector<int32_t> &idx,
				std::vector<GL43BVHNode> &nodes, int first, int count, int depth) {
			int node_idx = (int)nodes.size();
			nodes.push_back({});

			AABB bounds = boxes[idx[first]];
			AABB cent_bounds(bounds.position + bounds.size * 0.5f, Vector3());
			for (int i = 1; i < count; i++) {
				const AABB &b = boxes[idx[first + i]];
				bounds.merge_with(b);
				cent_bounds.expand_to(b.position + b.size * 0.5f);
			}

			GL43BVHNode &node = nodes[node_idx];
			node.min_x = (float)bounds.position.x;
			node.min_y = (float)bounds.position.y;
			node.min_z = (float)bounds.position.z;
			Vector3 bmax = bounds.position + bounds.size;
			node.max_x = (float)bmax.x;
			node.max_y = (float)bmax.y;
			node.max_z = (float)bmax.z;

			int axis = cent_bounds.get_longest_axis_index();
			float axis_len = (float)cent_bounds.size[axis];

			if (count <= 4 || depth >= 18 || axis_len < 1e-5f) {
				nodes[node_idx].left_or_first = first;
				nodes[node_idx].count = count;
				return node_idx;
			}

			float split = (float)(cent_bounds.position[axis] + cent_bounds.size[axis] * 0.5f);
			auto it = std::partition(idx.begin() + first, idx.begin() + first + count, [&](int32_t id) {
				float c = (float)(boxes[id].position[axis] + boxes[id].size[axis] * 0.5f);
				return c < split;
			});
			int mid = (int)(it - idx.begin());
			if (mid == first || mid == first + count) {
				mid = first + count / 2;
				std::nth_element(idx.begin() + first, idx.begin() + mid, idx.begin() + first + count, [&](int32_t a, int32_t b) {
					return (boxes[a].position[axis] + boxes[a].size[axis] * 0.5f) < (boxes[b].position[axis] + boxes[b].size[axis] * 0.5f);
				});
			}

			build(boxes, idx, nodes, first, mid - first, depth + 1);
			int right_idx = build(boxes, idx, nodes, mid, first + count - mid, depth + 1);
			nodes[node_idx].left_or_first = right_idx;
			nodes[node_idx].count = 0;
			return node_idx;
		}
	};

	BuildHelper::build(p_aabbs, r_indices, r_nodes, 0, n, 0);
}

void refit_bvh_bottom_up(std::vector<GL43BVHNode> &r_nodes, const void *p_tris, size_t p_tri_stride_bytes) {
	const uint8_t *base = (const uint8_t *)p_tris;
	for (int i = (int)r_nodes.size() - 1; i >= 0; i--) {
		GL43BVHNode &n = r_nodes[i];
		if (n.count > 0) {
			float mnx = 1e30f, mny = 1e30f, mnz = 1e30f;
			float mxx = -1e30f, mxy = -1e30f, mxz = -1e30f;
			for (int j = 0; j < n.count; j++) {
				const uint8_t *tri_ptr = base + (size_t)(n.left_or_first + j) * p_tri_stride_bytes;
				for (int c = 0; c < 3; c++) {
					const float *p = (const float *)(tri_ptr + (size_t)c * 16);
					if (p[0] < mnx) mnx = p[0];
					if (p[1] < mny) mny = p[1];
					if (p[2] < mnz) mnz = p[2];
					if (p[0] > mxx) mxx = p[0];
					if (p[1] > mxy) mxy = p[1];
					if (p[2] > mxz) mxz = p[2];
				}
			}
			n.min_x = mnx - 0.002f;
			n.min_y = mny - 0.002f;
			n.min_z = mnz - 0.002f;
			n.max_x = mxx + 0.002f;
			n.max_y = mxy + 0.002f;
			n.max_z = mxz + 0.002f;
		} else {
			const GL43BVHNode &L = r_nodes[i + 1];
			const GL43BVHNode &R = r_nodes[n.left_or_first];
			n.min_x = MIN(L.min_x, R.min_x);
			n.min_y = MIN(L.min_y, R.min_y);
			n.min_z = MIN(L.min_z, R.min_z);
			n.max_x = MAX(L.max_x, R.max_x);
			n.max_y = MAX(L.max_y, R.max_y);
			n.max_z = MAX(L.max_z, R.max_z);
		}
	}
}

} // namespace GL43BVHHelper

bool GL43TextureArrayPool::init_gl(const char *p_tag) {
	blit_program = GL43EffectHelpers::link_vf_program(GL43_TEX_BLIT_VERT_GLSL, GL43_TEX_BLIT_FRAG_GLSL, p_tag, "Blit");
	if (!blit_program) {
		return false;
	}

	GLuint vao = 0, fbo = 0, arr = 0;
	glGenVertexArrays(1, &vao);
	glGenFramebuffers(1, &fbo);
	glGenTextures(1, &arr);
	empty_vao = (uint32_t)vao;
	blit_fbo = (uint32_t)fbo;
	tex_array = (uint32_t)arr;

	uint8_t white[4] = { 255, 255, 255, 255 };
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D_ARRAY, (GLuint)tex_array);
	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 1, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

	return true;
}

void GL43TextureArrayPool::cleanup_gl() {
	if (tex_array) {
		GLuint t = (GLuint)tex_array;
		glDeleteTextures(1, &t);
		tex_array = 0;
	}
	if (blit_fbo) {
		GLuint f = (GLuint)blit_fbo;
		glDeleteFramebuffers(1, &f);
		blit_fbo = 0;
	}
	if (blit_program) {
		glDeleteProgram((GLuint)blit_program);
		blit_program = 0;
	}
	if (empty_vao) {
		GLuint v = (GLuint)empty_vao;
		glDeleteVertexArrays(1, &v);
		empty_vao = 0;
	}
	tex_array_allocated = false;
	allocated_res = 0;
	allocated_layers = 0;
	layer_map.clear();
	pinned_textures.clear();
	pending_blits.clear();
}

int32_t GL43TextureArrayPool::register_layer(const Ref<Texture> &p_tex, bool p_is_srgb, int p_max_layers) {
	if (p_tex.is_null() || !p_tex->get_rid().is_valid()) {
		return -1;
	}
	if (reset_needed) {
		layer_map.clear();
		pinned_textures.clear();
		pending_blits.clear();
	}
	uint64_t key = ((uint64_t)p_tex->get_rid().get_id() << 1) | (p_is_srgb ? 1ULL : 0ULL);
	auto it = layer_map.find(key);
	if (it != layer_map.end()) {
		return it->second;
	}
	if ((int)layer_map.size() >= p_max_layers) {
		return -1;
	}
	int32_t layer = (int32_t)layer_map.size();
	layer_map[key] = layer;
	pinned_textures.push_back(p_tex);
	pending_blits.push_back({ p_tex->get_rid(), layer, p_is_srgb });
	return layer;
}

void GL43TextureArrayPool::flush_blits(RasterizerStorageGLES3 *p_rs, int p_resolution, int p_max_layers, bool p_hdr_format) {
	if (reset_needed || (tex_array_allocated && (allocated_res != p_resolution || allocated_layers != p_max_layers))) {
		reset_needed = false;
		tex_array_allocated = false;
		layer_map.clear();
		pinned_textures.clear();
		pending_blits.clear();
		return;
	}

	if (pending_blits.empty()) {
		return;
	}

	if (!tex_array_allocated) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, (GLuint)tex_array);
		if (p_hdr_format) {
			glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA16F, p_resolution, p_resolution, p_max_layers, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
		} else {
			glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, p_resolution, p_resolution, p_max_layers, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		}
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
		tex_array_allocated = true;
		allocated_res = p_resolution;
		allocated_layers = p_max_layers;
	}

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glDisable(GL_SCISSOR_TEST);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

	glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)blit_fbo);
	glViewport(0, 0, p_resolution, p_resolution);
	glUseProgram((GLuint)blit_program);
	glBindVertexArray((GLuint)empty_vao);

	GLint loc_src = glGetUniformLocation((GLuint)blit_program, "uSrcTex");
	GLint loc_lin = glGetUniformLocation((GLuint)blit_program, "uToLinear");

	std::vector<PendingBlit> remaining;
	bool blitted_any = false;

	for (const PendingBlit &b : pending_blits) {
		if (b.layer < 0 || b.layer >= p_max_layers) {
			continue;
		}
		RasterizerStorageGLES3::Texture *gt = p_rs->texture_owner.getornull(b.tex_rid);
		if (gt) {
			gt = gt->get_ptr();
		}
		if (!gt || gt->tex_id == 0) {
			remaining.push_back(b);
			continue;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, gt->tex_id);

		int convert_srgb = 0;
		if (b.is_srgb) {
			GLint int_fmt = 0;
			glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &int_fmt);
			convert_srgb = !GL43EffectHelpers::is_srgb_gl_format((int32_t)int_fmt) ? 1 : 0;
		}

		glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, (GLuint)tex_array, 0, b.layer);
		GLenum draw_buf = GL_COLOR_ATTACHMENT0;
		glDrawBuffers(1, &draw_buf);

		glUniform1i(loc_src, 0);
		glUniform1i(loc_lin, convert_srgb);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		blitted_any = true;
	}

	pending_blits = std::move(remaining);

	if (blitted_any) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, (GLuint)tex_array);
		glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
		glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
