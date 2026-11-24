#ifndef AO_BAKE_SIMD_H
#define AO_BAKE_SIMD_H

#include <cstdint>

struct alignas(16) AOBVH4Node {
	float min_x[4];
	float min_y[4];
	float min_z[4];
	float max_x[4];
	float max_y[4];
	float max_z[4];
	int32_t child[4];
	int32_t leaf_blocks[4];
	uint32_t valid_mask;
	uint32_t _pad[3];
};

struct alignas(32) AOTri8 {
	float p0x[8];
	float p0y[8];
	float p0z[8];
	float e1x[8];
	float e1y[8];
	float e1z[8];
	float e2x[8];
	float e2y[8];
	float e2z[8];
	int32_t tri_idx[8];
};

float ao_trace_bvh4_avx(
		const AOBVH4Node *p_nodes,
		const AOTri8 *p_blocks,
		float p_rox, float p_roy, float p_roz,
		float p_rdx, float p_rdy, float p_rdz,
		float p_tmin, float p_tmax,
		int32_t *r_hit_tri = nullptr);

bool ao_occluded_bvh4_avx(
		const AOBVH4Node *p_nodes,
		const AOTri8 *p_blocks,
		float p_rox, float p_roy, float p_roz,
		float p_rdx, float p_rdy, float p_rdz,
		float p_tmin, float p_tmax);

#endif // AO_BAKE_SIMD_H
