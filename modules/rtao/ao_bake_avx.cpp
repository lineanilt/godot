#include "ao_bake_simd.h"

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__AVX__)

#include <immintrin.h>
#include <cmath>

float ao_trace_bvh4_avx(
		const AOBVH4Node *p_nodes,
		const AOTri8 *p_blocks,
		float p_rox, float p_roy, float p_roz,
		float p_rdx, float p_rdy, float p_rdz,
		float p_tmin, float p_tmax,
		int32_t *r_hit_tri) {

	float best_t = p_tmax;
	int32_t best_tri = -1;

	const __m128 rox4 = _mm_set1_ps(p_rox);
	const __m128 roy4 = _mm_set1_ps(p_roy);
	const __m128 roz4 = _mm_set1_ps(p_roz);

	const __m128 invx4 = _mm_set1_ps(1.0f / (std::abs(p_rdx) < 1e-8f ? (p_rdx < 0.0f ? -1e-8f : 1e-8f) : p_rdx));
	const __m128 invy4 = _mm_set1_ps(1.0f / (std::abs(p_rdy) < 1e-8f ? (p_rdy < 0.0f ? -1e-8f : 1e-8f) : p_rdy));
	const __m128 invz4 = _mm_set1_ps(1.0f / (std::abs(p_rdz) < 1e-8f ? (p_rdz < 0.0f ? -1e-8f : 1e-8f) : p_rdz));
	const __m128 tmin4_clamp = _mm_set1_ps(p_tmin);

	const __m256 rox8 = _mm256_set1_ps(p_rox);
	const __m256 roy8 = _mm256_set1_ps(p_roy);
	const __m256 roz8 = _mm256_set1_ps(p_roz);
	const __m256 rdx8 = _mm256_set1_ps(p_rdx);
	const __m256 rdy8 = _mm256_set1_ps(p_rdy);
	const __m256 rdz8 = _mm256_set1_ps(p_rdz);

	const __m256 zeros8 = _mm256_setzero_ps();
	const __m256 ones8 = _mm256_set1_ps(1.0f);
	const __m256 eps8 = _mm256_set1_ps(1e-7f);
	const __m256 neg_zero8 = _mm256_set1_ps(-0.0f);
	const __m256 tmin8 = _mm256_set1_ps(p_tmin);
	__m256 best8 = _mm256_set1_ps(best_t);

	int stack_idx[64];
	float stack_t[64];
	int sp = 0;
	stack_idx[0] = 0;
	stack_t[0] = 0.0f;
	sp = 1;

	while (sp > 0) {
		--sp;
		if (stack_t[sp] >= best_t) {
			continue;
		}

		const AOBVH4Node &node = p_nodes[stack_idx[sp]];

		__m128 tx0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_x), rox4), invx4);
		__m128 tx1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_x), rox4), invx4);
		__m128 tmin4 = _mm_min_ps(tx0, tx1);
		__m128 tmax4 = _mm_max_ps(tx0, tx1);

		__m128 ty0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_y), roy4), invy4);
		__m128 ty1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_y), roy4), invy4);
		tmin4 = _mm_max_ps(tmin4, _mm_min_ps(ty0, ty1));
		tmax4 = _mm_min_ps(tmax4, _mm_max_ps(ty0, ty1));

		__m128 tz0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_z), roz4), invz4);
		__m128 tz1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_z), roz4), invz4);
		tmin4 = _mm_max_ps(tmin4, _mm_min_ps(tz0, tz1));
		tmax4 = _mm_min_ps(tmax4, _mm_max_ps(tz0, tz1));

		tmin4 = _mm_max_ps(tmin4, tmin4_clamp);
		__m128 hit4 = _mm_and_ps(_mm_cmpge_ps(tmax4, tmin4), _mm_cmplt_ps(tmin4, _mm_set1_ps(best_t)));
		int hit_mask = _mm_movemask_ps(hit4) & (int)node.valid_mask;
		if (hit_mask == 0) {
			continue;
		}

		alignas(16) float tmin_arr[4];
		_mm_store_ps(tmin_arr, tmin4);

		int internal_c[4];
		float internal_t[4];
		int n_internal = 0;

		for (int k = 0; k < 4; k++) {
			if (!((hit_mask >> k) & 1)) {
				continue;
			}
			if (node.leaf_blocks[k] > 0) {
				if (tmin_arr[k] >= best_t) {
					continue;
				}
				const int first_blk = node.child[k];
				const int blk_count = node.leaf_blocks[k];
				for (int b = 0; b < blk_count; b++) {
					const AOTri8 &tri = p_blocks[first_blk + b];

					__m256 e1x = _mm256_load_ps(tri.e1x);
					__m256 e1y = _mm256_load_ps(tri.e1y);
					__m256 e1z = _mm256_load_ps(tri.e1z);
					__m256 e2x = _mm256_load_ps(tri.e2x);
					__m256 e2y = _mm256_load_ps(tri.e2y);
					__m256 e2z = _mm256_load_ps(tri.e2z);

					__m256 pvx = _mm256_sub_ps(_mm256_mul_ps(rdy8, e2z), _mm256_mul_ps(rdz8, e2y));
					__m256 pvy = _mm256_sub_ps(_mm256_mul_ps(rdz8, e2x), _mm256_mul_ps(rdx8, e2z));
					__m256 pvz = _mm256_sub_ps(_mm256_mul_ps(rdx8, e2y), _mm256_mul_ps(rdy8, e2x));

					__m256 det = _mm256_add_ps(
							_mm256_add_ps(_mm256_mul_ps(e1x, pvx), _mm256_mul_ps(e1y, pvy)),
							_mm256_mul_ps(e1z, pvz));
					__m256 abs_det = _mm256_andnot_ps(neg_zero8, det);
					__m256 valid = _mm256_cmp_ps(abs_det, eps8, _CMP_GE_OQ);
					if (_mm256_movemask_ps(valid) == 0) {
						continue;
					}

					__m256 safe_det = _mm256_blendv_ps(ones8, det, valid);
					__m256 inv_det = _mm256_div_ps(ones8, safe_det);

					__m256 tvx = _mm256_sub_ps(rox8, _mm256_load_ps(tri.p0x));
					__m256 tvy = _mm256_sub_ps(roy8, _mm256_load_ps(tri.p0y));
					__m256 tvz = _mm256_sub_ps(roz8, _mm256_load_ps(tri.p0z));

					__m256 u = _mm256_mul_ps(
							_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(tvx, pvx), _mm256_mul_ps(tvy, pvy)), _mm256_mul_ps(tvz, pvz)),
							inv_det);
					valid = _mm256_and_ps(valid, _mm256_and_ps(_mm256_cmp_ps(u, zeros8, _CMP_GE_OQ), _mm256_cmp_ps(u, ones8, _CMP_LE_OQ)));
					if (_mm256_movemask_ps(valid) == 0) {
						continue;
					}

					__m256 qvx = _mm256_sub_ps(_mm256_mul_ps(tvy, e1z), _mm256_mul_ps(tvz, e1y));
					__m256 qvy = _mm256_sub_ps(_mm256_mul_ps(tvz, e1x), _mm256_mul_ps(tvx, e1z));
					__m256 qvz = _mm256_sub_ps(_mm256_mul_ps(tvx, e1y), _mm256_mul_ps(tvy, e1x));

					__m256 v = _mm256_mul_ps(
							_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rdx8, qvx), _mm256_mul_ps(rdy8, qvy)), _mm256_mul_ps(rdz8, qvz)),
							inv_det);
					__m256 uv = _mm256_add_ps(u, v);
					valid = _mm256_and_ps(valid, _mm256_and_ps(_mm256_cmp_ps(v, zeros8, _CMP_GE_OQ), _mm256_cmp_ps(uv, ones8, _CMP_LE_OQ)));
					if (_mm256_movemask_ps(valid) == 0) {
						continue;
					}

					__m256 t = _mm256_mul_ps(
							_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(e2x, qvx), _mm256_mul_ps(e2y, qvy)), _mm256_mul_ps(e2z, qvz)),
							inv_det);
					valid = _mm256_and_ps(valid, _mm256_and_ps(_mm256_cmp_ps(t, tmin8, _CMP_GT_OQ), _mm256_cmp_ps(t, best8, _CMP_LT_OQ)));

					if (_mm256_movemask_ps(valid) != 0) {
						best8 = _mm256_blendv_ps(best8, t, valid);
						__m128 lo = _mm256_castps256_ps128(best8);
						__m128 hi = _mm256_extractf128_ps(best8, 1);
						__m128 m4 = _mm_min_ps(lo, hi);
						__m128 m2 = _mm_min_ps(m4, _mm_movehl_ps(m4, m4));
						__m128 m1 = _mm_min_ss(m2, _mm_shuffle_ps(m2, m2, 1));
						best_t = _mm_cvtss_f32(m1);
						best8 = _mm256_set1_ps(best_t);
						if (r_hit_tri) {
							int hit_lane_mask = _mm256_movemask_ps(_mm256_and_ps(valid, _mm256_cmp_ps(t, best8, _CMP_EQ_OQ)));
							if (hit_lane_mask != 0) {
								for (int l = 0; l < 8; l++) {
									if ((hit_lane_mask >> l) & 1) {
										best_tri = tri.tri_idx[l];
										break;
									}
								}
							}
						}
					}
				}
			} else {
				internal_c[n_internal] = node.child[k];
				internal_t[n_internal] = tmin_arr[k];
				n_internal++;
			}
		}

		// Sort descending so nearest child is popped from stack first
		for (int i = 0; i < n_internal - 1; i++) {
			for (int j = i + 1; j < n_internal; j++) {
				if (internal_t[i] < internal_t[j]) {
					int tc = internal_c[i]; internal_c[i] = internal_c[j]; internal_c[j] = tc;
					float tt = internal_t[i]; internal_t[i] = internal_t[j]; internal_t[j] = tt;
				}
			}
		}
		for (int i = 0; i < n_internal; i++) {
			if (sp < 64) {
				stack_idx[sp] = internal_c[i];
				stack_t[sp] = internal_t[i];
				sp++;
			}
		}
	}

	if (r_hit_tri) {
		*r_hit_tri = best_tri;
	}
	return best_t;
}

bool ao_occluded_bvh4_avx(
		const AOBVH4Node *p_nodes,
		const AOTri8 *p_blocks,
		float p_rox, float p_roy, float p_roz,
		float p_rdx, float p_rdy, float p_rdz,
		float p_tmin, float p_tmax) {

	const __m128 rox4 = _mm_set1_ps(p_rox);
	const __m128 roy4 = _mm_set1_ps(p_roy);
	const __m128 roz4 = _mm_set1_ps(p_roz);

	const __m128 invx4 = _mm_set1_ps(1.0f / (std::abs(p_rdx) < 1e-8f ? (p_rdx < 0.0f ? -1e-8f : 1e-8f) : p_rdx));
	const __m128 invy4 = _mm_set1_ps(1.0f / (std::abs(p_rdy) < 1e-8f ? (p_rdy < 0.0f ? -1e-8f : 1e-8f) : p_rdy));
	const __m128 invz4 = _mm_set1_ps(1.0f / (std::abs(p_rdz) < 1e-8f ? (p_rdz < 0.0f ? -1e-8f : 1e-8f) : p_rdz));
	const __m128 tmin4_clamp = _mm_set1_ps(p_tmin);
	const __m128 tmax4_clamp = _mm_set1_ps(p_tmax);

	const __m256 rox8 = _mm256_set1_ps(p_rox);
	const __m256 roy8 = _mm256_set1_ps(p_roy);
	const __m256 roz8 = _mm256_set1_ps(p_roz);
	const __m256 rdx8 = _mm256_set1_ps(p_rdx);
	const __m256 rdy8 = _mm256_set1_ps(p_rdy);
	const __m256 rdz8 = _mm256_set1_ps(p_rdz);

	const __m256 zeros8 = _mm256_setzero_ps();
	const __m256 ones8 = _mm256_set1_ps(1.0f);
	const __m256 eps8 = _mm256_set1_ps(1e-7f);
	const __m256 neg_zero8 = _mm256_set1_ps(-0.0f);
	const __m256 tmin8 = _mm256_set1_ps(p_tmin);
	const __m256 tmax8 = _mm256_set1_ps(p_tmax);

	int stack_idx[64];
	stack_idx[0] = 0;
	int sp = 1;

	while (sp > 0) {
		const AOBVH4Node &node = p_nodes[stack_idx[--sp]];

		__m128 tx0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_x), rox4), invx4);
		__m128 tx1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_x), rox4), invx4);
		__m128 tmin4 = _mm_min_ps(tx0, tx1);
		__m128 tmax4 = _mm_max_ps(tx0, tx1);

		__m128 ty0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_y), roy4), invy4);
		__m128 ty1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_y), roy4), invy4);
		tmin4 = _mm_max_ps(tmin4, _mm_min_ps(ty0, ty1));
		tmax4 = _mm_min_ps(tmax4, _mm_max_ps(ty0, ty1));

		__m128 tz0 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.min_z), roz4), invz4);
		__m128 tz1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(node.max_z), roz4), invz4);
		tmin4 = _mm_max_ps(tmin4, _mm_min_ps(tz0, tz1));
		tmax4 = _mm_min_ps(tmax4, _mm_max_ps(tz0, tz1));

		tmin4 = _mm_max_ps(tmin4, tmin4_clamp);
		__m128 hit4 = _mm_and_ps(_mm_cmpge_ps(tmax4, tmin4), _mm_cmplt_ps(tmin4, tmax4_clamp));
		int hit_mask = _mm_movemask_ps(hit4) & (int)node.valid_mask;
		if (hit_mask == 0) {
			continue;
		}

		for (int k = 0; k < 4; k++) {
			if (!((hit_mask >> k) & 1)) {
				continue;
			}
			if (node.leaf_blocks[k] > 0) {
				const int first_blk = node.child[k];
				const int blk_count = node.leaf_blocks[k];
				for (int b = 0; b < blk_count; b++) {
					const AOTri8 &tri = p_blocks[first_blk + b];

					__m256 e1x = _mm256_load_ps(tri.e1x);
					__m256 e1y = _mm256_load_ps(tri.e1y);
					__m256 e1z = _mm256_load_ps(tri.e1z);
					__m256 e2x = _mm256_load_ps(tri.e2x);
					__m256 e2y = _mm256_load_ps(tri.e2y);
					__m256 e2z = _mm256_load_ps(tri.e2z);

					__m256 pvx = _mm256_sub_ps(_mm256_mul_ps(rdy8, e2z), _mm256_mul_ps(rdz8, e2y));
					__m256 pvy = _mm256_sub_ps(_mm256_mul_ps(rdz8, e2x), _mm256_mul_ps(rdx8, e2z));
					__m256 pvz = _mm256_sub_ps(_mm256_mul_ps(rdx8, e2y), _mm256_mul_ps(rdy8, e2x));

					__m256 det = _mm256_add_ps(
							_mm256_add_ps(_mm256_mul_ps(e1x, pvx), _mm256_mul_ps(e1y, pvy)),
							_mm256_mul_ps(e1z, pvz));
					__m256 abs_det = _mm256_andnot_ps(neg_zero8, det);
					__m256 valid = _mm256_cmp_ps(abs_det, eps8, _CMP_GE_OQ);
					if (_mm256_movemask_ps(valid) == 0) {
						continue;
					}

					__m256 safe_det = _mm256_blendv_ps(ones8, det, valid);
					__m256 inv_det = _mm256_div_ps(ones8, safe_det);

					__m256 tvx = _mm256_sub_ps(rox8, _mm256_load_ps(tri.p0x));
					__m256 tvy = _mm256_sub_ps(roy8, _mm256_load_ps(tri.p0y));
					__m256 tvz = _mm256_sub_ps(roz8, _mm256_load_ps(tri.p0z));

					__m256 u = _mm256_mul_ps(
							_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(tvx, pvx), _mm256_mul_ps(tvy, pvy)), _mm256_mul_ps(tvz, pvz)),
							inv_det);
					valid = _mm256_and_ps(valid, _mm256_and_ps(_mm256_cmp_ps(u, zeros8, _CMP_GE_OQ), _mm256_cmp_ps(u, ones8, _CMP_LE_OQ)));
					if (_mm256_movemask_ps(valid) == 0) {
						continue;
					}

					__m256 qvx = _mm256_sub_ps(_mm256_mul_ps(tvy, e1z), _mm256_mul_ps(tvz, e1y));
					__m256 qvy = _mm256_sub_ps(_mm256_mul_ps(tvz, e1x), _mm256_mul_ps(tvx, e1z));
					__m256 qvz = _mm256_sub_ps(_mm256_mul_ps(tvx, e1y), _mm256_mul_ps(tvy, e1x));

					__m256 v = _mm256_mul_ps(
							_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rdx8, qvx), _mm256_mul_ps(rdy8, qvy)), _mm256_mul_ps(rdz8, qvz)),
							inv_det);
					__m256 uv = _mm256_add_ps(u, v);
					valid = _mm256_and_ps(valid, _mm256_and_ps(_mm256_cmp_ps(v, zeros8, _CMP_GE_OQ), _mm256_cmp_ps(uv, ones8, _CMP_LE_OQ)));
					if (_mm256_movemask_ps(valid) == 0) {
						continue;
					}

					__m256 t = _mm256_mul_ps(
							_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(e2x, qvx), _mm256_mul_ps(e2y, qvy)), _mm256_mul_ps(e2z, qvz)),
							inv_det);
					valid = _mm256_and_ps(valid, _mm256_and_ps(_mm256_cmp_ps(t, tmin8, _CMP_GT_OQ), _mm256_cmp_ps(t, tmax8, _CMP_LT_OQ)));
					if (_mm256_movemask_ps(valid) != 0) {
						return true;
					}
				}
			} else if (sp < 64) {
				stack_idx[sp++] = node.child[k];
			}
		}
	}

	return false;
}

#else

float ao_trace_bvh4_avx(
		const AOBVH4Node *p_nodes,
		const AOTri8 *p_blocks,
		float p_rox, float p_roy, float p_roz,
		float p_rdx, float p_rdy, float p_rdz,
		float p_tmin, float p_tmax,
		int32_t *r_hit_tri) {
	if (r_hit_tri) {
		*r_hit_tri = -1;
	}
	return p_tmax;
}

bool ao_occluded_bvh4_avx(
		const AOBVH4Node *p_nodes,
		const AOTri8 *p_blocks,
		float p_rox, float p_roy, float p_roz,
		float p_rdx, float p_rdy, float p_rdz,
		float p_tmin, float p_tmax) {
	return false;
}

#endif
