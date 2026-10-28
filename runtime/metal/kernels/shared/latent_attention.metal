#include "metal/abi/LatentAttention.h"
#include "metal/kernels/common/kv_extent.h"

#include <metal_stdlib>

using namespace metal;

// Latent attention (metal/abi/LatentAttention.h, DeepSeek-V4.1's), as
// DeepSeek's model.py computes it (sparse_attn; in
// dev/tools/deepseek41_reference.py Forward.attention): each head's scores
// over the row's keys, scaled; their softmax beside the head's sink logit,
// which takes its share of the denominator and no value; and the sum of the
// keys, each its own value, by those weights. Three passes: the scores of
// every head and key, their softmax in place, and the weighted sums, over
// parts of the keys added by a fourth when a pass splits them.

constant constexpr uint kDim = LATENT_ATTENTION_DIM;
constant constexpr uint kThreads = LATENT_ATTENTION_THREADS;
constant constexpr uint kKeyBlock = LATENT_ATTENTION_KEY_BLOCK;
constant constexpr uint kHeadBlock = LATENT_ATTENTION_HEAD_BLOCK;
constant constexpr uint kKeysPerSimdgroup = kKeyBlock / (kThreads / 32);

// The value of LATENT_INDEX_NONE (metal/abi/LatentIndex.h), which ends a
// row's selection.
constant constexpr uint kNoGroup = 0xFFFFFFFFu;

// Where a pass's compressed keys are: a pool of latents, a key its index-th,
// or a sequence's pages, a key the latent at the place of the position its
// index names, through the sequence's page table (abi/KvExtent.h).
struct PooledKeys {
  using Source = device const bfloat *;
  static device const bfloat *at(Source pool, constant LatentAttentionParams &, uint index) {
    return pool + ulong(index) * kDim;
  }
};
struct PagedKeys {
  using Source = device const SplashKvPage *;
  static device const bfloat *at(Source pages, constant LatentAttentionParams &p, uint position) {
    const SplashKvPage entry = pages[position / SplashKvPageTokens];
    const uint page = splash_kv_page_index(entry);
    return reinterpret_cast<device const bfloat *>(splash_kv_extent(entry, page) + p.kv.offset) +
           splash_kv_key_index<1, kDim>(page, 0, position % SplashKvPageTokens, 0);
  }
};

// Key `index` of a row's list: a window latent, of the ring before the step
// below slot p.ring and of the step's past it, or a compressed one.
template <class Keys>
inline device const bfloat *latent_key(device const bfloat *window, device const bfloat *ring,
                                       typename Keys::Source compressed, constant LatentAttentionParams &p,
                                       uint index) {
  if (!(index & LATENT_ATTENTION_WINDOW_KEY)) return Keys::at(compressed, p, index);
  const uint slot = index & ~LATENT_ATTENTION_WINDOW_KEY;
  return slot < p.ring ? ring + ulong(slot) * kDim : window + ulong(slot - p.ring) * kDim;
}

// The scaled scores of every head of a row with LATENT_ATTENTION_KEY_BLOCK
// of its keys: each simdgroup takes some of the keys, each lane the heads
// lane and lane + 32.
template <class Keys>
inline void latent_scores(device const bfloat *q, device const bfloat *window, typename Keys::Source compressed,
                          device const uint *indices, device const uint *counts, device float *scores,
                          constant LatentAttentionParams &p, device const bfloat *ring, uint2 group, uint lane,
                          uint sg) {
  const uint row = group.x;
  if (row >= p.rows) return;
  const uint count = counts[row];
  device const bfloat *row_q = q + ulong(row) * p.heads * kDim;
  for (uint i = 0; i < kKeysPerSimdgroup; ++i) {
    const uint k = group.y * kKeyBlock + sg * kKeysPerSimdgroup + i;
    if (k >= count) return;
    device const bfloat *key = latent_key<Keys>(window, ring, compressed, p, indices[ulong(row) * p.key_stride + k]);
    for (uint h = lane; h < p.heads; h += 32) {
      device const bfloat *head = row_q + ulong(h) * kDim;
      float sum = 0.0f;
      for (uint d = 0; d < kDim; d += 4)
        sum += dot(float4(*reinterpret_cast<device const bfloat4 *>(head + d)),
                   float4(*reinterpret_cast<device const bfloat4 *>(key + d)));
      scores[(ulong(row) * p.heads + h) * p.key_stride + k] = sum * p.scale;
    }
  }
}
kernel void latent_attention_scores(device const bfloat *q [[buffer(0)]], device const bfloat *window [[buffer(1)]],
                                    device const bfloat *compressed [[buffer(2)]],
                                    device const uint *indices [[buffer(3)]], device const uint *counts [[buffer(4)]],
                                    device float *scores [[buffer(5)]],
                                    constant LatentAttentionParams &p [[buffer(7)]],
                                    device const bfloat *ring [[buffer(6)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  latent_scores<PooledKeys>(q, window, compressed, indices, counts, scores, p, ring, group, lane, sg);
}
kernel void latent_attention_scores_paged(device const bfloat *q [[buffer(0)]],
                                          device const bfloat *window [[buffer(1)]],
                                          device const SplashKvPage *pages [[buffer(2)]],
                                          device const uint *indices [[buffer(3)]],
                                          device const uint *counts [[buffer(4)]],
                                          device float *scores [[buffer(5)]],
                                          constant LatentAttentionParams &p [[buffer(7)]],
                                          device const bfloat *ring [[buffer(6)]],
                                          uint2 group [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]],
                                          uint sg [[simdgroup_index_in_threadgroup]]) {
  latent_scores<PagedKeys>(q, window, pages, indices, counts, scores, p, ring, group, lane, sg);
}

// Each row's key list (LatentAttentionKeysParams): its window's slots in
// position order, then its selected groups, and their count. A threadgroup
// per row; its threads count the selection's groups, which end at its first
// LATENT_INDEX_NONE.
kernel void latent_attention_keys(device const uint *positions [[buffer(0)]],
                                  device const uint *selected [[buffer(1)]],
                                  device uint *indices [[buffer(2)]],
                                  device uint *counts [[buffer(3)]],
                                  constant LatentAttentionKeysParams &p [[buffer(4)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint partials[kThreads / 32];
  if (row >= p.rows) return;
  const uint position = positions[row], window = min(p.window, position + 1);
  device uint *out = indices + ulong(row) * p.key_stride;
  for (uint i = tid; i < window; i += kThreads) {
    const uint key = position + 1 - window + i;
    const uint slot = !p.split ? key % p.ring : key < p.committed ? key % p.ring : p.ring + key - p.committed;
    out[i] = slot | LATENT_ATTENTION_WINDOW_KEY;
  }
  device const uint *groups = selected + ulong(row) * p.selected;
  uint count = 0;
  for (uint i = tid; i < p.selected; i += kThreads) count += groups[i] != kNoGroup;
  count = simd_sum(count);
  if (lane == 0) partials[sg] = count;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint total = 0;
  for (uint g = 0; g < kThreads / 32; ++g) total += partials[g];
  for (uint i = tid; i < total; i += kThreads) out[window + i] = p.ratio ? (groups[i] + 1) * p.ratio - 1 : groups[i];
  if (tid == 0) counts[row] = window + total;
}

// A head's scores of a row in place by their softmax beside its sink: exp(s -
// peak) over their sum plus exp(sink - peak), peak the largest of them and
// the sink. A simdgroup per row and head.
kernel void latent_attention_softmax(device float *scores [[buffer(0)]],
                                     device const uint *counts [[buffer(1)]],
                                     device const float *sinks [[buffer(2)]],
                                     constant LatentAttentionParams &p [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.x, h = group.y;
  if (row >= p.rows || h >= p.heads) return;
  const uint count = counts[row];
  device float *s = scores + (ulong(row) * p.heads + h) * p.key_stride;
  const float sink = sinks[h];
  float peak = sink;
  for (uint k = lane; k < count; k += 32) peak = max(peak, s[k]);
  peak = simd_max(peak);
  float sum = 0.0f;
  for (uint k = lane; k < count; k += 32) sum += exp(s[k] - peak);
  sum = simd_sum(sum) + exp(sink - peak);
  const float inverse = 1.0f / sum;
  for (uint k = lane; k < count; k += 32) s[k] = exp(s[k] - peak) * inverse;
}

// Each head's output of a row: its keys summed by its weights, fp32; with
// p.splits parts, part `split` of it, over that part of the keys, at
// out [rows][splits][heads][LATENT_ATTENTION_DIM]. A threadgroup per row,
// LATENT_ATTENTION_HEAD_BLOCK heads and part; a thread takes four values of
// each of the heads, so each key's value is read once for them.
template <class Keys>
inline void latent_values(device const float *weights, device const bfloat *window, typename Keys::Source compressed,
                          device const uint *indices, device const uint *counts, device float *out,
                          constant LatentAttentionParams &p, device const bfloat *ring, uint3 group, uint tid) {
  const uint row = group.x, first = group.y * kHeadBlock, split = group.z, d = tid * 4;
  if (row >= p.rows || first >= p.heads || split >= p.splits || d >= kDim) return;
  const uint heads = min(kHeadBlock, p.heads - first);
  const uint count = counts[row], begin = split * count / p.splits, end = (split + 1) * count / p.splits;
  device const uint *row_indices = indices + ulong(row) * p.key_stride;
  device const float *row_weights = weights + (ulong(row) * p.heads + first) * p.key_stride;
  float4 sums[kHeadBlock];
  for (uint h = 0; h < kHeadBlock; ++h) sums[h] = 0.0f;
  for (uint k = begin; k < end; ++k) {
    const float4 value =
        float4(*reinterpret_cast<device const bfloat4 *>(latent_key<Keys>(window, ring, compressed, p, row_indices[k]) + d));
    for (uint h = 0; h < kHeadBlock; ++h)
      if (h < heads) sums[h] = fma(float4(row_weights[ulong(h) * p.key_stride + k]), value, sums[h]);
  }
  device float *row_out = out + ((ulong(row) * p.splits + split) * p.heads + first) * kDim + d;
  for (uint h = 0; h < heads; ++h) *reinterpret_cast<device float4 *>(row_out + ulong(h) * kDim) = sums[h];
}
kernel void latent_attention_values(device const float *weights [[buffer(0)]],
                                    device const bfloat *window [[buffer(1)]],
                                    device const bfloat *compressed [[buffer(2)]],
                                    device const uint *indices [[buffer(3)]], device const uint *counts [[buffer(4)]],
                                    device float *out [[buffer(5)]], constant LatentAttentionParams &p [[buffer(7)]],
                                    device const bfloat *ring [[buffer(6)]],
                                    uint3 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  latent_values<PooledKeys>(weights, window, compressed, indices, counts, out, p, ring, group, tid);
}
kernel void latent_attention_values_paged(device const float *weights [[buffer(0)]],
                                          device const bfloat *window [[buffer(1)]],
                                          device const SplashKvPage *pages [[buffer(2)]],
                                          device const uint *indices [[buffer(3)]],
                                          device const uint *counts [[buffer(4)]], device float *out [[buffer(5)]],
                                          constant LatentAttentionParams &p [[buffer(7)]],
                                          device const bfloat *ring [[buffer(6)]],
                                          uint3 group [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]]) {
  latent_values<PagedKeys>(weights, window, pages, indices, counts, out, p, ring, group, tid);
}

// Each head's output of a row from the values pass's parts, parts [rows]
// [splits][heads][LATENT_ATTENTION_DIM], added in part order. A thread per
// four values.
kernel void latent_attention_sum(device const float *parts [[buffer(0)]],
                                 device float *out [[buffer(1)]],
                                 constant LatentAttentionParams &p [[buffer(2)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, i = (group.y * kThreads + tid) * 4, width = p.heads * kDim;
  if (row >= p.rows || i >= width) return;
  device const float *part = parts + ulong(row) * p.splits * width + i;
  float4 sum = 0.0f;
  for (uint s = 0; s < p.splits; ++s) sum += *reinterpret_cast<device const float4 *>(part + ulong(s) * width);
  *reinterpret_cast<device float4 *>(out + ulong(row) * width + i) = sum;
}

// A sequence's ring after a step (LatentRingParams): a threadgroup per slot,
// a thread four values of its latent.
kernel void latent_ring_update(device const bfloat *ring_in [[buffer(0)]], device const bfloat *step [[buffer(1)]],
                               device bfloat *ring_out [[buffer(2)]], device const uint *retained [[buffer(3)]],
                               constant LatentRingParams &p [[buffer(4)]], uint slot [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  const uint d = tid * 4;
  if (slot >= p.ring || d >= kDim) return;
  const uint kept = p.retained ? min(retained[p.lane], p.rows) : p.rows, end = p.committed + kept;
  device const bfloat *from = ring_in + ulong(slot) * kDim;
  if (end) {
    const uint last = end - 1, back = (last + p.ring - slot) % p.ring;
    if (last >= back && last - back >= p.committed) from = step + ulong(last - back - p.committed) * kDim;
  }
  *reinterpret_cast<device bfloat4 *>(ring_out + ulong(slot) * kDim + d) =
      *reinterpret_cast<device const bfloat4 *>(from + d);
}

// A paired KV source's open group after a verify (LatentOpenParams): a thread
// per value of its kv and gate projections.
kernel void latent_open_commit(device const float *kv [[buffer(0)]], device const float *gate [[buffer(1)]],
                               device float *open [[buffer(2)]], device const uint *retained [[buffer(3)]],
                               constant LatentOpenParams &p [[buffer(4)]], uint d [[thread_position_in_grid]]) {
  const uint kept = retained[p.lane];
  if (d >= p.width || !((p.committed + kept) & 1)) return;
  const ulong row = ulong(kept) * p.width;
  open[d] = kv[row + d];
  open[p.width + d] = gate[row + d];
}
