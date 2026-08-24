#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common.h"

#include "params.h"

#include "sm90/decode/sparse_fp8/splitkv_mla.h"
#include "sm100/decode/head64/kernel.h"
#include "sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "smxx/decode/combine/combine.h"

// Feature set of sparse decoding kernels
enum class DecodeFeatures : int {
    HEAD_64,
    HEAD_128,

    HEAD_DIM_576,
    HEAD_DIM_512,

    V32_KVCACHE_FORMAT,
    MODEL1_KVCACHE_FORMAT,

    ATTN_SINK,
    TOPK_LENGTH,
    EXTRA_KVCACHE,
    EXTRA_TOPK_LENGTH
};

struct DecodeImplMeta {
    int num_sm_parts;
    int fixed_overhead_num_blocks;
    int block_size_topk;
    int active_h_q;
};

struct DecodeWorkload {
    int b;
    int s_q;
    int h_q;
    int d_qk;
    int page_block_size;
    int topk;
    int extra_page_block_size;
    int extra_topk;
    bool have_topk_length;
    bool have_extra_kcache;
    bool have_extra_topk_length;
};

class DecodeImplBase : public ImplBase<
    SparseAttnDecodeParams,
    DecodeFeatures
> {
public:
    virtual DecodeImplMeta get_meta(const DecodeWorkload &workload) = 0;
};

static int positive_env_or_zero(const char *name) {
    const char *value = std::getenv(name);
    if (!value) return 0;
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : 0;
}

static const char *workload_signature(const DecodeWorkload &w) {
    if (!w.have_extra_kcache && w.topk == 2048) return "glm";
    if (!w.have_extra_kcache && w.topk == 128) return "dsv4-swa";
    if (w.have_extra_kcache && w.topk == 128 && w.extra_topk == 1024)
        return "dsv4-c4a";
    if (w.have_extra_kcache && w.topk == 128 && w.extra_topk == 8192)
        return "dsv4-c128a";
    return "other";
}

// This first table is intentionally conservative and remains tunable while
// repeated multi-context sweeps are running.  Unsupported or high-load shapes
// always preserve the upstream scheduler.  Final accepted values are replaced
// only after every repeated sample is positive and the median clears the gate.
static int dynamic_num_sm_parts(
    const DecodeWorkload &w,
    int upstream_num_sm_parts
) {
    const int query_tiles = w.b * w.s_q;
    const char *signature = workload_signature(w);
    int cap = upstream_num_sm_parts;
    if (std::strcmp(signature, "glm") == 0 && w.d_qk == 576) {
        if (w.s_q == 1 && w.b == 1) cap = 32;
    } else if (std::strcmp(signature, "dsv4-swa") == 0) {
        if (query_tiles == 8) cap = 32;
        else if (query_tiles == 32) cap = 96;
        else if (query_tiles == 64) cap = 64;
    } else if (std::strcmp(signature, "dsv4-c128a") == 0) {
        if (query_tiles == 8) cap = 96;
        else if (query_tiles == 32) cap = 32;
    }
    return std::min(upstream_num_sm_parts, cap);
}

static int select_num_sm_parts(
    const DecodeWorkload &w,
    int upstream_num_sm_parts,
    const char **policy_out
) {
    const char *policy = std::getenv("WS58_FLASHMLA_POLICY");
    if (policy && (std::strcmp(policy, "dynamic") == 0 ||
                   std::strcmp(policy, "partition-only") == 0)) {
        *policy_out = "dynamic";
        return dynamic_num_sm_parts(w, upstream_num_sm_parts);
    }
    if (!policy || std::strcmp(policy, "baseline") == 0 ||
        std::strcmp(policy, "active-combine-only") == 0) {
        *policy_out = "baseline";
        return upstream_num_sm_parts;
    }
    if (std::strcmp(policy, "fixed") == 0) {
        int cap = positive_env_or_zero("WS58_FLASHMLA_FIXED_CAP");
        if (!cap) cap = positive_env_or_zero("WS58_C04_MAX_SM_PARTS");
        *policy_out = "fixed";
        return cap ? std::min(upstream_num_sm_parts, cap) : upstream_num_sm_parts;
    }
    *policy_out = "invalid-baseline-fallback";
    return upstream_num_sm_parts;
}

static int select_active_h_q(
    const DecodeWorkload &w,
    const char *policy
) {
    // The split kernel retains the padded 64/128-head ABI.  Only the generic
    // combine kernel is allowed to skip padded heads.  Keep this opt-in because
    // FlashMLA itself cannot infer the unpadded TP-local head count from q.
    if (!policy || (std::strcmp(policy, "dynamic") != 0 &&
                    std::strcmp(policy, "active-combine-only") != 0)) {
        return w.h_q;
    }
    const int requested = positive_env_or_zero("WS58_FLASHMLA_ACTIVE_HEADS");
    if (!requested || requested > w.h_q || requested % 8 != 0) {
        return w.h_q;
    }
    return requested;
}

class Decode_Sm90_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64,
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(const DecodeWorkload &w) override {
        Arch arch = Arch();
        const int upstream_num_sm_parts =
            std::max(arch.num_sms / w.s_q / (w.h_q / 64), 1);
        const char *policy = nullptr;
        const int selected_num_sm_parts =
            select_num_sm_parts(w, upstream_num_sm_parts, &policy);
        const char *requested_policy = std::getenv("WS58_FLASHMLA_POLICY");
        const int active_h_q = select_active_h_q(w, requested_policy);
        static std::atomic<int> log_count{0};
        const int ticket = log_count.fetch_add(1, std::memory_order_relaxed);
        if (ticket < 128) {
            std::fprintf(
                stderr,
                "WS58_FLASHMLA_DYNAMIC_HIT policy=%s signature=%s b=%d s_q=%d "
                "query_tiles=%d h_q=%d d_qk=%d page=%d topk=%d extra_page=%d "
                "extra_topk=%d have_topk_length=%d have_extra_topk_length=%d "
                "upstream=%d selected=%d active_h_q=%d\n",
                policy, workload_signature(w), w.b, w.s_q, w.b * w.s_q,
                w.h_q, w.d_qk, w.page_block_size, w.topk,
                w.extra_page_block_size, w.extra_topk,
                static_cast<int>(w.have_topk_length),
                static_cast<int>(w.have_extra_topk_length),
                upstream_num_sm_parts, selected_num_sm_parts, active_h_q
            );
            std::fflush(stderr);
        }
        return {
            selected_num_sm_parts,
            5,
            64,
            active_h_q
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
            DISPATCH_NUM_HEADS(params.h_q, NUM_HEADS, [&]() {
                sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE, NUM_HEADS>(params);
            });
        });
    }
};

class Decode_Sm100_Head64_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(const DecodeWorkload &w) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / w.s_q, 1),
            5,
            64,
            w.h_q
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
            sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE>(params);
        });
    }
};


// An implementation that calls the head64 kernel twice to process head128
// Necessary for running V3.2 shape (i.e. h = 128, d_qk = 576) on SM100f
class Decode_Sm100_Head64x2_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(const DecodeWorkload &w) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / w.s_q, 1),
            5,
            64,
            w.h_q
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
            for (int start_head_idx = 0; start_head_idx < 128; start_head_idx += 64) {
                SparseAttnDecodeParams cur_params = params;
                cur_params.q += start_head_idx * params.stride_q_h_q;
                if (cur_params.attn_sink) {
                    cur_params.attn_sink += start_head_idx;
                }
                cur_params.lse += start_head_idx;
                cur_params.out += start_head_idx * params.stride_o_h_q;
                cur_params.lse_accum += start_head_idx;
                cur_params.o_accum += start_head_idx * params.stride_o_accum_h_q;
                cur_params.h_q = 64;
                sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE>(cur_params);
            }
        });
    }
};


class Decode_Sm100_Head128_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(const DecodeWorkload &w) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / w.s_q / 2, 1),
            3,
            64,
            w.h_q
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        sm100::fwd_for_small_topk::head128::run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, 512>(params);
    }
};

static std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>>
sparse_attn_decode_interface(
    const Tensor &q,   // [b, s_q, h_q, d_qk]
    const Tensor &kv,   // [num_blocks, page_block_size, h_k, d_qk]
    const Tensor &indices,    // [b, s_q, topk]
    const std::optional<Tensor> &topk_length,   // [b, s_q]
    const std::optional<Tensor> &attn_sink, // [h_q]
    std::optional<Tensor> tile_scheduler_metadata,    // num_sm_parts x (DecodingSchedMetaSize/4)
    std::optional<Tensor> num_splits,                 // batch_size + 1
    const std::optional<Tensor> &extra_kv,
    const std::optional<Tensor> &extra_indices,
    const std::optional<Tensor> &extra_topk_length,
    int64_t d_v,
    double sm_scale,
    const std::optional<Tensor> &out_
) {
    using bf16 = cutlass::bfloat16_t;

    // Check the architecture
    Arch arch = Arch();

    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);

    int b = q.size(0);
    int s_q = q.size(1);
    int h_q = q.size(2);
    int d_qk = q.size(3);
    int num_blocks = kv.size(0);
    int page_block_size = kv.size(1);
    int h_kv = kv.size(2);
    int topk = indices.size(2);

    bool have_topk_length = topk_length.has_value();
    bool have_extra_kcache = extra_kv.has_value();
    bool have_extra_topk_length = extra_topk_length.has_value();
    bool have_attn_sink = attn_sink.has_value();

    int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;
    if (have_extra_kcache) {
        extra_num_blocks = extra_kv->size(0);
        extra_page_block_size = extra_kv->size(1);
    }
    if (extra_indices.has_value()) {
        extra_topk = extra_indices->size(-1);
    }

    // metadata sanity check
    STD_TORCH_CHECK(b > 0);
    STD_TORCH_CHECK(s_q > 0);
    STD_TORCH_CHECK(h_q > 0);
    STD_TORCH_CHECK(h_kv == 1, "Currently only MQA (i.e. h_kv == 1) is supported for sparse decoding");
    STD_TORCH_CHECK(d_qk == 576 || d_qk == 512, "Only head_size_k == 576 or 512 is supported for sparse decoding");
    STD_TORCH_CHECK(d_v == 512, "Only head_size_v == 512 is supported for sparse decoding");
    STD_TORCH_CHECK(topk > 0);

    if (have_extra_kcache) {
        STD_TORCH_CHECK(extra_indices.has_value(), "extra_indices_in_kvcache must be provided when extra_kcache is provided for sparse attention");
    } else {
        STD_TORCH_CHECK(!extra_indices.has_value(), "extra_indices_in_kvcache must not be provided when extra_k_cache is not provided");
        STD_TORCH_CHECK(!extra_topk_length.has_value(), "extra_topk_length must not be provided when extra_k_cache is not provided");
    }

    // Check device
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);

    // Check data type
    KU_CHECK_DTYPE(q, ScalarType::BFloat16);
    STD_TORCH_CHECK(kv.scalar_type() == ScalarType::Float8_e4m3fn || kv.scalar_type() == ScalarType::Char || kv.scalar_type() == ScalarType::Byte, "key must have dtype fp8_e4m3fn, int8 or uint8");
    if (extra_kv.has_value()) {
        STD_TORCH_CHECK(extra_kv->scalar_type() == ScalarType::Float8_e4m3fn || extra_kv->scalar_type() == ScalarType::Char || extra_kv->scalar_type() == ScalarType::Byte, "extra k cache must have dtype fp8_e4m3fn, int8 or uint8");
    }
    KU_CHECK_DTYPE(indices, ScalarType::Int);
    KU_CHECK_DTYPE(topk_length, ScalarType::Int);
    KU_CHECK_DTYPE(attn_sink, ScalarType::Float);
    KU_CHECK_DTYPE(tile_scheduler_metadata, ScalarType::Int);
    KU_CHECK_DTYPE(num_splits, ScalarType::Int);
    KU_CHECK_DTYPE(extra_indices, ScalarType::Int);
    KU_CHECK_DTYPE(extra_topk_length, ScalarType::Int);
    
    // Check layout
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);

    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);

    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);
    
    // Check shape
    KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
    {
        int bytes_per_token;
        if (d_qk == 576 && d_v == 512) {
            // V3.2 style
            bytes_per_token = 512 + 64*2 + (512/128)*4;
        } else if (d_qk == 512 && d_v == 512) {
            // MODEL1 style
            bytes_per_token = 448 + 64*2 + (448/64)*1 + 1;
        } else {
            STD_TORCH_CHECK(false, "Unsupported head sizes for is_fp8_kvcache == True");
        }
        KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, bytes_per_token);
        KU_CHECK_SHAPE(extra_kv, extra_num_blocks, extra_page_block_size, h_kv, bytes_per_token);
        STD_TORCH_CHECK(kv.stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True for kv cache");
        if (extra_kv.has_value()) {
            STD_TORCH_CHECK(extra_kv->stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True for extra kv cache");
        }
    }
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
    KU_CHECK_SHAPE(extra_topk_length, b);

    torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());

    Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        KU_CHECK_DTYPE(out, ScalarType::BFloat16);
        KU_CHECK_SHAPE(out, b, s_q, h_q, d_v);
        KU_CHECK_LAST_DIM_CONTIGUOUS(out);
        KU_CHECK_DEVICE(out);
    } else {
        out = torch::stable::new_empty(q, {b, s_q, h_q, d_v});
    }
    Tensor lse = torch::stable::new_empty(q, {b, s_q, h_q}, ScalarType::Float);

    ModelType model_type;
    if (d_qk == 576) {
        model_type = ModelType::V32;
    } else if (d_qk == 512) {
        model_type = ModelType::MODEL1;
    } else {
        STD_TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }

    std::vector<DecodeFeatures> features;
    if (h_q == 64) {
        features.push_back(DecodeFeatures::HEAD_64);
    } else if (h_q == 128) {
        features.push_back(DecodeFeatures::HEAD_128);
    } else {
        STD_TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (d_qk == 576) {
        features.push_back(DecodeFeatures::HEAD_DIM_576);
    } else if (d_qk == 512) {
        features.push_back(DecodeFeatures::HEAD_DIM_512);
    } else {
        STD_TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }
    if (model_type == ModelType::V32) {
        features.push_back(DecodeFeatures::V32_KVCACHE_FORMAT);
    } else if (model_type == ModelType::MODEL1) {
        features.push_back(DecodeFeatures::MODEL1_KVCACHE_FORMAT);
    } else {
        STD_TORCH_CHECK(false, "Unsupported model type: ", (int)model_type);
    }
    if (have_attn_sink) {
        features.push_back(DecodeFeatures::ATTN_SINK);
    }
    if (have_topk_length) {
        features.push_back(DecodeFeatures::TOPK_LENGTH);
    }
    if (have_extra_kcache) {
        features.push_back(DecodeFeatures::EXTRA_KVCACHE);
    }
    if (have_extra_topk_length) {
        features.push_back(DecodeFeatures::EXTRA_TOPK_LENGTH);
    }

    DecodeImplBase* impl;
    if (arch.is_sm100f()) {
        if (h_q == 64) {
            impl = new Decode_Sm100_Head64_Impl();
        } else if (h_q == 128) {
            if (d_qk == 576) {
                impl = new Decode_Sm100_Head64x2_Impl();
            } else if (d_qk == 512) {
                impl = new Decode_Sm100_Head128_Impl();
            } else {
                STD_TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
            }
        } else {
            STD_TORCH_CHECK(false, "Unsupported h_q: ", h_q);
        }
    } else if (arch.is_sm90a()) {
        impl = new Decode_Sm90_Impl();
    } else {
        STD_TORCH_CHECK(false, "Unsupported architecture for sparse decode fwd");
    }

    DecodeImplMeta impl_meta = impl->get_meta({
        b,
        s_q,
        h_q,
        d_qk,
        page_block_size,
        topk,
        extra_page_block_size,
        extra_topk,
        have_topk_length,
        have_extra_kcache,
        have_extra_topk_length,
    });

    SparseAttnDecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, d_v,
        sm_scale, sm_scale * LOG_2_E,
        num_blocks, page_block_size, topk,
        model_type,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        (float*)lse.data_ptr(),
        (bf16*)out.data_ptr(),

        extra_num_blocks, extra_page_block_size, extra_topk,
        ku::get_optional_tensor_ptr<bf16>(extra_kv),
        ku::get_optional_tensor_ptr<int>(extra_indices),
        ku::get_optional_tensor_ptr<int>(extra_topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)), int64_stride_to_int(q.stride(2)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        int64_stride_to_int(lse.stride(0)), int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)), int64_stride_to_int(out.stride(2)),

        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(0)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(1)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(0)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(1)) : 0,
        get_current_cuda_stream(q)
    };

    // Get MLA metadata if necessary
    Tensor o_accum, lse_accum;
    if (!tile_scheduler_metadata.has_value()) {
        tile_scheduler_metadata = torch::stable::new_empty(q, {impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4}, ScalarType::Int);
        num_splits = torch::stable::new_empty(q, {b+1}, ScalarType::Int);
        KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
        KU_CHECK_CONTIGUOUS(num_splits);

        GetDecodeSchedMetaParams get_sched_meta_params = {
            b, s_q,
            impl_meta.block_size_topk,
            impl_meta.fixed_overhead_num_blocks,
            topk,
            extra_topk,
            ku::get_optional_tensor_ptr<int>(topk_length),
            ku::get_optional_tensor_ptr<int>(extra_topk_length),
            nullptr,
            (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
            num_splits->mutable_data_ptr<int>(),
            impl_meta.num_sm_parts,
            get_current_cuda_stream(q)
        };
        smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
    }
    // Stick the metadata pointers to `params`
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DTYPE(tile_scheduler_metadata, ScalarType::Int);
    KU_CHECK_DTYPE(num_splits, ScalarType::Int);
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);
    KU_CHECK_SHAPE(tile_scheduler_metadata, impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/sizeof(int));
    KU_CHECK_SHAPE(num_splits, b+1);
    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
    params.num_splits_ptr = num_splits->mutable_data_ptr<int>();
    params.num_sm_parts = impl_meta.num_sm_parts;

    // Allocate intermediate buffers for split-KV
    const int total_num_splits = b + impl_meta.num_sm_parts;
    lse_accum = torch::stable::new_empty(q, {total_num_splits, s_q, h_q}, ScalarType::Float);
    o_accum = torch::stable::new_empty(q, {total_num_splits, s_q, h_q, d_v}, ScalarType::Float);
    KU_CHECK_CONTIGUOUS(lse_accum);
    KU_CHECK_CONTIGUOUS(o_accum);
    params.lse_accum = lse_accum.mutable_data_ptr<float>();
    params.o_accum = o_accum.mutable_data_ptr<float>();
    params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
    params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
    params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
    params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
    params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));

    impl->run(params, features);
    
    CombineParams combine_params = {
        b, s_q, impl_meta.active_h_q, d_v,

        params.lse,
        params.out,
        params.stride_lse_b, params.stride_lse_s_q,
        params.stride_o_b, params.stride_o_s_q, params.stride_o_h_q,

        params.lse_accum,
        params.o_accum,
        params.stride_lse_accum_split, params.stride_lse_accum_s_q,
        params.stride_o_accum_split, params.stride_o_accum_s_q, params.stride_o_accum_h_q,

        params.tile_scheduler_metadata_ptr,
        params.num_splits_ptr,
        params.num_sm_parts,

        ku::get_optional_tensor_ptr<float>(attn_sink),
        get_current_cuda_stream(q)
    };
    smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);

    delete impl;

    return {out, torch::stable::transpose(lse, 1, 2), tile_scheduler_metadata, num_splits};
}
