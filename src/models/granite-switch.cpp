#include "models.h"

#include <sstream>

void llama_model_granite_switch::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);
    ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale, false);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE,             hparams.f_attention_scale, false);

    ml.get_key(LLM_KV_ADAPTER_COUNT, hparams.n_adapters);

    if (hparams.n_adapters > 0) {
        ml.get_arr("graniteswitch.adapter_ranks", hparams.adapter_ranks_arr);
        ml.get_key(LLM_KV_MAX_LORA_RANK, hparams.max_lora_rank);
        ml.get_arr("graniteswitch.adapter_token_ids", hparams.adapter_token_ids_arr);
        ml.get_arr("graniteswitch.adapter_substitute_token_ids", hparams.adapter_substitute_token_ids_arr);
        ml.get_key(LLM_KV_CONTROL_TOKEN_GAIN, hparams.f_control_token_gain);
        ml.get_key(LLM_KV_SWITCH_HEAD_DIM, hparams.n_switch_head_dim);
        ml.get_key(LLM_KV_PROJECTION_HEAD_DIM, hparams.n_projection_head_dim);
    }

    // Initialize adapter arrays to -1 (sentinel value)
    hparams.adapter_token_ids_arr.fill(-1);
    hparams.adapter_substitute_token_ids_arr.fill(-1);

    // Granite Switch has no MoE experts - override any inherited values
    hparams.n_expert = 0;
    hparams.n_expert_used = 0;

    // Granite uses rope_finetuned as a switch for rope, so default to true
    bool rope_finetuned = true;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    switch (hparams.n_layer()) {
        case 41: type = LLM_TYPE_3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_granite_switch::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // Load embedded adapters from GGUF
    // Format: blk.{layer}.{module}.lora_{a|b}.adapter_{id}.weight
    if (hparams.n_adapters > 0 && hparams.max_lora_rank > 0) {
        // First pass: collect all lora tensors by (module, lora_type, adapter_id)
        struct LoraTensorInfo {
            std::string module;
            bool is_a;  // true = lora_a, false = lora_b
            int adapter_id;
            ggml_tensor * tensor = nullptr;
        };

        std::vector<LoraTensorInfo> lora_info;
        struct gguf_context * gguf = ml.metadata;

        for (int i = 0; i < gguf_get_n_tensors(gguf); i++) {
            const char * name = gguf_get_tensor_name(gguf, i);
            if (!name) continue;

            // Check if this is an adapter LoRA tensor
            // Format: blk.{layer}.{module}.lora_{a|b}.adapter_{id}.weight
            std::string n(name);

            // Split by '.'
            std::vector<std::string> parts;
            {
                std::string part;
                for (char c : n) {
                    if (c == '.') {
                        parts.push_back(std::move(part));
                        part.clear();
                    } else {
                        part += c;
                    }
                }
                parts.push_back(std::move(part));
            }

            // Expected: ["blk", "0", "module", "lora_a/b", "adapter_id", "weight"]
            if (parts.size() < 7) continue;
            if (parts[0] != "blk") continue;

            // Check if this is a lora tensor
            if (parts[3].find("lora_") != 0) continue;

            // Check if this is an adapter tensor
            if (parts[4].find("adapter_") != 0) continue;

            // Extract module name
            std::string module = parts[2];

            // Check if this is a valid LoRA module
            std::vector<std::string> valid_modules = {"attn_qkv", "attn_output", "ffn_gate", "ffn_up", "ffn_down"};
            bool valid = false;
            for (const auto & vm : valid_modules) {
                if (module == vm) { valid = true; break; }
            }
            if (!valid) continue;

            // Extract lora_type (a or b)
            std::string lora_type = parts[3].substr(5);  // Remove "lora_" prefix
            if (lora_type != "lora_a" && lora_type != "lora_b") continue;

            // Extract adapter_id from "adapter_N"
            std::string id_str = parts[4].substr(8);  // Remove "adapter_" prefix
            if (id_str.empty()) continue;

            int adapter_id;
            try {
                adapter_id = std::stoi(id_str);
            } catch (...) {
                continue;
            }
            if (adapter_id < 0 || adapter_id >= (int)hparams.n_adapters) continue;

            LoraTensorInfo info;
            info.module = module;
            info.is_a = (lora_type == "lora_a");
            info.adapter_id = adapter_id;
            info.tensor = ml.get_tensor_meta(name);
            lora_info.push_back(info);
        }

        // Build adapter maps
        std::vector<std::unique_ptr<llama_adapter_lora>> adapter_loras(hparams.n_adapters);

        // Group by module for each adapter
        struct ModuleLora {
            std::string full_key;  // "module.weight"
            ggml_tensor * a = nullptr;
            ggml_tensor * b = nullptr;
        };

        // Use a map keyed by (adapter_id, module) to collect tensors
        std::map<std::pair<int, std::string>, ModuleLora> module_map;

        for (const auto & info : lora_info) {
            auto key = std::make_pair(info.adapter_id, info.module);
            auto & ml_item = module_map[key];
            ml_item.full_key = info.module + ".weight";

            if (info.is_a) {
                ml_item.a = info.tensor;
            } else {
                ml_item.b = info.tensor;
            }
        }

        // Create adapter lora objects
        for (const auto & [key, ml_item] : module_map) {
            int adapter_id = key.first;

            if (!adapter_loras[adapter_id]) {
                adapter_loras[adapter_id] = std::make_unique<llama_adapter_lora>(this);
            }

            if (ml_item.a && ml_item.b) {
                adapter_loras[adapter_id]->ab_map[ml_item.full_key + ".lora_a"] =
                    llama_adapter_lora_weight(ml_item.a, nullptr);
                auto & existing = adapter_loras[adapter_id]->ab_map[ml_item.full_key + ".lora_b"];
                existing.a = nullptr;
                existing.b = ml_item.b;
            }
        }

        // Store embedded adapters
        for (auto & alora : adapter_loras) {
            if (alora && !alora->ab_map.empty()) {
                alora->alpha = 1.0f;
                embedded_loras.push_back(alora.release());
            }
        }
    }

    // Layer 0 is the switch layer - no standard decoder tensors
    // Decoder layers start at index 1
    for (int i = 1; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 1 ? TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 1 ? TENSOR_DUPLICATED : 0));
        }
        else {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 1 ? TENSOR_DUPLICATED : 0));
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_granite_switch::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_granite_switch::graph::graph(
    const llama_model & model,
    const llm_graph_params & params)
    : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // Build input embeddings
    ggml_tensor * inp_pos = nullptr;
    if (hparams.rope_finetuned) {
        inp_pos = build_inp_pos();
    }
    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // For Granite Switch, lora_mask will be applied to decoder layers
    // TODO: In a full implementation, compute adapter_indices from switch layer
    //       and convert to lora_mask here
    ggml_tensor * lora_mask = nullptr;

    // Decoder layers start at index 1 (layer 0 is the switch layer)
    for (int il = 1; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                n_embd_head, hparams.n_head(il), hparams.n_head_kv(il), il);

        if (hparams.rope_finetuned) {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        cur = build_attn(inp_attn,
                model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // ffn
        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

         cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
