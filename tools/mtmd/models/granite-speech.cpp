#include "models.h"

#include <cmath>
#include <algorithm>

// Build Shaw's relative position attention scores
// Computes: pos_attn = einsum("h c d, c r d -> h c r", Q, rel_pos_emb)
// rel_pos_emb parameter has shape [2*context-1, d_head]
// Q has shape [d_head, n_head, seq_len]
// Returns pos_attn [seq_len_q, seq_len_k, n_head] (to match content attention shape)
ggml_tensor * clip_graph_granite_speech::build_shaw_rel_pos_attn(
        ggml_tensor * rel_pos_emb,
        ggml_tensor * Q,  // [d_head, n_head, seq_len]
        int seq_len,
        int il) {
    // rel_pos_emb: [2*context-1, d_head] e.g. [399, 128]

    // Build relative position indices tensor (filled in by clip_image_batch_encode)
    // For positions i,j: index = clamp(i - j + context_size - 1, 0, 2*context-2)
    ggml_tensor * rel_pos_idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, seq_len, seq_len);
    ggml_set_name(rel_pos_idx, "rel_pos_idx");
    ggml_set_input(rel_pos_idx);

    // Gather relative position embeddings: [seq_len*seq_len, d_head]
    ggml_tensor * rel_pos_gathered = ggml_get_rows(ctx0, rel_pos_emb,
                                                    ggml_reshape_1d(ctx0, rel_pos_idx, seq_len * seq_len));

    // Reshape gathered embeddings to [d_head, seq_len_k, seq_len_q]
    // The indices are laid out as [q][k] so after gather we have [q*k, d_head]
    // Reshape to [d_head, seq_len_k, seq_len_q]
    rel_pos_gathered = ggml_reshape_3d(ctx0, rel_pos_gathered, d_head, seq_len, seq_len);

    // Compute pos_attn = einsum("h c d, c r d -> h c r", Q, rel_pos)
    // where h=n_head, c=seq_len_q (query position), r=seq_len_k (key position), d=d_head
    //
    // We need to compute: for each query position c:
    //   pos_attn[h, c, r] = sum_d(Q[h, c, d] * rel_pos[c, r, d])
    //
    // ggml_mul_mat(A, B) computes C[i,j,k] = sum_m(A[m,i,k] * B[m,j,k])
    // i.e., contracts dim 0, batches over dim 2
    //
    // rel_pos_gathered is [d_head, seq_len_k, seq_len_q]
    // We need to process per-query: for query c, we want rel_pos[:, :, c] with shape [d_head, seq_len_k]
    // But ggml_mul_mat batches over the LAST dimension, so we need rel_pos as [d_head, seq_len_k, n_head]
    // This doesn't directly work because rel_pos doesn't have n_head dimension.
    //
    // Alternative approach: repeat rel_pos for each head, or restructure the computation.
    // Since rel_pos is the same for all heads, we can:
    // 1. Compute Q @ rel_pos.T for all positions, then broadcast across heads
    //
    // Actually, the cleanest approach is to compute this per-head using a loop conceptually,
    // but in practice we can use broadcasting.
    //
    // Simpler approach: reshape and use mul_mat with proper broadcasting
    // For each position c, we want: Q[:, c, :].T @ rel_pos[:, :, c]
    // = [n_head, d_head] @ [d_head, seq_len_k] = [n_head, seq_len_k]
    //
    // If we permute rel_pos to [d_head, seq_len_q, seq_len_k] (swapping last two dims)
    // Then batch over seq_len_q with Q_rearranged [d_head, seq_len_q, n_head]:
    // But the batch dims don't match (seq_len_k vs n_head).

    // Let's try a different approach: compute the einsum by expanding and element-wise ops
    // This is less efficient but correct.
    //
    // Q_expanded: [d_head, n_head, seq_len_q, 1] -> broadcast to [d_head, n_head, seq_len_q, seq_len_k]
    // rel_pos_expanded: [d_head, 1, seq_len_q, seq_len_k] (after permute)
    // Multiply and sum over d_head

    // Permute rel_pos from [d_head, seq_len_k, seq_len_q] to [d_head, seq_len_q, seq_len_k]
    ggml_tensor * rel_pos_perm = ggml_cont(ctx0, ggml_permute(ctx0, rel_pos_gathered, 0, 2, 1, 3));
    // rel_pos_perm: [d_head, seq_len_q, seq_len_k]

    // Expand Q to 4D: [d_head, n_head, seq_len_q, 1]
    ggml_tensor * Q_4d = ggml_reshape_4d(ctx0, Q, d_head, n_head, seq_len, 1);

    // Expand rel_pos to 4D: [d_head, n_head, seq_len_q, seq_len_k]
    ggml_tensor * rel_pos_4d = ggml_repeat_4d(ctx0,
        ggml_reshape_4d(ctx0, rel_pos_perm, d_head, 1, seq_len, seq_len),
        d_head, n_head, seq_len, seq_len);

    // Element-wise multiply with broadcasting: [d_head, n_head, seq_len_q, seq_len_k]
    ggml_tensor * product = ggml_mul(ctx0, rel_pos_4d, Q_4d);

    // Sum over d_head (dimension 0) to get [n_head, seq_len_q, seq_len_k]
    ggml_tensor * pos_attn = ggml_sum_rows(ctx0, product);
    // sum_rows sums over dim 0, result: [1, n_head, seq_len_q, seq_len_k]
    // pos_attn = ggml_reshape_3d(ctx0, pos_attn, n_head, seq_len, seq_len);

    // Permute to match content attention shape [seq_len_q, seq_len_k, n_head]
    pos_attn = ggml_cont(ctx0, ggml_permute(ctx0, pos_attn, 3, 2, 1, 0));

    cb(pos_attn, "shaw_pos_attn.{}", il);
    return pos_attn;
}

// Build Q-Former layer: self-attention + cross-attention + FFN
ggml_tensor * clip_graph_granite_speech::build_qformer_layer(
        ggml_tensor * queries,
        ggml_tensor * encoder_out,
        const qformer_layer & layer,
        int il) {
    const int qf_n_head = 16;  // BLIP-2 Q-Former typically uses 16 heads
    const int qf_d_head = queries->ne[0] / qf_n_head;
    const float qf_kq_scale = 1.0f / sqrtf((float)qf_d_head);

    // Self-attention on queries
    ggml_tensor * residual = queries;
    {
        ggml_tensor * Q = ggml_mul_mat(ctx0, layer.self_attn_q_w, queries);
        if (layer.self_attn_q_b) {
            Q = ggml_add(ctx0, Q, layer.self_attn_q_b);
        }

        ggml_tensor * K = ggml_mul_mat(ctx0, layer.self_attn_k_w, queries);
        if (layer.self_attn_k_b) {
            K = ggml_add(ctx0, K, layer.self_attn_k_b);
        }

        ggml_tensor * V = ggml_mul_mat(ctx0, layer.self_attn_v_w, queries);
        if (layer.self_attn_v_b) {
            V = ggml_add(ctx0, V, layer.self_attn_v_b);
        }

        Q = ggml_reshape_3d(ctx0, Q, qf_d_head, qf_n_head, Q->ne[1]);
        K = ggml_reshape_3d(ctx0, K, qf_d_head, qf_n_head, K->ne[1]);
        V = ggml_reshape_3d(ctx0, V, qf_d_head, qf_n_head, V->ne[1]);

        ggml_tensor * attn_out = build_attn(
            layer.self_attn_o_w, layer.self_attn_o_b,
            Q, K, V, nullptr, qf_kq_scale, il);

        // Residual + LayerNorm
        queries = ggml_add(ctx0, residual, attn_out);
        queries = build_norm(queries, layer.self_attn_ln_w, layer.self_attn_ln_b,
                            NORM_TYPE_NORMAL, hparams.proj_layernorm_eps, il);
        cb(queries, "qformer.{}.self_attn", il);
    }

    // Cross-attention: queries attend to encoder output
    residual = queries;
    {
        ggml_tensor * Q = ggml_mul_mat(ctx0, layer.cross_attn_q_w, queries);
        if (layer.cross_attn_q_b) {
            Q = ggml_add(ctx0, Q, layer.cross_attn_q_b);
        }

        ggml_tensor * K = ggml_mul_mat(ctx0, layer.cross_attn_k_w, encoder_out);
        if (layer.cross_attn_k_b) {
            K = ggml_add(ctx0, K, layer.cross_attn_k_b);
        }

        ggml_tensor * V = ggml_mul_mat(ctx0, layer.cross_attn_v_w, encoder_out);
        if (layer.cross_attn_v_b) {
            V = ggml_add(ctx0, V, layer.cross_attn_v_b);
        }

        Q = ggml_reshape_3d(ctx0, Q, qf_d_head, qf_n_head, Q->ne[1]);
        K = ggml_reshape_3d(ctx0, K, qf_d_head, qf_n_head, K->ne[1]);
        V = ggml_reshape_3d(ctx0, V, qf_d_head, qf_n_head, V->ne[1]);

        ggml_tensor * attn_out = build_attn(
            layer.cross_attn_o_w, layer.cross_attn_o_b,
            Q, K, V, nullptr, qf_kq_scale, il);

        // Residual + LayerNorm
        queries = ggml_add(ctx0, residual, attn_out);
        queries = build_norm(queries, layer.cross_attn_ln_w, layer.cross_attn_ln_b,
                            NORM_TYPE_NORMAL, hparams.proj_layernorm_eps, il);
        cb(queries, "qformer.{}.cross_attn", il);
    }

    // FFN
    residual = queries;
    {
        queries = build_ffn(queries,
                           layer.ffn_up_w, layer.ffn_up_b,
                           nullptr, nullptr,
                           layer.ffn_down_w, layer.ffn_down_b,
                           FFN_GELU, il);

        // Residual + LayerNorm
        queries = ggml_add(ctx0, residual, queries);
        queries = build_norm(queries, layer.ffn_ln_w, layer.ffn_ln_b,
                            NORM_TYPE_NORMAL, hparams.proj_layernorm_eps, il);
        cb(queries, "qformer.{}.ffn", il);
    }

    return queries;
}

ggml_cgraph * clip_graph_granite_speech::build() {
    // Input: frame-stacked mel spectrogram [n_features=160, n_frames]
    const int n_frames = img.nx;
    const int n_features = img.ny;  // Should be 160 after frame stacking

    ggml_tensor * inp = build_inp_raw(1);
    cb(inp, "input", -1);

    // Input is [n_frames, n_features, 1] from build_inp_raw
    // Transpose to [n_features, n_frames] for linear projection
    ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, inp, n_frames, n_features)));
    cb(cur, "input_transposed", -1);

    // Linear input projection: [160, n_frames] -> [1024, n_frames]
    cur = ggml_mul_mat(ctx0, model.input_proj_w, cur);
    if (model.input_proj_b) {
        cur = ggml_add(ctx0, cur, model.input_proj_b);
    }
    cb(cur, "input_proj", -1);

    const int seq_len = cur->ne[1];

    // Layer with extra outputs in the middle
    const int mid_layer = hparams.n_layer / 2;

    // Conformer encoder layers
    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = cur;
        cb(cur, "encoder.layer.{}.in", il);

        // Feed-forward 1 (half-step)
        {
            ggml_tensor * ff1 = build_norm(cur, layer.ff_norm_w, layer.ff_norm_b, NORM_TYPE_NORMAL, eps, il);
            cb(ff1, "encoder.{}.ff1_norm", il);

            ff1 = build_ffn(ff1,
                           layer.ff_up_w, layer.ff_up_b,
                           nullptr, nullptr,
                           layer.ff_down_w, layer.ff_down_b,
                           FFN_SILU, il);
            cb(ff1, "encoder.{}.ff1", il);

            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, ff1, 0.5f));
        }

        // Multi-head self-attention with Shaw's relative position
        {
            ggml_tensor * attn_in = build_norm(residual, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
            cb(attn_in, "encoder.{}.attn_norm", il);

            // Q, K, V projections
            ggml_tensor * Q = ggml_mul_mat(ctx0, layer.q_w, attn_in);
            if (layer.q_b) {
                Q = ggml_add(ctx0, Q, layer.q_b);
            }
            Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, seq_len);

            ggml_tensor * K = ggml_mul_mat(ctx0, layer.k_w, attn_in);
            if (layer.k_b) {
                K = ggml_add(ctx0, K, layer.k_b);
            }
            K = ggml_reshape_3d(ctx0, K, d_head, n_head, seq_len);

            ggml_tensor * V = ggml_mul_mat(ctx0, layer.v_w, attn_in);
            if (layer.v_b) {
                V = ggml_add(ctx0, V, layer.v_b);
            }
            V = ggml_reshape_3d(ctx0, V, d_head, n_head, seq_len);

            // Permute for attention computation
            // Q, K: [d_head, n_head, seq_len] -> permute for matmul
            ggml_tensor * Q_perm = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));  // [d_head, seq_len, n_head]
            ggml_tensor * K_perm = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));  // [d_head, seq_len, n_head]
            ggml_tensor * V_perm = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // [seq_len, n_head, d_head]

            // Content attention: Q @ K^T
            // mul_mat(K, Q) with batching over n_head gives [seq_len, seq_len, n_head]
            ggml_tensor * content_attn = ggml_mul_mat(ctx0, K_perm, Q_perm);
            cb(content_attn, "encoder.{}.content_attn", il);

            // Shaw's relative position attention: einsum("h c d, c r d -> h c r", Q, rel_pos)
            // Returns [seq_len_query, seq_len_key, n_head]
            ggml_tensor * pos_attn = build_shaw_rel_pos_attn(layer.rel_pos_emb_w, Q, seq_len, il);

            // Combine content and position attention, then scale
            ggml_tensor * scores = ggml_add(ctx0, content_attn, pos_attn);
            scores = ggml_scale(ctx0, scores, kq_scale);
            cb(scores, "encoder.{}.attn_scores", il);

            ggml_tensor * attn = ggml_soft_max(ctx0, scores);
            ggml_tensor * attn_out = ggml_mul_mat(ctx0, attn, V_perm);
            attn_out = ggml_permute(ctx0, attn_out, 2, 0, 1, 3);
            attn_out = ggml_cont_2d(ctx0, attn_out, n_embd, seq_len);

            // Output projection
            attn_out = ggml_mul_mat(ctx0, layer.o_w, attn_out);
            if (layer.o_b) {
                attn_out = ggml_add(ctx0, attn_out, layer.o_b);
            }
            cb(attn_out, "encoder.{}.attn_out", il);

            residual = ggml_add(ctx0, residual, attn_out);
        }

        // Post-attention norm (if present)
        if (layer.ln_2_w) {
            residual = build_norm(residual, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        }

        // Convolution module
        {
            ggml_tensor * conv_in = build_norm(residual, layer.norm_conv_w, layer.norm_conv_b, NORM_TYPE_NORMAL, eps, il);
            cb(conv_in, "encoder.{}.conv_norm", il);

            // Up projection (GLU gate)
            // conv_up_w pre-reshaped at conversion: [in_ch, out_ch] = [1024, 4096]
            // conv_in is [n_embd=1024, seq_len]
            // mul_mat([1024, 4096], [1024, seq_len]) contracts ne[0]=1024, result is [4096, seq_len]
            ggml_tensor * conv = ggml_mul_mat(ctx0, layer.conv_up_w, conv_in);
            if (layer.conv_up_b) {
                conv = ggml_add(ctx0, conv, layer.conv_up_b);
            }
            cb(conv, "encoder.{}.conv_up", il);

            // GLU: split and apply sigmoid gate
            {
                int64_t d = conv->ne[0] / 2;
                ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, conv, d, conv->ne[1], conv->nb[1], d * conv->nb[0]));
                conv = ggml_mul(ctx0, ggml_view_2d(ctx0, conv, d, conv->ne[1], conv->nb[1], 0), gate);
            }
            cb(conv, "encoder.{}.conv_glu", il);

            // Transpose for 1D convolution: [channels, seq_len] -> [seq_len, channels]
            conv = ggml_cont(ctx0, ggml_transpose(ctx0, conv));

            // Depthwise convolution with same padding (kernel_size typically 31)
            // Using ggml_ssm_conv with manual padding (same as conformer.cpp pattern)
            const int kernel_size = layer.conv_dw_w->ne[0];
            const int pad = kernel_size / 2;
            conv = ggml_pad(ctx0, conv, pad, 0, 0, 0);
            conv = ggml_roll(ctx0, conv, pad, 0, 0, 0);
            conv = ggml_pad(ctx0, conv, pad, 0, 0, 0);
            conv = ggml_ssm_conv(ctx0, conv, layer.conv_dw_w);
            if (layer.conv_dw_b) {
                conv = ggml_add(ctx0, conv, layer.conv_dw_b);
            }
            cb(conv, "encoder.{}.conv_dw", il);

            // Batch norm (weights already folded into conv_norm)
            conv = ggml_add(ctx0, ggml_mul(ctx0, conv, layer.conv_norm_w), layer.conv_norm_b);
            conv = ggml_silu(ctx0, conv);

            // Down projection
            // conv_down_w pre-reshaped at conversion: [in_ch, out_ch]
            conv = ggml_mul_mat(ctx0, layer.conv_down_w, conv);
            if (layer.conv_down_b) {
                conv = ggml_add(ctx0, conv, layer.conv_down_b);
            }
            cb(conv, "encoder.{}.conv", il);

            residual = ggml_add(ctx0, residual, conv);
        }

        // Feed-forward 2 (half-step)
        {
            ggml_tensor * ff2 = build_norm(residual, layer.ff_norm_1_w, layer.ff_norm_1_b, NORM_TYPE_NORMAL, eps, il);
            cb(ff2, "encoder.{}.ff2_norm", il);

            ff2 = build_ffn(ff2,
                           layer.ff_up_1_w, layer.ff_up_1_b,
                           nullptr, nullptr,
                           layer.ff_down_1_w, layer.ff_down_1_b,
                           FFN_SILU, il);
            cb(ff2, "encoder.{}.ff2", il);

            cur = ggml_add(ctx0, residual, ggml_scale(ctx0, ff2, 0.5f));
        }

        cb(cur, "encoder.layer.{}.out", il);

        // Perform extra outputs for middle layer
        if (il == mid_layer - 1) {
            ggml_tensor * hidden_states_mid = ggml_mul_mat(ctx0, model.pre_encode_out_w, cur);
            if (model.pre_encode_out_b) {
                hidden_states_mid = ggml_add(ctx0, hidden_states_mid, model.pre_encode_out_b);
            }
            hidden_states_mid = ggml_soft_max(ctx0, hidden_states_mid);
            hidden_states_mid = ggml_mul_mat(ctx0, model.out_mid_w, hidden_states_mid);
            if (model.out_mid_b) {
                hidden_states_mid = ggml_add(ctx0, hidden_states_mid, model.out_mid_b);
            }
            cur = ggml_add(ctx0, cur, hidden_states_mid);
            cb(cur, "middle.out", il);
        }
    }

    // Q-Former projector
    {
        const int window_size = hparams.proj_window_size > 0 ? hparams.proj_window_size : 15;
        const int downsample_rate = hparams.proj_downsample_rate > 0 ? hparams.proj_downsample_rate : 5;
        const int n_queries_per_window = window_size / downsample_rate;  // 3

        // Pad encoder output to multiple of window_size
        const int n_enc_frames = cur->ne[1];
        const int n_pad = (window_size - (n_enc_frames % window_size)) % window_size;
        if (n_pad > 0) {
            cur = ggml_pad(ctx0, cur, 0, n_pad, 0, 0);
        }
        const int n_windows = (n_enc_frames + n_pad) / window_size;
        const int n_total_queries = n_queries_per_window * n_windows;

        // Reshape encoder output for windowed attention
        // [n_embd=256, n_frames] -> [n_embd, window_size, n_windows]
        ggml_tensor * encoder_windows = ggml_reshape_3d(ctx0, cur, cur->ne[0], window_size, n_windows);
        cb(encoder_windows, "qformer.encoder_windows", -1);

        // Input LayerNorm on encoder output
        encoder_windows = build_norm(ggml_reshape_2d(ctx0, encoder_windows, cur->ne[0], window_size * n_windows),
                                     model.qf_ln_w, model.qf_ln_b,
                                     NORM_TYPE_NORMAL, hparams.proj_layernorm_eps, -1);
        encoder_windows = ggml_reshape_3d(ctx0, encoder_windows, cur->ne[0], window_size, n_windows);

        // Expand query embeddings for all windows
        // model.qf_query: [qf_hidden=1024, n_queries_per_window=3]
        // Repeat for each window: [qf_hidden, n_queries_per_window, n_windows]
        ggml_tensor * queries = ggml_repeat_4d(ctx0, model.qf_query,
                                               model.qf_query->ne[0],
                                               model.qf_query->ne[1],
                                               n_windows, 1);
        cb(queries, "qformer.queries", -1);

        // Process through Q-Former layers
        for (int il = 0; il < hparams.proj_n_layer; il++) {
            const auto & layer = model.qformer_layers[il];

            // For windowed cross-attention, we process each window independently
            // Reshape for per-window processing
            queries = ggml_reshape_2d(ctx0, queries, queries->ne[0], n_total_queries);
            ggml_tensor * enc_flat = ggml_reshape_2d(ctx0, encoder_windows, encoder_windows->ne[0], window_size * n_windows);

            queries = build_qformer_layer(queries, enc_flat, layer, il);

            queries = ggml_reshape_3d(ctx0, queries, queries->ne[0], n_queries_per_window, n_windows);
        }

        // Flatten queries: [qf_hidden, n_queries_per_window, n_windows] -> [qf_hidden, n_total_queries]
        queries = ggml_reshape_2d(ctx0, queries, queries->ne[0], n_total_queries);
        cb(queries, "qformer.queries_out", -1);

        // Final projection to LLM hidden size: [qf_hidden=1024] -> [llm_hidden=2048]
        cur = ggml_mul_mat(ctx0, model.qf_out_w, queries);
        if (model.qf_out_b) {
            cur = ggml_add(ctx0, cur, model.qf_out_b);
        }
        cb(cur, "qformer.projected", -1);
    }

    cb(cur, "projected", -1);
    ggml_build_forward_expand(gf, cur);

    return gf;
}
