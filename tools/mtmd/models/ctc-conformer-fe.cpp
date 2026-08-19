#include "models.h"

// ctc-conformer front-end: the real encoder (conformer blocks, subsampling, CTC head) lives in
// the paired native llama_model (src/models/ctc-conformer.cpp) instead - this mmproj is
// "transformer-less" (see conversion/granite.py's CtcConformerFrontendMmprojModel), doing only
// the single learned input_linear projection up to the native model's working hidden size,
// mirroring gemma4ua's single-projection audio front-end.
ggml_cgraph * clip_graph_ctc_conformer_fe::build() {
    ggml_tensor * inp = build_inp_raw(1);
    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    cur = build_mm(model.inp_proj_w, cur);
    cur = ggml_add(ctx0, cur, model.inp_proj_b);
    cb(cur, "inp_linear", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
