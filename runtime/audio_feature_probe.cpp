// Offline feature extraction from the existing frozen audio tower, no language forward pass.
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
static void logger(enum ggml_log_level, const char * text, void *) { std::fputs(text, stderr); }
struct capture { std::vector<float> values; int64_t rows = 0; int calls = 0; };
static bool collect(ggml_tensor * tensor, bool ask, void * data) {
    if (std::strcmp(tensor->name, "ffn_gelu_erf") != 0) { return ask ? false : true; }
    if (ask) { return true; }
    auto & value = *static_cast<capture *>(data);
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] != 1280 || !ggml_is_contiguous(tensor)) { return false; }
    value.rows += tensor->ne[1];
    const size_t offset = value.values.size();
    value.values.resize(offset + ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, value.values.data() + offset, 0, ggml_nbytes(tensor));
    value.calls++;
    return true;
}
int main(int argc, char ** argv) {
    if (argc != 5) { return 1; }
    common_init(); llama_log_set(logger,nullptr); mtmd_helper_log_set(logger,nullptr); llama_backend_init();
    auto lp = llama_model_default_params(); lp.vocab_only = true; lp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1],lp);
    if (!model) { return 2; }
    capture values;
    auto mp = mtmd_context_params_default(); mp.use_gpu = true; mp.warmup = false;
    mp.cb_eval = collect; mp.cb_eval_user_data = &values;
    mtmd_context * ctx = mtmd_init_from_file(argv[2],model,mp);
    if (!ctx || !mtmd_support_audio(ctx)) { return 3; }
    std::cout << "R\n" << std::flush;
    std::string go; if (!std::getline(std::cin,go) || go != "GO") { return 4; }
    std::ifstream paths(argv[3]); std::string path; int index = 0;
    while (std::getline(paths,path)) {
        auto ref = mtmd_helper_bitmap_init_from_file(ctx,path.c_str(),false,mtmd_helper_init_opt_default());
        if (!ref.bitmap) { return 5; }
        const std::string marker = mtmd_default_marker();
        mtmd_input_text text{marker.c_str(),marker.size(),false,true};
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        const mtmd_bitmap * bp = ref.bitmap;
        if (mtmd_tokenize(ctx,chunks,&text,&bp,1) != 0) { return 6; }
        values.values.clear(); values.calls = 0; values.rows = 0; size_t tokens = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const auto * chunk = mtmd_input_chunks_get(chunks,i);
            if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) { continue; }
            if (mtmd_encode_chunk(ctx,chunk) != 0) { return 7; }
            tokens += mtmd_input_chunk_get_n_tokens(chunk);
        }
        mtmd_input_chunks_free(chunks); mtmd_bitmap_free(ref.bitmap);
        if (values.calls < 1 || values.values.empty()) { return 8; }
        const std::string filename = std::string(argv[4]) + "/" + std::to_string(index++) + ".f32";
        std::ofstream out(filename,std::ios::binary);
        out.write(reinterpret_cast<const char *>(values.values.data()),values.values.size()*sizeof(float));
        if (!out.good()) { return 9; }
        std::cout << values.rows << " " << tokens << "\n" << std::flush;
    }
    mtmd_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
}
