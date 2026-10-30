// Evaluation helper for the existing Qwen3-TTS projected speaker encoder.
// Use the scoped Python qualifier; it proves GPU residency before sending GO.
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
static void logger(enum ggml_log_level, const char * text, void *) { std::fputs(text, stderr); }
int main(int argc,char **argv) {
    if(argc!=5) return 1;
    common_init(); llama_log_set(logger,nullptr); mtmd_helper_log_set(logger,nullptr); llama_backend_init();
    auto params=llama_model_default_params(); params.n_gpu_layers=-1;
    llama_model * model=llama_model_load_from_file(argv[1],params);
    if(!model) return 2;
    auto mp=mtmd_context_params_default(); mp.use_gpu=true;
    mtmd_context * ctx=mtmd_init_from_file(argv[2],model,mp);
    if(!ctx || !mtmd_support_audio(ctx)) return 3;
    std::cout << "R\n" << std::flush;
    std::string go; if(!std::getline(std::cin,go)||go!="GO") return 4;
    std::ifstream paths(argv[3]); std::ofstream output(argv[4],std::ios::binary);
    std::string path;
    while(std::getline(paths,path)) {
        auto ref=mtmd_helper_bitmap_init_from_file(ctx,path.c_str(),false,mtmd_helper_init_opt_default());
        if(!ref.bitmap) return 5;
        const std::string marker=mtmd_default_marker();
        mtmd_input_text text{marker.c_str(),marker.size(),false,true};
        mtmd_input_chunks * chunks=mtmd_input_chunks_init();
        const mtmd_bitmap * bp=ref.bitmap;
        if(mtmd_tokenize(ctx,chunks,&text,&bp,1)!=0) return 6;
        bool found=false;
        for(size_t i=0;i<mtmd_input_chunks_size(chunks);++i) {
            const auto * chunk=mtmd_input_chunks_get(chunks,i);
            if(mtmd_input_chunk_get_type(chunk)!=MTMD_INPUT_CHUNK_TYPE_AUDIO) continue;
            if(mtmd_encode_chunk(ctx,chunk)!=0) return 7;
            const size_t width=llama_model_n_embd_inp(model)*mtmd_input_chunk_get_n_tokens(chunk);
            if(width!=2048) return 8;
            output.write(reinterpret_cast<const char *>(mtmd_get_output_embd(ctx)),width*sizeof(float));
            found=true; break;
        }
        mtmd_input_chunks_free(chunks); mtmd_bitmap_free(ref.bitmap);
        if(!found) return 9;
    }
    output.close(); mtmd_free(ctx); llama_model_free(model); llama_backend_free();
    return output.good()?0:10;
}
