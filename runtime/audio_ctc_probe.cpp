// Execute the qualified small head on native cached encoder features, with CUDA only.
// This qualification tool neither loads a reasoning model nor enables a runtime ASR route.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool valid, const char * message) {
    if (!valid) { throw std::runtime_error(message); }
}

struct head {
    gguf_context * file = nullptr;
    ggml_context * weights = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ~head() {
        if (allocator) { ggml_gallocr_free(allocator); }
        if (buffer) { ggml_backend_buffer_free(buffer); }
        if (weights) { ggml_free(weights); }
        if (file) { gguf_free(file); }
        if (backend) { ggml_backend_free(backend); }
    }
    ggml_tensor * tensor(const char * name) {
        auto * value = ggml_get_tensor(weights, name);
        require(value != nullptr && value->type == GGML_TYPE_F32, "missing or non-F32 head tensor");
        return value;
    }
    void load(const char * path) {
        gguf_init_params params{true, &weights};
        file = gguf_init_from_file(path, params);
        require(file != nullptr, "cannot load head metadata");
        auto schema = gguf_find_key(file,"robit.audio_bundle.schema");
        if (schema >= 0) {
            require(gguf_get_kv_type(file,schema) == GGUF_TYPE_STRING &&
                std::strcmp(gguf_get_val_str(file,schema),"robit.ollama-audio-bridge.v2") == 0,
                "unsupported indexed bundle");
            require(gguf_select_tensor_view(file,"a.h.c.",
                "robit.audio_bundle.component.audio_ctc_head.kv.",&weights),
                "bundle has no valid audio CTC head view");
        }
        auto key = gguf_find_key(file, "general.architecture");
        require(key >= 0 && gguf_get_kv_type(file, key) == GGUF_TYPE_STRING, "missing architecture");
        require(std::strcmp(gguf_get_val_str(file, key), "robit_audio_ctc") == 0, "wrong head architecture");
        require(gguf_get_n_tensors(file) == 8, "wrong head inventory");
        const struct { const char * name; int64_t x, y, z; } shapes[] = {
            {"norm.weight",1280,1,1}, {"norm.bias",1280,1,1},
            {"input.weight",1280,512,1}, {"input.bias",512,1,1},
            {"temporal.weight",3,512,512}, {"temporal.bias",512,1,1},
            {"output.weight",512,2048,1}, {"output.bias",2048,1,1}};
        for (const auto & shape : shapes) {
            auto * t = tensor(shape.name);
            require(t->ne[0] == shape.x && t->ne[1] == shape.y && t->ne[2] == shape.z && t->ne[3] == 1, "wrong head shape");
        }
        ggml_backend_load_all();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto device = ggml_backend_dev_get(i);
            if (std::strncmp(ggml_backend_dev_name(device), "CUDA", 4) == 0) {
                require(backend == nullptr, "qualification requires one visible CUDA device");
                backend = ggml_backend_dev_init(device, nullptr);
            }
        }
        require(backend != nullptr, "CUDA backend required; no CPU fallback");
        buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
        require(buffer != nullptr, "head weight allocation failed");
        std::ifstream input(path, std::ios::binary);
        for (int64_t i = 0; i < gguf_get_n_tensors(file); ++i) {
            auto * t = tensor(gguf_get_tensor_name(file, i));
            std::vector<char> raw(ggml_nbytes(t));
            input.seekg(gguf_get_data_offset(file) + gguf_get_tensor_offset(file, i));
            input.read(raw.data(), raw.size());
            require(bool(input), "head tensor read failed");
            ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
        }
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    ggml_tensor * matmul(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
        // An extra zero input dimension preserves the trained operation while
        // avoiding the pinned backend's aligned custom TF32 MMA fast path.
        auto * y = ggml_mul_mat(ctx,ggml_pad(ctx,w,1,0,0,0),ggml_pad(ctx,x,1,0,0,0));
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return y;
    }
    std::vector<float> compute(const std::vector<float> & input) {
        const int64_t frames = input.size()/1280;
        require(frames >= 1 && frames <= 640 && input.size()%1280 == 0, "invalid feature dimensions");
        ggml_init_params params{2*1024*1024, nullptr, true};
        auto * ctx = ggml_init(params);
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1280, frames);
        ggml_set_input(x);
        auto * normalized = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx,x,1e-5f), tensor("norm.weight")), tensor("norm.bias"));
        auto * h = ggml_gelu_erf(ctx, ggml_add(ctx, matmul(ctx,tensor("input.weight"),normalized), tensor("input.bias")));
        auto * transposed = ggml_cont(ctx, ggml_transpose(ctx,h));
        auto * columns = ggml_im2col(ctx,tensor("temporal.weight"),transposed,1,0,2,0,1,0,false,GGML_TYPE_F32);
        auto * convolution = matmul(ctx,ggml_reshape_2d(ctx,tensor("temporal.weight"),1536,512),ggml_reshape_2d(ctx,columns,1536,frames+2));
        auto * causal = ggml_view_2d(ctx,convolution,512,frames,convolution->nb[1],0);
        auto * biased = ggml_add(ctx,causal,tensor("temporal.bias"));
        auto * combined = ggml_gelu_erf(ctx,ggml_add(ctx,h,biased));
        auto * output = ggml_add(ctx,matmul(ctx,tensor("output.weight"),combined),tensor("output.bias"));
        ggml_set_output(output);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,output);
        require(ggml_gallocr_alloc_graph(allocator,graph), "head graph allocation failed");
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        require(ggml_backend_graph_compute(backend,graph) == GGML_STATUS_SUCCESS, "CUDA head evaluation failed");
        std::vector<float> logits(2048*frames);
        ggml_backend_tensor_get(output,logits.data(),0,logits.size()*sizeof(float));
        ggml_free(ctx);
        return logits;
    }
};

int main(int argc, char ** argv) {
    if (argc != 4) { return 1; }
    try {
        head model; model.load(argv[1]);
        model.compute(std::vector<float>(640*1280,0)); // Reserve the maximum qualification graph before readiness.
        std::cout << "R\n" << std::flush;
        std::string go; if (!std::getline(std::cin,go) || go != "GO") { return 2; }
        std::ifstream paths(argv[2]); std::string path; int index = 0;
        while (std::getline(paths,path)) {
            std::ifstream input(path,std::ios::binary|std::ios::ate);
            require(bool(input),"cannot read feature file");
            const auto bytes = input.tellg();
            require(bytes > 0 && bytes <= 640*1280*4 && bytes%(1280*4) == 0, "invalid feature file size");
            std::vector<float> values(size_t(bytes)/sizeof(float));
            input.seekg(0); input.read(reinterpret_cast<char *>(values.data()),bytes);
            require(bool(input),"feature read failed");
            auto started = std::chrono::steady_clock::now();
            auto logits = model.compute(values);
            auto elapsed = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            const auto name = std::string(argv[3])+"/"+std::to_string(index++)+".f32";
            std::ofstream output(name,std::ios::binary);
            output.write(reinterpret_cast<const char *>(logits.data()),logits.size()*sizeof(float));
            require(bool(output),"cannot write logits");
            std::cout << values.size()/1280 << " " << elapsed << "\n" << std::flush;
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr,"%s\n",error.what()); return 3;
    }
    return 0;
}
