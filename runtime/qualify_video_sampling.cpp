// Exercise the actual server parser, sampler and tokenizer with a vocabulary
// and CUDA projector. No language-model forward pass is performed.
#include "server-common.h"
#include "ggml-backend.h"
#include <csignal>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>

int main(int argc, char ** argv) {
    if (argc != 6) return 2;
    std::signal(SIGPIPE, SIG_IGN);
    ggml_backend_load_all(); llama_backend_init();
    auto params = llama_model_default_params(); params.vocab_only = true;
    auto * model = llama_model_load_from_file(argv[1], params);
    if (!model) return 3;
    auto options = mtmd_context_params_default();
    options.use_gpu = true; options.warmup = false; options.n_threads = 4;
    options.media_marker = get_media_marker();
    auto * ctx = mtmd_init_from_file(argv[2], model, options);
    if (!ctx) return 4;
    { std::ofstream marker(argv[5]); marker << getpid(); }
    bool admitted = false;
    for (int attempt = 0; attempt < 1200; ++attempt) {
        if (std::ifstream(std::string(argv[5]) + ".go").good()) { admitted = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!admitted) return 5;
    server_chat_params chat = {};
    chat.use_jinja = true; chat.allow_image = true; chat.allow_audio = true; chat.allow_video = true;
    chat.tmpls = common_chat_templates_init(model, "");
    chat.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    auto init = mtmd_helper_init_opt_default(); init.video_params.ffmpeg_bin_dir = argv[4];
    std::ifstream request_file(argv[3]);
    std::string request_text((std::istreambuf_iterator<char>(request_file)), {});
    json requests = json::parse(request_text);
    json results = json::array();
    for (const auto & request : requests) {
        json result = {{"case", request.at("case")}};
        try {
            auto body = request.at("body"); std::vector<raw_buffer> files;
            auto parsed = oaicompat_chat_params_parse(body, chat, files);
            auto sampling = json_value(parsed, "media_video_sampling", json::object());
            auto tokens = process_mtmd_prompt(ctx, parsed.at("prompt").get<std::string>(), files, init, true, sampling);
            result["input_tokens"] = tokens.size(); result["sampling_by_file"] = sampling;
            result["videos"] = json::array();
            for (size_t index = 0; index < files.size(); ++index) {
                auto video_options = init.video_params; auto key = std::to_string(index);
                if (sampling.contains(key)) {
                    const auto & entry = sampling.at(key);
                    video_options.fps_target = json_value(entry, "fps", video_options.fps_target);
                    video_options.max_frames = json_value(entry, "max_frames", 0);
                    video_options.align_timestamps = true; video_options.timestamp_interval_ms = 1;
                }
                auto * video = mtmd_helper_video_init_from_buf(ctx, files[index].data(), files[index].size(), video_options);
                if (!video) throw std::runtime_error("video decoder failed");
                auto info = mtmd_helper_video_get_info(video);
                json frames = json::array(), timestamps = json::array();
                mtmd_bitmap * bitmap = nullptr; char * text = nullptr;
                while (mtmd_helper_video_read_next(video, &bitmap, &text) == 0) {
                    if (text) { timestamps.push_back(text); free(text); }
                    if (bitmap) {
                        const auto * pixels = mtmd_bitmap_get_data(bitmap);
                        auto count = mtmd_bitmap_get_n_bytes(bitmap); double rgb[3] = {};
                        for (size_t i = 0; i + 2 < count; i += 3) for (int channel = 0; channel < 3; ++channel) rgb[channel] += pixels[i + channel];
                        for (double & value : rgb) value /= (double)count / 3;
                        json channels = json::array();
                        for (double value : rgb) channels.push_back(value);
                        frames.push_back(channels); mtmd_bitmap_free(bitmap);
                    }
                }
                mtmd_helper_video_free(video);
                result["videos"].push_back({{"fps", info.fps}, {"frames_rgb_mean", frames}, {"timestamp_text", timestamps}});
            }
            result["accepted"] = true;
        } catch (const std::exception & error) {
            result["accepted"] = false; result["error"] = error.what();
        }
        results.push_back(std::move(result));
    }
    std::cout << results.dump(2) << std::endl;
    mtmd_free(ctx); llama_model_free(model); llama_backend_free();
}
