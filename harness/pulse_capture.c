#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pulse/pulseaudio.h>

static bool write_pcm(const void *data, size_t bytes) {
    if (data) {
        return fwrite(data, 1, bytes, stdout) == bytes;
    }
    /* A PulseAudio hole has a length but no samples. Preserve its duration. */
    const char silence[4096] = {0};
    while (bytes) {
        size_t chunk = bytes > sizeof silence ? sizeof silence : bytes;
        if (fwrite(silence, 1, chunk, stdout) != chunk) {
            return false;
        }
        bytes -= chunk;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 4 || !argv[1][0]) {
        fprintf(stderr, "Usage: %s SOURCE SAMPLE_RATE CHANNELS\n", argv[0]);
        return 2;
    }
    char *end;
    unsigned long rate = strtoul(argv[2], &end, 10);
    if (*end || rate < 1 || rate > PA_RATE_MAX) {
        return 2;
    }
    unsigned long channels = strtoul(argv[3], &end, 10);
    if (*end || channels < 1 || channels > PA_CHANNELS_MAX) {
        return 2;
    }
    setvbuf(stdout, NULL, _IONBF, 0);

    pa_mainloop *loop = pa_mainloop_new();
    if (!loop) {
        return 1;
    }
    pa_context *context = pa_context_new(pa_mainloop_get_api(loop), "Omni pinned microphone");
    pa_stream *stream = NULL;
    if (!context || pa_context_connect(context, NULL, PA_CONTEXT_NOFLAGS, NULL) < 0) {
        goto done;
    }
    while (pa_context_get_state(context) != PA_CONTEXT_READY) {
        if (!PA_CONTEXT_IS_GOOD(pa_context_get_state(context)) ||
                pa_mainloop_iterate(loop, 1, NULL) < 0) {
            goto done;
        }
    }

    const pa_sample_spec format = {PA_SAMPLE_S16LE, (uint32_t)rate, (uint8_t)channels};
    stream = pa_stream_new(context, "Omni native capture", &format, NULL);
    if (!stream) {
        goto done;
    }
    pa_buffer_attr buffer = {
        .maxlength = (uint32_t)-1, .tlength = (uint32_t)-1, .prebuf = (uint32_t)-1,
        .minreq = (uint32_t)-1, .fragsize = pa_usec_to_bytes(20000, &format),
    };
    if (pa_stream_connect_record(stream, argv[1], &buffer,
                                 PA_STREAM_DONT_MOVE | PA_STREAM_ADJUST_LATENCY) < 0) {
        goto done;
    }
    while (PA_STREAM_IS_GOOD(pa_stream_get_state(stream)) &&
            PA_CONTEXT_IS_GOOD(pa_context_get_state(context))) {
        if (pa_stream_get_state(stream) == PA_STREAM_READY) {
            const char *source = pa_stream_get_device_name(stream);
            if (!source || strcmp(source, argv[1]) != 0) {
                fprintf(stderr, "PulseAudio capture lost its selected source\n");
                goto done;
            }
            size_t readable = pa_stream_readable_size(stream);
            if (readable == (size_t)-1) {
                goto done;
            }
            if (readable) {
                const void *fragment = NULL;
                size_t bytes = 0;
                if (pa_stream_peek(stream, &fragment, &bytes) < 0) {
                    goto done;
                }
                if (bytes) {
                    if (!write_pcm(fragment, bytes) || pa_stream_drop(stream) < 0) {
                        goto done;
                    }
                    continue;
                }
            }
        }
        if (pa_mainloop_iterate(loop, 1, NULL) < 0) {
            goto done;
        }
    }

done:
    if (context) {
        fprintf(stderr, "PulseAudio capture ended: %s\n", pa_strerror(pa_context_errno(context)));
    }
    if (stream) {
        pa_stream_unref(stream);
    }
    if (context) pa_context_unref(context);
    pa_mainloop_free(loop);
    return 1;
}
