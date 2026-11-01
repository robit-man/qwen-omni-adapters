package server

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner/model/audio"
)

// A native-only stream:false request returns a complete, seekable file. The
// default and stream:true keep upstream's immediate streamed PCM/WAV behavior.
// The Qwen3-TTS maximum 4096 codec frames fit within this payload bound.
const omniCompleteSpeechPCMBytes = 16 << 20

func omniSpeechResponse(c *gin.Context, speaker llm.Speaker, req llm.SpeechRequest, format string, stream *bool) {
	if stream == nil || *stream {
		streamSpeech(c, speaker, req, format)
		return
	}
	var pcm bytes.Buffer
	err := speaker.Speech(c.Request.Context(), req, func(chunk []byte) error {
		if err := c.Request.Context().Err(); err != nil {
			return err
		}
		if len(chunk)%2 != 0 {
			return errors.New("complete speech received invalid PCM16 data")
		}
		if len(chunk) > omniCompleteSpeechPCMBytes-pcm.Len() {
			return errors.New("complete speech exceeds the 16 MiB PCM limit")
		}
		_, err := pcm.Write(chunk)
		return err
	})
	if c.Request.Context().Err() != nil {
		return
	}
	if err == nil && pcm.Len() == 0 {
		err = errors.New("speech model produced no audio")
	}
	if err != nil {
		status, message := http.StatusInternalServerError, err.Error()
		var statusErr api.StatusError
		if errors.As(err, &statusErr) {
			status, message = statusErr.StatusCode, statusErr.ErrorMessage
		}
		openAIError(c, status, message)
		return
	}
	var header []byte
	if format == "wav" {
		header = audio.WAVHeader(llm.SpeechSampleRate, uint32(pcm.Len()))
	}
	var writer http.ResponseWriter = c.Writer
	if wrapped, ok := writer.(interface{ Unwrap() http.ResponseWriter }); ok {
		writer = wrapped.Unwrap()
	}
	controller := http.NewResponseController(writer)
	if err := controller.SetWriteDeadline(time.Now().Add(30 * time.Second)); err != nil {
		openAIError(c, http.StatusInternalServerError, err.Error())
		return
	}
	defer controller.SetWriteDeadline(time.Time{})
	c.DataFromReader(http.StatusOK, int64(len(header)+pcm.Len()), "audio/"+format,
		io.MultiReader(bytes.NewReader(header), bytes.NewReader(pcm.Bytes())), nil)
}
