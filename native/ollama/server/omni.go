package server

import (
	"context"
	"errors"
	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/types/model"
	"net/http"
	"time"
)

type omniTranscriber interface {
	OmniTranscribe(context.Context, []byte, float64) (string, error)
}

func (s *Server) omniTranscription(c *gin.Context, form audioTextForm, task renderers.AudioTask) {
	if task != renderers.AudioTranscribe || form.request.Prompt != "" {
		openAIError(c, http.StatusBadRequest, "native Omni transcription preserves its trained extraction prompt; translation and custom prompts are unsupported")
		return
	}
	clips, err := splitAudio(form.audio, 30*time.Second)
	if err != nil {
		openAIError(c, http.StatusBadRequest, err.Error())
		return
	}
	r, _, _, err := s.scheduleRunner(c.Request.Context(), form.model, "", []model.Capability{model.CapabilityAudio}, nil, nil, nil)
	if err != nil {
		status, msg := scheduleError(form.model, err)
		openAIError(c, status, msg)
		return
	}
	runner, ok := r.(omniTranscriber)
	if !ok {
		openAIError(c, http.StatusNotImplemented, "loaded runner lacks native Omni transcription")
		return
	}
	out := audioTextWriter{c: c, req: form.request, format: form.responseFormat, stream: form.stream}
	for _, clip := range clips {
		text, err := runner.OmniTranscribe(c.Request.Context(), clip.wav(), form.temperature)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return
			}
			out.fail(http.StatusInternalServerError, err.Error())
			return
		}
		if err := out.add(text, clip.duration); err != nil {
			return
		}
	}
	out.done()
}
