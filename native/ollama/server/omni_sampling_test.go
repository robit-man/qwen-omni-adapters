package server

import (
	"github.com/gin-gonic/gin"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOmniInvalidSamplingFailsBeforeModelScheduling(t *testing.T) {
	// A zero Server has no scheduler. Invalid requests must return400 without
	// touching model resolution, starting a runner or dereferencing a scheduler.
	gin.SetMode(gin.TestMode)
	for _, options := range []string{`{"seed":-2}`, `{"seed":2147483648}`, `{"top_k":-1}`, `{"top_p":1.1}`, `{"temperature":2.1}`} {
		recorder := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(recorder)
		ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/audio/speech", strings.NewReader(`{"model":"intentionally-missing","input":"Test.","omni":`+options+`}`))
		ctx.Request.Header.Set("Content-Type", "application/json")
		(&Server{}).SpeechHandler(ctx)
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("options%s: got%d: %s", options, recorder.Code, recorder.Body.String())
		}
	}
}
