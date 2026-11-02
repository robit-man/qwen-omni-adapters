package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/llm"
)

type omniCompleteTestSpeaker func(context.Context, llm.SpeechRequest, func([]byte) error) error

func (fn omniCompleteTestSpeaker) Speech(ctx context.Context, request llm.SpeechRequest, callback func([]byte) error) error {
	return fn(ctx, request, callback)
}

func TestOmniCompleteSpeechModes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	yes, no := true, false
	pcm := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	for _, test := range []struct {
		name     string
		format   string
		stream   *bool
		complete bool
	}{{"default-wav", "wav", nil, false}, {"streamed-wav", "wav", &yes, false}, {"complete-wav", "wav", &no, true}, {"default-pcm", "pcm", nil, false}, {"complete-pcm", "pcm", &no, true}} {
		t.Run(test.name, func(t *testing.T) {
			speaker := omniCompleteTestSpeaker(func(ctx context.Context, request llm.SpeechRequest, callback func([]byte) error) error {
				if err := callback(pcm[:4]); err != nil {
					return err
				}
				return callback(pcm[4:])
			})
			router := gin.New()
			router.POST("/speech", func(c *gin.Context) {
				omniSpeechResponse(c, speaker, llm.SpeechRequest{Input: "test"}, test.format, test.stream)
			})
			server := httptest.NewServer(router)
			defer server.Close()
			response, err := http.Post(server.URL+"/speech", "application/json", nil)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			body, err := io.ReadAll(response.Body)
			if err != nil || response.StatusCode != 200 {
				t.Fatal(response.Status, err)
			}
			if test.complete && response.ContentLength != int64(len(body)) {
				t.Fatal("complete body length missing", response.ContentLength, len(body))
			}
			if response.Header.Get("Content-Type") != "audio/"+test.format {
				t.Fatal(response.Header)
			}
			if test.format == "wav" {
				if len(body) != 44+len(pcm) || string(body[:4]) != "RIFF" || string(body[8:12]) != "WAVE" {
					t.Fatal("invalid WAV")
				}
				if binary.LittleEndian.Uint32(body[24:28]) != 24000 || binary.LittleEndian.Uint16(body[22:24]) != 1 || binary.LittleEndian.Uint16(body[34:36]) != 16 {
					t.Fatal("WAV format changed")
				}
				dataBytes, riffBytes := binary.LittleEndian.Uint32(body[40:44]), binary.LittleEndian.Uint32(body[4:8])
				if test.complete {
					if dataBytes != uint32(len(pcm)) || riffBytes != uint32(len(body)-8) {
						t.Fatal("complete WAV has unknown or inaccurate lengths", dataBytes, riffBytes)
					}
				} else if dataBytes != math.MaxUint32 || riffBytes != math.MaxUint32 {
					t.Fatal("upstream streaming WAV behavior changed")
				}
				body = body[44:]
			}
			if !bytes.Equal(body, pcm) {
				t.Fatal("PCM payload changed")
			}
		})
	}
}

func TestOmniCompleteSpeechRejectsPartialEmptyOddAndOversizedOutput(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, name := range []string{"decoder-failure", "empty", "odd-PCM", "over-limit"} {
		t.Run(name, func(t *testing.T) {
			speaker := omniCompleteTestSpeaker(func(ctx context.Context, request llm.SpeechRequest, callback func([]byte) error) error {
				switch name {
				case "decoder-failure":
					if err := callback([]byte{1, 2}); err != nil {
						return err
					}
					return errors.New("decoder failed after partial audio")
				case "empty":
					return nil
				case "odd-PCM":
					return callback([]byte{1})
				case "over-limit":
					chunk := make([]byte, 1<<20)
					for i := 0; i < 17; i++ {
						if err := callback(chunk); err != nil {
							return err
						}
					}
					return nil
				}
				panic("unreachable")
			})
			router := gin.New()
			router.POST("/speech", func(c *gin.Context) {
				stream := false
				omniSpeechResponse(c, speaker, llm.SpeechRequest{}, "wav", &stream)
			})
			server := httptest.NewServer(router)
			defer server.Close()
			response, err := http.Post(server.URL+"/speech", "application/json", nil)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			body, err := io.ReadAll(response.Body)
			if err != nil {
				t.Fatal(err)
			}
			if response.StatusCode != 500 || strings.HasPrefix(response.Header.Get("Content-Type"), "audio/") || bytes.HasPrefix(body, []byte("RIFF")) {
				t.Fatal("partial audio exposed as complete", response.Status, string(body))
			}
		})
	}
}

func TestOmniCompleteSpeechCancellationReachesGenerator(t *testing.T) {
	gin.SetMode(gin.TestMode)
	started, stopped := make(chan struct{}), make(chan struct{})
	speaker := omniCompleteTestSpeaker(func(ctx context.Context, request llm.SpeechRequest, callback func([]byte) error) error {
		close(started)
		select {
		case <-ctx.Done():
			close(stopped)
			return ctx.Err()
		case <-time.After(5 * time.Second):
			return errors.New("test cancellation not delivered")
		}
	})
	router := gin.New()
	router.POST("/speech", func(c *gin.Context) {
		stream := false
		omniSpeechResponse(c, speaker, llm.SpeechRequest{}, "wav", &stream)
	})
	server := httptest.NewServer(router)
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/speech", nil)
	if err != nil {
		t.Fatal(err)
	}
	result := make(chan error, 1)
	go func() {
		response, err := http.DefaultClient.Do(request)
		if response != nil {
			response.Body.Close()
		}
		result <- err
	}()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("generator not started")
	}
	cancel()
	select {
	case <-stopped:
	case <-time.After(2 * time.Second):
		t.Fatal("disconnected complete request retained generator")
	}
	select {
	case err := <-result:
		if err == nil {
			t.Fatal("cancelled request succeeded")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("client did not cancel")
	}
}

func TestOmniInvalidStreamTypeRejectedBeforeScheduling(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/audio/speech", strings.NewReader(`{"model":"missing","input":"Test.","stream":"false"}`))
	ctx.Request.Header.Set("Content-Type", "application/json")
	(&Server{}).SpeechHandler(ctx)
	if recorder.Code != http.StatusBadRequest {
		t.Fatal("invalid stream field reached scheduler", recorder.Code)
	}
}
