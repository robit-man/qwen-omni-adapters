package llm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"golang.org/x/sync/semaphore"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestOmniFrameBoundsAndTruncation(t *testing.T) {
	for _, n := range []int{0, 2, 30720} {
		var b bytes.Buffer
		var h [9]byte
		h[0] = 'A'
		binary.LittleEndian.PutUint64(h[1:], uint64(n))
		b.Write(h[:])
		b.Write(make([]byte, n))
		kind, data, err := omniFrame(&b)
		if err != nil || kind != 'A' || len(data) != n {
			t.Fatal(kind, len(data), err)
		}
	}
	for _, raw := range [][]byte{{'A'}, {'A', 3, 0, 0, 0, 0, 0, 0, 0, 1, 2}, {'A', 0, 0, 64, 0, 0, 0, 0, 0}} {
		if _, _, err := omniFrame(bytes.NewReader(raw)); err == nil {
			t.Fatal("accepted oversized or incomplete frame")
		}
	}
}
func TestOmniArtifactRejectsMutationWithRestoredMtime(t *testing.T) {
	p := filepath.Join(t.TempDir(), "weights")
	os.WriteFile(p, []byte("original"), 0600)
	h := sha256.Sum256([]byte("original"))
	a := OmniArtifact{p, hex.EncodeToString(h[:])}
	if err := verifyOmniArtifact(a); err != nil {
		t.Fatal(err)
	}
	before, _ := os.Stat(p)
	os.WriteFile(p, []byte("tampered"), 0600)
	os.Chtimes(p, before.ModTime(), before.ModTime())
	if err := verifyOmniArtifact(a); err == nil {
		after, _ := os.Stat(p)
		t.Logf("before=%#v after=%#v cache=%#v", before.Sys(), after.Sys(), omniVerified.Files[a.Path+":"+a.SHA256])
		t.Fatal("accepted mutated weights with restored timestamp")
	}
}
func TestOmniAdapterScopeIsExplicit(t *testing.T) {
	s := &llamaServerRunner{}
	s.launch.config.Omni = &OmniConfig{AudioAdapter: &OmniArtifact{}}
	if a := s.omniLoras(false); len(a) != 1 || a[0].Scale != 0 {
		t.Fatal(a)
	}
	if a := s.omniLoras(true); len(a) != 1 || a[0].Scale != 1 {
		t.Fatal(a)
	}
	s.launch.config.Omni = nil
	if s.omniLoras(true) != nil {
		t.Fatal("ordinary model gained audio adapter")
	}
}
func TestOmniNativeASRUsesTrainedPromptAndNoCache(t *testing.T) {
	var captured map[string]any
	fixture := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/completion" {
			t.Error(r.URL.Path)
		}
		json.NewDecoder(r.Body).Decode(&captured)
		json.NewEncoder(w).Encode(map[string]any{"content": "<speech_transcript>Hello there.</speech_transcript><audio_observation>Rain.</audio_observation>"})
	}))
	defer fixture.Close()
	u, _ := url.Parse(fixture.URL)
	port, _ := strconv.Atoi(u.Port())
	s := &llamaServerRunner{port: port, sem: semaphore.NewWeighted(1), mediaMarker: "marker"}
	s.launch.config.Omni = &OmniConfig{Python: "/bin/true", SystemPrompt: "system", AudioSuffix: "suffix", AudioAdapter: &OmniArtifact{}}
	text, err := s.OmniTranscribe(context.Background(), []byte("wave"), 0)
	if err != nil || text != "Hello there." {
		t.Fatal(text, err)
	}
	if captured["cache_prompt"] != false || captured["stream"] != false {
		t.Fatal(captured)
	}
	prompt := captured["prompt"].(map[string]any)
	if prompt["prompt_string"] != "<|im_start|>system\nsystem<|im_end|>\n<|im_start|>user\nmarkersuffix" {
		t.Fatal(prompt)
	}
	if strings.Contains(text, "Rain") {
		t.Fatal("acoustic observation leaked into user transcript")
	}
	lora := captured["lora"].([]any)[0].(map[string]any)
	if lora["scale"] != float64(1) {
		t.Fatal(lora)
	}
}
func TestOmniNativeASRRejectsMalformedEvidence(t *testing.T) {
	for _, content := range []string{"plain fabricated transcript", "<speech_transcript>unfinished", "<speech_transcript>A</speech_transcript><speech_transcript>B</speech_transcript>"} {
		fixture := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewEncoder(w).Encode(map[string]any{"content": content})
		}))
		u, _ := url.Parse(fixture.URL)
		port, _ := strconv.Atoi(u.Port())
		s := &llamaServerRunner{port: port, sem: semaphore.NewWeighted(1)}
		s.launch.config.Omni = &OmniConfig{Python: "/bin/true"}
		if _, err := s.OmniTranscribe(context.Background(), []byte("wave"), 0); err == nil {
			t.Fatal(content)
		}
		fixture.Close()
	}
}
