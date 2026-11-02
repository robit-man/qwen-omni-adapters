package llm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestOmniConditioningBoundedLRUExpiryAndCopies(t *testing.T) {
	now := time.Unix(1000, 0)
	c := newOmniConditioningCache()
	c.maxEntries, c.maxBytes, c.now = 2, 6, func() time.Time { return now }
	a, b, d := sha256.Sum256([]byte("a")), sha256.Sum256([]byte("b")), sha256.Sum256([]byte("d"))
	input := []byte("abc")
	c.put(a, input)
	input[0] = 'X'
	c.put(b, []byte("def"))
	value, ok := c.get(a)
	if !ok || string(value) != "abc" {
		t.Fatal("insert retained caller memory")
	}
	value[0] = 'Y'
	c.put(d, []byte("ghi"))
	if _, ok := c.get(b); ok {
		t.Fatal("least recently used entry survived")
	}
	if value, ok := c.get(a); !ok || string(value) != "abc" {
		t.Fatal("lookup exposed mutable cached data")
	}
	c.put(b, []byte("too large"))
	if c.bytes != 6 || len(c.entries) != 2 {
		t.Fatal("oversized input changed bounded cache")
	}
	c.put(a, []byte("fresh"))
	if c.bytes != 5 || len(c.entries) != 1 {
		t.Fatal("byte-budget eviction failed")
	}
	now = now.Add(omniConditioningTTL)
	if _, ok := c.get(a); ok || c.bytes != 0 || len(c.entries) != 0 {
		t.Fatal("expired conditioning survived lookup")
	}
	c.put(a, nil)
	if len(c.entries) != 0 {
		t.Fatal("empty result cached")
	}
}

func TestOmniConditioningKeyInvalidatesEveryReferenceDependency(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "reference.wav")
	if err := os.WriteFile(file, []byte("reference-A"), 0600); err != nil {
		t.Fatal(err)
	}
	base := OmniConfig{ManifestDigest: "manifest-A", Sidecar: OmniArtifact{SHA256: "weights-A"}, TTS: OmniArtifact{SHA256: "TTS-A"}, ReferenceEncoder: OmniArtifact{SHA256: "encoder-A"}}
	key, err := omniConditioningKey(&base, file, "Reference words.", "en")
	if err != nil {
		t.Fatal(err)
	}
	for _, change := range []struct {
		name   string
		modify func(*OmniConfig)
	}{
		{"bundle", func(c *OmniConfig) { c.ManifestDigest = "manifest-B" }},
		{"checkpoint", func(c *OmniConfig) { c.Sidecar.SHA256 = "weights-B" }},
		{"TTS binary", func(c *OmniConfig) { c.TTS.SHA256 = "TTS-B" }},
		{"reference encoder", func(c *OmniConfig) { c.ReferenceEncoder.SHA256 = "encoder-B" }},
	} {
		t.Run(change.name, func(t *testing.T) {
			cfg := base
			change.modify(&cfg)
			actual, err := omniConditioningKey(&cfg, file, "Reference words.", "en")
			if err != nil || actual == key {
				t.Fatal("dependency omitted", err)
			}
		})
	}
	for _, pair := range [][2]string{{"Different words.", "en"}, {"Reference words.", "fr"}} {
		actual, err := omniConditioningKey(&base, file, pair[0], pair[1])
		if err != nil || actual == key {
			t.Fatal("transcript/language omitted", err)
		}
	}
	if err := os.WriteFile(file, []byte("reference-B"), 0600); err != nil {
		t.Fatal(err)
	}
	actual, err := omniConditioningKey(&base, file, "Reference words.", "en")
	if err != nil || actual == key {
		t.Fatal("reference content omitted", err)
	}
	if _, err := omniConditioningKey(&base, file, strings.Repeat("x", 16385), "en"); err == nil {
		t.Fatal("oversized transcript accepted")
	}
}

func TestOmniConditioningHitBypassesEncoderAndKeepsPrivateRequestFile(t *testing.T) {
	cfg := &OmniConfig{conditioning: newOmniConditioningCache()}
	voice := filepath.Join(t.TempDir(), "reference.wav")
	if err := os.WriteFile(voice, []byte("reference bytes hashed before cache lookup"), 0600); err != nil {
		t.Fatal(err)
	}
	key, err := omniConditioningKey(cfg, voice, "Private reference transcript.", "en")
	if err != nil {
		t.Fatal(err)
	}
	encoded := []byte(`{"schema":"robit.qwen3tts.reference-prompt.v1","reference_text":"Private reference transcript.","codes":[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]}`)
	cfg.conditioning.put(key, encoded)
	dir := t.TempDir()
	// No encoder, Python proof or native binary exists in cfg. A hit must not
	// invoke one; this is a CPU control-flow test, not a generation qualification.
	file, err := omniCachedReferencePrompt(context.Background(), cfg, voice, "Private reference transcript.", "en", dir)
	if err != nil {
		t.Fatal(err)
	}
	actual, err := os.ReadFile(file)
	if err != nil || !bytes.Equal(actual, encoded) {
		t.Fatal("conditioning file differs", err)
	}
	stat, err := os.Stat(file)
	if err != nil || stat.Mode().Perm() != 0600 {
		t.Fatal("reference file is not private", err)
	}
	other := newOmniConditioningCache()
	if _, ok := other.get(key); ok {
		t.Fatal("conditioning shared with a different loaded configuration")
	}
	public, err := json.Marshal(cfg)
	if err != nil || bytes.Contains(public, []byte("Private reference")) || bytes.Contains(public, []byte("conditioning")) {
		t.Fatal("private conditioning serialized", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := omniCachedReferencePrompt(ctx, cfg, voice, "Private reference transcript.", "en", t.TempDir()); err == nil {
		t.Fatal("cancelled request reused conditioning")
	}
}

func TestOmniConditioningConcurrentAccessRemainsBounded(t *testing.T) {
	c := newOmniConditioningCache()
	var wait sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func(worker int) {
			defer wait.Done()
			for i := 0; i < 128; i++ {
				key := sha256.Sum256([]byte{byte(worker), byte(i)})
				c.put(key, bytes.Repeat([]byte{byte(worker)}, 100))
				c.get(key)
			}
		}(worker)
	}
	wait.Wait()
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.entries) > omniConditioningEntries || c.bytes > omniConditioningBytes || c.bytes != 100*len(c.entries) {
		t.Fatal("cache budget/accounting failed")
	}
}
