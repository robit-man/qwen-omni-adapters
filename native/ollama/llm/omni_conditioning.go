package llm

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

const (
	omniConditioningEntries   = 8
	omniConditioningBytes     = 1 << 20
	omniConditioningTTL       = 10 * time.Minute
	omniNormalizationVersion  = "ffmpeg-soxr20-pcm16-round-even-v1"
	omniConditioningAlgorithm = "qwen3tts-full-reference-context-v1"
)

// Only immutable reference JSON is retained. Generation helpers, samplers,
// decoded PCM and request text never enter this cache. Each loaded config owns
// its cache; unloading the runner makes the cache eligible for collection.
type omniConditioningEntry struct {
	key     [32]byte
	value   []byte
	expires time.Time
}

type omniConditioningCache struct {
	mu         sync.Mutex
	entries    map[[32]byte]*list.Element
	lru        list.List
	bytes      int
	maxEntries int
	maxBytes   int
	ttl        time.Duration
	now        func() time.Time
}

func newOmniConditioningCache() *omniConditioningCache {
	return &omniConditioningCache{entries: make(map[[32]byte]*list.Element), maxEntries: omniConditioningEntries, maxBytes: omniConditioningBytes, ttl: omniConditioningTTL, now: time.Now}
}

func (c *omniConditioningCache) remove(element *list.Element) {
	entry := element.Value.(*omniConditioningEntry)
	c.bytes -= len(entry.value)
	delete(c.entries, entry.key)
	c.lru.Remove(element)
}

func (c *omniConditioningCache) expire(now time.Time) {
	for element := c.lru.Back(); element != nil; {
		previous := element.Prev()
		if !now.Before(element.Value.(*omniConditioningEntry).expires) {
			c.remove(element)
		}
		element = previous
	}
}

func (c *omniConditioningCache) get(key [32]byte) ([]byte, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.expire(c.now())
	element, ok := c.entries[key]
	if !ok {
		return nil, false
	}
	c.lru.MoveToFront(element)
	return append([]byte(nil), element.Value.(*omniConditioningEntry).value...), true
}

func (c *omniConditioningCache) put(key [32]byte, value []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()
	c.expire(now)
	if len(value) == 0 || len(value) > c.maxBytes || c.maxEntries < 1 {
		return
	}
	if element, ok := c.entries[key]; ok {
		c.remove(element)
	}
	for len(c.entries) >= c.maxEntries || c.bytes+len(value) > c.maxBytes {
		c.remove(c.lru.Back())
	}
	entry := &omniConditioningEntry{key: key, value: append([]byte(nil), value...), expires: now.Add(c.ttl)}
	c.entries[key] = c.lru.PushFront(entry)
	c.bytes += len(value)
}

func omniConditioningKey(cfg *OmniConfig, speaker, text, language string) ([32]byte, error) {
	var empty [32]byte
	if len(text) > 16384 {
		return empty, errors.New("reference transcript exceeds 16 KiB")
	}
	file, err := os.Open(speaker)
	if err != nil {
		return empty, err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return empty, err
	}
	if !stat.Mode().IsRegular() {
		return empty, errors.New("voice reference must be a regular file")
	}
	h := sha256.New()
	_, err = io.Copy(h, file)
	if err != nil {
		return empty, err
	}
	// JSON framing prevents collisions between adjacent variable-length fields.
	metadata, err := json.Marshal([]string{hex.EncodeToString(h.Sum(nil)), text,
		cfg.ManifestDigest, cfg.Sidecar.SHA256, cfg.TTS.SHA256, cfg.ReferenceEncoder.SHA256,
		omniNormalizationVersion, language, omniConditioningAlgorithm})
	if err != nil {
		return empty, err
	}
	return sha256.Sum256(metadata), nil
}

func omniCachedReferencePrompt(ctx context.Context, cfg *OmniConfig, speaker, text, language, dir string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	key, err := omniConditioningKey(cfg, speaker, text, language)
	if err != nil {
		return "", err
	}
	if cfg.conditioning != nil {
		if encoded, ok := cfg.conditioning.get(key); ok {
			file := filepath.Join(dir, "reference.json")
			return file, os.WriteFile(file, encoded, 0600)
		}
	}
	file, err := omniReferencePrompt(ctx, cfg, speaker, text, dir)
	if err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if cfg.conditioning != nil {
		encoded, err := os.ReadFile(file)
		if err != nil {
			return "", err
		}
		cfg.conditioning.put(key, encoded)
	}
	return file, nil
}
