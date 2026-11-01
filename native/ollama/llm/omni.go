package llm

// Native Ollama execution for the exact-pair-bound Robit Omni bundle. The
// Python installer validates the GGUF inventory/ABI and writes this immutable
// digest-addressed configuration. Generation itself uses native workers only.
import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/ollama/ollama/api"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// OmniAuxiliaryReserve is admission headroom for the on-demand native TTS/ref
// graphs. It is not reported as actual resident model memory.
const OmniAuxiliaryReserve uint64 = 8 << 30

type OmniArtifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}
type OmniConfig struct {
	Schema           string                  `json:"schema"`
	ManifestDigest   string                  `json:"manifest_digest"`
	Model            OmniArtifact            `json:"model"`
	Projector        OmniArtifact            `json:"projector"`
	Sidecar          OmniArtifact            `json:"sidecar"`
	AudioAdapter     *OmniArtifact           `json:"audio_adapter,omitempty"`
	Engine           OmniArtifact            `json:"engine"`
	TTS              OmniArtifact            `json:"tts"`
	ReferenceEncoder OmniArtifact            `json:"reference_encoder"`
	ProofScript      OmniArtifact            `json:"proof_script"`
	Python           string                  `json:"python"`
	Libraries        []OmniArtifact          `json:"libraries"`
	Voices           map[string]OmniArtifact `json:"voices"`
	SystemPrompt     string                  `json:"system_prompt"`
	AudioSuffix      string                  `json:"audio_suffix"`
	PromptSHA256     string                  `json:"prompt_sha256"`
}

type omniFileIdentity struct {
	Path           string
	Size           int64
	ModTime        time.Time
	Info           os.FileInfo
	SystemIdentity string
	VerifiedAt     time.Time
}

var omniVerified = struct {
	sync.Mutex
	Files map[string]omniFileIdentity
}{Files: make(map[string]omniFileIdentity)}

func verifyOmniArtifact(a OmniArtifact) error {
	if !filepath.IsAbs(a.Path) || len(a.SHA256) != 64 {
		return errors.New("invalid native Omni artifact identity")
	}
	if _, err := hex.DecodeString(a.SHA256); err != nil {
		return err
	}
	f, err := os.Open(a.Path)
	if err != nil {
		return err
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil {
		return err
	}
	if !before.Mode().IsRegular() {
		return errors.New("native Omni artifact is not a regular file")
	}
	omniVerified.Lock()
	defer omniVerified.Unlock()
	key := a.Path + ":" + a.SHA256
	if v, ok := omniVerified.Files[key]; ok && os.SameFile(v.Info, before) && v.Size == before.Size() && v.ModTime.Equal(before.ModTime()) && v.SystemIdentity == fmt.Sprintf("%#v", before.Sys()) && time.Since(v.VerifiedAt) > time.Second {
		return nil
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	after, err := f.Stat()
	if err != nil {
		return err
	}
	if !os.SameFile(before, after) || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) || hex.EncodeToString(h.Sum(nil)) != a.SHA256 {
		return errors.New("native Omni artifact changed or failed SHA256 verification")
	}
	omniVerified.Files[key] = omniFileIdentity{a.Path, after.Size(), after.ModTime(), after, fmt.Sprintf("%#v", after.Sys()), time.Now()}
	return nil
}

func LoadOmniConfig(digest, model string, projectors []string) (*OmniConfig, error) {
	root := os.Getenv("OLLAMA_OMNI_CONFIG_DIR")
	if root == "" {
		return nil, nil
	}
	key := strings.TrimPrefix(digest, "sha256:")
	if len(key) != 64 {
		return nil, errors.New("native Omni manifest digest is invalid")
	}
	if _, err := hex.DecodeString(key); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(root, key+".json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if len(data) > 128<<10 {
		return nil, errors.New("native Omni configuration exceeds128KiB")
	}
	var cfg OmniConfig
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return nil, err
	}
	if cfg.Schema != "robit.omni.native-ollama.v1" || strings.TrimPrefix(cfg.ManifestDigest, "sha256:") != key || cfg.Model.Path != model || len(projectors) != 1 || cfg.Projector.Path != projectors[0] {
		return nil, errors.New("native Omni configuration does not match this exact model/projector/manifest")
	}
	expected := sha256.Sum256([]byte(cfg.SystemPrompt + "\x00" + cfg.AudioSuffix))
	if cfg.PromptSHA256 != "9f73862652e0226ec3f9690f0a783d1c21dc1113285b4dc18d0edd51f2766758" || hex.EncodeToString(expected[:]) != cfg.PromptSHA256 {
		return nil, errors.New("native Omni training prompt binding mismatch")
	}
	artifacts := append([]OmniArtifact{cfg.Model, cfg.Projector, cfg.Sidecar, cfg.Engine, cfg.TTS, cfg.ReferenceEncoder, cfg.ProofScript}, cfg.Libraries...)
	if cfg.AudioAdapter != nil {
		artifacts = append(artifacts, *cfg.AudioAdapter)
	}
	if len(cfg.Voices) == 0 {
		return nil, errors.New("native Omni requires at least one bundled voice reference")
	}
	for _, a := range cfg.Voices {
		artifacts = append(artifacts, a)
	}
	for _, a := range artifacts {
		if err := verifyOmniArtifact(a); err != nil {
			return nil, fmt.Errorf("native Omni artifact verification: %w", err)
		}
	}
	return &cfg, nil
}

type OmniSpeakerAudio struct {
	MIMEType string `json:"mime_type"`
	Encoding string `json:"encoding"`
	Data     string `json:"data"`
}
type OmniSpeechOptions struct {
	SpeakerAudio  *OmniSpeakerAudio `json:"speaker_audio,omitempty"`
	ReferenceText string            `json:"reference_text,omitempty"`
	Language      string            `json:"language,omitempty"`
	Temperature   *float64          `json:"temperature,omitempty"`
	TopK          *int              `json:"top_k,omitempty"`
	TopP          *float64          `json:"top_p,omitempty"`
	Seed          *int              `json:"seed,omitempty"`
	MaxFrames     *int              `json:"max_frames,omitempty"`
}

type omniLora struct {
	ID    int     `json:"id"`
	Scale float32 `json:"scale"`
}

func (s *llamaServerRunner) omniLoras(audio bool) []omniLora {
	if s.launch.config.Omni == nil || s.launch.config.Omni.AudioAdapter == nil {
		return nil
	}
	scale := float32(0)
	if audio {
		scale = 1
	}
	return []omniLora{{0, scale}}
}

// OmniTranscribe preserves the exact trained prefix/suffix and enables LoRA
// only for this acoustic perception pass. It shares the existing primary PID.
func (s *llamaServerRunner) OmniTranscribe(ctx context.Context, wav []byte, temperature float64) (string, error) {
	cfg := s.launch.config.Omni
	if cfg == nil {
		return "", errors.New("native Omni runner is not configured")
	}
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return "", err
	}
	defer s.sem.Release(1)
	if err := omniGPUProof(ctx, cfg, s.Pid()); err != nil {
		return "", err
	}
	prompt := "<|im_start|>system\n" + cfg.SystemPrompt + "<|im_end|>\n<|im_start|>user\n" + s.llamaServerMediaMarker() + cfg.AudioSuffix
	body := map[string]any{"prompt": llamaServerMultimodalPrompt{PromptString: prompt, MultimodalData: []string{base64.StdEncoding.EncodeToString(wav)}}, "stream": false, "cache_prompt": false, "n_predict": 384, "temperature": temperature, "repeat_penalty": 1.1, "lora": s.omniLoras(true)}
	data, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("http://127.0.0.1:%d/completion", s.port), bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := s.httpClient().Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return "", fmt.Errorf("native Omni transcription returnedHTTP%d", res.StatusCode)
	}
	var response struct {
		Content      string `json:"content"`
		StoppedLimit bool   `json:"stopped_limit"`
	}
	dec := json.NewDecoder(io.LimitReader(res.Body, 1<<20))
	if err := dec.Decode(&response); err != nil {
		return "", err
	}
	if response.StoppedLimit {
		return "", errors.New("native Omni transcription exceeded its generation bound")
	}
	const start = "<speech_transcript>"
	const end = "</speech_transcript>"
	if strings.Count(response.Content, start) != 1 || strings.Count(response.Content, end) != 1 {
		return "", errors.New("native Omni returned malformed speech evidence")
	}
	_, tail, _ := strings.Cut(response.Content, start)
	text, _, ok := strings.Cut(tail, end)
	if !ok {
		return "", errors.New("native Omni returned unclosed speech evidence")
	}
	return strings.TrimSpace(text), nil
}

func omniFrame(r io.Reader) (byte, []byte, error) {
	var h [9]byte
	if _, err := io.ReadFull(r, h[:]); err != nil {
		return 0, nil, err
	}
	n := binary.LittleEndian.Uint64(h[1:])
	if n > 2<<20 {
		return 0, nil, errors.New("native TTS frame exceeds2MiB")
	}
	data := make([]byte, int(n))
	_, err := io.ReadFull(r, data)
	return h[0], data, err
}

func omniCommand(ctx context.Context, cfg *OmniConfig, binary string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, binary, args...)
	// Parent Ollama already runs inside the exact admitted GPU scope. Inherit it
	// unchanged; never perform our own GPU placement or select an extra GPU.
	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+filepath.Dir(cfg.Engine.Path)+string(os.PathListSeparator)+os.Getenv("LD_LIBRARY_PATH"))
	return cmd
}

func omniGPUProof(ctx context.Context, cfg *OmniConfig, pid int) error {
	if !filepath.IsAbs(cfg.Python) {
		return errors.New("native Omni GPU proof interpreter missing")
	}
	cmd := exec.CommandContext(ctx, cfg.Python, cfg.ProofScript.Path, fmt.Sprint(pid))
	if err := cmd.Run(); err != nil {
		return errors.New("native Omni worker failed accelerator residency proof")
	}
	return nil
}

func omniReferencePrompt(ctx context.Context, cfg *OmniConfig, speaker, text, dir string) (string, error) {
	if len(text) > 16384 {
		return "", errors.New("reference transcript exceeds16KiB")
	}
	normalized := filepath.Join(dir, "reference.f32")
	codes := filepath.Join(dir, "codes.i32")
	ffmpeg := exec.CommandContext(ctx, "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-protocol_whitelist", "file,pipe", "-i", speaker, "-vn", "-t", "31", "-af", "aresample=24000:resampler=soxr:precision=20:osf=flt", "-ac", "1", "-ar", "24000", "-f", "f32le", normalized)
	if err := ffmpeg.Run(); err != nil {
		return "", errors.New("voice-reference normalization failed")
	}
	rawReference, err := os.ReadFile(normalized)
	if err != nil {
		return "", err
	}
	if len(rawReference)%4 != 0 || len(rawReference) < 12000*4 || len(rawReference) > 720000*4 {
		return "", errors.New("reference must contain0.5to30seconds of audio")
	}
	for i := 0; i < len(rawReference); i += 4 {
		v := float64(math.Float32frombits(binary.LittleEndian.Uint32(rawReference[i:])))
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return "", errors.New("reference contains nonfinite audio")
		}
		v = math.RoundToEven(math.Max(-1, math.Min(32767.0/32768, v))*32768) / 32768
		binary.LittleEndian.PutUint32(rawReference[i:], math.Float32bits(float32(v)))
	}
	if err := os.WriteFile(normalized, rawReference, 0600); err != nil {
		return "", err
	}

	st, err := os.Stat(normalized)
	if err != nil {
		return "", err
	}
	samples := st.Size() / 4
	if samples < 1 || samples > 720000 || st.Size()%4 != 0 {
		return "", errors.New("reference must contain at most30seconds of audio")
	}
	// The native reference helper selects its indexed component by this basename.
	view := filepath.Join(dir, "tts-reference-encoder.gguf")
	if err := os.Symlink(cfg.Sidecar.Path, view); err != nil {
		return "", err
	}
	cmd := omniCommand(ctx, cfg, cfg.ReferenceEncoder.Path, view, normalized, codes)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", err
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return "", err
	}
	if err := cmd.Start(); err != nil {
		return "", err
	}
	defer func() { cmd.Process.Kill(); cmd.Wait() }()
	var ready [2]byte
	if _, err := io.ReadFull(stdout, ready[:]); err != nil || string(ready[:]) != "R\n" {
		return "", errors.New("native voice-reference encoder failed")
	}
	if err := omniGPUProof(ctx, cfg, cmd.Process.Pid); err != nil {
		return "", err
	}
	if _, err := io.WriteString(stdin, "GO\n"); err != nil {
		return "", err
	}
	stdin.Close()
	if err := cmd.Wait(); err != nil {
		return "", errors.New("native reference encoding failed")
	}
	raw, err := os.ReadFile(codes)
	if err != nil {
		return "", err
	}
	frames := int((samples + 1919) / 1920)
	if len(raw) != frames*16*4 {
		return "", errors.New("native reference code dimensions mismatch")
	}
	values := make([][]int, frames)
	for i := range values {
		values[i] = make([]int, 16)
		for j := range values[i] {
			v := int(int32(binary.LittleEndian.Uint32(raw[(i*16+j)*4:])))
			if v < 0 || v >= 2048 {
				return "", errors.New("native reference code out of range")
			}
			values[i][j] = v
		}
	}
	encoded, err := json.Marshal(map[string]any{"schema": "robit.qwen3tts.reference-prompt.v1", "reference_text": text, "codes": values})
	if err != nil {
		return "", err
	}
	file := filepath.Join(dir, "reference.json")
	return file, os.WriteFile(file, encoded, 0600)
}

func (s *llamaServerRunner) omniSpeech(ctx context.Context, req SpeechRequest, fn func([]byte) error) error {
	cfg := s.launch.config.Omni
	sampling, err := resolveOmniSampling(req.Omni)
	if err != nil {
		return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return err
	}
	defer s.sem.Release(1)
	ctx, cancel := context.WithTimeout(ctx, 180*time.Second)
	defer cancel()
	dir, err := os.MkdirTemp("", "ollama-omni-voice-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	o := req.Omni
	if o == nil {
		o = &OmniSpeechOptions{}
	}
	voice := req.Voice
	if voice == "" || voice == "default" {
		voice = "female"
	}
	speaker := ""
	if o.SpeakerAudio != nil {
		a := o.SpeakerAudio
		if a.MIMEType != "audio/wav" || a.Encoding != "base64" || len(a.Data) > 14<<20 {
			return errors.New("speaker_audio must be a bounded base64 WAV")
		}
		raw, err := base64.StdEncoding.Strict().DecodeString(a.Data)
		if err != nil || len(raw) > 10<<20 || len(raw) < 44 || string(raw[:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
			return errors.New("invalid reference WAV")
		}
		speaker = filepath.Join(dir, "speaker.wav")
		if err := os.WriteFile(speaker, raw, 0600); err != nil {
			return err
		}
	} else {
		a, ok := cfg.Voices[voice]
		if !ok {
			return errors.New("unknown saved voice")
		}
		if err := verifyOmniArtifact(a); err != nil {
			return err
		}
		speaker = a.Path
	}
	language := o.Language
	if language == "" {
		language = "en"
	}
	allowed := map[string]bool{"en": true, "zh": true, "ja": true, "ko": true, "de": true, "fr": true, "ru": true, "pt": true, "es": true, "it": true}
	if !allowed[language] {
		return errors.New("unsupported Qwen3-TTS language")
	}
	temp, topP, topK, seed, frames := sampling.Temperature, sampling.TopP, sampling.TopK, sampling.Seed, sampling.Frames
	reference := ""
	if o.ReferenceText != "" {
		reference, err = omniReferencePrompt(ctx, cfg, speaker, o.ReferenceText, dir)
		if err != nil {
			return err
		}
	}
	args := []string{"-m", cfg.Sidecar.Path, "--mmproj", cfg.Sidecar.Path, "--prompt", "", "--output", filepath.Join(dir, "unused.wav"), "--tts-speaker-file", speaker, "--tts-lang", language, "--ctx-size", "4096", "--n-predict", fmt.Sprint(frames), "--gpu-layers", "999", "--temp", fmt.Sprint(temp), "--top-k", fmt.Sprint(topK), "--top-p", fmt.Sprint(topP), "--seed", fmt.Sprint(seed), "--tts-stream", "--tts-stream-frames", "4", "--tts-persistent"}
	// Bound context by complete UTF8 input/reference and generation, never truncate.
	required := len([]byte(req.Input)) + frames + 256
	if reference != "" {
		required += len([]byte(o.ReferenceText)) + 750 + 64
	}
	contextSize := 4096
	for contextSize < required {
		contextSize *= 2
	}
	if contextSize > 16384 {
		return errors.New("native speech context budget exceeded")
	}
	for i := range args {
		if args[i] == "--ctx-size" {
			args[i+1] = fmt.Sprint(contextSize)
		}
	}
	if reference != "" {
		args = append(args, "--tts-reference-prompt-file", reference)
	}
	cmd := omniCommand(ctx, cfg, cfg.TTS.Path, args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	defer func() { cancel(); cmd.Process.Kill(); cmd.Wait() }()
	kind, data, err := omniFrame(stdout)
	if err != nil || kind != 'R' || len(data) != 0 {
		return errors.New("native TTS did not emit ready")
	}
	if err := omniGPUProof(ctx, cfg, cmd.Process.Pid); err != nil {
		return err
	}
	encoded := base64.StdEncoding.EncodeToString([]byte(req.Input)) + "\n"
	if _, err := io.WriteString(stdin, encoded); err != nil {
		return err
	}
	stdin.Close()
	samples := 0
	for {
		kind, data, err = omniFrame(stdout)
		if err != nil {
			return fmt.Errorf("native TTS incomplete stream: %w", err)
		}
		switch kind {
		case 'A':
			if len(data) == 0 || len(data)%2 != 0 {
				return errors.New("invalid native PCM16 frame")
			}
			samples += len(data) / 2
			if err := fn(data); err != nil {
				return err
			}
		case 'D':
			if len(data) != 0 || samples == 0 {
				return errors.New("native TTS empty or malformed completion")
			}
			if err := cmd.Wait(); err != nil {
				return errors.New("native TTS failed after completion")
			}
			return nil
		case 'E':
			return errors.New("native TTS generation failed")
		default:
			return errors.New("native TTS unexpected frame")
		}
	}
}
