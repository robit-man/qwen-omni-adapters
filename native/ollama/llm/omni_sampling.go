package llm

import (
	"errors"
	"math"
)

type omniSampling struct {
	Temperature float64 `json:"temperature"`
	TopK        int     `json:"top_k"`
	TopP        float64 `json:"top_p"`
	Seed        int     `json:"seed"`
	Frames      int     `json:"max_frames"`
}

func resolveOmniSampling(options *OmniSpeechOptions) (omniSampling, error) {
	resolved := omniSampling{Temperature: .7, TopK: 40, TopP: .9, Seed: 42, Frames: 512}
	if options != nil {
		if options.Temperature != nil {
			resolved.Temperature = *options.Temperature
		}
		if options.TopK != nil {
			resolved.TopK = *options.TopK
		}
		if options.TopP != nil {
			resolved.TopP = *options.TopP
		}
		if options.Seed != nil {
			resolved.Seed = *options.Seed
		}
		if options.MaxFrames != nil {
			resolved.Frames = *options.MaxFrames
		}
	}
	if math.IsNaN(resolved.Temperature) || math.IsInf(resolved.Temperature, 0) || resolved.Temperature < 0 || resolved.Temperature > 2 {
		return resolved, errors.New("temperature must be between0and2")
	}
	if resolved.TopK < 0 || resolved.TopK > 1000 {
		return resolved, errors.New("top_k must be between0and1000")
	}
	if math.IsNaN(resolved.TopP) || math.IsInf(resolved.TopP, 0) || resolved.TopP < 0 || resolved.TopP > 1 {
		return resolved, errors.New("top_p must be between0and1")
	}
	if resolved.Seed < -1 || resolved.Seed > 2147483647 {
		return resolved, errors.New("seed must be -1or a32-bit nonnegative integer")
	}
	if resolved.Frames < 1 || resolved.Frames > 4096 {
		return resolved, errors.New("max_frames must be between1and4096")
	}
	return resolved, nil
}

// ValidateOmniSpeechOptions runs before scheduling/loading a model. Sampling
// bounds preserve the existing TTS contract, including zero-valued filters.
func ValidateOmniSpeechOptions(options *OmniSpeechOptions) error {
	_, err := resolveOmniSampling(options)
	return err
}
