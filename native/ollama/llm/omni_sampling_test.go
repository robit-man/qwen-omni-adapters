package llm

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestOmniSamplingMatchesExistingRuntime(t *testing.T) {
	data, err := os.ReadFile("testdata/omni-sampling-baseline.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Name     string            `json:"name"`
			Options  OmniSpeechOptions `json:"options"`
			Valid    bool              `json:"valid"`
			Expected omniSampling      `json:"expected"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) != 12 {
		t.Fatal("incomplete existing-runtime fixture")
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			actual, err := resolveOmniSampling(&c.Options)
			if (err == nil) != c.Valid {
				t.Fatalf("validity mismatch: %v", err)
			}
			if c.Valid && actual != c.Expected {
				t.Fatalf("got%+v want%+v", actual, c.Expected)
			}
		})
	}
}
func TestOmniSamplingRejectsNonfiniteNumbers(t *testing.T) {
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if ValidateOmniSpeechOptions(&OmniSpeechOptions{Temperature: &value}) == nil {
			t.Fatal("nonfinite temperature accepted")
		}
		if ValidateOmniSpeechOptions(&OmniSpeechOptions{TopP: &value}) == nil {
			t.Fatal("nonfinite top_p accepted")
		}
	}
}
func TestOmniSamplingOmittedVersusExplicitZero(t *testing.T) {
	defaults, err := resolveOmniSampling(nil)
	if err != nil || defaults.TopK != 40 || defaults.TopP != .9 || defaults.Seed != 42 {
		t.Fatal(defaults, err)
	}
	zero := 0
	zeroFloat := 0.0
	resolved, err := resolveOmniSampling(&OmniSpeechOptions{TopK: &zero, TopP: &zeroFloat})
	if err != nil || resolved.TopK != 0 || resolved.TopP != 0 {
		t.Fatal(resolved, err)
	}
}
