package tokenizer_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/tiktoken-go/tokenizer"
)

type testCase struct {
	text string
	ids  []uint
}

func TestO200kBase(t *testing.T) {
	tok, err := tokenizer.Get(tokenizer.O200kBase)
	if err != nil {
		t.Fatalf("can't create tokenizer: %v", err)
	}

	tests := []testCase{
		{text: "'RE", ids: []uint{6, 1099}},
		{text: "hello world", ids: []uint{24912, 2375}},
		{text: "hello  world", ids: []uint{24912, 220, 2375}},
		{text: "hello   world", ids: []uint{24912, 256, 2375}},
		{text: "supercalifragilistic", ids: []uint{17789, 5842, 366, 17764, 311, 6207}},
		{text: "We know what we are, but know not what we may be.", ids: []uint{2167, 1761, 1412, 581, 553, 11, 889, 1761, 625, 1412, 581, 1340, 413, 13}},
	}

	runTests(t, tok, tests)
}

func TestCl100kBase(t *testing.T) {
	tok, err := tokenizer.Get(tokenizer.Cl100kBase)
	if err != nil {
		t.Fatalf("can't create tokenizer: %v", err)
	}

	tests := []testCase{
		{text: "'RE", ids: []uint{95253}},
		{text: "hello world", ids: []uint{15339, 1917}},
		{text: "hello  world", ids: []uint{15339, 220, 1917}},
		{text: "hello   world", ids: []uint{15339, 256, 1917}},
		{text: "supercalifragilistic", ids: []uint{13066, 3035, 278, 333, 4193, 321, 4633}},
		{text: "We know what we are, but know not what we may be.", ids: []uint{1687, 1440, 1148, 584, 527, 11, 719, 1440, 539, 1148, 584, 1253, 387, 13}},
	}

	runTests(t, tok, tests)
}

func TestR50kBase(t *testing.T) {
	tok, err := tokenizer.Get(tokenizer.R50kBase)
	if err != nil {
		t.Fatalf("can't create tokenizer: %v", err)
	}

	tests := []testCase{
		{text: "hello world", ids: []uint{31373, 995}},
		{text: "hello  world", ids: []uint{31373, 220, 995}},
		{text: "hello   world", ids: []uint{31373, 220, 220, 995}},
		{text: "supercalifragilistic", ids: []uint{16668, 9948, 361, 22562, 346, 2569}},
		{text: "We know what we are, but know not what we may be.", ids: []uint{1135, 760, 644, 356, 389, 11, 475, 760, 407, 644, 356, 743, 307, 13}},
	}

	runTests(t, tok, tests)
}

func TestP50kBase(t *testing.T) {
	tok, err := tokenizer.Get(tokenizer.P50kBase)
	if err != nil {
		t.Fatalf("can't create tokenizer: %v", err)
	}

	tests := []testCase{
		{text: "hello world", ids: []uint{31373, 995}},
		{text: "hello  world", ids: []uint{31373, 220, 995}},
		{text: "hello   world", ids: []uint{31373, 50257, 995}},
		{text: "supercalifragilistic", ids: []uint{16668, 9948, 361, 22562, 346, 2569}},
		{text: "We know what we are, but know not what we may be.", ids: []uint{1135, 760, 644, 356, 389, 11, 475, 760, 407, 644, 356, 743, 307, 13}},
	}

	runTests(t, tok, tests)
}

func TestConcurrentCountEncodeDecode(t *testing.T) {
	inputs := []string{
		"",
		"hello world",
		"supercalifragilistic",
		"We're testing 123456789!\n  Multiple\tspaces.\r\n",
		"こんにちは世界 — café",
	}

	for _, encoding := range []tokenizer.Encoding{
		tokenizer.O200kBase,
		tokenizer.Cl100kBase,
		tokenizer.R50kBase,
		tokenizer.P50kBase,
		tokenizer.P50kEdit,
	} {
		t.Run(string(encoding), func(t *testing.T) {
			reference, err := tokenizer.Get(encoding)
			if err != nil {
				t.Fatalf("can't create reference tokenizer: %v", err)
			}
			wantIDs := make([][]uint, len(inputs))
			wantTokens := make([][]string, len(inputs))
			for i, input := range inputs {
				wantIDs[i], wantTokens[i], err = reference.Encode(input)
				if err != nil {
					t.Fatalf("error encoding reference %q: %v", input, err)
				}
			}

			// Keep the shared instance unused until the parallel calls begin,
			// so lazy initialization is also exercised by the race detector.
			shared, err := tokenizer.Get(encoding)
			if err != nil {
				t.Fatalf("can't create shared tokenizer: %v", err)
			}
			for worker := range 24 {
				t.Run(fmt.Sprintf("worker-%d", worker), func(t *testing.T) {
					t.Parallel()
					for iteration := range 10 {
						for offset := range inputs {
							i := (worker + iteration + offset) % len(inputs)
							input := inputs[i]
							switch worker % 3 {
							case 0:
								count, err := shared.Count(input)
								if err != nil {
									t.Fatalf("error counting %q: %v", input, err)
								}
								if count != len(wantIDs[i]) {
									t.Fatalf("count for %q: want %d, got %d", input, len(wantIDs[i]), count)
								}
							case 1:
								ids, tokens, err := shared.Encode(input)
								if err != nil {
									t.Fatalf("error encoding %q: %v", input, err)
								}
								if !slices.Equal(ids, wantIDs[i]) {
									t.Fatalf("IDs for %q: want %v, got %v", input, wantIDs[i], ids)
								}
								if !slices.Equal(tokens, wantTokens[i]) {
									t.Fatalf("tokens for %q: want %q, got %q", input, wantTokens[i], tokens)
								}
							case 2:
								text, err := shared.Decode(wantIDs[i])
								if err != nil {
									t.Fatalf("error decoding %q: %v", input, err)
								}
								if text != input {
									t.Fatalf("decoded text: want %q, got %q", input, text)
								}
							}
						}
					}
				})
			}
		})
	}
}

func runTests(t *testing.T, tok tokenizer.Codec, tests []testCase) {
	for _, test := range tests {
		t.Run(test.text, func(t *testing.T) {
			ids, _, err := tok.Encode(test.text)
			if err != nil {
				t.Fatalf("error encoding: %v", err)
			}
			if !sliceEqual(ids, test.ids) {
				t.Errorf("encoding mismatch - want: %v got: %v", test.ids, ids)
			}

			text, err := tok.Decode(ids)
			if err != nil {
				t.Fatalf("error decoding: %v", err)
			}
			if text != test.text {
				t.Errorf("decoding mismatch - want: %s got: %s", test.text, text)
			}

			count, err := tok.Count(test.text)
			if err != nil {
				t.Fatalf("error counting: %v", err)
			}
			if count != len(test.ids) {
				t.Errorf("count mismatch - want: %d got: %d", len(test.ids), count)
			}
		})
	}
}

func sliceEqual(a, b []uint) bool {
	if len(a) != len(b) {
		return false
	}
	for i, elem := range a {
		if elem != b[i] {
			return false
		}
	}
	return true
}
