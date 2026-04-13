package main

import (
	"os"
	"testing"

	"github.com/hybridgroup/yzma/pkg/llama"
)

func TestParseGemma4ToolCall(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantOK   bool
		wantName string
		wantArgs map[string]string
	}{
		{
			name:     "simple city",
			input:    `<|tool_call>call:get_weather{location:<|"|>Miami, FL<|"|>}<tool_call|>`,
			wantOK:   true,
			wantName: "get_weather",
			wantArgs: map[string]string{"location": "Miami, FL"},
		},
		{
			name:     "city with state",
			input:    `<|tool_call>call:get_weather{location:<|"|>New York, NY<|"|>}<tool_call|>`,
			wantOK:   true,
			wantName: "get_weather",
			wantArgs: map[string]string{"location": "New York, NY"},
		},
		{
			name:     "with surrounding text",
			input:    `Let me check that for you. <|tool_call>call:get_weather{location:<|"|>Tampa, FL<|"|>}<tool_call|>`,
			wantOK:   true,
			wantName: "get_weather",
			wantArgs: map[string]string{"location": "Tampa, FL"},
		},
		{
			name:     "multiple arguments",
			input:    `<|tool_call>call:search{city:<|"|>Denver<|"|>,state:<|"|>CO<|"|>}<tool_call|>`,
			wantOK:   true,
			wantName: "search",
			wantArgs: map[string]string{"city": "Denver", "state": "CO"},
		},
		{
			name:   "no tool call",
			input:  "I can only provide weather for US territories.",
			wantOK: false,
		},
		{
			name:   "empty response",
			input:  "",
			wantOK: false,
		},
		{
			name:     "newline in value",
			input:    "<|tool_call>call:get_weather{location:<|\"|>Winter Haven,\nFL<|\"|>}<tool_call|>",
			wantOK:   true,
			wantName: "get_weather",
			wantArgs: map[string]string{"location": "Winter Haven,\nFL"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			call, ok := parseGemma4ToolCall(tt.input)
			if ok != tt.wantOK {
				t.Fatalf("parseGemma4ToolCall() ok = %v, want %v", ok, tt.wantOK)
			}
			if !ok {
				return
			}
			if call.Function.Name != tt.wantName {
				t.Errorf("name = %q, want %q", call.Function.Name, tt.wantName)
			}
			for k, want := range tt.wantArgs {
				got, exists := call.Function.Arguments[k]
				if !exists {
					t.Errorf("missing argument %q", k)
				} else if got != want {
					t.Errorf("argument %q = %q, want %q", k, got, want)
				}
			}
		})
	}
}

// TestGemma4NativeTokens verifies whether the loaded GGUF model's vocabulary
// recognizes Gemma 4's native control tokens as single special tokens.
//
// If a token is recognized, Tokenize with parseSpecial=true returns exactly
// 1 token (the special token ID). If it's NOT recognized, the literal text
// gets split into multiple regular tokens (e.g. "<", "|", "turn", ">").
//
// Run with:
//
//	YZMA_LIB=<path> YZMA_TEST_MODEL=<path-to-gemma4.gguf> go test -v -run TestGemma4NativeTokens ./weather-tool/
//
// The test skips (not fails) if the env vars aren't set, so it won't break
// CI or casual `go test ./...` runs.
func TestGemma4NativeTokens(t *testing.T) {
	libPath := os.Getenv("YZMA_LIB")
	if libPath == "" {
		t.Skip("YZMA_LIB not set, skipping")
	}
	modelPath := os.Getenv("YZMA_TEST_MODEL")
	if modelPath == "" {
		t.Skip("YZMA_TEST_MODEL not set, skipping")
	}

	// Load library and model
	if err := llama.Load(libPath); err != nil {
		t.Fatalf("llama.Load: %v", err)
	}
	llama.LogSet(llama.LogSilent())
	llama.Init()
	defer llama.Close()

	model, err := llama.ModelLoadFromFile(modelPath, llama.ModelDefaultParams())
	if err != nil {
		t.Fatalf("ModelLoadFromFile: %v", err)
	}
	defer llama.ModelFree(model)

	vocab := llama.ModelGetVocab(model)

	// Gemma 4 native tokens (from https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
	gemma4Tokens := []string{
		"<|turn>",
		"<turn|>",
		"<|tool>",
		"<tool|>",
		"<|tool_call>",
		"<tool_call|>",
		"<|tool_response>",
		"<tool_response|>",
		`<|"|>`,
		"<|think|>",
		"<|channel>",
		"<channel|>",
	}

	// Legacy Gemma 2/3 tokens (for comparison)
	legacyTokens := []string{
		"<start_of_turn>",
		"<end_of_turn>",
		"<bos>",
	}

	t.Log("=== Gemma 4 native tokens ===")
	for _, tok := range gemma4Tokens {
		tokens := llama.Tokenize(vocab, tok, false, true)
		status := "RECOGNIZED (single special token)"
		if len(tokens) != 1 {
			status = "NOT recognized (split into text tokens)"
		}
		t.Logf("  %-25s -> %d token(s), IDs: %-20v  %s", tok, len(tokens), tokens, status)
	}

	t.Log("")
	t.Log("=== Legacy Gemma 2/3 tokens (for comparison) ===")
	for _, tok := range legacyTokens {
		tokens := llama.Tokenize(vocab, tok, false, true)
		status := "RECOGNIZED (single special token)"
		if len(tokens) != 1 {
			status = "NOT recognized (split into text tokens)"
		}
		t.Logf("  %-25s -> %d token(s), IDs: %-20v  %s", tok, len(tokens), tokens, status)
	}
}
