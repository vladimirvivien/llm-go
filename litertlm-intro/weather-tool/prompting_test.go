package main

import "testing"

func TestParseGemma4ToolCall(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantOK   bool
		wantName string
		wantArgs map[string]string
	}{
		{
			name:     "documented form: <|\"|> quoted, : separator",
			input:    `<|tool_call>call:get_weather{location:<|"|>Miami, FL<|"|>}<tool_call|>`,
			wantOK:   true,
			wantName: "get_weather",
			wantArgs: map[string]string{"location": "Miami, FL"},
		},
		{
			name:     "observed form: standard quoted, = separator",
			input:    `<|tool_call>call:get_weather{location="Miami, FL"}<tool_call|>`,
			wantOK:   true,
			wantName: "get_weather",
			wantArgs: map[string]string{"location": "Miami, FL"},
		},
		{
			name:     "valid JSON form: standard quoted keys and values",
			input:    `<|tool_call>call:get_weather{"location":"Miami, FL"}<tool_call|>`,
			wantOK:   true,
			wantName: "get_weather",
			wantArgs: map[string]string{"location": "Miami, FL"},
		},
		{
			name:     "with surrounding text",
			input:    `Let me check that for you. <|tool_call>call:get_weather{location:<|"|>Tampa, FL<|"|>}<tool_call|>`,
			wantOK:   true,
			wantName: "get_weather",
			wantArgs: map[string]string{"location": "Tampa, FL"},
		},
		{
			name:     "multiple arguments, native form",
			input:    `<|tool_call>call:search{city:<|"|>Denver<|"|>,state:<|"|>CO<|"|>}<tool_call|>`,
			wantOK:   true,
			wantName: "search",
			wantArgs: map[string]string{"city": "Denver", "state": "CO"},
		},
		{
			name:     "multiple arguments, = separator",
			input:    `<|tool_call>call:search{city="Denver",state="CO"}<tool_call|>`,
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
