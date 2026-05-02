package main

import (
	"fmt"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// getToolDefinitions declares the get_weather tool in the OpenAI /
// Anthropic function-calling schema. The Chat API renders this into
// the model's native tool-declaration format internally.
func getToolDefinitions() []litertlm.Tool {
	return []litertlm.Tool{
		{
			Type: "function",
			Function: litertlm.ToolFunction{
				Name:        "get_weather",
				Description: "Get the weather forecast for a US location by city and state name",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "City and state to get weather for, e.g. 'Washington, DC'",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}
}

// executeToolCall dispatches a model-emitted tool call. The return
// value is JSON-marshaled directly by Chat.SendToolResult, so a struct
// or map[string]any is what we want — not a pre-serialised string.
func executeToolCall(call litertlm.ToolCall) (any, error) {
	switch call.Function.Name {
	case "get_weather":
		location, err := stringArg(call.Function.Arguments, "location")
		if err != nil {
			return nil, err
		}
		forecast, err := GetForecast(location)
		if err != nil {
			return nil, fmt.Errorf("weather lookup failed: %w", err)
		}
		return forecast, nil
	default:
		return nil, fmt.Errorf("unknown function: %s", call.Function.Name)
	}
}

// stringArg pulls a string-valued argument out of the model's
// tool-call. *Reply already strips Gemma 4's <|"|> quote markers, so
// no further sanitisation is needed.
func stringArg(args map[string]any, key string) (string, error) {
	v, ok := args[key]
	if !ok {
		return "", fmt.Errorf("missing %q argument", key)
	}
	s, ok := v.(string)
	if !ok || s == "" {
		return "", fmt.Errorf("argument %q must be a non-empty string (got %T)", key, v)
	}
	return s, nil
}
