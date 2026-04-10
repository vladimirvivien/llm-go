package main

import (
	"encoding/json"
	"fmt"

	"github.com/hybridgroup/yzma/pkg/message"
)

// Tool represents a tool definition for the LLM.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction represents a function definition.
type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

func getToolDefinitions() []Tool {
	return []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "get_weather",
				Description: "Get the weather forecast for a US location by city and state name",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
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

// formatToolsForPrompt formats the tool definitions as a JSON string for the system prompt.
func formatToolsForPrompt(tools []Tool) string {
	toolsJSON, err := json.MarshalIndent(tools, "", "  ")
	if err != nil {
		return "[]"
	}
	return string(toolsJSON)
}

// executeToolCall executes a tool call and returns the result.
func executeToolCall(call message.ToolCall) (string, error) {
	switch call.Function.Name {
	case "get_weather":
		location, ok := call.Function.Arguments["location"]
		if !ok || location == "" {
			return "", fmt.Errorf("missing 'location' argument")
		}
		forecast, err := GetForecast(location)
		if err != nil {
			return "", fmt.Errorf("weather lookup failed: %w", err)
		}
		result, err := json.Marshal(forecast)
		if err != nil {
			return "", fmt.Errorf("marshal forecast: %w", err)
		}
		return string(result), nil
	default:
		return "", fmt.Errorf("unknown function: %s", call.Function.Name)
	}
}
