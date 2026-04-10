# Proposal: US Weather Forecast Tool with Yzma + Gemma 4

## Overview

Reimplement the GenKit-based weather tool application using **yzma** for fully local inference with the **Gemma 4 E4B** model (`~/models/gemma-4-E4B-it-Q8_0.gguf`). The app retains the same user-facing behavior — a TUI that answers US weather questions using NWS data — but runs entirely on-device via llama.cpp, with no cloud API keys required.

## Architecture Comparison

| Aspect | GenKit (current) | Yzma (proposed) |
|---|---|---|
| Model | Gemini 3 Flash (cloud) | Gemma 4 E4B (local GGUF) |
| Inference | Google AI API | llama.cpp via yzma (in-process) |
| Tool calling | GenKit framework (`ai.WithTools`) | Manual: system prompt + `<tool_call>` XML parsing |
| Dependencies | GenKit, Google AI plugin | yzma `pkg/llama`, `pkg/message`, `pkg/template` |
| API key | `GEMINI_API_KEY` required | None (local model) |
| Env var | — | `YZMA_LIB` (path to llama.cpp shared lib) |

## Proposed File Structure

```
yzma-intro/
├── go.mod
├── proposal.md
├── main.go        # Entry point: yzma init, model load, TUI, orchestration
├── weather.go     # Weather data fetching (reused from genkit version, mostly unchanged)
└── tools.go       # Tool definitions, tool call execution, tool conversation loop
```

## Implementation Plan

### 1. `weather.go` — Reuse As-Is

Copy `weather.go` from the GenKit version with minimal changes:
- `Forecast`, `ForecastPeriod`, `NWSGridPoint` structs
- `GetForecast()`, `geocode()`, `getGridPoint()`, `getForecast()` functions
- Update the `User-Agent` header from `genkit-weather-tool/1.0` to `yzma-weather-tool/1.0`

No other changes needed — the NWS/Nominatim logic is framework-agnostic.

### 2. `tools.go` — Tool Definitions and Execution

Define the weather tool using the yzma tool schema pattern:

```go
type Tool struct {
    Type     string       `json:"type"`
    Function ToolFunction `json:"function"`
}

type ToolFunction struct {
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Parameters  map[string]interface{} `json:"parameters"`
}
```

**Tool definition:**
```go
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
                            "description": "City and state to get weather for (e.g. 'Washington, DC')",
                        },
                    },
                    "required": []string{"location"},
                },
            },
        },
    }
}
```

**Tool execution** — dispatch on function name, extract `location` argument, call `GetForecast()`, return JSON-serialized `Forecast`:
```go
func executeToolCall(call message.ToolCall) (string, error) {
    switch call.Function.Name {
    case "get_weather":
        location := call.Function.Arguments["location"]
        forecast, err := GetForecast(location)
        if err != nil {
            return "", err
        }
        result, _ := json.Marshal(forecast)
        return string(result), nil
    default:
        return "", fmt.Errorf("unknown tool: %s", call.Function.Name)
    }
}
```

### 3. `main.go` — Yzma Setup, TUI, and Orchestration

#### a) Yzma initialization
```go
llama.Load(os.Getenv("YZMA_LIB"))
llama.LogSet(llama.LogSilent())
llama.Init()

model, _ := llama.ModelLoadFromFile(
    filepath.Join(os.Getenv("HOME"), "models", "gemma-4-E4B-it-Q8_0.gguf"),
    llama.ModelDefaultParams(),
)
defer llama.ModelFree(model)

params := llama.ContextDefaultParams()
params.NCtx = 4096
ctx, _ := llama.InitFromModel(model, params)
defer llama.Free(ctx)

vocab := llama.ModelGetVocab(model)
chatTemplate := llama.ModelChatTemplate(model, "")
```

#### b) Sampler setup
```go
sp := llama.DefaultSamplerParams()
sp.Temp = 0.1  // Low temperature for factual weather responses
sampler := llama.NewSampler(model, llama.DefaultSamplers, sp)
defer llama.SamplerFree(sampler)
```

#### c) TUI (Charmbracelet — same as GenKit version)
- `huh.NewForm` to capture user input
- `spinner.New()` wrapping the inference + tool-calling loop
- `glamour.Render()` for Markdown output
- `lipgloss` styled box for final display

#### d) Tool-calling conversation loop

Follow the yzma multitool pattern:

1. Build system prompt with tool definitions JSON and the same behavioral instructions from the GenKit version (US-only, weather-only, Markdown output)
2. Construct messages: `system` + `user`
3. Loop (max 5 iterations):
   - `template.Apply(chatTemplate, messages, true)` to format the prompt
   - Tokenize, decode, and sample to generate a response
   - `message.ParseToolCalls(response)` to detect `<tool_call>` tags
   - If tool calls found: execute `get_weather`, append `message.Tool` + `message.ToolResponse` to messages, clear KV cache, loop
   - If no tool calls: break — response is the final answer
4. Render the final response as Markdown in TUI

### 4. System Prompt

```
You are a helpful weather assistant that ONLY handles weather requests
for US territories (the 50 states, DC, Puerto Rico, US Virgin Islands,
Guam, American Samoa, and Northern Mariana Islands).

You have access to the following tools:
{tools_json}

When you need to use a tool, respond with a tool call in the following format:
<tool_call>
{"name": "function_name", "arguments": {"arg1": "value1"}}
</tool_call>

If the user asks about a non-US location, politely decline.
If the user asks something unrelated to weather, politely decline.
Do NOT call get_weather for non-US locations.

For valid US weather requests, use the get_weather tool, then provide
a clear, well-formatted Markdown summary with current conditions
and upcoming forecast periods.
After receiving tool results, provide a final answer. Do not include
tool calls in your final answer.
```

## Key Differences from GenKit Version

1. **No API key needed** — runs fully local on Apple Silicon via Metal
2. **Manual tool-call loop** — GenKit abstracts tool calling; yzma requires explicit parsing of `<tool_call>` XML tags and re-prompting with tool results
3. **KV cache management** — must call `llama.MemoryClear()` before re-tokenizing the full conversation on each loop iteration
4. **Token generation** — manual decode/sample loop instead of a single `genkit.Generate()` call
5. **Template system** — uses the model's built-in Jinja chat template via `template.Apply()` instead of GenKit's message formatting

## Configuration

| Setting | Value | Rationale |
|---|---|---|
| Model | `~/models/gemma-4-E4B-it-Q8_0.gguf` | Gemma 4 E4B instruction-tuned, Q8 quantization |
| Context size | 4096 | Sufficient for system prompt + tool defs + 5 forecast periods |
| Temperature | 0.1 | Low for factual, deterministic weather responses |
| Max predict | 1024 | Enough for a full Markdown weather summary |
| Max tool iterations | 5 | Safety bound; weather queries should need exactly 1 tool call |

## Dependencies

```
github.com/hybridgroup/yzma          # llama.cpp Go bindings (pkg/llama, pkg/message, pkg/template)
github.com/charmbracelet/glamour     # Markdown rendering
github.com/charmbracelet/huh         # TUI forms
github.com/charmbracelet/lipgloss    # Styled output
```

## Runtime Requirements

- `YZMA_LIB` environment variable pointing to the compiled llama.cpp shared library
- macOS arm64 with Metal acceleration (Apple Silicon)
- `~/models/gemma-4-E4B-it-Q8_0.gguf` model file (already downloaded)
