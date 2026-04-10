package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/huh/spinner"
	"github.com/charmbracelet/lipgloss"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
	"github.com/hybridgroup/yzma/pkg/template"
)

const (
	systemPromptTemplate = `
You are a helpful weather assistant that ONLY handles weather requests 
for US territories (the 50 states, DC, Puerto Rico, US Virgin Islands, Guam, 
American Samoa, and Northern Mariana Islands). If the user asks about 
a non-US location, politely let them know that this application only supports 
weather forecasts for US territories. If the user asks something unrelated to 
weather, politely let them know that this application only handles weather requests. 
Do NOT call the get_weather tool for non-US locations. For valid US weather requests, 
use the get_weather tool to fetch real forecast data, then provide a clear, 
well-formatted summary. 

**Format your response in Markdown** with sections for current 
conditions and upcoming forecast periods. Use weather related emojis to help convey
conditions:

* Use ☀️ - for sunny forcast
* Use 🌤️ - for partial sunny condition
* Use 🌦️ - for sunny followed by rainy conditions
* Use ☁️ - for cloudy no rain conditions 
* Use 🌧️ - for mostly cloudy with possible rain conditions
* Use ⛈️ - for cloudy, rainy, with possible thunderstorm conditions
* Use 🌩️ - for cloudy, rainy, and lightning conditions
* Use 🌨️ - for cloudy and snow conditions
* Use 🌪️ - for possible tornadoes

	
You have access to the following tools:

%s

When you need to use a tool, respond with a tool call in the following format:
<tool_call>
{"name": "function_name", "arguments": {"arg1": "value1"}}
</tool_call>

After receiving tool results, provide a final answer to the user. 
DO NOT include tool calls in your final answer.
`
)

var (
	modelFile   string
	libPath     string
	verbose     bool
	maxTokens   int
	predictSize int
	temperature float64
)

func main() {
	// Parse flags
	home, _ := os.UserHomeDir()
	defaultModel := filepath.Join(home, "models", "gemma-4-E4B-it-Q8_0.gguf")

	flag.StringVar(&modelFile, "model", defaultModel, "path to GGUF model file")
	flag.StringVar(&libPath, "lib", os.Getenv("YZMA_LIB"), "path to llama.cpp compiled library (or set YZMA_LIB)")
	flag.BoolVar(&verbose, "v", false, "verbose logging")
	flag.IntVar(&maxTokens, "max-tokens", 4096, "max tokens in context")
	flag.IntVar(&predictSize, "n", 2048, "max tokens to predict per generation")
	flag.Float64Var(&temperature, "temperature", 0.1, "prediction temperature")
	flag.Parse()

	if libPath == "" {
		slog.Error("Error: provide -lib flag or set YZMA_LIB environment variable")
		os.Exit(1)
	}

	// Load llama.cpp library and initialize
	if err := llama.Load(libPath); err != nil {
		slog.Error("Failed to load llama library", "err", err)
		os.Exit(1)
	}

	if !verbose {
		llama.LogSet(llama.LogSilent())
	}

	// Initialize
	llama.Init()
	defer llama.Close()

	// Load model from model file
	model, err := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	if err != nil {
		slog.Error("Failed to load model", "err", err)
		os.Exit(1)
	}
	defer llama.ModelFree(model)

	// Create context
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(maxTokens)
	ctxParams.NBatch = 2048
	ctxParams.NUbatch = 2048

	ctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		slog.Error("Failed to create context", "err", err)
		os.Exit(1)
	}
	defer llama.Free(ctx)

	// Load model vocabulary and chat template
	vocab := llama.ModelGetVocab(model)
	chatTmpl := llama.ModelChatTemplate(model, "")
	if chatTmpl == "" {
		slog.Error("Model has no embedded chat template")
		os.Exit(1)
	}

	// Create sampler for model predictions
	sp := llama.DefaultSamplerParams()
	sp.Temp = float32(temperature)

	sampler := llama.NewSampler(model, llama.DefaultSamplers, sp)
	defer llama.SamplerFree(sampler)

	// Get tool definitions
	tools := getToolDefinitions()

	prompt, err := createForm()
	if err != nil {
		slog.Error("Failed to get user prompt", "err", err)
		os.Exit(1)
	}

	// Re-print the user's query so it stays visible after the form clears
	fmt.Printf("Asking: %s\n\n", prompt)

	err = spinner.New().
		Title("Getting weather info ...").
		Action(func() {
			if err := runConversation(ctx, vocab, sampler, chatTmpl, tools, prompt); err != nil {
				slog.Error("Conversation failed", "err", err)
				os.Exit(1)
			}
		}).Run()
	if err != nil {
		slog.Error("Spinner errorv", "err", err)
	}
}

// Create TUI form to get user prompt
func createForm() (string, error) {
	var prompt string

	form := huh.NewForm(
		huh.NewGroup(
			huh.NewNote().
				Title("US Weather Forecast (powered by Gemma + NWS)").
				Description("Ask about the weather for any US territory (Ctrl+C to cancel)"),

			huh.NewInput().
				Title("Ask about weather in any US location").
				Placeholder("e.g. What's the weather like in Washington, DC?").
				Value(&prompt).
				Validate(func(str string) error {
					if str == "" {
						return fmt.Errorf("please enter a weather related question")
					}
					return nil
				}),
		),
	)

	if err := form.Run(); err != nil {
		return "", fmt.Errorf("running form: %w", err)
	}

	return prompt, nil
}

// Return system prompt
func getSystemPrompt(toolSpec string) string {
	return fmt.Sprintf(systemPromptTemplate, toolSpec)
}

func runConversation(
	ctx llama.Context,
	vocab llama.Vocab,
	sampler llama.Sampler,
	chatTemplate string,
	tools []Tool,
	userPrompt string,
) error {
	toolsJSON := formatToolsForPrompt(tools)
	systemPrompt := getSystemPrompt(toolsJSON)

	// Initialize messages with system prompt and user question
	chatMessages := []message.Message{
		message.Chat{
			Role:    "system",
			Content: systemPrompt,
		},
		message.Chat{
			Role:    "user",
			Content: userPrompt,
		},
	}

	maxIterations := 5

	for i := range maxIterations {
		// Apply template and generate response
		prompt, err := template.Apply(chatTemplate, chatMessages, true)
		if err != nil {
			return fmt.Errorf("conversation: failed to apply template: %w", err)
		}

		if prompt == "" {
			return fmt.Errorf("conversation: failed to apply chat template")
		}

		if verbose {
			fmt.Printf("\n=== Iteration %d Prompt (%d chars) ===\n%s\n===========================\n", i+1, len(prompt), prompt)
		}

		// Clear KV cache before each full-prompt generation
		mem, err := llama.GetMemory(ctx)
		if err != nil {
			return fmt.Errorf("conversation: KV cache error: %w", err)
		}
		llama.MemoryClear(mem, true)

		tokens := llama.Tokenize(vocab, prompt, false, true)
		if len(tokens) == 0 {
			return fmt.Errorf("conversation: tokenization produced no tokens")
		}
		if verbose {
			fmt.Printf("Token count: %d\n", len(tokens))
		}

		// Generate prompt response
		response := generate(ctx, vocab, sampler, tokens)

		if verbose {
			fmt.Printf("\n=== Iteration %d Response ===\n%s\n=============================\n", i+1, response)
		}

		toolCalls := message.ParseToolCalls(response)

		// No tool calls — this is the final answer.
		if len(toolCalls) == 0 {
			renderMarkdown(response)
			return nil
		}

		// Process tool calls found
		for _, call := range toolCalls {
			argsJSON, err := json.Marshal(call.Function.Arguments)
			if err != nil {
				return fmt.Errorf("conversation tool call: %w", err)
			}

			result, err := executeToolCall(call)
			if err != nil {
				return fmt.Errorf("conversation tool call: %w", err)
			}
			if verbose {
				fmt.Printf("Tool call: %s(%s) => %s\n", call.Function.Name, string(argsJSON), result[:min(len(result), 200)])
			}

			// Add tool call and response to messages
			chatMessages = append(chatMessages, message.Tool{
				Role:      "assistant",
				ToolCalls: []message.ToolCall{call},
			})
			chatMessages = append(chatMessages, message.ToolResponse{
				Role:    "tool",
				Name:    call.Function.Name,
				Content: result,
			})
		}

	}
	slog.Warn("Maximum tool-call iterations reached without final answer", "max", maxIterations)
	return nil
}

// renderMarkdown renders a response string as styled Markdown in a bordered box.
func renderMarkdown(text string) {
	out, err := glamour.Render(text, "dark")
	if err != nil {
		fmt.Println(text)
		return
	}

	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("63")).
		Padding(0, 1)

	fmt.Println(box.Render(strings.TrimSpace(out)))
}

// generate runs token-by-token inference for up to predictSize tokens or until
// the model emits an end-of-generation token, then returns the decoded response.
func generate(ctx llama.Context, vocab llama.Vocab, sampler llama.Sampler, tokens []llama.Token) string {
	var response strings.Builder

	batch := llama.BatchGetOne(tokens)
	llama.Decode(ctx, batch)

	for pos := int32(0); pos < int32(predictSize); pos += batch.NTokens {
		token := llama.SamplerSample(sampler, ctx, -1)

		if llama.VocabIsEOG(vocab, token) {
			break
		}

		buf := make([]byte, 256)
		n := llama.TokenToPiece(vocab, token, buf, 0, false)
		response.Write(buf[:n])

		batch = llama.BatchGetOne([]llama.Token{token})
		llama.Decode(ctx, batch)
	}

	return strings.TrimSpace(response.String())
}
