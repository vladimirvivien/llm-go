// weather-tool runs a small tool-using agent against a local Gemma 4
// .litertlm model: the user asks about US weather, the model emits a
// Gemma 4 native <|tool_call>, the program calls the National Weather
// Service API for real forecast data, and the model produces a styled
// Markdown summary using the result.
//
// This file uses litertlm-go's high-level Client API for the engine
// lifecycle and per-call generation. The hand-rendered Gemma 4 native
// prompts (prompting.go) and the regex-based tool-call parser stay —
// they're the showcase for how to drive the model at the native-token
// level when you don't want to go through the higher-level Chat
// helper. For a fully high-level tool-calling flow, see the sibling
// weather-tool-convo example.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/huh/spinner"
	"github.com/charmbracelet/lipgloss"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

var (
	modelFile       string
	libPath         string
	backend         string
	verbose         bool
	temperature     float64
	engineMaxTokens int
	promptFlag      string
)

// maxTokens caps the per-call output budget for the pass-2
// (final-Markdown) generation. The model's natural output here is
// ~200-250 tokens; 512 is a safety belt, not a binding cap. To
// actually shave wall-clock time, trim the system prompt's output
// template instead.
const maxTokens = 512

func main() {
	home, _ := os.UserHomeDir()
	defaultModel := filepath.Join(home, "models", "gemma-4-E2B-it.litertlm")
	defaultLib := filepath.Join(home, "include", "litertlm", "lib")

	flag.StringVar(&modelFile, "model", defaultModel, "path to .litertlm model file")
	flag.StringVar(&libPath, "lib", envOr("LITERTLM_LIB", defaultLib), "directory holding the LiteRT-LM shared libraries")
	flag.StringVar(&backend, "backend", envOr("LITERTLM_BACKEND", "cpu"), "inference backend (cpu | gpu)")
	flag.BoolVar(&verbose, "v", false, "verbose logging")
	flag.Float64Var(&temperature, "temperature", 0.5, "prediction temperature (0 = greedy)")
	flag.IntVar(&engineMaxTokens, "max", 4096, "max total tokens for the engine (prompt + output)")
	flag.StringVar(&promptFlag, "prompt", "", "prompt to use directly (skip the TUI form)")
	flag.Parse()

	ctx := context.Background()

	opts := []litertlm.Option{
		litertlm.WithLib(libPath),
		litertlm.WithModel(modelFile),
		litertlm.WithBackend(backend),
		litertlm.WithMaxTokens(engineMaxTokens),
	}
	if verbose {
		opts = append(opts, litertlm.WithLogLevel(litertlm.LogInfo))
	}
	if temperature > 0 {
		sampler := litertlm.DefaultSamplerParams()
		sampler.Type = litertlm.SamplerTopP
		sampler.Temperature = float32(temperature)
		sampler.TopP = 0.95
		opts = append(opts, litertlm.WithDefaultSampler(sampler))
	}

	loadStart := time.Now()
	client, err := litertlm.New(ctx, opts...)
	if err != nil {
		slog.Error("failed to construct litertlm client", "err", err)
		os.Exit(1)
	}
	defer client.Close()
	slog.Info("model loaded", "model", modelFile, "elapsed", time.Since(loadStart).Round(time.Millisecond))

	tools := getToolDefinitions()

	var prompt string
	if promptFlag != "" {
		prompt = promptFlag
	} else {
		var err error
		prompt, err = createForm()
		if err != nil {
			slog.Error("failed to get user prompt", "err", err)
			os.Exit(1)
		}
	}

	fmt.Printf("Asking: %s\n\n", prompt)

	err = spinner.New().
		Title("Getting weather info ...").
		Action(func() {
			if err := runConversation(ctx, client, tools, prompt); err != nil {
				slog.Error("conversation failed", "err", err)
				os.Exit(1)
			}
		}).Run()
	if err != nil {
		slog.Error("spinner error", "err", err)
	}
}

// createForm shows a TUI form to collect the user's weather question.
func createForm() (string, error) {
	var prompt string

	form := huh.NewForm(
		huh.NewGroup(
			huh.NewNote().
				Title("US Weather Forecast (powered by Gemma 4 + LiteRT-LM + NWS)").
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

// runConversation drives the two-pass tool-calling flow on top of
// Client.Generate:
//
//  1. Render system + tool declarations + user prompt with an open
//     model turn. Generate. Parse a <|tool_call> from the output.
//  2. If a tool call is present, execute it, splice the call and a
//     <|tool_response> back into the same open model turn, generate
//     again, and render the final answer as styled Markdown.
//
// Each Client.Generate call opens a fresh internal Session, so the
// two passes are naturally independent — the equivalent of the
// per-pass session reset the prior low-level version did by hand.
//
// Per-phase wall-clock times are printed at the end so the caller can
// see where wall time is spent (model vs. NWS HTTP).
func runConversation(ctx context.Context, client *litertlm.Client, tools []Tool, userPrompt string) error {
	systemPrompt := getSystemPrompt()
	toolDeclarations := renderToolDeclarations(tools)

	prompt := renderGemma4Prompt(systemPrompt, toolDeclarations, userPrompt)
	if verbose {
		fmt.Printf("\n=== Pass 1 Prompt (%d chars) ===\n%s\n===========================\n", len(prompt), prompt)
	}
	pass1Start := time.Now()
	// Pass 1 is intentionally uncapped — its output is just the
	// <|tool_call>...<tool_call|> block, naturally short (~50 tokens).
	response, err := generate(ctx, client, prompt)
	pass1Dur := time.Since(pass1Start)
	if err != nil {
		return fmt.Errorf("pass 1 generation: %w", err)
	}
	if verbose {
		fmt.Printf("\n=== Pass 1 Response ===\n%s\n=============================\n", response)
	}

	call, found := parseGemma4ToolCall(response)
	if !found {
		if strings.Contains(response, "<|tool_call>") {
			return fmt.Errorf("model emitted a tool call but parsing failed: %s", response)
		}
		renderMarkdown(response)
		printTimings(pass1Dur, 0, 0)
		return nil
	}

	toolStart := time.Now()
	result, err := executeToolCall(call)
	toolDur := time.Since(toolStart)
	if err != nil {
		result = fmt.Sprintf("Error: %v", err)
	}
	if verbose {
		fmt.Printf("Tool call: %s(%v) => %s\n", call.Function.Name, call.Function.Arguments, result[:min(len(result), 200)])
	}

	// Use the model's raw pass-1 output verbatim as the tool_call text
	// in pass 2. Whichever encoding shape the model chose
	// (`<|"|>...<|"|>`, `="..."`, or proper JSON) is preserved exactly,
	// keeping the model's own conditioning consistent.
	toolCallText := strings.TrimSpace(response)
	toolResponseText := fmt.Sprintf(
		"<|tool_response>response:%s{%s}<tool_response|>\n",
		call.Function.Name, result)

	prompt = renderGemma4PromptWithToolResult(
		systemPrompt, toolDeclarations, userPrompt,
		toolCallText, toolResponseText)
	if verbose {
		fmt.Printf("\n=== Pass 2 Prompt (%d chars) ===\n%s\n===========================\n", len(prompt), prompt)
	}
	pass2Start := time.Now()
	// Pass 2 produces the rendered Markdown forecast — the wall-time
	// dominator at default settings. Cap output tokens so the model
	// can't ramble.
	response, err = generate(ctx, client, prompt, litertlm.WithMaxOutputTokens(maxTokens))
	pass2Dur := time.Since(pass2Start)
	if err != nil {
		return fmt.Errorf("pass 2 generation: %w", err)
	}
	if verbose {
		fmt.Printf("\n=== Pass 2 Response ===\n%s\n=============================\n", response)
	}

	renderMarkdown(response)
	printTimings(pass1Dur, toolDur, pass2Dur)
	return nil
}

// printTimings writes a short per-phase wall-clock breakdown so users
// can see where the run spent its time. Zero durations (e.g. when the
// model answered without invoking a tool) are reported as "n/a".
func printTimings(pass1, tool, pass2 time.Duration) {
	round := func(d time.Duration) string {
		if d == 0 {
			return "n/a"
		}
		return d.Round(time.Millisecond).String()
	}
	total := pass1 + tool + pass2
	fmt.Println()
	fmt.Println("Timings:")
	fmt.Printf("  pass 1 (model): %s\n", round(pass1))
	fmt.Printf("  tool   (NWS):   %s\n", round(tool))
	fmt.Printf("  pass 2 (model): %s\n", round(pass2))
	fmt.Printf("  total:          %s\n", total.Round(time.Millisecond))
}

// generate is a thin wrapper around Client.Generate that trims the
// response. The trim keeps the existing behaviour where the parser is
// fed text without trailing whitespace, which the regex relies on.
// Per-call opts (e.g. WithMaxOutputTokens) are forwarded as-is.
func generate(ctx context.Context, client *litertlm.Client, prompt string, opts ...litertlm.GenOption) (string, error) {
	out, err := client.Generate(ctx, prompt, opts...)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(out), nil
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

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
