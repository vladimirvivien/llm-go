// weather-tool-chat is a high-level twin of ../weather-tool. Same demo
// (US weather Q&A backed by the National Weather Service), but built
// on litertlm-go's Chat API:
//
//   - tools are declared as []litertlm.Tool (OpenAI-style schema),
//   - the chat template renders the model's native tool-declaration
//     syntax for us,
//   - the model's tool call comes back as a structured *Reply, and
//   - the result is fed back via Chat.SendToolResult.
//
// No hand-rolled <|tool_call> parser, no Gemma 4 native-token wrangling.
// Compare main.go here with ../weather-tool/main.go + prompting.go to
// see what the higher-level API absorbs.
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
			if err := runChat(ctx, client, prompt); err != nil {
				slog.Error("chat failed", "err", err)
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
				Title("US Weather Forecast (Gemma 4 + LiteRT-LM Chat API + NWS)").
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

// runChat drives the tool-using flow on the high-level Chat API:
//
//  1. Open a Chat with system prompt + tool declarations.
//  2. Send the user message; if the reply has a tool_call, execute it
//     and SendToolResult back. The Chat preserves history internally,
//     so the second turn already sees the user message and the model's
//     prior tool_call.
//  3. Render the final natural-language answer as styled Markdown.
//
// Per-phase wall-clock times are printed so the caller can see where
// time is spent (model passes vs. NWS HTTP).
func runChat(ctx context.Context, client *litertlm.Client, userPrompt string) error {
	chat, err := client.NewChat(ctx,
		litertlm.WithSystemPrompt(getSystemPrompt()),
		litertlm.WithTools(getToolDefinitions()),
	)
	if err != nil {
		return fmt.Errorf("new chat: %w", err)
	}
	defer chat.Close()

	pass1Start := time.Now()
	reply, err := chat.Send(ctx, userPrompt)
	pass1Dur := time.Since(pass1Start)
	if err != nil {
		return fmt.Errorf("pass 1 send: %w", err)
	}
	if verbose {
		fmt.Printf("\n=== Pass 1 Reply ===\n%s\n=============================\n", reply.Raw())
	}

	if !reply.HasToolCalls() {
		// Model answered directly (e.g. politely declined a non-US
		// request). No tool dispatch needed.
		renderMarkdown(reply.Text())
		printTimings(pass1Dur, 0, 0)
		return nil
	}

	call := reply.ToolCalls()[0]
	if verbose {
		fmt.Printf("Tool call: %s(%v)\n", call.Function.Name, call.Function.Arguments)
	}

	toolStart := time.Now()
	result, err := executeToolCall(call)
	toolDur := time.Since(toolStart)
	if err != nil {
		// Surface the error back to the model so it can apologise.
		result = map[string]string{"error": err.Error()}
	}
	if verbose {
		fmt.Printf("Tool result: %v\n", truncate(fmt.Sprintf("%v", result), 200))
	}

	pass2Start := time.Now()
	final, err := chat.SendToolResult(ctx, call.Function.Name, result)
	pass2Dur := time.Since(pass2Start)
	if err != nil {
		return fmt.Errorf("pass 2 send tool result: %w", err)
	}
	if verbose {
		fmt.Printf("\n=== Pass 2 Reply ===\n%s\n=============================\n", final.Raw())
	}

	renderMarkdown(final.Text())
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

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
