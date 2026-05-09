// weather-autotool is the auto-dispatch twin of weather-tool-chat. The
// get_weather tool is registered with litertlm.RegisterTool; the
// framework dispatches it when the model invokes it and returns the
// post-tool natural-language answer in a single Chat.Send call.
package main

import (
	"context"
	"errors"
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
	maxToolHops     int
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
	flag.IntVar(&maxToolHops, "max-hops", 4, "max tool-call round-trips per Send")
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
	defer func() { _ = client.Close() }()
	slog.Info("model loaded", "model", modelFile, "elapsed", time.Since(loadStart).Round(time.Millisecond))

	weather, err := registerWeatherTool(client)
	if err != nil {
		slog.Error("failed to register weather tool", "err", err)
		os.Exit(1)
	}

	var prompt string
	if promptFlag != "" {
		prompt = promptFlag
	} else {
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
			if err := runChat(ctx, client, weather, prompt); err != nil {
				slog.Error("chat failed", "err", err)
				os.Exit(1)
			}
		}).Run()
	if err != nil {
		slog.Error("spinner error", "err", err)
	}
}

func createForm() (string, error) {
	var prompt string

	form := huh.NewForm(
		huh.NewGroup(
			huh.NewNote().
				Title("US Weather Forecast (Gemma 4 + LiteRT-LM Auto-Dispatch + NWS)").
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

func runChat(ctx context.Context, client *litertlm.Client, weather litertlm.ToolDefinition, userPrompt string) error {
	chat, err := client.NewChat(ctx,
		litertlm.WithSystemPrompt(getSystemPrompt()),
		litertlm.WithTool(weather),
		litertlm.WithMaxToolHops(maxToolHops),
	)
	if err != nil {
		return fmt.Errorf("new chat: %w", err)
	}
	defer func() { _ = chat.Close() }()

	start := time.Now()
	reply, err := chat.Send(ctx, userPrompt)
	dur := time.Since(start)
	if err != nil {
		var hops *litertlm.ToolHopsError
		if errors.As(err, &hops) {
			return fmt.Errorf("tool hop cap exceeded after %d iterations; last reply: %s", hops.Hops, hops.LastReply.Raw())
		}
		return fmt.Errorf("send: %w", err)
	}

	if verbose {
		fmt.Printf("\n=== Final Reply ===\n%s\n=============================\n", reply.Raw())
	}

	renderMarkdown(reply.Text())
	fmt.Println()
	fmt.Printf("Total wall-clock: %s\n", dur.Round(time.Millisecond))
	return nil
}

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
