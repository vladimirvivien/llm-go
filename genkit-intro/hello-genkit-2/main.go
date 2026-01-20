package main

import (
	"context"
	"log"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	anthropicplugin "github.com/firebase/genkit/go/plugins/anthropic"
)

func main() {
	ctx := context.Background()

	// Initialize a GenKit context with the Anthropic plugin.
	gkx := genkit.Init(
		ctx,
		// Anthropic Plugin automatically searches for ANTHROPIC_API_KEY env variable
		genkit.WithPlugins(&anthropicplugin.Anthropic{}),
	)

	// Next, define a Claude model instance to use
	claude := anthropicplugin.Model(gkx, "claude-sonnet-4-5-20250929")

	// Generate sends a prompt to instruct the LLM to perform a task.
	resp, err := genkit.Generate(
		ctx,
		gkx, // Use the configured GenKit context
		ai.WithModel(claude),
		// Required model configuration for Anthropic models
		ai.WithConfig(&anthropic.MessageNewParams{
			Temperature: anthropic.Float(1),
			MaxTokens:   1024,
		}),
		ai.WithPrompt("Say Hello in 5 languages"),
	)
	if err != nil {
		log.Fatal(err)
	}

	log.Println(resp.Text())
}
