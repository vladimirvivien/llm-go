package main

import (
	"context"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
	ctx := context.Background()

	// First, initialize a GenKit context with a Google AI as LLM provider
	gkx := genkit.Init(
		ctx,
		// Google AI Plugin automatically looks for GEMINI_API_KEY env variable
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
	)

	// Next, define a Google Gemini model instance to use
	gemini := googlegenai.GoogleAIModel(gkx, "gemini-3-flash-preview")

	// Define a prompt to instruct the LLM to generate content.
	resp, err := genkit.Generate(
		ctx,
		gkx, // Use the configured GenKit context
		ai.WithModel(gemini),
		ai.WithPrompt("Say Hello in 5 languages"),
	)
	if err != nil {
		log.Fatal(err)
	}

	log.Println(resp.Text())
}
