package main

import (
	"context"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

// Greeting represents a greeting message in a specific language
type Greeting struct {
	Language string `json:"language"`
	Text     string `json:"text"`
}

// Greetings is a collection of Greeting
type Greetings struct {
	Greetings []Greeting `json:"greetings"`
}

func main() {
	ctx := context.Background()

	// First, initialize a GenKit context with a Google AI plugin
	gkx := genkit.Init(
		ctx,
		// Google AI Plugin automatically looks for GEMINI_API_KEY env variable
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
	)

	// Next, define a Google Gemini model instance to use
	gemini := googlegenai.GoogleAIModel(gkx, "gemini-3-flash-preview")

	// Define a prompt to instruct the LLM to generate greetings.
	// genkit.GenerateData automatically prepares the prompt to use
	// the provided struct to instruct the LLM to return structured data
	// that fits the specified type.
	greets, _, err := genkit.GenerateData[Greetings](
		ctx,
		gkx, // Use the configured GenKit context
		ai.WithModel(gemini),
		ai.WithPrompt("Say Hello in 5 languages"),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Greetings: %v\n", greets)
}
