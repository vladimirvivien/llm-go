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

// var count = 5
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
	greets := genkit.GenerateDataStream[Greeting](
		ctx,
		gkx, // Use the configured GenKit context
		ai.WithModel(gemini),
		ai.WithPrompt(fmt.Sprintf("Generate %d greetings in different languages", 10)),
	)

	for greet, err := range greets {
		fmt.Printf("Received greeting chunk: %+v\n", greet)
		if err != nil {
			log.Fatalf("greet stream: %v", err)
		}
		if greet.Done {
			break
		}

		// if greet.Chunk.Language != "" {
		// 	fmt.Printf("Greeting in %s: %s\n", greet.Chunk.Language, greet.Chunk.Text)
		// }
	}
}
