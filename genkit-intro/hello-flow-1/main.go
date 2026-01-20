package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

type Greeting struct {
	LanguageFamily string
	Language       string
	Message        string
	History        string
}

type GreetRequest struct {
	LanguageFamily string
	Languages      []string
}

type GreetResponse struct {
	Greetings []Greeting
}

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

	greetFlow := genkit.DefineFlow[*GreetRequest, *GreetResponse](
		gkx,
		"greet_flow",
		func(ctx context.Context, req *GreetRequest) (*GreetResponse, error) {
			// setup the prompt
			prompt := fmt.Sprintf(
				"Generate greetings in this language array: [%s]",
				strings.Join(req.Languages, ", "),
			)

			// Send prompt to LLM and automatically parse structured response
			resp, _, err := genkit.GenerateData[GreetResponse](
				ctx,
				gkx, // Use the configured GenKit context
				ai.WithSystem(
					"You're a helpful assistant that generates greetings/salutations "+
						"with greeting history from different languages."+
						"When no specific language is provided, generate 5 greetings "+
						"in diverse language families.",
				),
				ai.WithModel(gemini),
				ai.WithPrompt(prompt),
			)
			if err != nil {
				return nil, err
			}

			return resp, nil
		},
	)
}
