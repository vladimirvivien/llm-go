package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/huh/spinner"
	"github.com/charmbracelet/lipgloss"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

// WeatherInput is the tool input type — the LLM fills this when calling the tool.
type WeatherInput struct {
	Location string `json:"location" jsonschema:"description=City and state to get weather for"`
}

func main() {
	ctx := context.Background()

	// Initialize a GenKit context with a Google AI plugin
	gkx := genkit.Init(
		ctx,
		// Google AI Plugin automatically looks for GEMINI_API_KEY env variable
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
	)

	// Next, define a Google Gemini model instance to use
	gemini := googlegenai.GoogleAIModel(gkx, "gemini-3-flash-preview")

	// Define the weather tool that the LLM can call
	weatherTool := genkit.DefineTool(gkx, "get_weather",
		"Get the weather forecast for a US location by city and state name",
		func(ctx *ai.ToolContext, input WeatherInput) (*Forecast, error) {
			return GetForecast(input.Location)
		},
	)

	// TUI form to capture user input
	var prompt string

	form := huh.NewForm(
		huh.NewGroup(
			huh.NewNote().
				Title("US Weather Forecast (powered by Gemini + NWS)").
				Description("Ask about the weather for any US territory (Ctrl+C to cancel)"),

			huh.NewInput().
				Title("Ask about US weather").
				Placeholder("e.g. What's the weather like in Washington, DC?").
				Value(&prompt).
				Validate(func(str string) error {
					if str == "" {
						return fmt.Errorf("please enter a question")
					}
					return nil
				}),
		),
	)

	if err := form.Run(); err != nil {
		os.Exit(0)
	}

	// Re-print the user's query so it stays visible after the form clears
	fmt.Printf("Ask about the weather: %s\n\n", prompt)

	// Generate response using Gemini with the weather tool
	var resp *ai.ModelResponse
	err := spinner.New().
		Title("Fetching weather forecast...").
		Action(func() {
			var err error
			resp, err = genkit.Generate(ctx, gkx,
				ai.WithModel(gemini),
				ai.WithTools(weatherTool),
				ai.WithSystem(
					"You are a helpful weather assistant that ONLY handles weather requests "+
						"for US territories (the 50 states, DC, Puerto Rico, US Virgin Islands, Guam, "+
						"American Samoa, and Northern Mariana Islands). "+
						"If the user asks about a non-US location, politely let them know that this "+
						"application only supports weather forecasts for US territories. "+
						"If the user asks something unrelated to weather, politely let them know that "+
						"this application only handles weather requests. "+
						"Do NOT call the get_weather tool for non-US locations. "+
						"For valid US weather requests, use the get_weather tool to fetch real forecast data, "+
						"then provide a clear, well-formatted summary. "+
						"Format your response in Markdown with sections for current conditions "+
						"and upcoming forecast periods.",
				),
				ai.WithPrompt(prompt),
			)
			if err != nil {
				log.Fatalf("Generate error: %v", err)
			}
		}).Run()
	if err != nil {
		log.Fatalf("Spinner error: %v", err)
	}

	// Render the Markdown response with Glamour
	out, err := glamour.Render(resp.Text(), "dark")
	if err != nil {
		log.Fatalf("Rendering error: %v", err)
	}

	// Display output in a rounded border box
	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("63")).
		Padding(0, 1)

	fmt.Println(box.Render(strings.TrimSpace(out)))
}
