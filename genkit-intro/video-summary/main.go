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
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

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

	// Define a Huh form to capture video URL and prompt
	var videoURL string
	var userPrompt string

	form := huh.NewForm(
		huh.NewGroup(
			huh.NewNote().
				Title("Gemini 3 YouTube Video Analyzer").
				Description("Provide a YouTube video URL to analyze (Ctrl+C to cancel)"),

			huh.NewInput().
				Title("Video URL").
				Placeholder("https://www.youtube.com/watch?v=...").
				Value(&videoURL).
				Validate(func(str string) error {
					if !strings.Contains(str, "youtube.com") && !strings.Contains(str, "youtu.be") {
						return fmt.Errorf("that doesn't look like a YouTube URL")
					}
					return nil
				}),

			huh.NewInput().
				Title("Prompt").
				Placeholder("Summarize this video").
				Value(&userPrompt),
		),
	)

	// Display and capture form input
	if err := form.Run(); err != nil {
		os.Exit(0)
	}

	// Set default prompt if empty
	if userPrompt == "" {
		userPrompt = "Please summarize the following video details."
	}

	// Request content generation as a stream
	stream := genkit.GenerateStream(
		ctx,
		gkx, // Use the configured GenKit context
		ai.WithModel(gemini),
		ai.WithSystem(
			"You're an expert video content analyzer. "+
				"Provide detailed and insightful summaries. "+
				"Format your response in well-structured Markdown."+
				"Include a title, sections, bullet points, and highlights as appropriate.",
		),
		ai.WithMessages(
			ai.NewUserMessage(
				ai.NewMediaPart("video/mp4", videoURL),
				ai.NewTextPart(userPrompt),
			),
		),
	)

	var response strings.Builder
	err := spinner.New().
		Title("Analyzing video (this may take a while)...").
		Action(func() {
			for text, err := range stream {
				if err != nil {
					log.Fatalf("Response stream error: %v", err)
				}
				if text.Done {
					break
				}
				response.WriteString(text.Chunk.Text())
			}
		}).Run()
	if err != nil {
		log.Fatalf("Spinner error: %v", err)
	}
	// Render the Markdown response with Glamour
	out, err := glamour.Render(response.String(), "dark")
	if err != nil {
		log.Fatalf("Rendering error: %v", err)
	}

	fmt.Print(out)
}
