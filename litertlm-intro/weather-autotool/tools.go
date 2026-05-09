package main

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

type WeatherIn struct {
	Location string `json:"location" description:"City and state to get weather for, e.g. 'Washington, DC'"`
}

func registerWeatherTool(client *litertlm.Client) (litertlm.ToolDefinition, error) {
	return litertlm.RegisterTool(client, "get_weather",
		"Get the weather forecast for a US location by city and state name",
		func(ctx context.Context, in WeatherIn) (*Forecast, error) {
			if in.Location == "" {
				return nil, fmt.Errorf("location is required")
			}
			slog.Info("tool invoked", "tool", "get_weather", "location", in.Location)
			return GetForecast(in.Location)
		},
		litertlm.WithToolPolicy(litertlm.ToolPolicyInformOnError),
	)
}
