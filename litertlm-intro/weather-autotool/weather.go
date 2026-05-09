package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"time"
)

var httpClient = &http.Client{Timeout: 30 * time.Second}

type Forecast struct {
	Location string           `json:"location"`
	Periods  []ForecastPeriod `json:"periods"`
}

type ForecastPeriod struct {
	Name             string `json:"name"`
	Temperature      int    `json:"temperature"`
	TemperatureUnit  string `json:"temperatureUnit"`
	WindSpeed        string `json:"windSpeed"`
	WindDirection    string `json:"windDirection"`
	ShortForecast    string `json:"shortForecast"`
	DetailedForecast string `json:"detailedForecast"`
}

type NWSGridPoint struct {
	ForecastURL string `json:"forecast"`
	City        string `json:"city"`
	State       string `json:"state"`
}

func GetForecast(location string) (*Forecast, error) {
	lat, lon, err := geocode(location)
	if err != nil {
		return nil, fmt.Errorf("geocode %q: %w", location, err)
	}

	grid, err := getGridPoint(lat, lon)
	if err != nil {
		return nil, fmt.Errorf("grid point: %w", err)
	}

	periods, err := getForecast(grid.ForecastURL)
	if err != nil {
		return nil, fmt.Errorf("forecast: %w", err)
	}

	return &Forecast{
		Location: fmt.Sprintf("%s, %s", grid.City, grid.State),
		Periods:  periods,
	}, nil
}

func geocode(location string) (lat, lon float64, err error) {
	u := fmt.Sprintf(
		"https://nominatim.openstreetmap.org/search?q=%s&format=jsonv2&limit=1",
		url.QueryEscape(location),
	)

	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		return 0, 0, err
	}
	req.Header.Set("User-Agent", "litertlm-weather-autotool/1.0")

	resp, err := httpClient.Do(req)
	if err != nil {
		return 0, 0, err
	}
	defer func() { _ = resp.Body.Close() }()

	var results []struct {
		Lat string `json:"lat"`
		Lon string `json:"lon"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		return 0, 0, err
	}
	if len(results) == 0 {
		return 0, 0, fmt.Errorf("location not found: %s", location)
	}

	var latF, lonF float64
	if _, err := fmt.Sscanf(results[0].Lat, "%f", &latF); err != nil {
		return 0, 0, fmt.Errorf("parse lat: %w", err)
	}
	if _, err := fmt.Sscanf(results[0].Lon, "%f", &lonF); err != nil {
		return 0, 0, fmt.Errorf("parse lon: %w", err)
	}

	latF = math.Round(latF*10000) / 10000
	lonF = math.Round(lonF*10000) / 10000

	return latF, lonF, nil
}

func getGridPoint(lat, lon float64) (*NWSGridPoint, error) {
	u := fmt.Sprintf("https://api.weather.gov/points/%.4f,%.4f", lat, lon)

	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "litertlm-weather-autotool/1.0")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	var result struct {
		Properties struct {
			Forecast         string `json:"forecast"`
			RelativeLocation struct {
				Properties struct {
					City  string `json:"city"`
					State string `json:"state"`
				} `json:"properties"`
			} `json:"relativeLocation"`
		} `json:"properties"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if result.Properties.Forecast == "" {
		return nil, fmt.Errorf("no forecast URL returned for %.4f,%.4f", lat, lon)
	}

	return &NWSGridPoint{
		ForecastURL: result.Properties.Forecast,
		City:        result.Properties.RelativeLocation.Properties.City,
		State:       result.Properties.RelativeLocation.Properties.State,
	}, nil
}

func getForecast(forecastURL string) ([]ForecastPeriod, error) {
	req, err := http.NewRequest("GET", forecastURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "litertlm-weather-autotool/1.0")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	var result struct {
		Properties struct {
			Periods []ForecastPeriod `json:"periods"`
		} `json:"properties"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	periods := result.Properties.Periods
	if len(periods) > 5 {
		periods = periods[:5]
	}

	return periods, nil
}
