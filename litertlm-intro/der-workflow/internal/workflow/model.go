package workflow

import (
	"context"
	"fmt"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

type ModelConfig struct {
	LibPath       string
	ModelPath     string
	Backend       string
	VisionBackend string
	MaxTokens     int
}

func LoadModel(ctx context.Context, cfg ModelConfig) (*litertlm.Client, error) {
	if cfg.LibPath == "" {
		return nil, fmt.Errorf("LITERTLM_LIB / -lib is required")
	}
	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	opts := []litertlm.Option{
		litertlm.WithLib(cfg.LibPath),
		litertlm.WithModel(cfg.ModelPath),
		litertlm.WithBackend(cfg.Backend),
		litertlm.WithMaxTokens(cfg.MaxTokens),
	}
	if cfg.VisionBackend != "" {
		opts = append(opts, litertlm.WithVisionBackend(cfg.VisionBackend))
	}
	return litertlm.New(ctx, opts...)
}
