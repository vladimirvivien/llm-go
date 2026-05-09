package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/vladimirvivien/exception-workflow/internal/api"
	"github.com/vladimirvivien/exception-workflow/internal/intake"
	"github.com/vladimirvivien/exception-workflow/internal/stages"
	"github.com/vladimirvivien/exception-workflow/internal/store"
	"github.com/vladimirvivien/exception-workflow/internal/ui"
	"github.com/vladimirvivien/exception-workflow/internal/workflow"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	home, _ := os.UserHomeDir()

	var (
		dbPath               = flag.String("db", "exception.db", "path to sqlite database file")
		dataDir              = flag.String("data", "data", "directory for uploaded form images")
		watchDir             = flag.String("watch", "", "directory to watch for dropped form files (empty disables)")
		addr                 = flag.String("addr", ":8080", "HTTP listen address")
		modelPath            = flag.String("model", filepath.Join(home, "models", "gemma-4-E2B-it.litertlm"), "path to .litertlm model file")
		libPath              = flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libraries")
		backend              = flag.String("backend", "cpu", "text inference backend (cpu | gpu)")
		visionBackend        = flag.String("vision-backend", "cpu", "vision inference backend (cpu | gpu)")
		maxTokens            = flag.Int("max-tokens", 8192, "engine max tokens (prompt + output)")
		workers              = flag.Int("workers", 1, "number of workflow workers")
		autoApproveThreshold = flag.Int("auto-approve-threshold", 100, "score >= threshold auto-approves")
		noModel              = flag.Bool("no-model", false, "skip model load (for development)")
	)
	flag.Parse()

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})))

	if err := os.MkdirAll(*dataDir, 0o755); err != nil {
		slog.Error("create data dir", "err", err)
		os.Exit(1)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	st, err := store.Open(ctx, *dbPath)
	if err != nil {
		slog.Error("open store", "err", err)
		os.Exit(1)
	}
	defer func() { _ = st.Close() }()
	slog.Info("store opened", "db", *dbPath)

	var client *litertlm.Client
	if !*noModel {
		loadStart := time.Now()
		c, err := workflow.LoadModel(ctx, workflow.ModelConfig{
			LibPath:       *libPath,
			ModelPath:     *modelPath,
			Backend:       *backend,
			VisionBackend: *visionBackend,
			MaxTokens:     *maxTokens,
		})
		if err != nil {
			slog.Error("load model", "err", err)
			os.Exit(1)
		}
		client = c
		defer func() { _ = client.Close() }()
		slog.Info("model loaded", "model", *modelPath, "elapsed", time.Since(loadStart).Round(time.Millisecond))
	} else {
		slog.Warn("model load skipped (--no-model)")
	}

	var stageList []workflow.Stage
	if client != nil {
		stageList = []workflow.Stage{
			&stages.PreValidate{Client: client},
			&stages.Extract{Client: client},
			&stages.PostValidate{Client: client},
			&stages.Decide{Client: client, Store: st, AutoApproveScore: *autoApproveThreshold},
		}
	}

	runner := workflow.NewRunner(st, stageList, workflow.RunnerConfig{
		Workers:              *workers,
		AutoApproveThreshold: *autoApproveThreshold,
	})
	runner.Start(ctx)
	defer runner.Close()

	if *watchDir != "" {
		w := &intake.Watcher{Dir: *watchDir, DataDir: *dataDir, Store: st, Runner: runner}
		go func() {
			if err := w.Run(ctx); err != nil {
				slog.Error("watcher exited", "err", err)
			}
		}()
	}

	apiSrv := &api.Server{Store: st, Runner: runner, DataDir: *dataDir}
	uiSrv, err := ui.NewServer(st, runner, *dataDir)
	if err != nil {
		slog.Error("ui init", "err", err)
		os.Exit(1)
	}

	root := http.NewServeMux()
	root.Handle("/api/", apiSrv.Routes())
	root.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	})
	root.Handle("/", uiSrv.Routes())

	srv := &http.Server{
		Addr:              *addr,
		Handler:           root,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		slog.Info("http server listening", "addr", *addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("http serve", "err", err)
		}
	}()

	<-ctx.Done()
	slog.Info("shutting down")
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		slog.Error("http shutdown", "err", err)
	}
	fmt.Fprintln(os.Stderr, "bye")
}
