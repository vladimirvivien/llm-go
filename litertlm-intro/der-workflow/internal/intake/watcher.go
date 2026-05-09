package intake

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/google/uuid"

	"github.com/vladimirvivien/exception-workflow/internal/store"
	"github.com/vladimirvivien/exception-workflow/internal/workflow"
)

type Watcher struct {
	Dir     string
	DataDir string
	Store   *store.Store
	Runner  *workflow.Runner
}

var validExts = map[string]bool{
	".png":  true,
	".jpg":  true,
	".jpeg": true,
	".webp": true,
	".gif":  true,
	".bmp":  true,
}

func (w *Watcher) Run(ctx context.Context) error {
	if err := os.MkdirAll(w.Dir, 0o755); err != nil {
		return fmt.Errorf("create watch dir: %w", err)
	}

	if err := w.scanExisting(ctx); err != nil {
		slog.Warn("watcher initial scan", "err", err)
	}

	fsw, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("new watcher: %w", err)
	}
	defer func() { _ = fsw.Close() }()

	if err := fsw.Add(w.Dir); err != nil {
		return fmt.Errorf("watch %s: %w", w.Dir, err)
	}
	slog.Info("watcher running", "dir", w.Dir)

	for {
		select {
		case <-ctx.Done():
			return nil
		case event, ok := <-fsw.Events:
			if !ok {
				return nil
			}
			if event.Has(fsnotify.Create) || event.Has(fsnotify.Write) {
				time.Sleep(150 * time.Millisecond)
				if err := w.handle(ctx, event.Name); err != nil {
					slog.Error("handle drop", "path", event.Name, "err", err)
				}
			}
		case err, ok := <-fsw.Errors:
			if !ok {
				return nil
			}
			slog.Error("watcher error", "err", err)
		}
	}
}

func (w *Watcher) scanExisting(ctx context.Context) error {
	entries, err := os.ReadDir(w.Dir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if err := w.handle(ctx, filepath.Join(w.Dir, e.Name())); err != nil {
			slog.Error("scan existing", "path", e.Name(), "err", err)
		}
	}
	return nil
}

func (w *Watcher) handle(ctx context.Context, path string) error {
	ext := strings.ToLower(filepath.Ext(path))
	if !validExts[ext] {
		return nil
	}
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if info.IsDir() {
		return nil
	}

	id := uuid.NewString()
	dst, err := w.copyToData(path, id, ext)
	if err != nil {
		return err
	}

	req := &store.Request{
		ID:        id,
		Source:    store.SourceWatch,
		Filename:  filepath.Base(path),
		ImagePath: dst,
		Status:    store.StatusQueued,
	}
	if err := w.Store.InsertRequest(ctx, req); err != nil {
		return fmt.Errorf("insert request: %w", err)
	}
	if err := w.Store.AppendAudit(ctx, id, "system", "submitted", fmt.Sprintf("source=watch filename=%s", filepath.Base(path))); err != nil {
		slog.Warn("audit submitted", "err", err)
	}
	if err := w.Runner.Enqueue(id); err != nil {
		return fmt.Errorf("enqueue: %w", err)
	}
	if err := os.Remove(path); err != nil {
		slog.Warn("remove dropped file", "path", path, "err", err)
	}
	slog.Info("watcher submitted", "id", id, "filename", filepath.Base(path))
	return nil
}

func (w *Watcher) copyToData(src, id, ext string) (string, error) {
	if err := os.MkdirAll(w.DataDir, 0o755); err != nil {
		return "", err
	}
	dst := filepath.Join(w.DataDir, id+ext)
	in, err := os.Open(src)
	if err != nil {
		return "", err
	}
	defer func() { _ = in.Close() }()
	out, err := os.Create(dst)
	if err != nil {
		return "", err
	}
	defer func() { _ = out.Close() }()
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(out, h), in); err != nil {
		_ = os.Remove(dst)
		return "", err
	}
	_ = hex.EncodeToString(h.Sum(nil))
	return dst, nil
}
