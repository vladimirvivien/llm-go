package api

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vladimirvivien/exception-workflow/internal/store"
	"github.com/vladimirvivien/exception-workflow/internal/workflow"
)

type Server struct {
	Store   *store.Store
	Runner  *workflow.Runner
	DataDir string
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("POST /api/requests", s.upload)
	mux.HandleFunc("GET /api/requests", s.list)
	mux.HandleFunc("GET /api/requests/{id}", s.detail)
	mux.HandleFunc("GET /api/requests/{id}/image", s.image)
	mux.HandleFunc("POST /api/requests/{id}/approve", s.approve)
	mux.HandleFunc("POST /api/requests/{id}/reject", s.reject)
	mux.HandleFunc("GET /api/metrics", s.metrics)
	mux.HandleFunc("GET /api/queues", s.queues)
	return mux
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok\n"))
}

func (s *Server) upload(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(20 << 20); err != nil {
		httpError(w, http.StatusBadRequest, fmt.Errorf("parse multipart: %w", err))
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		httpError(w, http.StatusBadRequest, fmt.Errorf("read file: %w", err))
		return
	}
	defer func() { _ = file.Close() }()

	id := uuid.NewString()
	imagePath, err := s.persistUpload(file, header.Filename, id)
	if err != nil {
		httpError(w, http.StatusInternalServerError, err)
		return
	}

	req := &store.Request{
		ID:        id,
		Source:    store.SourceUpload,
		Filename:  header.Filename,
		ImagePath: imagePath,
		Status:    store.StatusQueued,
	}
	if err := s.Store.InsertRequest(r.Context(), req); err != nil {
		httpError(w, http.StatusInternalServerError, err)
		return
	}
	if err := s.Store.AppendAudit(r.Context(), id, "system", "submitted", fmt.Sprintf("source=upload filename=%s", header.Filename)); err != nil {
		httpError(w, http.StatusInternalServerError, err)
		return
	}
	if err := s.Runner.Enqueue(id); err != nil {
		httpError(w, http.StatusServiceUnavailable, err)
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"id": id, "status": req.Status})
}

func (s *Server) persistUpload(src io.Reader, filename, id string) (string, error) {
	if err := os.MkdirAll(s.DataDir, 0o755); err != nil {
		return "", fmt.Errorf("data dir: %w", err)
	}
	ext := strings.ToLower(filepath.Ext(filename))
	if ext == "" {
		ext = ".png"
	}
	dst := filepath.Join(s.DataDir, fmt.Sprintf("%s%s", id, ext))
	out, err := os.Create(dst)
	if err != nil {
		return "", fmt.Errorf("create file: %w", err)
	}
	defer func() { _ = out.Close() }()
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(out, h), src); err != nil {
		_ = os.Remove(dst)
		return "", fmt.Errorf("write file: %w", err)
	}
	_ = hex.EncodeToString(h.Sum(nil))
	return dst, nil
}

func (s *Server) list(w http.ResponseWriter, r *http.Request) {
	status := store.Status(r.URL.Query().Get("status"))
	lane := store.ReviewLane(r.URL.Query().Get("lane"))
	requests, err := s.Store.ListRequests(r.Context(), status, lane, 200)
	if err != nil {
		httpError(w, http.StatusInternalServerError, err)
		return
	}
	out := make([]map[string]any, 0, len(requests))
	for _, req := range requests {
		out = append(out, requestSummary(req))
	}
	writeJSON(w, http.StatusOK, map[string]any{"requests": out})
}

func (s *Server) detail(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	req, err := s.Store.GetRequest(r.Context(), id)
	if err != nil {
		httpError(w, http.StatusNotFound, err)
		return
	}
	stages, err := s.Store.StageRunsFor(r.Context(), id)
	if err != nil {
		httpError(w, http.StatusInternalServerError, err)
		return
	}
	audits, err := s.Store.AuditFor(r.Context(), id)
	if err != nil {
		httpError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"request": requestSummary(req),
		"stages":  stagesToJSON(stages),
		"audit":   auditToJSON(audits),
	})
}

func (s *Server) image(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	req, err := s.Store.GetRequest(r.Context(), id)
	if err != nil {
		httpError(w, http.StatusNotFound, err)
		return
	}
	http.ServeFile(w, r, req.ImagePath)
}

type humanAction struct {
	Actor string `json:"actor"`
	Note  string `json:"note"`
}

func (s *Server) approve(w http.ResponseWriter, r *http.Request) {
	s.humanDecision(w, r, true)
}

func (s *Server) reject(w http.ResponseWriter, r *http.Request) {
	s.humanDecision(w, r, false)
}

func (s *Server) humanDecision(w http.ResponseWriter, r *http.Request, approve bool) {
	id := r.PathValue("id")
	var body humanAction
	if r.ContentLength > 0 {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			httpError(w, http.StatusBadRequest, fmt.Errorf("decode body: %w", err))
			return
		}
	}
	if body.Actor == "" {
		body.Actor = "human"
	}

	var err error
	if approve {
		err = workflow.ApproveByHuman(r.Context(), s.Store, id, body.Actor, body.Note)
	} else {
		err = workflow.RejectByHuman(r.Context(), s.Store, id, body.Actor, body.Note)
	}
	if err != nil {
		httpError(w, http.StatusInternalServerError, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) metrics(w http.ResponseWriter, r *http.Request) {
	m, err := s.Store.Metrics(r.Context())
	if err != nil {
		httpError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, m)
}

func (s *Server) queues(w http.ResponseWriter, r *http.Request) {
	extraction, err := s.Store.ListRequests(r.Context(), store.StatusPendingReview, store.LaneExtraction, 100)
	if err != nil {
		httpError(w, http.StatusInternalServerError, err)
		return
	}
	decision, err := s.Store.ListRequests(r.Context(), store.StatusPendingReview, store.LaneDecision, 100)
	if err != nil {
		httpError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"extraction": summaryList(extraction),
		"decision":   summaryList(decision),
	})
}

func summaryList(rs []*store.Request) []map[string]any {
	out := make([]map[string]any, 0, len(rs))
	for _, r := range rs {
		out = append(out, requestSummary(r))
	}
	return out
}

func requestSummary(r *store.Request) map[string]any {
	m := map[string]any{
		"id":         r.ID,
		"filename":   r.Filename,
		"source":     r.Source,
		"status":     r.Status,
		"created_at": r.CreatedAt.Format(time.RFC3339),
		"updated_at": r.UpdatedAt.Format(time.RFC3339),
	}
	if r.ReviewLane != "" {
		m["review_lane"] = r.ReviewLane
	}
	if r.Score.Valid {
		m["score"] = r.Score.Int64
	}
	return m
}

func stagesToJSON(rs []*store.StageRun) []map[string]any {
	out := make([]map[string]any, 0, len(rs))
	for _, r := range rs {
		m := map[string]any{
			"id":         r.ID,
			"stage":      r.Stage,
			"started_at": r.StartedAt.Format(time.RFC3339),
			"decision":   r.Decision,
		}
		if r.FinishedAt.Valid {
			m["finished_at"] = r.FinishedAt.Time.Format(time.RFC3339)
		}
		if r.LatencyMS.Valid {
			m["latency_ms"] = r.LatencyMS.Int64
		}
		if r.InputJSON.Valid {
			m["input"] = json.RawMessage(r.InputJSON.String)
		}
		if r.OutputJSON.Valid {
			m["output"] = json.RawMessage(r.OutputJSON.String)
		}
		if r.RawText.Valid {
			m["raw"] = r.RawText.String
		}
		if r.Notes.Valid {
			m["notes"] = r.Notes.String
		}
		out = append(out, m)
	}
	return out
}

func auditToJSON(as []*store.AuditEntry) []map[string]any {
	out := make([]map[string]any, 0, len(as))
	for _, a := range as {
		m := map[string]any{
			"actor":      a.Actor,
			"action":     a.Action,
			"created_at": a.CreatedAt.Format(time.RFC3339),
		}
		if a.Detail.Valid {
			m["detail"] = a.Detail.String
		}
		out = append(out, m)
	}
	return out
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func httpError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}
