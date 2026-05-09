package ui

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"fmt"
	"html/template"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/uuid"

	"github.com/vladimirvivien/exception-workflow/internal/store"
	"github.com/vladimirvivien/exception-workflow/internal/workflow"
)

//go:embed templates/*.html
var templatesFS embed.FS

type Server struct {
	Store   *store.Store
	Runner  *workflow.Runner
	DataDir string

	dashboardTpl *template.Template
	requestsTpl  *template.Template
	detailTpl    *template.Template
	queuesTpl    *template.Template
	uploadTpl    *template.Template
}

func NewServer(st *store.Store, runner *workflow.Runner, dataDir string) (*Server, error) {
	s := &Server{Store: st, Runner: runner, DataDir: dataDir}
	if err := s.parseTemplates(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Server) parseTemplates() error {
	funcs := template.FuncMap{
		"statusClass":   statusClass,
		"decisionClass": decisionClass,
	}

	for _, t := range []struct {
		field **template.Template
		page  string
	}{
		{&s.dashboardTpl, "dashboard.html"},
		{&s.requestsTpl, "requests.html"},
		{&s.detailTpl, "detail.html"},
		{&s.queuesTpl, "queues.html"},
		{&s.uploadTpl, "upload.html"},
	} {
		tpl, err := template.New("layout").Funcs(funcs).ParseFS(templatesFS,
			"templates/layout.html",
			"templates/requests.html",
			"templates/"+t.page,
		)
		if err != nil {
			return fmt.Errorf("parse %s: %w", t.page, err)
		}
		*t.field = tpl
	}
	return nil
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", s.dashboard)
	mux.HandleFunc("GET /requests", s.requestList)
	mux.HandleFunc("GET /requests/{id}", s.requestDetail)
	mux.HandleFunc("GET /requests/{id}/image", s.requestImage)
	mux.HandleFunc("POST /requests/{id}/approve", s.approve)
	mux.HandleFunc("POST /requests/{id}/reject", s.reject)
	mux.HandleFunc("GET /queues", s.queues)
	mux.HandleFunc("GET /upload", s.uploadForm)
	mux.HandleFunc("POST /upload", s.uploadSubmit)
	return mux
}

type card struct {
	Label, Value, Sub string
}

func (s *Server) dashboard(w http.ResponseWriter, r *http.Request) {
	m, err := s.Store.Metrics(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	recent, err := s.Store.ListRequests(r.Context(), "", "", 10)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	cards := []card{
		{Label: "Total", Value: fmt.Sprintf("%d", m.Total)},
		{Label: "Queued / running", Value: fmt.Sprintf("%d", m.Queued)},
		{Label: "Approved", Value: fmt.Sprintf("%d", m.ByStatus[string(store.StatusApproved)])},
		{Label: "Pending review", Value: fmt.Sprintf("%d", m.ByStatus[string(store.StatusPendingReview)])},
	}

	data := map[string]any{
		"Cards":    cards,
		"ByStatus": m.ByStatus,
		"ByLane":   m.ByLane,
		"Recent":   recent,
	}
	if err := s.dashboardTpl.ExecuteTemplate(w, "layout", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (s *Server) requestList(w http.ResponseWriter, r *http.Request) {
	statusFilter := store.Status(r.URL.Query().Get("status"))
	laneFilter := store.ReviewLane(r.URL.Query().Get("lane"))

	requests, err := s.Store.ListRequests(r.Context(), statusFilter, laneFilter, 200)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	data := map[string]any{
		"Requests": requests,
		"Filter": map[string]string{
			"Status": string(statusFilter),
			"Lane":   string(laneFilter),
		},
		"Statuses": []store.Status{
			store.StatusQueued, store.StatusRunning, store.StatusApproved,
			store.StatusRejected, store.StatusPendingReview,
		},
		"Lanes": []store.ReviewLane{store.LaneExtraction, store.LaneDecision},
	}
	if err := s.requestsTpl.ExecuteTemplate(w, "layout", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (s *Server) requestDetail(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	req, err := s.Store.GetRequest(r.Context(), id)
	if err != nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	stages, err := s.Store.StageRunsFor(r.Context(), id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	audit, err := s.Store.AuditFor(r.Context(), id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	data := map[string]any{
		"Request": req,
		"Stages":  stages,
		"Audit":   audit,
	}
	if err := s.detailTpl.ExecuteTemplate(w, "layout", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (s *Server) requestImage(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	req, err := s.Store.GetRequest(r.Context(), id)
	if err != nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	http.ServeFile(w, r, req.ImagePath)
}

func (s *Server) approve(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := workflow.ApproveByHuman(r.Context(), s.Store, id, "human", "approved via UI"); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	http.Redirect(w, r, "/requests/"+id, http.StatusSeeOther)
}

func (s *Server) reject(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := workflow.RejectByHuman(r.Context(), s.Store, id, "human", "rejected via UI"); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	http.Redirect(w, r, "/requests/"+id, http.StatusSeeOther)
}

func (s *Server) queues(w http.ResponseWriter, r *http.Request) {
	extraction, err := s.Store.ListRequests(r.Context(), store.StatusPendingReview, store.LaneExtraction, 100)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	decision, err := s.Store.ListRequests(r.Context(), store.StatusPendingReview, store.LaneDecision, 100)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	data := map[string]any{
		"Extraction": extraction,
		"Decision":   decision,
	}
	if err := s.queuesTpl.ExecuteTemplate(w, "layout", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (s *Server) uploadForm(w http.ResponseWriter, r *http.Request) {
	data := map[string]any{"Message": r.URL.Query().Get("msg")}
	if err := s.uploadTpl.ExecuteTemplate(w, "layout", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (s *Server) uploadSubmit(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(20 << 20); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer func() { _ = file.Close() }()

	id := uuid.NewString()
	imagePath, err := s.persist(file, header.Filename, id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
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
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	_ = s.Store.AppendAudit(r.Context(), id, "system", "submitted", fmt.Sprintf("source=upload filename=%s", header.Filename))
	if err := s.Runner.Enqueue(id); err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	http.Redirect(w, r, "/requests/"+id, http.StatusSeeOther)
}

func (s *Server) persist(src io.Reader, filename, id string) (string, error) {
	if err := os.MkdirAll(s.DataDir, 0o755); err != nil {
		return "", err
	}
	ext := strings.ToLower(filepath.Ext(filename))
	if ext == "" {
		ext = ".png"
	}
	dst := filepath.Join(s.DataDir, id+ext)
	out, err := os.Create(dst)
	if err != nil {
		return "", err
	}
	defer func() { _ = out.Close() }()
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(out, h), src); err != nil {
		_ = os.Remove(dst)
		return "", err
	}
	_ = hex.EncodeToString(h.Sum(nil))
	return dst, nil
}

func statusClass(s store.Status) string {
	switch s {
	case store.StatusApproved:
		return "bg-emerald-100 text-emerald-800"
	case store.StatusRejected:
		return "bg-rose-100 text-rose-800"
	case store.StatusPendingReview:
		return "bg-amber-100 text-amber-800"
	case store.StatusRunning:
		return "bg-blue-100 text-blue-800"
	case store.StatusQueued:
		return "bg-slate-200 text-slate-700"
	}
	return "bg-slate-100 text-slate-700"
}

func decisionClass(d store.Decision) string {
	switch d {
	case store.DecisionPass:
		return "stage-pass"
	case store.DecisionFail:
		return "stage-fail"
	case store.DecisionRouteReview:
		return "stage-route"
	}
	return ""
}
