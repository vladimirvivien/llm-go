package workflow

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/vladimirvivien/exception-workflow/internal/store"
)

type Stage interface {
	Name() store.Stage
	Run(ctx context.Context, in StageInput) (StageOutput, error)
}

type StageInput struct {
	RequestID  string
	ImagePath  string
	Filename   string
	Extracted  *ExtractedRequest
	PrevOutput map[string]any
}

type Outcome string

const (
	OutcomeAdvance     Outcome = "advance"
	OutcomeReject      Outcome = "reject"
	OutcomeRouteReview Outcome = "route-review"
	OutcomeApprove     Outcome = "approve"
)

type StageOutput struct {
	Outcome   Outcome
	Lane      store.ReviewLane
	Score     int
	Output    any
	Raw       string
	Notes     string
	Extracted *ExtractedRequest
}

type Runner struct {
	store             *store.Store
	stages            []Stage
	queue             chan string
	workers           int
	autoApproveCutoff int
	wg                sync.WaitGroup
	mu                sync.Mutex
	closed            bool
}

type RunnerConfig struct {
	Workers              int
	AutoApproveThreshold int
	QueueSize            int
}

func NewRunner(st *store.Store, stages []Stage, cfg RunnerConfig) *Runner {
	if cfg.Workers <= 0 {
		cfg.Workers = 1
	}
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 64
	}
	if cfg.AutoApproveThreshold <= 0 {
		cfg.AutoApproveThreshold = 100
	}
	return &Runner{
		store:             st,
		stages:            stages,
		queue:             make(chan string, cfg.QueueSize),
		workers:           cfg.Workers,
		autoApproveCutoff: cfg.AutoApproveThreshold,
	}
}

func (r *Runner) Start(ctx context.Context) {
	for i := 0; i < r.workers; i++ {
		r.wg.Add(1)
		go r.workerLoop(ctx, i)
	}
}

func (r *Runner) Close() {
	r.mu.Lock()
	if !r.closed {
		close(r.queue)
		r.closed = true
	}
	r.mu.Unlock()
	r.wg.Wait()
}

func (r *Runner) Enqueue(requestID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return fmt.Errorf("runner closed")
	}
	select {
	case r.queue <- requestID:
		return nil
	default:
		return fmt.Errorf("queue full")
	}
}

func (r *Runner) workerLoop(ctx context.Context, id int) {
	defer r.wg.Done()
	log := slog.With("worker", id)
	for {
		select {
		case <-ctx.Done():
			return
		case requestID, ok := <-r.queue:
			if !ok {
				return
			}
			r.process(ctx, log, requestID)
		}
	}
}

func (r *Runner) process(ctx context.Context, log *slog.Logger, requestID string) {
	req, err := r.store.GetRequest(ctx, requestID)
	if err != nil {
		log.Error("get request", "id", requestID, "err", err)
		return
	}
	log = log.With("request", requestID, "filename", req.Filename)
	log.Info("workflow start")

	if len(r.stages) == 0 {
		log.Warn("no stages configured; leaving request queued")
		return
	}

	if err := r.store.UpdateRequest(ctx, req.ID, store.StatusRunning, store.LaneNone, sql.NullInt64{}); err != nil {
		log.Error("mark running", "err", err)
		return
	}

	in := StageInput{
		RequestID: req.ID,
		ImagePath: req.ImagePath,
		Filename:  req.Filename,
	}

	for _, stage := range r.stages {
		stageLog := log.With("stage", string(stage.Name()))
		started := time.Now()
		runID, err := r.store.StartStageRun(ctx, req.ID, stage.Name(), serializeInput(in))
		if err != nil {
			stageLog.Error("start stage_run", "err", err)
			return
		}

		out, runErr := stage.Run(ctx, in)
		decision := store.DecisionPass
		notes := out.Notes
		if runErr != nil {
			decision = store.DecisionFail
			notes = runErr.Error()
		} else {
			switch out.Outcome {
			case OutcomeReject:
				decision = store.DecisionFail
			case OutcomeRouteReview:
				decision = store.DecisionRouteReview
			}
		}

		outputJSON := serializeOutput(out.Output)
		if finishErr := r.store.FinishStageRun(ctx, runID, decision, outputJSON, out.Raw, notes, started); finishErr != nil {
			stageLog.Error("finish stage_run", "err", finishErr)
		}
		stageLog.Info("stage done", "decision", decision, "latency_ms", time.Since(started).Milliseconds())

		if runErr != nil {
			_ = r.store.UpdateRequest(ctx, req.ID, store.StatusRejected, store.LaneNone, sql.NullInt64{})
			_ = r.store.AppendAudit(ctx, req.ID, "system", "stage-error", fmt.Sprintf("%s: %v", stage.Name(), runErr))
			return
		}

		switch out.Outcome {
		case OutcomeReject:
			_ = r.store.UpdateRequest(ctx, req.ID, store.StatusRejected, store.LaneNone, sql.NullInt64{})
			_ = r.store.AppendAudit(ctx, req.ID, "system", "rejected", fmt.Sprintf("%s: %s", stage.Name(), out.Notes))
			return
		case OutcomeRouteReview:
			_ = r.store.UpdateRequest(ctx, req.ID, store.StatusPendingReview, out.Lane, scoreNullable(out.Score))
			_ = r.store.AppendAudit(ctx, req.ID, "system", "routed-review", fmt.Sprintf("%s -> %s: %s", stage.Name(), out.Lane, out.Notes))
			return
		case OutcomeApprove:
			_ = r.store.UpdateRequest(ctx, req.ID, store.StatusApproved, store.LaneNone, scoreNullable(out.Score))
			_ = r.store.AppendAudit(ctx, req.ID, "system", "approved", out.Notes)
			return
		}

		if out.Extracted != nil {
			in.Extracted = out.Extracted
		}
	}

	stageLog := log.With("stage", "<end>")
	stageLog.Warn("workflow ended without terminal outcome")
}

func serializeInput(in StageInput) string {
	v := map[string]any{
		"request_id": in.RequestID,
		"image_path": in.ImagePath,
		"filename":   in.Filename,
	}
	if in.Extracted != nil {
		v["extracted"] = in.Extracted
	}
	b, _ := json.Marshal(v)
	return string(b)
}

func serializeOutput(o any) string {
	if o == nil {
		return ""
	}
	b, err := json.Marshal(o)
	if err != nil {
		return fmt.Sprintf(`{"_marshal_error":%q}`, err.Error())
	}
	return string(b)
}

func scoreNullable(s int) sql.NullInt64 {
	if s <= 0 {
		return sql.NullInt64{}
	}
	return sql.NullInt64{Valid: true, Int64: int64(s)}
}

func ApproveByHuman(ctx context.Context, st *store.Store, requestID, actor, note string) error {
	if err := st.UpdateRequest(ctx, requestID, store.StatusApproved, store.LaneNone, sql.NullInt64{}); err != nil {
		return err
	}
	return st.AppendAudit(ctx, requestID, actor, "human-approved", note)
}

func RejectByHuman(ctx context.Context, st *store.Store, requestID, actor, note string) error {
	if err := st.UpdateRequest(ctx, requestID, store.StatusRejected, store.LaneNone, sql.NullInt64{}); err != nil {
		return err
	}
	return st.AppendAudit(ctx, requestID, actor, "human-rejected", note)
}
