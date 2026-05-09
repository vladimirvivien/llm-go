package store

import (
	"database/sql"
	"time"
)

type Status string

const (
	StatusQueued        Status = "queued"
	StatusRunning       Status = "running"
	StatusApproved      Status = "approved"
	StatusRejected      Status = "rejected"
	StatusPendingReview Status = "pending-review"
)

type ReviewLane string

const (
	LaneNone       ReviewLane = ""
	LaneExtraction ReviewLane = "extraction"
	LaneDecision   ReviewLane = "decision"
)

type Source string

const (
	SourceUpload Source = "upload"
	SourceWatch  Source = "watch"
)

type Stage string

const (
	StagePreValidate  Stage = "pre-validate"
	StageExtract      Stage = "extract"
	StagePostValidate Stage = "post-validate"
	StageDecide       Stage = "decide"
)

type Decision string

const (
	DecisionPass        Decision = "pass"
	DecisionFail        Decision = "fail"
	DecisionRouteReview Decision = "route-review"
)

type Request struct {
	ID         string
	Source     Source
	Filename   string
	ImagePath  string
	Status     Status
	ReviewLane ReviewLane
	Score      sql.NullInt64
	CreatedAt  time.Time
	UpdatedAt  time.Time
}

type StageRun struct {
	ID         int64
	RequestID  string
	Stage      Stage
	StartedAt  time.Time
	FinishedAt sql.NullTime
	LatencyMS  sql.NullInt64
	InputJSON  sql.NullString
	OutputJSON sql.NullString
	RawText    sql.NullString
	Decision   Decision
	Notes      sql.NullString
}

type AuditEntry struct {
	ID        int64
	RequestID string
	Actor     string
	Action    string
	Detail    sql.NullString
	CreatedAt time.Time
}

type FormularyEntry struct {
	DrugName    string
	OnFormulary bool
	RequiresPA  bool
	StepTherapy []string
}
