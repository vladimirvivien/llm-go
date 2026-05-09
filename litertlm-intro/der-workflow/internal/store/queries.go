package store

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"
)

func (s *Store) InsertRequest(ctx context.Context, r *Request) error {
	now := Now()
	r.CreatedAt = now
	r.UpdatedAt = now
	_, err := s.DB.ExecContext(ctx, `
		INSERT INTO requests (id, source, filename, image_path, status, review_lane, score, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		r.ID, string(r.Source), r.Filename, r.ImagePath, string(r.Status),
		nullableLane(r.ReviewLane), r.Score, r.CreatedAt, r.UpdatedAt,
	)
	return err
}

func (s *Store) UpdateRequest(ctx context.Context, id string, status Status, lane ReviewLane, score sql.NullInt64) error {
	_, err := s.DB.ExecContext(ctx, `
		UPDATE requests SET status = ?, review_lane = ?, score = ?, updated_at = ? WHERE id = ?`,
		string(status), nullableLane(lane), score, Now(), id,
	)
	return err
}

func (s *Store) GetRequest(ctx context.Context, id string) (*Request, error) {
	row := s.DB.QueryRowContext(ctx, `
		SELECT id, source, filename, image_path, status, review_lane, score, created_at, updated_at
		FROM requests WHERE id = ?`, id)
	return scanRequest(row)
}

func (s *Store) ListRequests(ctx context.Context, status Status, lane ReviewLane, limit int) ([]*Request, error) {
	var conds []string
	var args []any
	if status != "" {
		conds = append(conds, "status = ?")
		args = append(args, string(status))
	}
	if lane != "" {
		conds = append(conds, "review_lane = ?")
		args = append(args, string(lane))
	}
	q := `SELECT id, source, filename, image_path, status, review_lane, score, created_at, updated_at FROM requests`
	if len(conds) > 0 {
		q += " WHERE " + strings.Join(conds, " AND ")
	}
	q += " ORDER BY created_at DESC"
	if limit > 0 {
		q += fmt.Sprintf(" LIMIT %d", limit)
	}
	rows, err := s.DB.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var out []*Request
	for rows.Next() {
		r, err := scanRequest(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

type scanner interface {
	Scan(dest ...any) error
}

func scanRequest(row scanner) (*Request, error) {
	var r Request
	var lane sql.NullString
	if err := row.Scan(&r.ID, &r.Source, &r.Filename, &r.ImagePath, &r.Status, &lane, &r.Score, &r.CreatedAt, &r.UpdatedAt); err != nil {
		return nil, err
	}
	if lane.Valid {
		r.ReviewLane = ReviewLane(lane.String)
	}
	return &r, nil
}

func nullableLane(lane ReviewLane) sql.NullString {
	if lane == "" {
		return sql.NullString{}
	}
	return sql.NullString{Valid: true, String: string(lane)}
}

func (s *Store) StartStageRun(ctx context.Context, requestID string, stage Stage, input string) (int64, error) {
	res, err := s.DB.ExecContext(ctx, `
		INSERT INTO stage_runs (request_id, stage, started_at, input_json, decision)
		VALUES (?, ?, ?, ?, ?)`,
		requestID, string(stage), Now(), nullableString(input), string(DecisionPass),
	)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

func (s *Store) FinishStageRun(ctx context.Context, id int64, decision Decision, output, raw, notes string, started time.Time) error {
	finished := Now()
	latency := finished.Sub(started).Milliseconds()
	_, err := s.DB.ExecContext(ctx, `
		UPDATE stage_runs SET finished_at = ?, latency_ms = ?, output_json = ?, raw_text = ?, notes = ?, decision = ?
		WHERE id = ?`,
		finished, latency, nullableString(output), nullableString(raw), nullableString(notes), string(decision), id,
	)
	return err
}

func (s *Store) StageRunsFor(ctx context.Context, requestID string) ([]*StageRun, error) {
	rows, err := s.DB.QueryContext(ctx, `
		SELECT id, request_id, stage, started_at, finished_at, latency_ms, input_json, output_json, raw_text, decision, notes
		FROM stage_runs WHERE request_id = ? ORDER BY id ASC`, requestID)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var out []*StageRun
	for rows.Next() {
		var sr StageRun
		if err := rows.Scan(&sr.ID, &sr.RequestID, &sr.Stage, &sr.StartedAt, &sr.FinishedAt, &sr.LatencyMS,
			&sr.InputJSON, &sr.OutputJSON, &sr.RawText, &sr.Decision, &sr.Notes); err != nil {
			return nil, err
		}
		out = append(out, &sr)
	}
	return out, rows.Err()
}

func (s *Store) AppendAudit(ctx context.Context, requestID, actor, action, detail string) error {
	_, err := s.DB.ExecContext(ctx, `
		INSERT INTO audit (request_id, actor, action, detail, created_at) VALUES (?, ?, ?, ?, ?)`,
		requestID, actor, action, nullableString(detail), Now(),
	)
	return err
}

func (s *Store) AuditFor(ctx context.Context, requestID string) ([]*AuditEntry, error) {
	rows, err := s.DB.QueryContext(ctx, `
		SELECT id, request_id, actor, action, detail, created_at
		FROM audit WHERE request_id = ? ORDER BY id ASC`, requestID)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var out []*AuditEntry
	for rows.Next() {
		var a AuditEntry
		if err := rows.Scan(&a.ID, &a.RequestID, &a.Actor, &a.Action, &a.Detail, &a.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, &a)
	}
	return out, rows.Err()
}

func (s *Store) GetFormularyEntry(ctx context.Context, drug string) (*FormularyEntry, error) {
	row := s.DB.QueryRowContext(ctx, `SELECT drug_name, on_formulary, requires_pa, step_therapy FROM formulary WHERE drug_name = ?`, strings.ToLower(drug))
	var e FormularyEntry
	var stepTherapy sql.NullString
	var onFormulary, requiresPA int
	if err := row.Scan(&e.DrugName, &onFormulary, &requiresPA, &stepTherapy); err != nil {
		return nil, err
	}
	e.OnFormulary = onFormulary == 1
	e.RequiresPA = requiresPA == 1
	if stepTherapy.Valid && stepTherapy.String != "" {
		e.StepTherapy = strings.Split(stepTherapy.String, ",")
		for i, s := range e.StepTherapy {
			e.StepTherapy[i] = strings.TrimSpace(s)
		}
	}
	return &e, nil
}

func nullableString(s string) sql.NullString {
	if s == "" {
		return sql.NullString{}
	}
	return sql.NullString{Valid: true, String: s}
}

type Metrics struct {
	ByStatus map[string]int
	ByLane   map[string]int
	Queued   int
	Total    int
}

func (s *Store) Metrics(ctx context.Context) (*Metrics, error) {
	m := &Metrics{ByStatus: map[string]int{}, ByLane: map[string]int{}}
	rows, err := s.DB.QueryContext(ctx, `SELECT status, COUNT(*) FROM requests GROUP BY status`)
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		var status string
		var n int
		if err := rows.Scan(&status, &n); err != nil {
			_ = rows.Close()
			return nil, err
		}
		m.ByStatus[status] = n
		m.Total += n
		if status == string(StatusQueued) || status == string(StatusRunning) {
			m.Queued += n
		}
	}
	_ = rows.Close()

	rows2, err := s.DB.QueryContext(ctx, `SELECT review_lane, COUNT(*) FROM requests WHERE review_lane IS NOT NULL GROUP BY review_lane`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows2.Close() }()
	for rows2.Next() {
		var lane string
		var n int
		if err := rows2.Scan(&lane, &n); err != nil {
			return nil, err
		}
		m.ByLane[lane] = n
	}
	return m, nil
}
