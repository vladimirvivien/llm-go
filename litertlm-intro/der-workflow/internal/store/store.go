package store

import (
	"context"
	"database/sql"
	"embed"
	"encoding/csv"
	"fmt"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

//go:embed migrations/*.sql
var migrationsFS embed.FS

//go:embed seed_formulary.csv
var seedFormularyCSV []byte

type Store struct {
	DB *sql.DB
}

func Open(ctx context.Context, dsn string) (*Store, error) {
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	db.SetMaxOpenConns(1)
	if _, err := db.ExecContext(ctx, "PRAGMA foreign_keys = ON;"); err != nil {
		return nil, fmt.Errorf("enable fk: %w", err)
	}
	if _, err := db.ExecContext(ctx, "PRAGMA journal_mode = WAL;"); err != nil {
		return nil, fmt.Errorf("set wal: %w", err)
	}

	s := &Store{DB: db}
	if err := s.migrate(ctx); err != nil {
		return nil, err
	}
	if err := s.seedFormulary(ctx); err != nil {
		return nil, fmt.Errorf("seed formulary: %w", err)
	}
	return s, nil
}

func (s *Store) Close() error {
	return s.DB.Close()
}

func (s *Store) migrate(ctx context.Context) error {
	entries, err := migrationsFS.ReadDir("migrations")
	if err != nil {
		return fmt.Errorf("read migrations: %w", err)
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".sql") {
			continue
		}
		body, err := migrationsFS.ReadFile("migrations/" + e.Name())
		if err != nil {
			return fmt.Errorf("read %s: %w", e.Name(), err)
		}
		if _, err := s.DB.ExecContext(ctx, string(body)); err != nil {
			return fmt.Errorf("apply %s: %w", e.Name(), err)
		}
	}
	return nil
}

func (s *Store) seedFormulary(ctx context.Context) error {
	var n int
	if err := s.DB.QueryRowContext(ctx, "SELECT COUNT(*) FROM formulary").Scan(&n); err != nil {
		return err
	}
	if n > 0 {
		return nil
	}

	r := csv.NewReader(strings.NewReader(string(seedFormularyCSV)))
	rows, err := r.ReadAll()
	if err != nil {
		return fmt.Errorf("parse seed csv: %w", err)
	}
	if len(rows) < 2 {
		return nil
	}

	tx, err := s.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	stmt, err := tx.PrepareContext(ctx, `INSERT INTO formulary(drug_name, on_formulary, requires_pa, step_therapy) VALUES (?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer func() { _ = stmt.Close() }()

	for i, row := range rows {
		if i == 0 {
			continue
		}
		if len(row) != 4 {
			return fmt.Errorf("seed row %d: want 4 cols, got %d", i, len(row))
		}
		stepTherapy := sql.NullString{}
		if row[3] != "" {
			stepTherapy.Valid = true
			stepTherapy.String = row[3]
		}
		if _, err := stmt.ExecContext(ctx, row[0], parseBool(row[1]), parseBool(row[2]), stepTherapy); err != nil {
			return fmt.Errorf("seed row %d: %w", i, err)
		}
	}
	return tx.Commit()
}

func parseBool(s string) int {
	if s == "1" || strings.EqualFold(s, "true") {
		return 1
	}
	return 0
}

func Now() time.Time {
	return time.Now().UTC()
}
