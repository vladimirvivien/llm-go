CREATE TABLE IF NOT EXISTS requests (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    filename      TEXT NOT NULL,
    image_path    TEXT NOT NULL,
    status        TEXT NOT NULL,
    review_lane   TEXT,
    score         INTEGER,
    created_at    DATETIME NOT NULL,
    updated_at    DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS requests_status_idx ON requests(status);
CREATE INDEX IF NOT EXISTS requests_lane_idx   ON requests(review_lane);

CREATE TABLE IF NOT EXISTS stage_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT NOT NULL REFERENCES requests(id),
    stage         TEXT NOT NULL,
    started_at    DATETIME NOT NULL,
    finished_at   DATETIME,
    latency_ms    INTEGER,
    input_json    TEXT,
    output_json   TEXT,
    raw_text      TEXT,
    decision      TEXT NOT NULL,
    notes         TEXT
);

CREATE INDEX IF NOT EXISTS stage_runs_request_idx ON stage_runs(request_id);

CREATE TABLE IF NOT EXISTS audit (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT NOT NULL REFERENCES requests(id),
    actor         TEXT NOT NULL,
    action        TEXT NOT NULL,
    detail        TEXT,
    created_at    DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS audit_request_idx ON audit(request_id);

CREATE TABLE IF NOT EXISTS formulary (
    drug_name     TEXT PRIMARY KEY,
    on_formulary  INTEGER NOT NULL,
    requires_pa   INTEGER NOT NULL,
    step_therapy  TEXT
);
