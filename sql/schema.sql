-- =============================================================
-- Running Coach — PostgreSQL Schema
-- Dates: stored as TEXT in ISO-8601 ("YYYY-MM-DD HH:MM:SS")
-- for compatibility with existing HealthKit ETL pipelines.
-- Coaching session state is managed by LangGraph AsyncPostgresSaver —
-- no coaching_sessions table needed here.
-- =============================================================

CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector for run_summaries embeddings


-- =============================================================
-- LAYER 0  —  Configuration / Reference
-- =============================================================

CREATE TABLE IF NOT EXISTS athlete_profile (
    id                   BIGSERIAL PRIMARY KEY,
    name                 TEXT,
    date_of_birth        TEXT NOT NULL,
    biological_sex       TEXT,
    blood_type           TEXT,
    height_cm            REAL,
    current_weight_kg    REAL,
    current_goal         TEXT,
    secondary_goal       TEXT,
    target_weight_kg     REAL,
    dietary_pref         TEXT,
    fitness_level        TEXT DEFAULT 'intermediate'
                         CHECK (fitness_level IN ('beginner', 'intermediate', 'advanced')),
    medical_conditions   TEXT,
    onboarding_complete  INTEGER DEFAULT 0,
    updated_at           TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS shoes (
    id               BIGSERIAL PRIMARY KEY,
    brand            TEXT NOT NULL,
    model            TEXT NOT NULL,
    color            TEXT,
    purchase_date    TEXT,
    retired_date     TEXT,
    notes            TEXT
);


-- =============================================================
-- LAYER 1  —  Core
-- =============================================================

CREATE TABLE IF NOT EXISTS workouts (
    id                      BIGSERIAL PRIMARY KEY,
    activity_type           TEXT    NOT NULL,
    start_date              TEXT    NOT NULL,
    end_date                TEXT    NOT NULL,
    duration_min            REAL,
    distance_km             REAL,
    active_calories         REAL,
    basal_calories          REAL,
    avg_heart_rate_bpm      REAL,
    min_heart_rate_bpm      REAL,
    max_heart_rate_bpm      REAL,
    avg_speed_kmh           REAL,
    min_speed_kmh           REAL,
    max_speed_kmh           REAL,
    step_count              INTEGER,
    indoor                  INTEGER,
    avg_mets                REAL,
    elevation_ascended_m    REAL,
    timezone                TEXT,
    weather_temp_c          REAL,
    weather_humidity_pct    REAL,
    training_stress_score   REAL,
    rpe                     INTEGER CHECK (rpe BETWEEN 1 AND 10),
    notes                   TEXT,
    shoe_id                 BIGINT REFERENCES shoes(id),
    gpx_file_path           TEXT,
    source_name             TEXT,
    created_at              TIMESTAMP DEFAULT NOW(),
    updated_at              TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workouts_activity_date
    ON workouts(activity_type, start_date);

CREATE TABLE IF NOT EXISTS strength_sets (
    id              BIGSERIAL PRIMARY KEY,
    workout_id      BIGINT  NOT NULL REFERENCES workouts(id) ON DELETE CASCADE,
    exercise_name   TEXT    NOT NULL,
    set_number      INTEGER NOT NULL DEFAULT 1,
    weight_kg       REAL,
    reps            INTEGER,
    duration_sec    REAL,
    rpe             INTEGER CHECK (rpe BETWEEN 1 AND 10),
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_strength_sets_workout   ON strength_sets(workout_id);
CREATE INDEX IF NOT EXISTS idx_strength_sets_exercise  ON strength_sets(exercise_name, created_at DESC);

CREATE TABLE IF NOT EXISTS running_form (
    workout_id                      BIGINT PRIMARY KEY REFERENCES workouts(id),
    ground_contact_avg_ms           REAL,
    ground_contact_min_ms           REAL,
    ground_contact_max_ms           REAL,
    vertical_oscillation_avg_cm     REAL,
    vertical_oscillation_min_cm     REAL,
    vertical_oscillation_max_cm     REAL,
    stride_length_avg_m             REAL,
    stride_length_min_m             REAL,
    stride_length_max_m             REAL,
    running_power_avg_w             REAL,
    running_power_min_w             REAL,
    running_power_max_w             REAL
);

CREATE TABLE IF NOT EXISTS daily_health (
    date                        TEXT PRIMARY KEY,
    resting_heart_rate_bpm      REAL,
    hrv_sdnn_ms                 REAL,
    vo2max_ml_kg_min            REAL,
    walking_hr_avg_bpm          REAL,
    hr_recovery_1min_bpm        REAL,
    spo2_pct                    REAL,
    respiratory_rate_rpm        REAL,
    body_mass_kg                REAL,
    active_calories             REAL,
    exercise_min                REAL,
    stand_hours                 INTEGER,
    step_count                  INTEGER,
    sleep_total_min             REAL,
    sleep_deep_min              REAL,
    sleep_rem_min               REAL,
    sleep_core_min              REAL,
    sleep_awake_min             REAL,
    daily_tss                   REAL,
    atl                         REAL,
    ctl                         REAL,
    tsb                         REAL
);


-- =============================================================
-- LAYER 1b  —  Injury Log
-- =============================================================

CREATE TABLE IF NOT EXISTS injuries (
    id                  BIGSERIAL PRIMARY KEY,
    onset_date          TEXT    NOT NULL,
    resolved_date       TEXT,
    body_part           TEXT    NOT NULL,
    side                TEXT    CHECK (side IN ('left', 'right', 'bilateral', 'central')),
    injury_type         TEXT,
    diagnosis           TEXT,
    cause               TEXT,
    severity            TEXT    NOT NULL CHECK (severity IN ('mild', 'moderate', 'severe')),
    status              TEXT    NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active', 'recovering', 'resolved')),
    pain_scale          INTEGER CHECK (pain_scale BETWEEN 0 AND 10),
    pain_context        TEXT    CHECK (pain_context IN ('workout', 'recovery', 'rest', 'both')),
    days_missed         INTEGER,
    return_to_run_date  TEXT,
    treatment           TEXT,
    related_workout_id  BIGINT REFERENCES workouts(id),
    notes               TEXT,
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_injuries_status_date
    ON injuries(status, onset_date DESC);

CREATE TABLE IF NOT EXISTS injury_checks (
    id           BIGSERIAL PRIMARY KEY,
    injury_id    BIGINT  NOT NULL REFERENCES injuries(id) ON DELETE CASCADE,
    check_date   TEXT    NOT NULL,
    pain_scale   INTEGER NOT NULL CHECK (pain_scale BETWEEN 0 AND 10),
    pain_context TEXT    NOT NULL
                 CHECK (pain_context IN ('workout', 'recovery', 'rest', 'both')),
    notes        TEXT,
    created_at   TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_injury_checks_injury
    ON injury_checks(injury_id, check_date DESC);


-- =============================================================
-- LAYER 1c  —  Training Plan
-- =============================================================

CREATE TABLE IF NOT EXISTS planned_workouts (
    id                     BIGSERIAL PRIMARY KEY,
    week_start             TEXT    NOT NULL,
    day_date               TEXT    NOT NULL,
    session_order          INTEGER NOT NULL DEFAULT 1,
    activity_type          TEXT    NOT NULL,
    workout_type           TEXT,
    description            TEXT    NOT NULL,
    target_distance_km     REAL,
    target_duration_min    REAL,
    intensity              TEXT    CHECK (intensity IN ('easy', 'moderate', 'hard', 'rest')),
    phase                  TEXT    CHECK (phase IN ('onboarding','base','build','peak','race','recovery','return_to_run')),
    is_assessment          INTEGER NOT NULL DEFAULT 0,
    notes                  TEXT,
    status                 TEXT    NOT NULL DEFAULT 'planned'
                           CHECK (status IN ('planned', 'completed', 'skipped', 'modified')),
    created_at             TIMESTAMP DEFAULT NOW(),
    updated_at             TIMESTAMP DEFAULT NOW(),
    deleted_at             TIMESTAMP
);

-- Partial unique index: enforced only on active (non-deleted) rows.
CREATE UNIQUE INDEX IF NOT EXISTS idx_planned_workouts_active_day_order
    ON planned_workouts(week_start, day_date, session_order)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_planned_workouts_week ON planned_workouts(week_start);

CREATE TABLE IF NOT EXISTS fitness_assessments (
    id               BIGSERIAL PRIMARY KEY,
    assessment_date  TEXT    NOT NULL,
    assessment_type  TEXT    NOT NULL,
    exercise_name    TEXT,
    metric_name      TEXT    NOT NULL,
    metric_value     REAL    NOT NULL,
    estimated_vdot   REAL,
    estimated_1rm_kg REAL,
    notes            TEXT,
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_assessments_type_date
    ON fitness_assessments(assessment_type, assessment_date DESC);
CREATE INDEX IF NOT EXISTS idx_assessments_exercise
    ON fitness_assessments(exercise_name, assessment_date DESC);


-- =============================================================
-- LAYER 2  —  Detail
-- =============================================================

CREATE TABLE IF NOT EXISTS activity_rings (
    date                        TEXT PRIMARY KEY,
    active_calories             REAL,
    active_calories_goal        REAL,
    exercise_min                INTEGER,
    exercise_min_goal           INTEGER,
    stand_hours                 INTEGER,
    stand_hours_goal            INTEGER
);

CREATE TABLE IF NOT EXISTS sleep_records (
    id              BIGSERIAL PRIMARY KEY,
    date            TEXT    NOT NULL,
    stage           TEXT    NOT NULL,
    start_time      TEXT    NOT NULL,
    end_time        TEXT    NOT NULL,
    duration_min    REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sleep_date ON sleep_records(date);

CREATE TABLE IF NOT EXISTS workout_laps (
    id              BIGSERIAL PRIMARY KEY,
    workout_id      BIGINT  NOT NULL REFERENCES workouts(id),
    event_type      TEXT,
    start_time      TEXT,
    duration_min    REAL
);

CREATE INDEX IF NOT EXISTS idx_laps_workout ON workout_laps(workout_id);

CREATE TABLE IF NOT EXISTS kilometer_splits (
    id                  BIGSERIAL PRIMARY KEY,
    workout_id          BIGINT  NOT NULL REFERENCES workouts(id),
    km_number           INTEGER NOT NULL,
    start_time          TEXT,
    duration_sec        REAL,
    distance_m          REAL,
    avg_speed_kmh       REAL,
    avg_heart_rate_bpm  REAL,
    min_heart_rate_bpm  REAL,
    max_heart_rate_bpm  REAL,
    avg_power_w         REAL,
    elevation_gain_m    REAL
);

CREATE INDEX IF NOT EXISTS idx_splits_workout ON kilometer_splits(workout_id);


-- =============================================================
-- LAYER 3  —  Time-series
-- =============================================================

CREATE TABLE IF NOT EXISTS gps_tracks (
    id              BIGSERIAL PRIMARY KEY,
    workout_id      BIGINT  NOT NULL REFERENCES workouts(id),
    ts              TEXT    NOT NULL,
    lat             REAL    NOT NULL,
    lon             REAL    NOT NULL,
    elevation_m     REAL,
    speed_ms        REAL,
    course_deg      REAL
);

CREATE INDEX IF NOT EXISTS idx_gps_workout_time ON gps_tracks(workout_id, ts);

CREATE TABLE IF NOT EXISTS health_records (
    id              BIGSERIAL PRIMARY KEY,
    metric_type     TEXT    NOT NULL,
    start_time      TEXT    NOT NULL,
    end_time        TEXT    NOT NULL,
    value           REAL    NOT NULL,
    unit            TEXT,
    workout_id      BIGINT REFERENCES workouts(id)
);

CREATE INDEX IF NOT EXISTS idx_hr_type_time ON health_records(metric_type, start_time);
CREATE INDEX IF NOT EXISTS idx_hr_workout   ON health_records(workout_id);


-- =============================================================
-- LAYER 4  —  LLM / Agent
-- =============================================================

-- Pre-generated natural-language summaries per run for RAG retrieval.
-- Uses pgvector for efficient nearest-neighbour search.
-- Dimension 1024 matches BAAI/bge-large-en-v1.5.
CREATE TABLE IF NOT EXISTS run_summaries (
    workout_id      BIGINT PRIMARY KEY REFERENCES workouts(id),
    summary_text    TEXT        NOT NULL,
    embedding       vector(1024),
    generated_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vdot_paces (
    vdot                INTEGER PRIMARY KEY,
    e_pace_slow_sec     INTEGER,
    e_pace_fast_sec     INTEGER,
    m_pace_sec          INTEGER,
    t_pace_sec          INTEGER,
    i_pace_sec          INTEGER,
    r_pace_sec          INTEGER
);

-- Conversation summaries (long-term coaching memory across sessions)
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id           BIGSERIAL PRIMARY KEY,
    summary_date TEXT      NOT NULL,
    week_start   TEXT      NOT NULL,
    domain       TEXT      NOT NULL,
    summary_type TEXT      NOT NULL,
    content      TEXT      NOT NULL,
    created_at   TIMESTAMP DEFAULT NOW(),
    UNIQUE(summary_date, domain, summary_type)
);


-- =============================================================
-- Convenience view
-- =============================================================

CREATE OR REPLACE VIEW v_running_overview AS
SELECT
    w.id,
    w.start_date,
    w.duration_min,
    w.distance_km,
    ROUND((w.duration_min / NULLIF(w.distance_km, 0))::numeric, 2) AS pace_min_per_km,
    w.avg_heart_rate_bpm,
    w.max_heart_rate_bpm,
    w.avg_speed_kmh,
    w.elevation_ascended_m,
    w.active_calories,
    w.training_stress_score,
    w.rpe,
    w.notes,
    rf.ground_contact_avg_ms,
    rf.vertical_oscillation_avg_cm,
    rf.stride_length_avg_m,
    rf.running_power_avg_w,
    dh.resting_heart_rate_bpm,
    dh.hrv_sdnn_ms,
    dh.vo2max_ml_kg_min,
    dh.atl,
    dh.ctl,
    dh.tsb,
    dh.sleep_total_min,
    dh.sleep_deep_min,
    s.brand  || ' ' || s.model AS shoe
FROM workouts w
LEFT JOIN running_form  rf ON rf.workout_id = w.id
LEFT JOIN daily_health  dh ON dh.date = substr(w.start_date, 1, 10)
LEFT JOIN shoes          s ON s.id = w.shoe_id
WHERE w.activity_type = 'running'
ORDER BY w.start_date DESC;
