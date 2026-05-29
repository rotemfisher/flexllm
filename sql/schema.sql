-- =============================================================
-- Running Coach — SQLite Schema
-- Dates: stored as TEXT in ISO-8601 ("YYYY-MM-DD HH:MM:SS")
-- so SQLite's built-in date() / strftime() functions work natively.
-- Coaching session state is managed by LangGraph SqliteSaver —
-- no coaching_sessions table needed here.
-- =============================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode   = WAL;   -- concurrent reads during writes
PRAGMA synchronous    = NORMAL;


-- =============================================================
-- LAYER 0  —  Configuration / Reference
-- =============================================================

CREATE TABLE IF NOT EXISTS athlete_profile (
    id                   INTEGER PRIMARY KEY,
    date_of_birth        TEXT NOT NULL,          -- "YYYY-MM-DD"
    biological_sex       TEXT,                   -- 'male' | 'female' | 'other'
    blood_type           TEXT,                   -- e.g. 'B-'
    height_cm            REAL,                   -- BMR calculation
    current_goal         TEXT,                   -- 'fat_loss' | 'muscle_gain' | 'marathon_prep' | '10k_prep' | 'maintenance' | ...
    secondary_goal       TEXT,                   -- optional second goal (e.g. 'muscle_gain' while training for 10k)
    target_weight_kg     REAL,
    dietary_pref         TEXT,                   -- 'vegan' | 'omnivore' | 'gluten-free' | ...
    fitness_level        TEXT DEFAULT 'intermediate'
                         CHECK (fitness_level IN ('beginner', 'intermediate', 'advanced')),
    onboarding_complete  INTEGER DEFAULT 0,      -- 0 = needs physical assessment, 1 = baseline established
    updated_at           TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

-- Shoe rack — lets the coach warn you when a pair is worn out
CREATE TABLE IF NOT EXISTS shoes (
    id               INTEGER PRIMARY KEY,
    brand            TEXT NOT NULL,
    model            TEXT NOT NULL,
    color            TEXT,
    purchase_date    TEXT,                   -- "YYYY-MM-DD"
    retired_date     TEXT,                   -- NULL = currently in rotation
    notes            TEXT
);


-- =============================================================
-- LAYER 1  —  Core  (always available to the LLM via tool calls)
-- =============================================================

-- One row per workout session
CREATE TABLE IF NOT EXISTS workouts (
    id                      INTEGER PRIMARY KEY,
    activity_type           TEXT    NOT NULL,   -- 'running' | 'strength' | 'swimming' | 'walking'
    start_date              TEXT    NOT NULL,   -- ISO-8601
    end_date                TEXT    NOT NULL,
    duration_min            REAL,
    distance_km             REAL,
    active_calories         REAL,
    basal_calories          REAL,

    -- Aggregate heart rate stats for the whole session
    avg_heart_rate_bpm      REAL,
    min_heart_rate_bpm      REAL,
    max_heart_rate_bpm      REAL,

    -- Aggregate pace / speed
    avg_speed_kmh           REAL,
    min_speed_kmh           REAL,
    max_speed_kmh           REAL,

    step_count              INTEGER,
    indoor                  INTEGER,            -- 0 | 1 (SQLite has no BOOLEAN)
    avg_mets                REAL,
    elevation_ascended_m    REAL,
    timezone                TEXT,

    -- Weather (present in older workouts from HealthKit metadata)
    weather_temp_c          REAL,
    weather_humidity_pct    REAL,

    -- Training load  (computed during ETL from duration + HR zones via TRIMP)
    -- TRIMP = duration_min * hr_ratio * exp(1.92 * hr_ratio)
    -- where hr_ratio = (avg_hr - resting_hr) / (max_hr - resting_hr)
    training_stress_score   REAL,

    -- Subjective  (filled in post-run, NULL until the user provides them)
    rpe                     INTEGER CHECK (rpe BETWEEN 1 AND 10),
    notes                   TEXT,

    -- Gear
    shoe_id                 INTEGER REFERENCES shoes(id),

    -- Source
    gpx_file_path           TEXT,               -- relative path to .gpx file
    source_name             TEXT,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    updated_at              TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_workouts_activity_date
    ON workouts(activity_type, start_date);

-- Per-set strength log — enables progressive overload tracking across sessions
CREATE TABLE IF NOT EXISTS strength_sets (
    id              INTEGER PRIMARY KEY,
    workout_id      INTEGER NOT NULL REFERENCES workouts(id) ON DELETE CASCADE,
    exercise_name   TEXT    NOT NULL,           -- 'squat' | 'bench_press' | 'deadlift' | 'pull_up' | ...
    set_number      INTEGER NOT NULL DEFAULT 1,
    weight_kg       REAL,                       -- NULL for bodyweight exercises
    reps            INTEGER,
    duration_sec    REAL,                       -- for timed holds (plank, wall-sit, etc.)
    rpe             INTEGER CHECK (rpe BETWEEN 1 AND 10),
    notes           TEXT,
    created_at      TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_strength_sets_workout
    ON strength_sets(workout_id);

CREATE INDEX IF NOT EXISTS idx_strength_sets_exercise
    ON strength_sets(exercise_name, created_at DESC);


-- Running-specific biomechanics per workout
CREATE TABLE IF NOT EXISTS running_form (
    workout_id                      INTEGER PRIMARY KEY REFERENCES workouts(id),
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

-- Daily fitness markers — the coach's "readiness dashboard"
CREATE TABLE IF NOT EXISTS daily_health (
    date                        TEXT PRIMARY KEY,   -- "YYYY-MM-DD"

    -- Fitness markers
    resting_heart_rate_bpm      REAL,
    hrv_sdnn_ms                 REAL,
    vo2max_ml_kg_min            REAL,
    walking_hr_avg_bpm          REAL,
    hr_recovery_1min_bpm        REAL,

    -- Recovery markers
    spo2_pct                    REAL,
    respiratory_rate_rpm        REAL,
    body_mass_kg                REAL,

    -- Activity totals (from ActivitySummary + Record aggregation)
    active_calories             REAL,
    exercise_min                REAL,
    stand_hours                 INTEGER,
    step_count                  INTEGER,

    -- Sleep totals for the preceding night (derived from sleep_records)
    sleep_total_min             REAL,
    sleep_deep_min              REAL,
    sleep_rem_min               REAL,
    sleep_core_min              REAL,
    sleep_awake_min             REAL,

    -- Training Load  (computed nightly after all workouts are logged)
    -- ATL = 7-day exp. weighted avg. of daily TSS  ("fitness fatigue")
    -- CTL = 42-day exp. weighted avg. of daily TSS  ("fitness base")
    -- TSB = CTL - ATL  ("form" — positive = fresh, negative = fatigued)
    daily_tss                   REAL,   -- sum of TSS across all workouts this day
    atl                         REAL,
    ctl                         REAL,
    tsb                         REAL
);


-- =============================================================
-- LAYER 1b  —  Injury Log  (manually filled by the athlete)
-- The coach reads active injuries before prescribing training.
-- =============================================================

CREATE TABLE IF NOT EXISTS injuries (
    id                  INTEGER PRIMARY KEY,

    -- Timeline
    onset_date          TEXT    NOT NULL,    -- "YYYY-MM-DD" first symptom
    resolved_date       TEXT,               -- NULL = still active

    -- Location
    body_part           TEXT    NOT NULL,    -- 'knee' | 'achilles' | 'shin' | 'hip' | 'foot' | ...
    side                TEXT    CHECK (side IN ('left', 'right', 'bilateral', 'central')),

    -- Classification
    injury_type         TEXT,               -- 'ITBS' | 'stress fracture' | 'plantar fasciitis' | ...
    diagnosis           TEXT,               -- formal medical diagnosis if seen by a clinician
    cause               TEXT,               -- 'overuse' | 'acute trauma' | 'overtraining' | 'unknown'

    -- Severity / status
    severity            TEXT    NOT NULL CHECK (severity IN ('mild', 'moderate', 'severe')),
    status              TEXT    NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active', 'recovering', 'resolved')),

    -- Pain detail at time of logging
    pain_scale          INTEGER CHECK (pain_scale BETWEEN 0 AND 10),
    pain_context        TEXT    CHECK (pain_context IN ('workout', 'recovery', 'rest', 'both')),

    -- Training impact
    days_missed         INTEGER,            -- total training days lost
    return_to_run_date  TEXT,               -- planned or actual date of return to running

    -- Management
    treatment           TEXT,               -- free text: physio, rest, medication, surgery, etc.

    -- Link to the workout where the injury happened (optional)
    related_workout_id  INTEGER REFERENCES workouts(id),

    notes               TEXT,

    created_at          TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

-- Fast lookup: active injuries for daily readiness check
CREATE INDEX IF NOT EXISTS idx_injuries_status_date
    ON injuries(status, onset_date DESC);

-- Pain check-ins: track how a single injury evolves day-to-day
-- One row per visit/self-assessment; enables trend queries for the coach agent.
CREATE TABLE IF NOT EXISTS injury_checks (
    id           INTEGER PRIMARY KEY,
    injury_id    INTEGER NOT NULL REFERENCES injuries(id) ON DELETE CASCADE,
    check_date   TEXT    NOT NULL,   -- "YYYY-MM-DD"
    pain_scale   INTEGER NOT NULL CHECK (pain_scale BETWEEN 0 AND 10),
    pain_context TEXT    NOT NULL
                 CHECK (pain_context IN ('workout', 'recovery', 'rest', 'both')),
    notes        TEXT,
    created_at   TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_injury_checks_injury
    ON injury_checks(injury_id, check_date DESC);


-- =============================================================
-- LAYER 1c  —  Training Plan  (written by agent, read back each session)
-- =============================================================

CREATE TABLE IF NOT EXISTS planned_workouts (
    id                     INTEGER PRIMARY KEY,
    week_start             TEXT    NOT NULL,        -- "YYYY-MM-DD" Monday of the plan week
    day_date               TEXT    NOT NULL,        -- "YYYY-MM-DD" target session date
    session_order          INTEGER NOT NULL DEFAULT 1, -- supports two-a-days
    activity_type          TEXT    NOT NULL,        -- 'running' | 'strength' | 'rest' | 'cross_training'
    workout_type           TEXT,                    -- 'easy' | 'tempo' | 'interval' | 'long_run' | 'recovery' | 'strength' | 'rest' | 'assessment'
    description            TEXT    NOT NULL,        -- what to do
    target_distance_km     REAL,
    target_duration_min    REAL,
    intensity              TEXT    CHECK (intensity IN ('easy', 'moderate', 'hard', 'rest')),
    phase                  TEXT    CHECK (phase IN ('onboarding','base','build','peak','race','recovery','return_to_run')),
    is_assessment          INTEGER NOT NULL DEFAULT 0,  -- 1 = progress test / physical exam session
    notes                  TEXT,                    -- coach rationale / cues
    status                 TEXT    NOT NULL DEFAULT 'planned'
                           CHECK (status IN ('planned', 'completed', 'skipped', 'modified')),
    created_at             TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    updated_at             TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_planned_workouts_day_order
    ON planned_workouts(week_start, day_date, session_order);

CREATE INDEX IF NOT EXISTS idx_planned_workouts_week
    ON planned_workouts(week_start);


-- Fitness assessments: physical exams + periodic progress tests
-- Onboarding uses assessment_type='onboarding_run' or 'onboarding_strength'.
-- Recurring tests (every 4–6 wks) use 'time_trial' or 'strength_1rm'.
CREATE TABLE IF NOT EXISTS fitness_assessments (
    id               INTEGER PRIMARY KEY,
    assessment_date  TEXT    NOT NULL,        -- "YYYY-MM-DD"
    assessment_type  TEXT    NOT NULL,        -- 'onboarding_run' | 'onboarding_strength' | 'time_trial' | 'strength_1rm' | 'cooper_test' | 'body_composition'
    exercise_name    TEXT,                    -- for strength: 'squat' | 'bench_press' | 'deadlift' | ...
    metric_name      TEXT    NOT NULL,        -- what was measured: 'time_sec' | 'distance_m' | 'weight_kg' | 'reps' | 'pace_min_per_km'
    metric_value     REAL    NOT NULL,        -- raw result
    estimated_vdot   REAL,                    -- derived from running tests
    estimated_1rm_kg REAL,                    -- derived from strength tests (Epley formula)
    notes            TEXT,
    created_at       TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_assessments_type_date
    ON fitness_assessments(assessment_type, assessment_date DESC);

CREATE INDEX IF NOT EXISTS idx_assessments_exercise
    ON fitness_assessments(exercise_name, assessment_date DESC);


-- =============================================================
-- LAYER 2  —  Detail  (queried on demand by tool calls)
-- =============================================================

-- Apple Activity Rings — daily goal tracking
CREATE TABLE IF NOT EXISTS activity_rings (
    date                        TEXT PRIMARY KEY,   -- "YYYY-MM-DD"
    active_calories             REAL,
    active_calories_goal        REAL,
    exercise_min                INTEGER,
    exercise_min_goal           INTEGER,
    stand_hours                 INTEGER,
    stand_hours_goal            INTEGER
);

-- Sleep stage breakdown per night
CREATE TABLE IF NOT EXISTS sleep_records (
    id              INTEGER PRIMARY KEY,
    date            TEXT    NOT NULL,   -- calendar date of the night ("YYYY-MM-DD")
    stage           TEXT    NOT NULL,   -- 'core' | 'deep' | 'rem' | 'awake' | 'in_bed'
    start_time      TEXT    NOT NULL,
    end_time        TEXT    NOT NULL,
    duration_min    REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sleep_date ON sleep_records(date);

-- Lap / interval segments from WorkoutEvent — Apple's native splits
-- (not necessarily 1 km; use kilometer_splits for uniform per-km rows)
CREATE TABLE IF NOT EXISTS workout_laps (
    id              INTEGER PRIMARY KEY,
    workout_id      INTEGER NOT NULL REFERENCES workouts(id),
    event_type      TEXT,               -- 'segment' | 'lap' | 'pause' | 'resume'
    start_time      TEXT,
    duration_min    REAL
);

CREATE INDEX IF NOT EXISTS idx_laps_workout ON workout_laps(workout_id);

-- Pre-computed uniform per-kilometer splits  (ETL derives these from GPX + HR)
-- Key for the LLM to answer "why did I slow down at km 4?" without scanning GPS
CREATE TABLE IF NOT EXISTS kilometer_splits (
    id                  INTEGER PRIMARY KEY,
    workout_id          INTEGER NOT NULL REFERENCES workouts(id),
    km_number           INTEGER NOT NULL,   -- 1-based sequential km within the run
    start_time          TEXT,
    duration_sec        REAL,
    distance_m          REAL,               -- usually ~1000; last split may be shorter
    avg_speed_kmh       REAL,
    avg_heart_rate_bpm  REAL,
    min_heart_rate_bpm  REAL,
    max_heart_rate_bpm  REAL,
    avg_power_w         REAL,
    elevation_gain_m    REAL
);

CREATE INDEX IF NOT EXISTS idx_splits_workout ON kilometer_splits(workout_id);


-- =============================================================
-- LAYER 3  —  Time-series  (analytics only — never raw-dumped to LLM)
-- =============================================================

-- Per-second GPS track points from .gpx files
-- 83 runs × ~2,400 pts ≈ 200k rows — reasonable for SQLite
CREATE TABLE IF NOT EXISTS gps_tracks (
    id              INTEGER PRIMARY KEY,
    workout_id      INTEGER NOT NULL REFERENCES workouts(id),
    ts              TEXT    NOT NULL,   -- ISO-8601 timestamp
    lat             REAL    NOT NULL,
    lon             REAL    NOT NULL,
    elevation_m     REAL,
    speed_ms        REAL,
    course_deg      REAL
);

CREATE INDEX IF NOT EXISTS idx_gps_workout_time ON gps_tracks(workout_id, ts);

-- Sub-minute health records  (HR, running speed, power, etc.)
-- workout_id NULL = background (non-workout) measurement
CREATE TABLE IF NOT EXISTS health_records (
    id              INTEGER PRIMARY KEY,
    metric_type     TEXT    NOT NULL,   -- 'heart_rate' | 'running_speed' | 'running_power' | ...
    start_time      TEXT    NOT NULL,
    end_time        TEXT    NOT NULL,
    value           REAL    NOT NULL,
    unit            TEXT,
    workout_id      INTEGER REFERENCES workouts(id)
);

CREATE INDEX IF NOT EXISTS idx_hr_type_time ON health_records(metric_type, start_time);
CREATE INDEX IF NOT EXISTS idx_hr_workout   ON health_records(workout_id);


-- =============================================================
-- LAYER 4  —  LLM / Agent
-- (coaching_sessions state is managed by LangGraph SqliteSaver)
-- =============================================================

-- Pre-generated natural-language summaries per run for RAG retrieval.
-- Embedding column stores a float32 binary vector (sqlite-vec format).
-- Typical dimension: 1536 (text-embedding-3-small) or 768 (smaller models).
CREATE TABLE IF NOT EXISTS run_summaries (
    workout_id      INTEGER PRIMARY KEY REFERENCES workouts(id),
    summary_text    TEXT        NOT NULL,
    embedding       BLOB,       -- sqlite-vec float32 vector: vec_f32(1024)  (BAAI/bge-large-en-v1.5)
    generated_at    TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

-- Daniels' VDOT training paces (seeded by etl/seed_vdot.py).
-- The VDOT tables in the PDF are raster images — not parseable — so we
-- store them as structured SQL for reliable tool-call lookups.
-- All pace columns are seconds-per-km.
CREATE TABLE IF NOT EXISTS vdot_paces (
    vdot                INTEGER PRIMARY KEY,
    e_pace_slow_sec     INTEGER,   -- slower end of Easy zone
    e_pace_fast_sec     INTEGER,   -- faster end of Easy zone
    m_pace_sec          INTEGER,   -- Marathon pace
    t_pace_sec          INTEGER,   -- Threshold / Tempo pace
    i_pace_sec          INTEGER,   -- Interval pace
    r_pace_sec          INTEGER    -- Repetition pace
);

-- =============================================================
-- Convenience view — flattens the most-queried columns.
-- The LLM tool can SELECT * FROM v_running_overview LIMIT 20
-- to get a rich snapshot without joining 4 tables.
-- =============================================================

CREATE VIEW IF NOT EXISTS v_running_overview AS
SELECT
    w.id,
    w.start_date,
    w.duration_min,
    w.distance_km,
    ROUND(w.duration_min / NULLIF(w.distance_km, 0), 2)    AS pace_min_per_km,
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
