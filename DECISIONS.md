# Architecture & Design Decisions

This document explains the non-obvious choices made while building FlexLLM —
a personal AI coaching system backed by Apple Health data and a local LLM,
serving four integrated roles: Trainer, Physiotherapist, Recovery Coach, and Dietitian.

---

## 1. Data store — SQLite, not Postgres

**Decision:** A single SQLite file at `data/personal/running.db`.

**Why:** This is one athlete's data. The largest table (`health_records`) holds
~540 K rows; the GPS table tops out around 200 K rows. SQLite handles both
without breaking a sweat, and the entire database is a single file you can copy,
back up, or inspect with DB Browser for SQLite.
WAL mode (`PRAGMA journal_mode = WAL`) lets reads and writes proceed concurrently,
which matters when the LLM agent is querying while the nightly sync is running.

**Trade-off accepted:** Postgres would be necessary if multiple athletes shared
the same instance or if write throughput were a concern. For a personal tool it
is overkill.

---

## 2. Schema layering (Layer 0 → 4)

**Decision:** Tables are grouped into semantic layers that map to query frequency and data volume.

| Layer | Tables | Access pattern |
|-------|--------|----------------|
| 0 — Config | `athlete_profile`, `shoes` | Read rarely; rarely changes |
| 1 — Core | `workouts`, `strength_sets`, `running_form`, `daily_health` | Every LLM tool call |
| 1b — Injury | `injuries`, `injury_checks` | Before every session prescription |
| 1c — Planning | `planned_workouts`, `fitness_assessments` | Written by agent, read back each session |
| 2 — Detail | `activity_rings`, `sleep_records`, `workout_laps`, `kilometer_splits` | On-demand |
| 3 — Time-series | `gps_tracks`, `health_records` | Analytics only — never dumped raw to LLM |
| 4 — LLM / Agent | `run_summaries`, `vdot_paces` | RAG retrieval & coaching paces |

**Why:** The LLM has a finite context window. Keeping the highest-frequency
query targets small and flat (Layers 1–1c) means the agent answers most questions
with a single tool call. Raw GPS and per-second HR live in Layer 3 so they never
accidentally end up in a prompt.

**`athlete_profile` holds the coaching contract:**
Key fields: `fitness_level` ('beginner'|'intermediate'|'advanced'),
`onboarding_complete` (0/1), `current_goal`, `secondary_goal`, `dietary_pref`,
`height_cm`, `target_weight_kg`. These drive the onboarding flow, multi-goal
phasing, and meal plan generation.

---

## 3. 4-Role coaching system — one agent, not four

**Decision:** A single LangGraph ReAct agent handles all four roles simultaneously
rather than separate specialist agents.

**Why:** The roles are deeply interdependent at every decision point.
- A knee injury (Physio) changes what the Trainer prescribes.
- A poor HRV reading (Recovery Coach) changes whether the Trainer's plan runs today.
- A strength phase (Trainer) changes the Dietitian's macro targets.

Splitting into four agents would require constant cross-agent state synchronisation.
A single agent with a well-ordered system prompt and 21 tools handles all roles
from one context window, reading athlete data once per session and making
holistic decisions.

**Trade-off accepted:** 21 tools is a large tool list for a local LLM.
Qwen2.5:32b handles it well; smaller models may struggle. The system prompt is
structured as an ordered decision tree (onboarding → session start → situation rules)
to reduce the model's search space.

---

## 4. Tool architecture — 21 tools, read/write separated

**Decision:** Tools are split into read tools (no side effects) and write tools (DB mutations).
All share a single config object for DB and vector store paths.

**Read tools (11):**
`get_onboarding_status`, `get_fitness_assessments`, `get_daily_readiness`,
`get_vdot_paces`, `get_active_injuries`, `get_injury_recovery_trend`,
`get_recent_workouts`, `get_recent_strength_sets`, `get_current_workout_plan`,
`get_nutrition_profile`, `get_progress_report`, `query_running_database`,
`search_coaching_books`

**Write tools (8):**
`log_fitness_assessment`, `log_injury`, `log_injury_checkin`,
`log_workout_rpe_and_notes`, `log_strength_sets`, `save_workout_plan`,
`update_planned_workout_status`

**Why separate:** Read tools use `file:{path}?mode=ro` (read-only SQLite URI) as
a runtime guard against accidental writes. Write tools use a plain connection.
All tools guard against `con.close()` NameError by initialising `con = None`
before the try block and checking `if con: con.close()` in finally.

**Config centralisation:** `src/config.py` resolves `DB_PATH` and `QDRANT_PATH`
to absolute paths via `Path(__file__).parent.parent` so tools work regardless of
the process working directory. Environment variables in `.env` can still override.

---

## 5. Onboarding flow — physical assessment before first plan

**Decision:** When `fitness_level = 'beginner'` and `onboarding_complete = 0`,
the agent builds a 2-day assessment plan instead of a training plan.

**Assessment protocol:**
- Day 1 (Running): 5-min walk → 10-min easy jog → 1km time trial at RPE 8.
  Result fed to `log_fitness_assessment(assessment_type='onboarding_run')`.
  System auto-estimates VDOT via reverse lookup into `vdot_paces` (find VDOT
  where `t_pace_sec` is closest to the measured pace).
- Day 3 (Strength): max bodyweight squats + push-ups → sub-maximal barbell test.
  Result fed to `log_fitness_assessment(assessment_type='onboarding_strength')`.
  System auto-estimates 1RM via Epley formula.

**Why:** Starting a beginner with a week of training before knowing their baseline
risks injury (too hard) or stagnation (too easy). Assessment first establishes the
coaching contract with real numbers, not guesses.

**Intermediates/Advanced:** Schedule a single 1-day assessment (time trial + 3RM
strength test) in week 1, then proceed with training.

---

## 6. Progress tests embedded in every plan

**Decision:** `planned_workouts` has an `is_assessment` flag and a `phase` column.
The system prompt instructs the agent to embed assessment sessions in every plan:
- Running time trial every 4 weeks (`is_assessment=1`, `workout_type='assessment'`)
- Strength 3RM test every 6 weeks (`is_assessment=1`, `workout_type='assessment'`)

**Why not ad-hoc testing:** If tests are not scheduled in the plan, they don't happen.
Embedding them forces a cadence. After the athlete completes a test and reports
results, `log_fitness_assessment` stores the outcome and `get_fitness_assessments`
can show VDOT or 1RM progression over the entire history — making progress
toward the goal visible and concrete.

**`fitness_assessments` table:** Stores raw result + derived value. VDOT is
estimated by reverse-lookup (pace → nearest `t_pace_sec` in `vdot_paces`).
1RM is estimated by Epley formula: `1RM = weight × (1 + reps / 30)`.

---

## 7. Multi-goal planning — phased blocks, not concurrent optimisation

**Decision:** When `athlete_profile.secondary_goal` exists alongside `current_goal`,
the agent detects conflicts and proposes phased training blocks rather than trying
to optimise both simultaneously.

**Example: 10k prep (primary) + muscle gain (secondary):**
- Phase A — Base (6–8 wks): moderate running + 2× hypertrophy strength/week
- Phase B — Build (6 wks): increase running quality, maintain strength
- Phase C — Peak (4–6 wks): 10k-specific intervals, strength maintenance only

**Caloric logic for fat loss + performance:**
- Hard training days: maintenance or slight surplus (performance protected)
- Easy/rest days: moderate deficit (fat loss achieved without impairing adaptation)
- `get_nutrition_profile` reads `active_calories` from `daily_health` to compute
  actual TDEE from Apple Health data rather than estimating from formulas alone.

**`planned_workouts.phase` column:** Tracks which block each session belongs to
('onboarding'|'base'|'build'|'peak'|'race'|'recovery'|'return_to_run').
This lets the agent reason about where in the macrocycle the athlete currently is.

---

## 8. Injury management — log, track, return-to-train protocol

**Decision:** Injuries have a full lifecycle: log → daily check-in → trend analysis → return protocol.

**Lifecycle:**
1. Athlete reports pain → `log_injury` writes to `injuries` with severity, side, pain scale.
2. Remaining week plan is replaced with recovery sessions via `save_workout_plan(phase='recovery')`.
   Skipped sessions are marked via `update_planned_workout_status`.
3. Daily: `log_injury_checkin` writes to `injury_checks` and updates the pain snapshot on `injuries`.
4. `get_injury_recovery_trend` reads the check-in history and fires a
   **RETURN-TO-TRAIN CLEARED** signal when pain ≤ 2 for 3 consecutive days.
5. Return uses 3 phases:
   - Phase 1 (wk 1): 30% of previous volume, easy only → `save_workout_plan(phase='return_to_run')`
   - Phase 2 (wk 2): 50% volume if pain stays ≤ 2
   - Phase 3 (wk 3+): 70% volume, one quality session reintroduced

**Why the check-in table rather than updating the injury record:**
`injury_checks` builds a time series of pain progression. A single updated field
on `injuries` would lose the trend. The trend is what tells the agent whether
to escalate, hold, or return — a snapshot cannot do that.

---

## 9. Strength tracking — per-set logging + Epley progressive overload

**Decision:** `strength_sets` stores one row per set (exercise, weight, reps, RPE).
`get_recent_strength_sets` fetches the last N sessions for a lift and recommends
the next session's weight automatically.

**Why per-set, not per-session:** Session-level aggregates (avg weight, total volume)
lose the information needed for progressive overload. The progression signal lives
in per-set data: if the athlete completed all prescribed reps at RPE ≤ 8, add
2.5–5 kg next session. That decision requires knowing the exact weight and reps
for each set, not a session average.

**Epley formula for estimated 1RM:** `1RM = weight × (1 + reps / 30)`
Used to track absolute strength progress across sessions even when the
rep scheme changes (5×5 one block, 3×8 the next). The estimated 1RM normalises
across schemes and makes progress visible in `get_fitness_assessments`.

---

## 10. Streaming XML parse — iterparse, not ElementTree.parse()

**Decision:** `etl/ingest_health.py` uses `xml.etree.ElementTree.iterparse`
(SAX-style) rather than loading the tree into memory.

**Why:** The Apple Health export (`ייצוא.xml`) is 2.8 million lines. Parsing it
with `ET.parse()` would require ~1–2 GB of RAM. `iterparse` processes elements
one at a time and keeps only the current subtree in memory.

**Key implementation detail — the "inside_workout" guard:**
`iterparse` fires `start` and `end` events for every element. If we call
`elem.clear()` on every `end` event, a Workout element's children
(`WorkoutStatistics`, `WorkoutEvent`, etc.) get wiped before the parent's `end`
fires — leaving empty attribute dicts. The fix is to track whether we are
currently inside a `<Workout>` container and suppress `clear()` on its children
until the parent's `end` event has been processed.

```
start Workout  → set inside_workout = True
end   WorkoutStatistics → skip clear() (inside_workout is True)
end   Workout  → process with children intact → clear() → inside_workout = False
```

---

## 11. Deduplication — unique indices + INSERT OR IGNORE

**Decision:** Five `UNIQUE` indices on natural keys, all inserts written as `INSERT OR IGNORE`.

| Table | Dedup key |
|-------|-----------|
| `workouts` | `(start_date, activity_type)` |
| `sleep_records` | `(start_time, end_time, stage)` |
| `health_records` | `(metric_type, start_time, end_time, value)` |
| `gps_tracks` | `(workout_id, ts)` |
| `workout_laps` | `(workout_id, event_type, start_time)` |

**Why INSERT OR IGNORE over check-then-insert:** Single round-trip, race-condition-free.
The unique index enforces the invariant at the engine level. The ingestion script
is therefore idempotent — re-running after a fresh Apple Health export only writes
truly new records.

---

## 12. Training Stress Score — TRIMP, not TSS/hrTSS

**Decision:** Compute stress load via TRIMP (Training Impulse) using the
Banister/Busso exponential HR-zone formula.

```
hr_ratio = (avg_hr − resting_hr) / (max_hr − resting_hr)
TRIMP    = duration_min × hr_ratio × e^(1.92 × hr_ratio)
```

**Why TRIMP over power-based TSS:** Running power data from Apple Watch only
exists from ~2023 onwards and is not available for strength training. HR data
is available for every workout since 2022. TRIMP gives a consistent, comparable
number across the entire history — including gym sessions, cycling, and walking.

**Why 1.92:** Banister's original coefficient, calibrated to the relationship
between HR zone and blood lactate in trained athletes. It weights Zone 4/5 work
exponentially more than Zone 1/2, matching the physiological cost.

**Defaults when data is missing:**
- Resting HR: 55 bpm (conservative for a trained athlete)
- Max HR: `220 − age` (estimated from `athlete_profile.date_of_birth`)

---

## 13. Training load — exponential decay ATL/CTL/TSB

**Decision:** Use the Performance Manager model (Coggan) with exponential decay constants.

```
ATL(today) = ATL(yesterday) × e^(−1/7)  + TSS × (1 − e^(−1/7))   # 7-day fatigue
CTL(today) = CTL(yesterday) × e^(−1/42) + TSS × (1 − e^(−1/42))  # 42-day fitness
TSB(today) = CTL − ATL                                              # form
```

**Why exponential decay over simple rolling average:** A rolling 7-day average
treats a hard session from 6 days ago identically to yesterday's session.
Exponential decay reflects the biological reality that stress and adaptation
decay continuously rather than dropping off a cliff at a fixed window.

**TSB thresholds used by the agent:**
- TSB > +25 → very fresh (may be under-training)
- TSB +5 to +25 → race-ready window
- TSB −10 to +5 → normal training
- TSB −25 to −10 → building / some accumulated fatigue
- TSB < −25 → HIGH FATIGUE → agent auto-replaces session with easy/rest

---

## 14. Coaching book embeddings — Qdrant + hybrid search + reranking

**Decision:** PDFs are parsed with Docling, chunked with a markdown-aware splitter,
embedded with `BAAI/bge-large-en-v1.5` (dense, 1024-dim) and `Qdrant/bm25` (sparse),
stored in Qdrant, and reranked with `BAAI/bge-reranker-large`.

**Why hybrid dense + sparse (not dense-only):**
Dense embeddings excel at semantic similarity ("threshold training benefits").
Sparse BM25 excels at exact term recall ("VDOT 52", "creatine loading protocol").
Reciprocal Rank Fusion (RRF) over both retrieval streams consistently outperforms
either alone on sports science literature, where both concept proximity and exact
terminology matter.

**Why reranking:** The RRF candidate pool is wider than the final `n_results`.
A cross-encoder reranker (`bge-reranker-large`) re-scores the top candidates with
full query-passage attention — catching passages that rank high in one stream but
are genuinely more relevant than their fusion score suggests.

**Why Docling over pdfplumber / pypdf:** The coaching PDFs (Daniels, Lore of
Running, NSCA, Burke & Deakin) are typeset books with two-column layouts, tables,
and figures. Docling's layout-aware model preserves reading order across columns
and marks tables as markdown pipe syntax — critical so the chunk splitter keeps
table rows together rather than splitting them mid-row.

**Why VDOT paces are in SQLite, not Qdrant:** The VDOT table is structured
reference data (vdot → seconds/km per zone). Semantic search is wrong for this.
An exact lookup `SELECT * FROM vdot_paces WHERE vdot = 52` is faster, cheaper,
and more reliable than a vector retrieval that might return a neighbouring row.

---

## 15. File layout

```
FlexLLM/
├── data/
│   ├── personal/                   # private — gitignored
│   │   ├── running.db              # main SQLite database (all 18 tables)
│   │   └── apple_health_export/
│   │       ├── ייצוא.xml           # raw Apple Health export
│   │       └── workout-routes/     # GPX files
│   ├── qdrant_db/                  # vector store — gitignored
│   └── model/                      # coaching PDFs — gitignored
├── etl/
│   ├── ingest_health.py            # Apple Health XML → SQLite
│   ├── embed_books.py              # Coaching PDFs → Qdrant (hybrid dense+sparse)
│   └── seed_vdot.py                # Daniels VDOT paces → SQLite
├── sql/
│   └── schema.sql                  # Single source of truth for the DB schema
├── src/
│   ├── agent/
│   │   └── coach_agent.py          # LangGraph ReAct agent + athlete context loader
│   ├── config.py                   # Centralised settings (absolute paths, model IDs)
│   ├── models/
│   │   └── agent_state.py          # CoachState TypedDict for LangGraph
│   ├── promts/
│   │   └── system_promt.py         # 4-role system prompt + all tool rules
│   └── tools/
│       ├── assessment_tool.py      # get_onboarding_status, log/get_fitness_assessments
│       ├── injury_tool.py          # get_active_injuries, get_injury_recovery_trend
│       ├── injury_write_tool.py    # log_injury, log_injury_checkin
│       ├── log_workout_feedback_tool.py  # log_workout_rpe_and_notes
│       ├── nutrition_tool.py       # get_nutrition_profile
│       ├── plan_tool.py            # save/get_workout_plan, update_planned_workout_status
│       ├── progress_tool.py        # get_progress_report
│       ├── rag_tool.py             # search_coaching_books (hybrid RAG)
│       ├── readiness_tool.py       # get_daily_readiness
│       ├── sql_tool.py             # query_running_database
│       ├── strength_tool.py        # log_strength_sets, get_recent_strength_sets
│       ├── vdot_tool.py            # get_vdot_paces
│       └── workout_history_tool.py # get_recent_workouts
├── tests/
│   └── test_ingest.py              # Smoke tests for the ETL pipeline
├── cli.py                          # Interactive CLI entry point
└── DECISIONS.md                    # This file
```

**Why `data/personal/` for the DB:** The database contains personal health data.
Keeping it co-located with the raw Apple Health export makes the gitignore rule
simple (`data/personal/`) and makes it obvious what is sensitive.
