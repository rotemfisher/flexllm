# Architecture & Design Decisions

This document explains the non-obvious choices made while building FlexLLM —
a personal running coach backed by your own Apple Health data and a local LLM.

---

## 1. Data store — SQLite, not Postgres

**Decision:** A single SQLite file at `data/personal/running.db`.

**Why:** This is one athlete's data. The largest table (`health_records`) holds
~540 K rows; the GPS table will top out around 200 K rows. SQLite handles both
without breaking a sweat, and the entire database is a single file you can copy,
back up, or inspect with DB Browser for SQLite.
WAL mode (`PRAGMA journal_mode = WAL`) lets reads and writes proceed concurrently,
which matters when the LLM agent is querying while the nightly sync is running.

**Trade-off accepted:** Postgres would be necessary if multiple athletes shared
the same instance or if write throughput were a concern. For a personal tool it
is overkill.

---

## 2. Schema layering (Layer 0 → 4)

**Decision:** Tables are grouped into four semantic layers that map to query
frequency and data volume.

| Layer | Tables | Access pattern |
|-------|--------|----------------|
| 0 — Config | `athlete_profile`, `shoes` | Read rarely; rarely changes |
| 1 — Core | `workouts`, `running_form`, `daily_health` | Every LLM tool call |
| 2 — Detail | `activity_rings`, `sleep_records`, `workout_laps`, `kilometer_splits` | On-demand |
| 3 — Time-series | `gps_tracks`, `health_records` | Analytics only — never dumped raw to the LLM |
| 4 — LLM / Agent | `run_summaries`, `vdot_paces` | RAG retrieval & coaching paces |

**Why:** The LLM has a finite context window. By keeping the highest-frequency
query targets small and flat (Layer 1), the agent can answer most questions with
a single `SELECT *` from `v_running_overview` without hitting token limits.
Raw GPS and per-second HR live in separate tables so they never end up in a
prompt by accident.

---

## 3. Streaming XML parse — iterparse, not ElementTree.parse()

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

## 4. Deduplication — unique indices + INSERT OR IGNORE

**Decision:** Five `UNIQUE` indices on natural keys, all inserts written as
`INSERT OR IGNORE`.

| Table | Dedup key |
|-------|-----------|
| `workouts` | `(start_date, activity_type)` |
| `sleep_records` | `(start_time, end_time, stage)` |
| `health_records` | `(metric_type, start_time, end_time, value)` |
| `gps_tracks` | `(workout_id, ts)` |
| `workout_laps` | `(workout_id, event_type, start_time)` |

**Why this key for workouts:** Apple Watch and iPhone can both record the same
session. They will produce two `<Workout>` elements in the export with the same
`startDate` and `workoutActivityType`. No other combination is a reliable
natural key — `sourceName` differs between devices and `creationDate` varies.

**Why INSERT OR IGNORE over check-then-insert:** It is a single round-trip to
the database instead of two, and it is race-condition-free. The unique index
enforces the invariant at the engine level regardless of application logic.

**Re-run safety:** Because every insert is idempotent, the ingestion script can
be run again after receiving a fresh export (e.g., after a sync with Apple
Health) and will only write the truly new records. Existing data is untouched.

---

## 5. Training Stress Score — TRIMP, not TSS/hrTSS

**Decision:** Compute stress load via TRIMP (Training Impulse) using the
Banister/Busso exponential HR-zone formula.

```
hr_ratio = (avg_hr − resting_hr) / (max_hr − resting_hr)
TRIMP    = duration_min × hr_ratio × e^(1.92 × hr_ratio)
```

**Why TRIMP over power-based TSS:** Running power data from Apple Watch only
exists from ~2023 onwards and is not available for strength training. HR data
is available for every workout since 2022. TRIMP gives a consistent,
comparable number across the entire history.

**Why 1.92:** This is Banister's original coefficient calibrated to the
relationship between HR zone and blood lactate in trained athletes. It weights
Zone 4/5 work exponentially more than Zone 1/2, matching the physiological
cost.

**Defaults used when data is missing:**
- Resting HR: 55 bpm (conservative for a trained runner)
- Max HR: `220 − age` (estimated from `athlete_profile.date_of_birth`)

---

## 6. Training load — exponential decay ATL/CTL/TSB

**Decision:** Use the Performance Manager model (Coggan) with exponential
decay constants.

```
ATL(today) = ATL(yesterday) × e^(−1/7)  + TSS × (1 − e^(−1/7))   # 7-day "fatigue"
CTL(today) = CTL(yesterday) × e^(−1/42) + TSS × (1 − e^(−1/42))  # 42-day "fitness"
TSB(today) = CTL − ATL                                             # "form"
```

**Why exponential decay over simple rolling average:** A rolling 7-day average
treats a hard session from 6 days ago identically to yesterday's session.
Exponential decay reflects the biological reality that stress and adaptation
decay continuously rather than dropping off a cliff at a fixed window.

**Positive TSB = fresh (rested more than trained recently).**
**Negative TSB = fatigued (accumulated training load outpaces recovery).**
Optimal race-day TSB is typically +5 to +25.

---

## 7. Coaching book embeddings — local model, ChromaDB

**Decision:** PDFs are parsed with Docling, chunked with a markdown-aware
splitter, embedded with `BAAI/bge-large-en-v1.5` (1024-dim, runs locally on
CPU/MPS), and stored in ChromaDB.

**Why local embeddings:** No API key, no cost, no data leaving the machine.
`bge-large-en-v1.5` scores comparably to `text-embedding-3-small` on MTEB
retrieval benchmarks at zero marginal cost per query.

**Why Docling over pdfplumber / pypdf:** The coaching PDFs (Daniels, Lore of
Running, NSCA) are typeset books with complex two-column layouts, tables, and
figures. Docling uses a layout-aware model that correctly preserves reading
order across columns and marks tables as markdown pipe syntax — critical for the
chunk splitter to keep table rows together rather than splitting them mid-row.

**Why VDOT paces are in SQLite, not ChromaDB:** The VDOT table is structured
reference data (vdot → seconds/km per zone). Semantic search is wrong for this
— you want an exact lookup by VDOT score. An LLM tool call to
`SELECT * FROM vdot_paces WHERE vdot = 52` is faster, cheaper, and more
reliable than a vector retrieval that might return a neighbouring row.

---

## 8. File layout

```
FlexLLM/
├── data/
│   ├── personal/            # private — gitignored
│   │   ├── running.db       # main SQLite database
│   │   └── apple_health_export/
│   │       ├── ייצוא.xml    # raw Apple Health export
│   │       └── workout-routes/  # GPX files
│   ├── chroma_db/           # vector store — gitignored
│   └── model/               # coaching PDFs — gitignored
├── etl/
│   ├── ingest_health.py     # Apple Health XML → SQLite (this ETL)
│   ├── embed_books.py       # Coaching PDFs → ChromaDB
│   └── seed_vdot.py         # Daniels VDOT paces → SQLite
├── sql/
│   └── schema.sql           # Single source of truth for the DB schema
├── src/                     # LangGraph agent, tool definitions, server
├── tests/
│   └── test_ingest.py       # Smoke tests for the ETL pipeline
└── DECISIONS.md             # This file
```

**Why `data/personal/` for the DB:** The database contains personal health data.
Keeping it co-located with the raw export makes the gitignore rule simple
(`data/personal/`) and makes it obvious what is sensitive.
