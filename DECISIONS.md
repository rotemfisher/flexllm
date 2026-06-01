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

## 3. 4-Role coaching system — four specialist agents with handoffs

**Decision:** Four specialist LangGraph agents (Trainer, Physiotherapist, Recovery Coach,
Dietitian) share a single `StateGraph`. Each agent has its own focused tool set and can
hand off to any other agent via LangGraph's `Command` API.

**Why four agents instead of one with all tools:**
The roles are deeply interdependent but a single agent carrying all tools means the LLM
must search 25+ tools on every turn — Qwen2.5:32b handles this poorly once tool names
start colliding. Splitting to four agents:
- Reduces the per-turn tool search space (7–16 tools per agent).
- Makes each system prompt smaller and role-coherent.
- Preserves cross-role continuity via shared `CoachState` and explicit handoff notes.

**Why not a supervisor / orchestrator agent:**
A supervisor would add a full LLM call just to decide routing — expensive and slow on
local hardware. The semantic router (see §17) handles first-turn routing with a 67MB
embedding model in ~10ms, and subsequent turns stay with the current agent until it
explicitly hands off.

**Trade-off accepted:** A handoff forces the LLM to emit a `*_transfer` tool call as its
only tool call in that turn. The conflict resolver in `graph.py` (`_resolve_tool_conflicts`)
strips any domain tool calls that accidentally appear alongside a handoff, so the graph
never enters an invalid state.

---

## 4. Tool architecture — per-agent tool sets, read/write separated

**Decision:** Each specialist agent gets a focused tool set. Tools are split into read
(no side effects) and write (DB mutations). All share a single `config` object for paths.

| Agent | Tools (incl. handoff) |
|-------|-----------------------|
| Trainer | 17 (`get_onboarding_status`, `log/get_fitness_assessments`, `get_vdot_paces`, `get_recent_workouts`, `log_workout_rpe_and_notes`, `get/log_strength_sets`, `save/get_current_workout_plan`, `replace_day_in_plan`, `update_planned_workout_status`, `get_progress_report`, `update_athlete_profile`, `search_coaching_books`, `query_running_database`, `trainer_transfer`) |
| Physiotherapist | 12 (`get_active_injuries`, `get_injury_recovery_trend`, `log_injury`, `log_injury_checkin`, `resolve_injury`, `get_recent_workouts`, `save_workout_plan`, `replace_day_in_plan`, `update_planned_workout_status`, `search_coaching_books`, `physio_transfer`) |
| Recovery Coach | 9 (`get_daily_readiness`, `get_recent_workouts`, `get_current_workout_plan`, `replace_day_in_plan`, `update_planned_workout_status`, `get_progress_report`, `search_coaching_books`, `recovery_transfer`) |
| Dietitian | 8 (`get_nutrition_profile`, `get_daily_readiness`, `get_recent_workouts`, `update_athlete_profile`, `search_coaching_books`, `query_running_database`, `dietitian_transfer`) |

**Why separate read/write:** Read tools open SQLite with `file:{path}?mode=ro` (read-only
URI) as a runtime guard against accidental mutations. Write tools use a plain connection.
All tools initialise `con = None` before the try block and check `if con: con.close()` in
finally to guard against `NameError` on early returns.

**Config centralisation:** `src/config.py` resolves `DB_PATH` and `QDRANT_PATH` to
absolute paths via `Path(__file__).parent.parent` so tools work regardless of CWD.
`.env` variables can still override any setting.

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
│   │   ├── running.db              # main SQLite + LangGraph checkpoint store
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
│   │   ├── coach_agent.py          # get_athlete_context() + build_coach_graph() entry points
│   │   ├── graph.py                # LangGraph StateGraph: 4 agent nodes + 4 tool nodes
│   │   ├── router.py               # Semantic entry router (bge-small cosine similarity)
│   │   ├── handoffs.py             # 4 transfer tools returning LangGraph Command objects
│   │   ├── memory.py               # SummaryStore: daily/weekly LLM-generated summaries
│   │   ├── prompts.py              # Per-agent system prompt builders
│   │   ├── trainer.py              # TRAINER_TOOLS list
│   │   ├── physiotherapist.py      # PHYSIO_TOOLS list
│   │   ├── recovery_coach.py       # RECOVERY_TOOLS list
│   │   └── dietitian.py            # DIETITIAN_TOOLS list
│   ├── config.py                   # Centralised settings (absolute paths, model IDs, LangSmith)
│   ├── tracing.py                  # setup_tracing() + re-export of @traceable
│   ├── models/
│   │   └── agent_state.py          # CoachState TypedDict for LangGraph
│   ├── promts/
│   │   └── system_promt.py         # Legacy monolithic system prompt (kept for reference)
│   └── tools/
│       ├── assessment_tool.py      # get_onboarding_status, log/get_fitness_assessments
│       ├── injury_tool.py          # get_active_injuries, get_injury_recovery_trend
│       ├── injury_write_tool.py    # log_injury, log_injury_checkin, resolve_injury
│       ├── log_workout_feedback_tool.py  # log_workout_rpe_and_notes
│       ├── nutrition_tool.py       # get_nutrition_profile
│       ├── plan_tool.py            # save/get_workout_plan, replace_day_in_plan, update_planned_workout_status
│       ├── profile_tool.py         # update_athlete_profile
│       ├── progress_tool.py        # get_progress_report
│       ├── rag_tool.py             # search_coaching_books (hybrid RAG)
│       ├── readiness_tool.py       # get_daily_readiness
│       ├── sql_tool.py             # query_running_database
│       ├── strength_tool.py        # log_strength_sets, get_recent_strength_sets
│       ├── vdot_tool.py            # get_vdot_paces
│       └── workout_history_tool.py # get_recent_workouts
├── tests/
│   └── test_ingest.py              # Smoke tests for the ETL pipeline
├── src/cli.py                      # Interactive CLI entry point (streaming output)
└── DECISIONS.md                    # This file
```

**Why `data/personal/` for the DB:** The database contains personal health data.
Keeping it co-located with the raw Apple Health export makes the gitignore rule
simple (`data/personal/`) and makes it obvious what is sensitive.

**`running.db` doubles as the LangGraph checkpoint store:** `SqliteSaver` writes its
`checkpoints` and `checkpoint_blobs` tables into the same file. No second DB file needed;
the WAL mode already handles concurrent access from the checkpointer and the tool layer.

---

## 16. Multi-agent graph topology

**Decision:** `src/agent/graph.py` builds a `StateGraph` with 4 agent nodes and 4 tool
nodes. Each agent owns an exclusive tool node so tools never cross-contaminate.

```
START
  │ (conditional — semantic router or active_agent)
  ├─▶ trainer         ──▶ trainer_tools   ──▶ trainer
  ├─▶ physiotherapist ──▶ physio_tools    ──▶ physiotherapist
  ├─▶ recovery_coach  ──▶ recovery_tools  ──▶ recovery_coach
  └─▶ dietitian       ──▶ dietitian_tools ──▶ dietitian
                                                │ (when last msg has no tool_calls)
                                               END
```

**Why separate tool nodes per agent:** LangGraph's `ToolNode` routes by the tool name in
the last `AIMessage.tool_calls`. If all agents shared one tool node, a handoff tool call
could be dispatched alongside the previous agent's domain tools — producing an invalid
`ToolMessage` sequence. Per-agent nodes guarantee that only the calling agent's tools are
ever invoked in that arm.

**`_MAX_HISTORY_MESSAGES = 30`** (in `graph.py`): The context sent to the LLM is trimmed
to the 30 most recent messages. This fits inside Qwen2.5:32b's `NUM_CTX=16384` budget even
with tool call pairs. The trim logic walks forward past any leading `ToolMessage` to avoid
an orphaned tool result without its parent `AIMessage`.

**Context window per agent node (`_make_agent_node`):**
```
[SystemMessage(role_prompt + athlete_context + handoff_reason)]
+ trimmed_history
```
The `handoff_reason` is appended to the system prompt when `state["handoff_reason"]`
is set, giving the receiving agent a compact clinical briefing without re-injecting the
full conversation.

---

## 17. Semantic entry router

**Decision:** `src/agent/router.py` routes the first turn of each session using cosine
similarity between the user's message and per-domain anchor sentences embedded with
`BAAI/bge-small-en-v1.5` (67MB ONNX, via `fastembed`).

**Two-phase routing:**
1. **Mid-session (active_agent set):** `route_entry` returns `state["active_agent"]`
   immediately — no embedding call. This preserves handoff decisions across turns.
2. **Session start (active_agent not set):** Embed the latest human message, compute
   cosine similarity against the mean embedding of each domain's anchor bank, pick the
   closest domain.

**Anchor bank design:** Each domain has 14–21 representative sentences covering diverse
phrasings of real user requests. The mean anchor vector approximates the centroid of that
domain's intent space. More diverse anchors → better coverage of edge-case phrasing.

**Why bge-small, not bge-large:** The router fires on every session start. bge-small
(67MB) loads in ~2s cold and embeds a sentence in <10ms. bge-large (1.3GB) is already
loaded for RAG; running it synchronously in the router on the same process would compete
for GPU VRAM. The routing task only needs coarse-grained intent classification, not the
fine-grained semantic similarity that RAG retrieval requires.

**Model and anchor embeddings are `lru_cache(maxsize=1)` cached** — loaded once per
process, never recomputed between turns.

---

## 18. Agent handoffs — LangGraph Command API

**Decision:** Each agent has one `*_transfer` tool that returns a `Command(goto=target,
update={...})`. LangGraph intercepts the `Command` and re-routes the graph without
returning to the calling agent node.

```python
# Example: trainer hands off to physiotherapist
return Command(
    goto="physiotherapist",
    update={"active_agent": "physiotherapist", "handoff_reason": "<clinical note>"}
)
```

**Why Command instead of a state flag + conditional edge:**
A state-flag approach would need the ToolNode to process the handoff tool, write a flag,
return to the agent node, then have the agent node emit no tool calls so a conditional
edge can route to the next agent. That is three extra graph steps and one extra LLM call.
`Command` exits the ToolNode directly to the target node — zero extra LLM calls.

**Conflict resolver (`_resolve_tool_conflicts` in `graph.py`):**
LLMs occasionally emit a handoff tool call alongside a domain tool call in the same
response despite instructions not to. Keeping both causes LangGraph to try to return a
`ToolMessage` next to a `Command`, which is undefined behaviour. The conflict resolver
inspects `response.tool_calls`, detects any handoff name in `_HANDOFF_TOOL_NAMES`, and
drops all non-handoff calls from that response before it reaches the ToolNode.

**Handoff state fields (in `CoachState`):**
- `active_agent` — persisted across turns so the router re-enters the same agent on the
  next human message. Reset to `None` only when the session ends.
- `handoff_reason` — a concise clinical note set by the transfer tool, injected into the
  receiving agent's system prompt, then cleared to `None` so it does not pollute
  subsequent turns.

---

## 19. LangSmith tracing

**Decision:** `src/tracing.py` activates LangSmith when `LANGSMITH_API_KEY` is present
in `.env`. LangChain/LangGraph read tracing config from `os.environ`, not from Pydantic
`Settings`, so `setup_tracing()` explicitly writes five env vars.

**Key env vars set by `setup_tracing()`:**

| Variable | Value |
|----------|-------|
| `LANGSMITH_API_KEY` | from `config.LANGSMITH_API_KEY` |
| `LANGCHAIN_TRACING_V2` | `"true"` |
| `LANGCHAIN_PROJECT` | `config.LANGCHAIN_PROJECT` (default `"flexllm-coach-local"`) |
| `LANGCHAIN_CALLBACKS_BACKGROUND` | `"true"` — traces ship in a background thread |
| `LANGCHAIN_TAGS` | `config.ENVIRONMENT` — tags every run for easy filter in the UI |

**Why background callbacks:** The main thread runs an interactive CLI. Blocking to flush
traces on each LLM call would add latency the user would feel. Background shipping means
trace data may be lost on a hard crash, but for a local personal tool that trade-off is
acceptable.

**`@traceable` decorator:** Re-exported from `src/tracing.py` as a single import point.
Applied to `get_athlete_context()` with `run_type="retriever"` so the LangSmith UI
groups it correctly alongside LangGraph's auto-instrumented node spans.

**Test isolation:** `setup_tracing(project="flexllm-test")` is called from `conftest.py`
so test traces land in a separate LangSmith project and never pollute production run
history.

**`ENVIRONMENT` config field** (`local` | `dev` | `staging` | `prod`): Tags every span.
Useful when the same LangSmith project receives traces from multiple deployment contexts.

**CLI metadata on every `graph.stream()` call** (`src/cli.py`):
```python
run_config = {
    "run_name": f"coaching-{THREAD_ID}",
    "tags": [config.ENVIRONMENT, "cli", config.MODEL_ID],
    "metadata": {
        "source": "cli",
        "session_id": session_id,   # uuid4 hex, unique per CLI invocation
        "thread_id": THREAD_ID,
        "model": config.MODEL_ID,
        "environment": config.ENVIRONMENT,
    },
}
```
This makes every LangSmith trace filterable by session, model, and environment without
any manual annotation inside the agent code.

---

## 20. Conversation memory — LLM-generated summaries

**Decision:** `src/agent/memory.py` compresses session messages into bullet-point daily
summaries, stored in a `conversation_summaries` SQLite table. At session start these
summaries are injected into `athlete_context` so every agent has long-term memory without
replaying the full raw message history.

**Why not raw message history:** The LangGraph `SqliteSaver` checkpointer persists raw
messages, but injecting hundreds of turns into each prompt would overflow the context
window quickly. Summaries trade recall precision for unlimited effective memory horizon.

**Summary hierarchy:**

| Type | Row per | Content | Trigger |
|------|---------|---------|---------|
| `daily` | `(date, domain)` | 3–5 bullet points from one session | End of each CLI session |
| `weekly` | `(week_start, "all")` | Cross-domain consolidated view | Regenerated when `daily` rows ≥ 2 and no weekly exists for the week |

**Dedup key:** `UNIQUE(summary_date, domain, summary_type) ON CONFLICT REPLACE` — re-running
end-of-session summarisation overwrites the existing daily note rather than creating
duplicates.

**`format_for_context()` injection order:**
```
=== THIS WEEK'S COACHING SUMMARY ===
<weekly text>

=== RECENT SESSION NOTES (last 7 days) ===
[2026-05-30 / trainer] • ...
[2026-05-31 / recovery_coach] • ...
```
The weekly summary gives the agent a stable high-level view; the daily notes give recency.
Both are prepended to the athlete profile block in `athlete_context`.

**Minimum message threshold:** `save_session_summary()` requires ≥ 4 messages before
summarising (system + human + AI reply + one more exchange). Shorter sessions — e.g. a
single quick question — are not worth a round-trip LLM summarisation call.

**Week start convention:** Israeli Sunday-based weeks (`_week_start(d)` returns `d -
timedelta(days=(d.weekday() + 1) % 7)`). Consistent with how the athlete thinks about
training weeks.

---

## 21. CoachState schema and checkpointing

**Decision:** `CoachState` is a minimal `TypedDict` with three fields:

```python
class CoachState(TypedDict):
    messages:        Annotated[Sequence[BaseMessage], add_messages]
    athlete_context: str   # injected once per session from get_athlete_context()
    active_agent:    str   # "trainer" | "physiotherapist" | "recovery_coach" | "dietitian"
    handoff_reason:  str   # set by transfer tools, cleared after one agent turn
```

**`add_messages` reducer:** LangGraph merges new messages into the existing list rather
than replacing it. This means each `graph.stream()` call appends to the checkpoint rather
than rewriting the full history — critical for performance as conversation grows.

**`athlete_context` is not checkpointed between sessions:** It is passed freshly on every
`graph.stream()` call. This means profile changes (new goal, updated fitness level) take
effect immediately without needing to invalidate the checkpoint. The checkpoint only
carries `messages` and the routing fields.

**`SqliteSaver` checkpointer:** `build_multi_agent_graph()` is a context manager that
opens a `SqliteSaver` from `config.DB_PATH` (the same `running.db`), compiles the graph
with it, yields, then closes the connection. This ensures the SQLite connection is never
left open after the CLI exits.

**Thread ID:** The CLI uses `THREAD_ID = "default"` — a single persistent conversation
thread per athlete. Every session continues from where the last one ended. The
`session_id` (UUID) in `run_config.metadata` distinguishes individual CLI invocations
within that thread in LangSmith without branching the checkpoint graph.
