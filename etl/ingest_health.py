#!/usr/bin/env python3
"""
etl/ingest_health.py  —  Streaming ETL: Apple Health Export XML → PostgreSQL

Parses the export in one forward pass (iterparse / SAX-style), then processes
GPX tracks, computes training load, and aggregates daily_health.

Deduplication strategy
──────────────────────
  * Workouts        UNIQUE(start_date, activity_type)
  * sleep_records   UNIQUE(start_time, end_time, stage)
  * health_records  UNIQUE(metric_type, start_time, end_time, value)
  * gps_tracks      UNIQUE(workout_id, ts)
  * workout_laps    UNIQUE(workout_id, event_type, start_time)

Every INSERT uses ON CONFLICT DO NOTHING against those indices, so the script
is safe to re-run any number of times: nothing is ever duplicated or lost.

Requires: the PostgreSQL schema (sql/schema.sql) must already be applied.

Usage:
    python etl/ingest_health.py
    python etl/ingest_health.py --xml /path/to/export.xml --export-dir /path/to/export
    DATABASE_URL=postgresql://user:pass@host/db python etl/ingest_health.py
"""

import argparse
import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from langsmith import traceable

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent
EXPORT_DIR = ROOT / "data" / "personal" / "apple_health_export"
XML_FILE   = EXPORT_DIR / "ייצוא.xml"

_DEFAULT_DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://localhost:5432/flexllm"
)

# ─── Lookup tables ────────────────────────────────────────────────────────────

ACTIVITY_TYPES: dict[str, str] = {
    "HKWorkoutActivityTypeRunning":                       "running",
    "HKWorkoutActivityTypeTraditionalStrengthTraining":   "strength",
    "HKWorkoutActivityTypeFunctionalStrengthTraining":    "strength",
    "HKWorkoutActivityTypeSwimming":                      "swimming",
    "HKWorkoutActivityTypeWalking":                       "walking",
    "HKWorkoutActivityTypeCycling":                       "cycling",
    "HKWorkoutActivityTypeHiking":                        "hiking",
    "HKWorkoutActivityTypeYoga":                          "yoga",
    "HKWorkoutActivityTypeMindAndBody":                   "mindfulness",
    "HKWorkoutActivityTypeElliptical":                    "elliptical",
    "HKWorkoutActivityTypeStairClimbing":                 "stair_climbing",
    "HKWorkoutActivityTypeHighIntensityIntervalTraining": "hiit",
    "HKWorkoutActivityTypeCrossTraining":                 "cross_training",
    "HKWorkoutActivityTypeCooldown":                      "cooldown",
    "HKWorkoutActivityTypeOther":                         "other",
}

SLEEP_STAGES: dict[str, str] = {
    "HKCategoryValueSleepAnalysisAsleepCore":        "core",
    "HKCategoryValueSleepAnalysisAsleepDeep":        "deep",
    "HKCategoryValueSleepAnalysisAsleepREM":         "rem",
    "HKCategoryValueSleepAnalysisAwake":             "awake",
    "HKCategoryValueSleepAnalysisInBed":             "in_bed",
    "HKCategoryValueSleepAnalysisAsleepUnspecified": "core",
}

WORKOUT_EVENTS: dict[str, str] = {
    "HKWorkoutEventTypeSegment": "segment",
    "HKWorkoutEventTypeLap":     "lap",
    "HKWorkoutEventTypePause":   "pause",
    "HKWorkoutEventTypeResume":  "resume",
    "HKWorkoutEventTypeMarker":  "marker",
}

# HK record types stored as time-series rows in health_records
TIMESERIES_TYPES: dict[str, str] = {
    "HKQuantityTypeIdentifierHeartRate":    "heart_rate",
    "HKQuantityTypeIdentifierRunningSpeed": "running_speed",
    "HKQuantityTypeIdentifierRunningPower": "running_power",
    "HKQuantityTypeIdentifierStepCount":    "step_count",
}

# HK record types whose latest daily value goes into daily_health columns
DAILY_SCALAR_TYPES: dict[str, str] = {
    "HKQuantityTypeIdentifierRestingHeartRate":           "resting_heart_rate_bpm",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":   "hrv_sdnn_ms",
    "HKQuantityTypeIdentifierVO2Max":                     "vo2max_ml_kg_min",
    "HKQuantityTypeIdentifierBodyMass":                   "body_mass_kg",
    "HKQuantityTypeIdentifierOxygenSaturation":           "spo2_pct",
    "HKQuantityTypeIdentifierRespiratoryRate":            "respiratory_rate_rpm",
    "HKQuantityTypeIdentifierWalkingHeartRateAverage":    "walking_hr_avg_bpm",
    "HKQuantityTypeIdentifierHeartRateRecoveryOneMinute": "hr_recovery_1min_bpm",
}

# Pre-build per-column UPDATE SQL at module load time.
# Column names come from the controlled dict above, so this is safe.
_DAILY_SCALAR_UPDATE: dict[str, str] = {
    col: f"UPDATE daily_health SET {col} = %s WHERE date = %s AND {col} IS NULL"
    for col in DAILY_SCALAR_TYPES.values()
}

_RING_COLUMN_UPDATE: dict[str, str] = {
    col: f"UPDATE daily_health SET {col} = %s WHERE date = %s AND {col} IS NULL"
    for col in ("active_calories", "exercise_min", "stand_hours")
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

_TZ_RE = re.compile(r'\s*([+-])(\d{2})(\d{2})$')


def _normalize_ts(s: str) -> str:
    s = s.strip().replace("Z", "+00:00")
    return _TZ_RE.sub(r'\1\2:\3', s)


def _ts(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        return (
            datetime.fromisoformat(_normalize_ts(s))
            .astimezone(timezone.utc)
            .strftime("%Y-%m-%d %H:%M:%S")
        )
    except ValueError:
        return None


def _date(utc: Optional[str]) -> Optional[str]:
    return utc[:10] if utc else None


def _f(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _meta_num(raw: Optional[str]) -> Optional[float]:
    return _f(raw.split()[0]) if raw else None


def _f_to_c(val: Optional[float]) -> Optional[float]:
    return (val - 32) * 5 / 9 if val is not None else None


def _dur_min(start_raw: str, end_raw: str) -> float:
    return (
        datetime.fromisoformat(_normalize_ts(end_raw))
        - datetime.fromisoformat(_normalize_ts(start_raw))
    ).total_seconds() / 60


# ─── Ingester ─────────────────────────────────────────────────────────────────

class HealthIngester:
    """
    Single-pass streaming ETL from an Apple Health Export directory into PostgreSQL.

    All table writes use ON CONFLICT DO NOTHING, so the instance can be run
    against an already-populated database without creating any duplicates.
    """

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self, database_url: str, xml: Path, export_dir: Path) -> None:
        self.xml        = xml
        self.export_dir = export_dir

        self.con = psycopg2.connect(database_url)
        self.cur = self.con.cursor()
        self._init_db()

        self.counts: dict[str, int] = dict.fromkeys(
            ["workouts", "laps", "running_form", "sleep",
             "health_rec", "activity_rings", "gps_tracks"],
            0,
        )

    def _init_db(self) -> None:
        """Create dedup unique indices (idempotent — schema must already exist)."""
        for stmt in [
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_workouts_start_type "
            "ON workouts(start_date, activity_type)",

            "CREATE UNIQUE INDEX IF NOT EXISTS ux_sleep_window "
            "ON sleep_records(start_time, end_time, stage)",

            "CREATE UNIQUE INDEX IF NOT EXISTS ux_health_rec "
            "ON health_records(metric_type, start_time, end_time, value)",

            "CREATE UNIQUE INDEX IF NOT EXISTS ux_gps_track "
            "ON gps_tracks(workout_id, ts)",

            "CREATE UNIQUE INDEX IF NOT EXISTS ux_laps "
            "ON workout_laps(workout_id, event_type, start_time)",
        ]:
            self.cur.execute(stmt)
        self.con.commit()

    # ── Element handlers ──────────────────────────────────────────────────────

    def _on_me(self, elem) -> None:
        self.cur.execute("SELECT COUNT(*) FROM athlete_profile")
        if self.cur.fetchone()[0]:
            return
        dob   = elem.get("HKCharacteristicTypeIdentifierDateOfBirth")
        sex   = elem.get("HKCharacteristicTypeIdentifierBiologicalSex", "")
        sex   = sex.replace("HKBiologicalSex", "").lower() or None
        blood = elem.get("HKCharacteristicTypeIdentifierBloodType", "")
        blood = blood.replace("HKBloodType", "") or None
        self.cur.execute(
            "INSERT INTO athlete_profile (date_of_birth, biological_sex, blood_type) "
            "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (dob, sex, blood),
        )
        self.con.commit()

    def _workout_stats(self, elem) -> dict:
        out: dict = {}
        for s in elem.findall("WorkoutStatistics"):
            t = s.get("type")
            if t:
                out[t] = {
                    "sum": _f(s.get("sum")),
                    "avg": _f(s.get("average")),
                    "min": _f(s.get("minimum")),
                    "max": _f(s.get("maximum")),
                    "unit": s.get("unit"),
                }
        return out

    def _workout_meta(self, elem) -> dict:
        return {
            m.get("key"): m.get("value")
            for m in elem.findall("MetadataEntry")
            if m.get("key")
        }

    def _on_workout(self, elem) -> None:
        hk_type  = elem.get("workoutActivityType", "")
        activity = ACTIVITY_TYPES.get(
            hk_type, hk_type.replace("HKWorkoutActivityType", "").lower()
        )
        start = _ts(elem.get("startDate"))
        end   = _ts(elem.get("endDate"))
        if not start or not end:
            return

        stats = self._workout_stats(elem)
        meta  = self._workout_meta(elem)

        dist_run  = stats.get("HKQuantityTypeIdentifierDistanceWalkingRunning", {})
        dist_swim = stats.get("HKQuantityTypeIdentifierDistanceSwimming", {})
        raw_dist  = _f(elem.get("totalDistance"))
        if raw_dist is None:
            if dist_run.get("sum") is not None:
                raw_dist = dist_run["sum"]
            elif dist_swim.get("sum") is not None:
                raw_dist = dist_swim["sum"] / 1000

        act_e      = stats.get("HKQuantityTypeIdentifierActiveEnergyBurned", {})
        bas_e      = stats.get("HKQuantityTypeIdentifierBasalEnergyBurned", {})
        active_cal = _f(elem.get("totalEnergyBurned")) or act_e.get("sum")
        basal_cal  = bas_e.get("sum")
        hr_s       = stats.get("HKQuantityTypeIdentifierHeartRate", {})
        sp_s       = stats.get("HKQuantityTypeIdentifierRunningSpeed", {})
        step_s     = stats.get("HKQuantityTypeIdentifierStepCount", {})
        steps      = int(step_s["sum"]) if step_s.get("sum") else None

        temp_c   = _f_to_c(_meta_num(meta.get("HKWeatherTemperature")))
        hum_raw  = _meta_num(meta.get("HKWeatherHumidity"))
        humidity = hum_raw / 100 if hum_raw is not None else None
        indoor   = int(meta.get("HKIndoorWorkout", "0") == "1")

        gpx_ref  = elem.find(".//FileReference")
        gpx_path = gpx_ref.get("path") if gpx_ref is not None else None

        self.cur.execute(
            """
            INSERT INTO workouts
                (activity_type, start_date, end_date, duration_min, distance_km,
                 active_calories, basal_calories,
                 avg_heart_rate_bpm, min_heart_rate_bpm, max_heart_rate_bpm,
                 avg_speed_kmh,     min_speed_kmh,      max_speed_kmh,
                 step_count, indoor, weather_temp_c, weather_humidity_pct,
                 gpx_file_path, source_name)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            (
                activity, start, end,
                _f(elem.get("duration")),
                raw_dist,
                active_cal, basal_cal,
                hr_s.get("avg"), hr_s.get("min"), hr_s.get("max"),
                sp_s.get("avg"), sp_s.get("min"), sp_s.get("max"),
                steps, indoor, temp_c, humidity,
                gpx_path, elem.get("sourceName"),
            ),
        )
        row = self.cur.fetchone()
        if row is None:
            return  # already in DB — skip children too

        self.counts["workouts"] += 1
        wid = row[0]

        if activity == "running":
            gc = stats.get("HKQuantityTypeIdentifierRunningGroundContactTime", {})
            vo = stats.get("HKQuantityTypeIdentifierRunningVerticalOscillation", {})
            sl = stats.get("HKQuantityTypeIdentifierRunningStrideLength", {})
            rp = stats.get("HKQuantityTypeIdentifierRunningPower", {})
            if any(s.get("avg") for s in (gc, vo, sl, rp)):
                self.cur.execute(
                    """
                    INSERT INTO running_form
                        (workout_id,
                         ground_contact_avg_ms,      ground_contact_min_ms,      ground_contact_max_ms,
                         vertical_oscillation_avg_cm, vertical_oscillation_min_cm, vertical_oscillation_max_cm,
                         stride_length_avg_m,         stride_length_min_m,         stride_length_max_m,
                         running_power_avg_w,         running_power_min_w,         running_power_max_w)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        wid,
                        gc.get("avg"), gc.get("min"), gc.get("max"),
                        vo.get("avg"), vo.get("min"), vo.get("max"),
                        sl.get("avg"), sl.get("min"), sl.get("max"),
                        rp.get("avg"), rp.get("min"), rp.get("max"),
                    ),
                )
                self.counts["running_form"] += 1

        for ev in elem.findall("WorkoutEvent"):
            etype = WORKOUT_EVENTS.get(ev.get("type", ""),
                                       (ev.get("type") or "").lower())
            edate = _ts(ev.get("date"))
            if etype and edate:
                self.cur.execute(
                    "INSERT INTO workout_laps "
                    "(workout_id, event_type, start_time, duration_min) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (wid, etype, edate, _f(ev.get("duration"))),
                )
                self.counts["laps"] += 1

    def _on_record(self, elem) -> None:
        rtype = elem.get("type", "")

        if rtype == "HKCategoryTypeIdentifierSleepAnalysis":
            stage = SLEEP_STAGES.get(elem.get("value", ""))
            if not stage:
                return
            start_raw = elem.get("startDate", "")
            end_raw   = elem.get("endDate", "")
            start = _ts(start_raw)
            end   = _ts(end_raw)
            if not start or not end:
                return
            self.cur.execute(
                "INSERT INTO sleep_records "
                "(date, stage, start_time, end_time, duration_min) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (_date(start), stage, start, end, _dur_min(start_raw, end_raw)),
            )
            self.counts["sleep"] += 1
            return

        if rtype in TIMESERIES_TYPES:
            metric = TIMESERIES_TYPES[rtype]
            start  = _ts(elem.get("startDate"))
            end    = _ts(elem.get("endDate"))
            val    = _f(elem.get("value"))
            if val is None or not start:
                return
            self.cur.execute(
                "INSERT INTO health_records "
                "(metric_type, start_time, end_time, value, unit) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (metric, start, end or start, val, elem.get("unit")),
            )
            self.counts["health_rec"] += 1
            return

        if rtype in DAILY_SCALAR_TYPES:
            col   = DAILY_SCALAR_TYPES[rtype]
            start = _ts(elem.get("startDate"))
            val   = _f(elem.get("value"))
            if val is None or not start:
                return
            if rtype == "HKQuantityTypeIdentifierOxygenSaturation" and val <= 1.0:
                val *= 100
            date = _date(start)
            self.cur.execute(
                "INSERT INTO daily_health (date) VALUES (%s) ON CONFLICT DO NOTHING",
                (date,),
            )
            self.cur.execute(_DAILY_SCALAR_UPDATE[col], (val, date))

    def _on_activity_summary(self, elem) -> None:
        date = elem.get("dateComponents")
        if not date:
            return
        ac  = _f(elem.get("activeEnergyBurned"))
        acg = _f(elem.get("activeEnergyBurnedGoal"))
        ex  = _f(elem.get("appleExerciseTime"))
        exg = _f(elem.get("appleExerciseTimeGoal"))
        st  = _f(elem.get("appleStandHours"))
        stg = _f(elem.get("appleStandHoursGoal"))

        self.cur.execute(
            "INSERT INTO activity_rings "
            "(date, active_calories, active_calories_goal, "
            " exercise_min, exercise_min_goal, stand_hours, stand_hours_goal) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (
                date, ac, acg, ex, exg,
                int(st) if st is not None else None,
                int(stg) if stg is not None else None,
            ),
        )
        self.counts["activity_rings"] += 1

        self.cur.execute(
            "INSERT INTO daily_health (date) VALUES (%s) ON CONFLICT DO NOTHING",
            (date,),
        )
        stand_val = int(st) if st is not None else None
        for col, val in [
            ("active_calories", ac),
            ("exercise_min",    ex),
            ("stand_hours",     stand_val),
        ]:
            if val is not None:
                self.cur.execute(_RING_COLUMN_UPDATE[col], (val, date))

    # ── XML streaming ─────────────────────────────────────────────────────────

    @traceable(name="etl:parse_apple_health_xml", run_type="tool")
    def _stream_xml(self) -> None:
        logger.info("Streaming %s …", self.xml.name)
        inside_workout = False
        n = 0

        with open(self.xml, "rb") as fh:
            for event, elem in ET.iterparse(fh, events=("start", "end")):
                if event == "start":
                    if elem.tag == "Workout":
                        inside_workout = True
                    continue

                tag = elem.tag

                if tag == "Me":
                    self._on_me(elem)
                elif tag == "Workout":
                    inside_workout = False
                    self._on_workout(elem)
                elif tag == "Record" and not inside_workout:
                    self._on_record(elem)
                elif tag == "ActivitySummary" and not inside_workout:
                    self._on_activity_summary(elem)

                if not inside_workout:
                    elem.clear()

                n += 1
                if n % 200_000 == 0:
                    self.con.commit()
                    logger.info(
                        "  %s elements | workouts=%d  sleep=%d  health_rec=%d",
                        f"{n:,}",
                        self.counts["workouts"],
                        self.counts["sleep"],
                        self.counts["health_rec"],
                    )

        self.con.commit()
        logger.info(
            "XML done.  %s elements\n"
            "  workouts=%d  laps=%d  running_form=%d\n"
            "  sleep=%d  health_rec=%d  activity_rings=%d",
            f"{n:,}",
            self.counts["workouts"], self.counts["laps"], self.counts["running_form"],
            self.counts["sleep"], self.counts["health_rec"], self.counts["activity_rings"],
        )

    # ── GPX streaming ─────────────────────────────────────────────────────────

    def _stream_gpx(self) -> None:
        NS = "{http://www.topografix.com/GPX/1/1}"
        self.cur.execute(
            "SELECT id, gpx_file_path FROM workouts WHERE gpx_file_path IS NOT NULL"
        )
        rows = self.cur.fetchall()
        logger.info("Parsing %d GPX files …", len(rows))

        FLUSH = 2000
        batch: list = []

        for wid, rel in rows:
            path = self.export_dir / rel.lstrip("/")
            if not path.exists():
                continue
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError:
                continue

            for pt in root.findall(f".//{NS}trkpt"):
                lat = _f(pt.get("lat"))
                lon = _f(pt.get("lon"))
                ele = _f(pt.findtext(f"{NS}ele"))
                t   = pt.findtext(f"{NS}time")
                ts  = _ts(t) if t else None
                ext = pt.find(f"{NS}extensions")
                spd = _f(ext.findtext("speed"))  if ext is not None else None
                crs = _f(ext.findtext("course")) if ext is not None else None

                if lat and lon and ts:
                    batch.append((wid, ts, lat, lon, ele, spd, crs))
                    if len(batch) >= FLUSH:
                        self._flush_gps(batch)
                        batch = []

        self._flush_gps(batch)
        logger.info("GPS tracks: %s", f"{self.counts['gps_tracks']:,}")

    def _flush_gps(self, batch: list) -> None:
        if not batch:
            return
        psycopg2.extras.execute_batch(
            self.cur,
            "INSERT INTO gps_tracks "
            "(workout_id, ts, lat, lon, elevation_m, speed_ms, course_deg) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            batch,
            page_size=1000,
        )
        self.con.commit()
        self.counts["gps_tracks"] += len(batch)

    # ── Training stress score (TRIMP) ─────────────────────────────────────────

    @traceable(name="etl:compute_training_stress", run_type="tool")
    def _compute_tss(self) -> None:
        logger.info("Computing TSS (TRIMP) …")
        self.cur.execute("SELECT date_of_birth FROM athlete_profile LIMIT 1")
        profile = self.cur.fetchone()
        max_hr = 190.0
        if profile and profile[0]:
            try:
                current_year = datetime.now(timezone.utc).year
                age = current_year - int(profile[0][:4])
                max_hr = 208.0 - 0.7 * age
            except Exception:
                pass

        self.cur.execute(
            """
            SELECT w.id, w.duration_min, w.avg_heart_rate_bpm, dh.resting_heart_rate_bpm
            FROM   workouts w
            LEFT JOIN daily_health dh ON dh.date = substr(w.start_date, 1, 10)
            WHERE  w.avg_heart_rate_bpm  IS NOT NULL
              AND  w.duration_min        IS NOT NULL
              AND  w.training_stress_score IS NULL
            """
        )
        rows = self.cur.fetchall()

        updates = []
        for wid, dur, avg_hr, rhr in rows:
            rhr   = rhr or 55.0
            ratio = max(0.0, min(1.0, (avg_hr - rhr) / (max_hr - rhr)))
            trimp = dur * ratio * math.exp(1.92 * ratio)
            updates.append((round(trimp, 2), wid))

        if updates:
            psycopg2.extras.execute_batch(
                self.cur,
                "UPDATE workouts SET training_stress_score = %s WHERE id = %s",
                updates,
                page_size=1000,
            )
            self.con.commit()
        logger.info("TSS set on %d workouts.", len(updates))

    # ── daily_health aggregation ──────────────────────────────────────────────

    @traceable(name="etl:aggregate_daily_health", run_type="tool")
    def _aggregate_daily(self) -> None:
        logger.info("Aggregating daily_health …")

        self.cur.execute(
            """
            UPDATE daily_health SET
                sleep_total_min = (SELECT SUM(duration_min) FROM sleep_records
                                   WHERE date = daily_health.date AND stage != 'in_bed'),
                sleep_deep_min  = (SELECT SUM(duration_min) FROM sleep_records
                                   WHERE date = daily_health.date AND stage = 'deep'),
                sleep_rem_min   = (SELECT SUM(duration_min) FROM sleep_records
                                   WHERE date = daily_health.date AND stage = 'rem'),
                sleep_core_min  = (SELECT SUM(duration_min) FROM sleep_records
                                   WHERE date = daily_health.date AND stage = 'core'),
                sleep_awake_min = (SELECT SUM(duration_min) FROM sleep_records
                                   WHERE date = daily_health.date AND stage = 'awake')
            WHERE EXISTS (SELECT 1 FROM sleep_records WHERE date = daily_health.date)
            """
        )

        self.cur.execute(
            """
            UPDATE daily_health SET
                step_count = (
                    SELECT CAST(SUM(value) AS INTEGER)
                    FROM   health_records
                    WHERE  metric_type = 'step_count'
                      AND  substr(start_time, 1, 10) = daily_health.date
                )
            WHERE EXISTS (
                SELECT 1 FROM health_records
                WHERE metric_type = 'step_count'
                  AND substr(start_time, 1, 10) = daily_health.date
            )
            """
        )

        self.cur.execute(
            """
            UPDATE daily_health SET
                daily_tss = (
                    SELECT COALESCE(SUM(training_stress_score), 0)
                    FROM   workouts
                    WHERE  substr(start_date, 1, 10) = daily_health.date
                      AND  training_stress_score IS NOT NULL
                )
            """
        )
        self.con.commit()

    # ── ATL / CTL / TSB ──────────────────────────────────────────────────────

    @traceable(name="etl:compute_atl_ctl_tsb", run_type="tool")
    def _compute_load(self) -> None:
        logger.info("Computing ATL / CTL / TSB …")
        self.cur.execute(
            "SELECT date, daily_tss FROM daily_health ORDER BY date"
        )
        rows = self.cur.fetchall()

        K7  = math.exp(-1 / 7)
        K42 = math.exp(-1 / 42)
        atl = ctl = 0.0
        ups = []
        for date, tss in rows:
            tss = tss or 0.0
            atl = atl * K7  + tss * (1 - K7)
            ctl = ctl * K42 + tss * (1 - K42)
            ups.append((round(atl, 2), round(ctl, 2), round(ctl - atl, 2), date))

        psycopg2.extras.execute_batch(
            self.cur,
            "UPDATE daily_health SET atl=%s, ctl=%s, tsb=%s WHERE date=%s",
            ups,
            page_size=1000,
        )
        self.con.commit()
        logger.info("Training load computed for %d days.", len(ups))

    # ── Entry point ───────────────────────────────────────────────────────────

    @traceable(name="etl:ingest_health_data", run_type="tool")
    def run(self) -> None:
        try:
            self._stream_xml()
            self._stream_gpx()
            self._compute_tss()
            self._aggregate_daily()
            self._compute_load()
        finally:
            self.cur.close()
            self.con.close()
        logger.info("Ingestion complete.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Stream Apple Health Export XML → PostgreSQL"
    )
    p.add_argument("--xml",          type=Path, default=XML_FILE,             metavar="PATH")
    p.add_argument("--export-dir",   type=Path, default=EXPORT_DIR,           metavar="PATH")
    p.add_argument("--database-url", type=str,  default=_DEFAULT_DATABASE_URL, metavar="URL")
    args = p.parse_args()

    ingester = HealthIngester(
        database_url=args.database_url,
        xml=args.xml,
        export_dir=args.export_dir,
    )
    ingester.run()


if __name__ == "__main__":
    main()
