#!/usr/bin/env python3
"""
etl/watch_health.py — File watcher: auto-ingests Apple Health export on change.

Watches the apple_health_export directory for any .xml file creation or
modification. When a change is detected the watcher debounces for
DEBOUNCE_SECONDS (to let the copy finish), then re-runs ingest_health.py.

Works both as a standalone process and as a Docker service.

Usage:
    python etl/watch_health.py
    python etl/watch_health.py --export-dir /path/to/export --db /path/to/running.db
    python etl/watch_health.py --once        # single ingest pass, then exit
    python etl/watch_health.py --debounce 5  # wait 5 s after last event
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).parent.parent

# Env-variable overrides so the Docker service can be configured without
# rebuilding the image.  CLI flags take precedence over env vars.
_DEFAULT_EXPORT_DIR = Path(
    os.environ.get("HEALTH_EXPORT_DIR", ROOT / "data" / "personal" / "apple_health_export")
)
_DEFAULT_DB_PATH = Path(
    os.environ.get("DB_PATH", ROOT / "data" / "personal" / "running.db")
)

DEBOUNCE_SECONDS = 3.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [health-watcher] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class HealthExportHandler(FileSystemEventHandler):
    """Watches for .xml changes inside the export directory and schedules an ingest."""

    def __init__(self, export_dir: Path, db_path: Path, debounce: float = DEBOUNCE_SECONDS):
        self._export_dir = export_dir
        self._db_path = db_path
        self._debounce = debounce
        self._last_event_time = 0.0
        self._pending = False

    # ── watchdog callbacks ────────────────────────────────────────────────────

    def on_created(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() == ".xml":
            logger.debug("Created: %s", event.src_path)
            self._schedule()

    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() == ".xml":
            logger.debug("Modified: %s", event.src_path)
            self._schedule()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _schedule(self) -> None:
        self._last_event_time = time.monotonic()
        self._pending = True

    def flush_if_due(self) -> bool:
        """Call from the main loop. Returns True when an ingest was triggered."""
        if not self._pending:
            return False
        if time.monotonic() - self._last_event_time < self._debounce:
            return False
        self._pending = False
        _run_ingest(self._export_dir, self._db_path)
        return True


def _run_ingest(export_dir: Path, db_path: Path) -> None:
    """Locate the XML file and launch ingest_health.py as a subprocess."""
    xml_candidates = sorted(export_dir.glob("*.xml"))
    if not xml_candidates:
        logger.warning("No .xml file found in %s — skipping ingest.", export_dir)
        return

    # Prefer export.xml / ייצוא.xml; fall back to any .xml present.
    xml_path = next(
        (p for p in xml_candidates if p.stem.lower() in ("export", "ייצוא")),
        xml_candidates[0],
    )

    logger.info("Ingesting %s → %s …", xml_path.name, db_path)
    ingest_script = Path(__file__).parent / "ingest_health.py"
    result = subprocess.run(
        [
            sys.executable, str(ingest_script),
            "--xml",        str(xml_path),
            "--db",         str(db_path),
            "--export-dir", str(export_dir),
        ],
        check=False,
    )
    if result.returncode == 0:
        logger.info("Ingest completed successfully.")
    else:
        logger.error("Ingest exited with code %d.", result.returncode)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Watch Apple Health export directory and auto-ingest on change"
    )
    p.add_argument("--export-dir", type=Path, default=_DEFAULT_EXPORT_DIR, metavar="PATH",
                   help="Directory containing the Apple Health XML export")
    p.add_argument("--db", type=Path, default=_DEFAULT_DB_PATH, metavar="PATH",
                   help="Target SQLite database path")
    p.add_argument("--debounce", type=float, default=DEBOUNCE_SECONDS, metavar="SECS",
                   help="Seconds to wait after the last event before triggering ingest "
                        "(allows slow file copies to finish)")
    p.add_argument("--once", action="store_true",
                   help="Run a single ingest pass now and exit (no file watching)")
    args = p.parse_args()

    if args.once:
        _run_ingest(args.export_dir, args.db)
        return

    if not args.export_dir.exists():
        logger.warning("Watch directory %s does not exist — creating it.", args.export_dir)
        args.export_dir.mkdir(parents=True, exist_ok=True)

    handler = HealthExportHandler(args.export_dir, args.db, args.debounce)
    observer = Observer()
    observer.schedule(handler, str(args.export_dir), recursive=False)
    observer.start()
    logger.info(
        "Watching %s for Apple Health export changes (debounce=%.1fs) …",
        args.export_dir, args.debounce,
    )

    try:
        while True:
            time.sleep(0.5)
            handler.flush_if_due()
    except KeyboardInterrupt:
        logger.info("Stopping watcher …")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
