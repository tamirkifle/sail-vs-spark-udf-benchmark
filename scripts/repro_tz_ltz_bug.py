#!/usr/bin/env python3
"""
Reproduces the session-timezone LTZ bug in Sail v0.6.0.

Issue: convert_tz-based LTZ functions (make_timestamp, try_make_timestamp,
       from_utc_timestamp, to_utc_timestamp, convert_timezone) use the host
       machine timezone for output timestamp metadata instead of the Spark
       session timezone.  When host and session differ, displayed wall-clock
       time and/or the underlying UTC epoch are wrong.

Host timezone:    America/Los_Angeles  (PDT = UTC-7 in May / PST = UTC-8 in Feb)
Session timezone: America/New_York     (EDT = UTC-4 in May / EST = UTC-5 in Feb)

Expected:  make_timestamp(2026, 2, 14, 12, 0, 0) in session TZ NY
           → 12:00 EST = 2026-02-14 17:00:00 UTC  (epoch 1739552400)

Run with:
    .venv/bin/sail spark run -f scripts/repro_tz_ltz_bug.py
"""

from __future__ import annotations

import sys
from datetime import datetime
import zoneinfo

# ── Session setup ─────────────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.getActiveSession()
if spark is None:
    print("ERROR: no active SparkSession — run via 'sail spark run -f ...'", file=sys.stderr)
    sys.exit(1)

SESSION_TZ = "America/New_York"
spark.conf.set("spark.sql.session.timeZone", SESSION_TZ)

# Verify the setting took effect
actual_tz = spark.conf.get("spark.sql.session.timeZone")
print(f"[setup] spark.sql.session.timeZone = {actual_tz}")

# ── Expected values ───────────────────────────────────────────────────────────
# make_timestamp(2026, 2, 14, 12, 0, 0) with session TZ = NY (EST = UTC-5):
#   wall clock 12:00:00 in NY → 2026-02-14 17:00:00 UTC
NY_TZ = zoneinfo.ZoneInfo("America/New_York")
WALL = (2026, 2, 14, 12, 0, 0)          # the literal passed to make_timestamp
EXPECTED_EPOCH = int(
    datetime(2026, 2, 14, 12, 0, 0, tzinfo=NY_TZ).timestamp()
)
EXPECTED_DISPLAY = "2026-02-14 12:00:00"

print(f"[setup] expected epoch   = {EXPECTED_EPOCH}  "
      f"({datetime.fromtimestamp(EXPECTED_EPOCH, tz=NY_TZ)})")
print()

# ── Helpers ───────────────────────────────────────────────────────────────────
PASSES: list[str] = []
FAILURES: list[str] = []


def epoch_of(df_or_sql: str) -> int:
    """Return the unix epoch of the first row/column of a SQL expression."""
    row = spark.sql(f"SELECT unix_timestamp(({df_or_sql}))").collect()[0][0]
    return int(row)


def display_of(df_or_sql: str) -> str:
    """Return the formatted display string of the first row/column."""
    row = spark.sql(
        f"SELECT date_format(({df_or_sql}), 'yyyy-MM-dd HH:mm:ss')"
    ).collect()[0][0]
    return str(row)


def check(name: str, sql_expr: str) -> None:
    """Run both epoch and display checks for a SQL expression."""
    print(f"  testing: {sql_expr}")
    try:
        # Show output so the raw value is visible in the log
        spark.sql(f"SELECT ({sql_expr}) AS ts").show(truncate=False)

        actual_epoch = epoch_of(sql_expr)
        actual_display = display_of(sql_expr)
    except Exception as exc:
        FAILURES.append(name)
        print(f"  [FAIL]  {name} — exception: {exc}\n")
        return

    epoch_ok = actual_epoch == EXPECTED_EPOCH
    display_ok = actual_display == EXPECTED_DISPLAY
    ok = epoch_ok and display_ok

    tag = "PASS" if ok else "FAIL"
    (PASSES if ok else FAILURES).append(name)
    print(f"  [{tag}]  {name}")

    if not epoch_ok:
        delta_h = (actual_epoch - EXPECTED_EPOCH) / 3600
        print(f"          epoch expected : {EXPECTED_EPOCH}")
        print(f"          epoch actual   : {actual_epoch}  (delta {delta_h:+.1f} h)")

    if not display_ok:
        print(f"          display expected : '{EXPECTED_DISPLAY}'")
        print(f"          display actual   : '{actual_display}'")

    print()


# ── make_timestamp ────────────────────────────────────────────────────────────
print("=" * 60)
print("make_timestamp")
print("=" * 60)
check(
    "make_timestamp",
    "make_timestamp(2026, 2, 14, 12, 0, 0)",
)

# ── try_make_timestamp ────────────────────────────────────────────────────────
print("=" * 60)
print("try_make_timestamp")
print("=" * 60)
check(
    "try_make_timestamp",
    "try_make_timestamp(2026, 2, 14, 12, 0, 0)",
)

# ── from_utc_timestamp ────────────────────────────────────────────────────────
# 2026-02-14 17:00:00 UTC → convert to NY → should display as 12:00:00
print("=" * 60)
print("from_utc_timestamp")
print("=" * 60)
check(
    "from_utc_timestamp",
    "from_utc_timestamp(TIMESTAMP '2026-02-14 17:00:00', 'America/New_York')",
)

# ── to_utc_timestamp ──────────────────────────────────────────────────────────
# 2026-02-14 12:00:00 NY → convert to UTC → should be 17:00:00 UTC
# We then wrap in from_utc_timestamp(…, NY) to get back the wall-clock for display.
print("=" * 60)
print("to_utc_timestamp  (round-trip via from_utc_timestamp)")
print("=" * 60)
check(
    "to_utc_timestamp",
    "from_utc_timestamp("
    "  to_utc_timestamp(TIMESTAMP '2026-02-14 12:00:00', 'America/New_York'),"
    "  'America/New_York'"
    ")",
)

# ── convert_timezone ──────────────────────────────────────────────────────────
# UTC 17:00 → NY → should display as 12:00:00
print("=" * 60)
print("convert_timezone")
print("=" * 60)
check(
    "convert_timezone",
    "convert_timezone('UTC', 'America/New_York', TIMESTAMP '2026-02-14 17:00:00')",
)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
for name in PASSES:
    print(f"  PASS  {name}")
for name in FAILURES:
    print(f"  FAIL  {name}")
print()
if FAILURES:
    print(f"RESULT: {len(FAILURES)} failure(s) — bug reproduced")
    sys.exit(1)
else:
    print(f"RESULT: all {len(PASSES)} checks passed")
