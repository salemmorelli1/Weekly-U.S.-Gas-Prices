#!/usr/bin/env python3
"""Offline EIA release preflight; a due release still requires fresh source data.

The reviewed schedule is pinned to EIA's published exceptions, including its
2025 Tuesday exceptions. It does not infer EIA dates from an exchange calendar.
No API clients or model packages are needed by the Ubuntu workflow bootstrap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
CALENDAR_SOURCE = "https://www.eia.gov/petroleum/gasdiesel/schedule.php"
CALENDAR_REVIEWED = "2026-10-13"
CALENDAR_FIRST_WEEK = date(2025, 1, 6)
CALENDAR_LAST_WEEK = date(2027, 1, 18)
SCHEDULE_EXPRESSIONS = {"30 11 * * 2", "30 11 * * 3"}

# (Monday reference week, alternate publication date, EIA holiday label).
HOLIDAY_RELEASES = {
    date(2025, 1, 20): (date(2025, 1, 21), "Martin Luther King Jr. Day / Inauguration"),
    date(2025, 2, 17): (date(2025, 2, 18), "President's Day"),
    date(2025, 5, 26): (date(2025, 5, 28), "Memorial Day"),
    date(2025, 9, 1): (date(2025, 9, 3), "Labor Day"),
    date(2025, 10, 13): (date(2025, 10, 15), "Columbus Day"),
    date(2025, 11, 10): (date(2025, 11, 12), "Veterans Day"),
    date(2026, 1, 19): (date(2026, 1, 21), "Martin Luther King Jr. Day"),
    date(2026, 2, 16): (date(2026, 2, 18), "President's Day"),
    date(2026, 5, 25): (date(2026, 5, 27), "Memorial Day"),
    date(2026, 9, 7): (date(2026, 9, 9), "Labor Day"),
    date(2026, 10, 12): (date(2026, 10, 14), "Columbus Day"),
    date(2027, 1, 18): (date(2027, 1, 20), "Martin Luther King Jr. Day"),
}


@dataclass(frozen=True)
class ReleaseDecision:
    run: bool
    reason: str
    reference_week: str
    forecast_target_week: str
    scheduled_release_at_eastern: str
    calendar_reviewed: str = CALENDAR_REVIEWED
    calendar_source: str = CALENDAR_SOURCE


def expected_release_at(reference_week: date) -> datetime:
    """Earliest scheduled time, not an attestation that FRED has updated."""
    if not isinstance(reference_week, date) or isinstance(reference_week, datetime):
        raise ValueError("reference_week must be a date")
    if reference_week.weekday() != 0:
        raise ValueError("reference_week must be Monday")
    if not CALENDAR_FIRST_WEEK <= reference_week <= CALENDAR_LAST_WEEK:
        raise ValueError(
            f"EIA calendar does not cover {reference_week}; refresh the reviewed "
            f"schedule at {CALENDAR_SOURCE} before automatic publication"
        )
    release_date = HOLIDAY_RELEASES.get(
        reference_week, (reference_week + timedelta(days=1), "")
    )[0]
    return datetime.combine(release_date, time(10), tzinfo=EASTERN)


def decide_release_run(
    *,
    now: datetime | None = None,
    event_name: str = "workflow_dispatch",
    schedule_expression: str = "",
    status: Mapping | None = None,
) -> ReleaseDecision:
    if event_name not in {"schedule", "workflow_dispatch"}:
        raise ValueError(f"unsupported release event: {event_name!r}")
    if event_name == "schedule" and schedule_expression not in SCHEDULE_EXPRESSIONS:
        raise ValueError(f"unrecognized production schedule: {schedule_expression!r}")
    if status is not None and not isinstance(status, Mapping):
        raise ValueError("pipeline status must be a JSON object")
    current = now if now is not None else datetime.now(EASTERN)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("release clock must include an explicit timezone")
    current = current.astimezone(EASTERN)
    reference = current.date() - timedelta(days=current.weekday())
    target = reference + timedelta(days=7)
    release_at = expected_release_at(reference)
    holiday = HOLIDAY_RELEASES.get(reference, (None, ""))[1]
    label = f" ({holiday})" if holiday else ""

    if current < release_at:
        run = False
        reason = f"before EIA scheduled release {release_at.isoformat()}{label}"
    elif (
        event_name == "schedule"
        and status is not None
        and status.get("status") == "SUCCESS"
        and status.get("source_observation_date") == reference.isoformat()
        and status.get("target_date") == target.isoformat()
    ):
        run = False
        reason = f"already published source {reference} for target {target}"
    else:
        run = True
        reason = f"EIA scheduled release is due for {reference}{label}; fresh data still required"
    return ReleaseDecision(
        run, reason, reference.isoformat(), target.isoformat(), release_at.isoformat()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("GASPRICE_ROOT") or Path(__file__).parent),
    )
    parser.add_argument(
        "--event",
        choices=("schedule", "workflow_dispatch"),
        default=os.environ.get("GITHUB_EVENT_NAME", "workflow_dispatch"),
    )
    parser.add_argument("--schedule", default=os.environ.get("SCHEDULE_EXPRESSION", ""))
    parser.add_argument(
        "--now", help="Explicit timezone-aware ISO clock for an offline diagnostic only"
    )
    args = parser.parse_args(argv)
    try:
        status_path = args.root / "artifacts_part3/pipeline_status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.exists()
            else None
        )
        decision = decide_release_run(
            now=datetime.fromisoformat(args.now) if args.now else None,
            event_name=args.event,
            schedule_expression=args.schedule,
            status=status,
        )
        print(json.dumps(asdict(decision), sort_keys=True))
        if os.environ.get("GITHUB_OUTPUT"):
            with open(
                os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(f"run={'true' if decision.run else 'false'}\n")
                handle.write(f"reason={decision.reason}\n")
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"Release preflight error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
