"""Real holiday/date regressions for the production release preflight."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pytest
import yaml

import gas_release_schedule as schedule
import gas_run_weekly_prediction as runner

ROOT = Path(__file__).resolve().parents[1]


def workflow_bash():
    """Use Git for Windows Bash, avoiding the unrelated System32 WSL launcher."""
    if os.name == "nt":
        git = shutil.which("git")
        assert git is not None, "Git for Windows is required for workflow verification"
        git_path = Path(git)
        candidates = [git_path.with_name("bash.exe")]
        candidates.extend(parent / "bin/bash.exe" for parent in list(git_path.parents)[:3])
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise AssertionError("Could not locate Git for Windows Bash beside git.exe")
    bash = shutil.which("bash")
    assert bash is not None, "Bash is required for workflow verification"
    return bash


@pytest.mark.parametrize(
    "clock",
    [
        "2026-10-13T16:46:00-04:00",  # Actual failed production run.
        "2026-10-14T09:59:00-04:00",  # Holiday release not due yet.
        "2026-10-06T09:59:00-04:00",  # Ordinary Tuesday before publication.
        "2026-10-14T01:00:00+00:00",  # Still the pre-release Tuesday in Eastern.
    ],
)
def test_direct_runner_skips_unpublished_release(clock):
    assert not runner.is_release_day(datetime.fromisoformat(clock))


@pytest.mark.parametrize(
    "clock",
    [
        "2026-10-14T11:30:00-04:00",
        "2026-10-06T11:30:00-04:00",
        "2025-01-21T11:30:00-05:00",  # EIA's explicit Tuesday exception.
    ],
)
def test_direct_runner_accepts_due_release(clock):
    assert runner.is_release_day(datetime.fromisoformat(clock))


@pytest.mark.parametrize(
    ("reference", "release"),
    [
        ("2025-01-20", "2025-01-21"),
        ("2025-02-17", "2025-02-18"),
        ("2025-05-26", "2025-05-28"),
        ("2025-09-01", "2025-09-03"),
        ("2025-10-13", "2025-10-15"),
        ("2025-11-10", "2025-11-12"),
        ("2026-01-19", "2026-01-21"),
        ("2026-02-16", "2026-02-18"),
        ("2026-05-25", "2026-05-27"),
        ("2026-09-07", "2026-09-09"),
        ("2026-10-12", "2026-10-14"),
        ("2027-01-18", "2027-01-20"),
    ],
)
def test_published_eia_exceptions(reference, release):
    expected = schedule.expected_release_at(date.fromisoformat(reference))
    assert expected.date() == date.fromisoformat(release)
    assert expected.hour == 10 and expected.minute == 0


@pytest.mark.parametrize(
    ("clock", "due", "release_at"),
    [
        ("2026-10-14T13:59:59+00:00", False, "2026-10-14T10:00:00-04:00"),
        ("2026-10-14T14:00:00+00:00", True, "2026-10-14T10:00:00-04:00"),
        ("2026-11-03T14:59:59+00:00", False, "2026-11-03T10:00:00-05:00"),
        ("2026-11-03T15:00:00+00:00", True, "2026-11-03T10:00:00-05:00"),
    ],
)
def test_release_clock_uses_eastern_and_dst(clock, due, release_at):
    result = schedule.decide_release_run(now=datetime.fromisoformat(clock))
    assert result.run is due
    assert result.scheduled_release_at_eastern == release_at


@pytest.mark.parametrize("reference", ["2024-12-30", "2027-01-25", "2026-10-13"])
def test_unreviewed_or_non_monday_calendar_is_rejected(reference):
    with pytest.raises(ValueError):
        schedule.expected_release_at(date.fromisoformat(reference))


def test_naive_clock_is_rejected():
    with pytest.raises(ValueError, match="timezone"):
        schedule.decide_release_run(now=datetime(2026, 10, 14, 11, 30))


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="unrecognized"):
        schedule.decide_release_run(
            now=datetime.fromisoformat("2026-10-14T11:30:00-04:00"),
            event_name="schedule",
            schedule_expression="30 11 * * 4",
        )


CURRENT_STATUS = {
    "status": "SUCCESS",
    "source_observation_date": "2026-10-12",
    "target_date": "2026-10-19",
}


@pytest.mark.parametrize(
    ("clock", "cron", "event", "status", "due"),
    [
        ("2026-10-13T16:46:00-04:00", "30 11 * * 2", "schedule", None, False),
        ("2026-10-14T11:30:00-04:00", "30 11 * * 3", "schedule", None, True),
        ("2026-10-14T11:30:00-04:00", "30 11 * * 3", "schedule", CURRENT_STATUS, False),
        ("2026-10-13T16:46:00-04:00", "", "workflow_dispatch", None, False),
        ("2026-10-14T11:30:00-04:00", "", "workflow_dispatch", CURRENT_STATUS, True),
        # Scheduler lateness must not turn a due source week into a silent skip.
        ("2026-10-15T11:30:00-04:00", "30 11 * * 2", "schedule", None, True),
        (
            "2026-10-14T11:30:00-04:00",
            "30 11 * * 3",
            "schedule",
            {**CURRENT_STATUS, "target_date": "2026-10-12"},
            True,
        ),
    ],
)
def test_actual_workflow_preflight_outputs(tmp_path, clock, cron, event, status, due):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/weekly-production.yml").read_text()
    )
    step = next(
        s for s in workflow["jobs"]["forecast"]["steps"] if s.get("id") == "release"
    )
    command = "python gas_release_schedule.py"
    assert step["run"].count(command) == 1
    if status is not None:
        path = tmp_path / "artifacts_part3/pipeline_status.json"
        path.parent.mkdir()
        path.write_text(json.dumps(status), encoding="utf-8")
    script = step["run"].replace(
        command,
        " ".join(
            [
                shlex.quote(Path(sys.executable).as_posix()),
                shlex.quote((ROOT / "gas_release_schedule.py").as_posix()),
                "--root",
                shlex.quote(tmp_path.as_posix()),
                "--now",
                shlex.quote(clock),
            ]
        ),
    )
    output = tmp_path / "github-outputs"
    env = dict(
        os.environ,
        GITHUB_EVENT_NAME=event,
        SCHEDULE_EXPRESSION=cron,
        GITHUB_OUTPUT=str(output),
    )
    result = subprocess.run(
        [workflow_bash(), "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    decision = json.loads(result.stdout)
    assert decision["run"] is due
    assert decision["reference_week"] == "2026-10-12"
    assert decision["forecast_target_week"] == "2026-10-19"
    assert f"run={'true' if due else 'false'}" in output.read_text().splitlines()


def test_bootstrap_runs_without_site_packages(tmp_path):
    import tzdata

    env = dict(os.environ, PYTHONTZPATH=str(Path(tzdata.__file__).parent / "zoneinfo"))
    env.pop("GITHUB_OUTPUT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "gas_release_schedule.py"),
            "--root",
            str(tmp_path),
            "--event",
            "schedule",
            "--schedule",
            "30 11 * * 2",
            "--now",
            "2026-10-13T16:46:00-04:00",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["run"] is False


def test_malformed_status_fails_preflight(tmp_path):
    path = tmp_path / "artifacts_part3/pipeline_status.json"
    path.parent.mkdir()
    path.write_text("[]", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "gas_release_schedule.py"),
            "--root",
            str(tmp_path),
            "--now",
            "2026-10-14T11:30:00-04:00",
            "--event",
            "workflow_dispatch",
        ],
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "JSON object" in result.stderr


@pytest.mark.parametrize(
    ("decision", "reference", "fresh", "target"),
    [
        ("2026-10-13", "2026-10-05", False, "2026-10-12"),
        ("2026-10-14", "2026-10-05", False, "2026-10-12"),
        ("2026-10-14", "2026-10-12", True, "2026-10-19"),
    ],
)
def test_part0_still_requires_observed_prospective_source(
    monkeypatch, decision, reference, fresh, target
):
    import pandas as pd
    import gas_part0_data_infrastructure as part0

    monkeypatch.setattr(part0, "eastern_today", lambda: pd.Timestamp(decision))
    frame = pd.DataFrame(
        {"week_date": pd.to_datetime([reference]), "gas_us_avg": [4.157]}
    )
    result = part0.check_freshness(frame, part0.Part0Config(min_history_weeks=1))
    assert result["data_freshness_ok"] is fresh
    assert result["forecast_target_week"] == target
