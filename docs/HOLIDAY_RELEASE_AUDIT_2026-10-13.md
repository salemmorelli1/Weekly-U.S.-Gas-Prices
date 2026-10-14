# October 13, 2026 holiday-release audit

Baseline main: `8e927212a666a42cc8b23339a1deca12262f166a`.
Baseline tree: `5cd2b58720bc15c3db71c9e24212f0274650db26`.

## Root cause and evidence

[Production run 38465456557](https://github.com/salemmorelli1/Weekly-U.S.-Gas-Prices/actions/runs/38465456557)
started October 13 at 16:46 Eastern. The workflow identified a normal Tuesday
release, although EIA explicitly scheduled the October 12 price observation
for **Wednesday, October 14** because of Columbus Day.

The [EIA holiday table](https://www.eia.gov/petroleum/gasdiesel/schedule.php)
is the governing source for this publication date. EIA normally publishes
Monday observations around 10 a.m. Tuesday Eastern. The table includes
Wednesday exceptions and two explicit Tuesday exceptions in early 2025.
An exchange trading calendar is not a substitute for this release calendar.

The failed run successfully fetched yfinance history (1,398 weekly rows).
FRED's latest required `GASREGCOVW` observation was October 5. Part 0 therefore
derived October 12 as its next target, which was already in the past on the
October 13 decision date. Its age/future-target guard correctly exited 1.
[FRED](https://fred.stlouisfed.org/series/GASREGCOVW) still showed October 5 at
the audit time; its displayed next-release date was October 13, illustrating
why an indicative downstream date cannot replace EIA's explicit holiday table.

The defect was the preflight's unconditional acceptance of scheduled Tuesdays,
plus the direct runner's weekday-only guard. The initial gate used a source
status check only on Wednesdays and had no EIA exception lookup. The previous
71 tests passed, but did not exercise the holiday release gate. The Ubuntu
image-migration notice was informational; the fatal exit came from Part 0.

## Corrections

- Add `gas_release_schedule.py`, a standard-library preflight shared by the
  actual production workflow and the direct runner. It needs timezone data
  from the operating system or the existing `tzdata` dependency, but no model
  packages or API clients. It performs no data fetches and writes no forecasts.
- Pin the published EIA exceptions and explicitly bound reviewed reference
  weeks to January 6, 2025 through January 18, 2027. Unreviewed weeks produce
  an error requiring a calendar refresh, rather than assuming a release date.
- Normalize aware clocks to Eastern, including DST, and skip before the
  scheduled release. The October 13 run becomes a successful **preflight skip**
  before dependency installation or API requests; October 14 can proceed.
- Retain eligibility for delayed scheduled jobs after the release time. A
  narrow execution-time window could otherwise suppress a needed retry.
- Suppress scheduled retries only when both the current source observation
  and its next Monday target match a successful status. A wrong target cannot
  suppress publication. Malformed status JSON fails the preflight.
- Manual workflow dispatch respects the scheduled release time. A direct
  `--force` retains its deliberate calendar/day override, while the required
  Part 0 freshness and prospective-target checks continue to apply.

The scheduled timestamp is an earliest expected publication time, **not proof
of source availability**. If FRED still lacks October 12 after the October 14
release, Part 0 must continue to fail. Increasing the freshness age threshold
would not fix a target already in the past and is not an appropriate remedy.

The authoritative target remains regular conventional gasoline, `GASREGCOVW`.
The observed-week, exact-target, frozen-ledger, same-run lineage, causal model
validation, sleeve evidence, and persistence fallback contracts are preserved.
No forecasts, outcomes, model artifacts, publication data, or release-manifest
bytes are changed by this patch.

## Verification

- Baseline: **71 tests passed** in the locked Python 3.12 CPU environment.
- Four new direct-run regressions failed before correction; three due-release
  cases already passed. The old workflow gate independently reproduced the
  incorrect `run=true` on the failed holiday Tuesday.
- Corrected suite: **111 tests passed**, including all 12 EIA exceptions,
  timezone/DST boundaries, actual workflow-shell output, late jobs, manual
  dispatch, scheduled duplicate suppression, malformed status, and a bootstrap
  process with Python site packages disabled.
- Part 0 regressions reject October 5 as a source on both October 13 and 14;
  an observed October 12 source on October 14 targets October 19 and is eligible.
- The existing disposable integration test continues to run the numerical
  pipeline and sleeves with synthetic inputs, then verifies the release ledger.
- Dependency consistency, fatal-error lint, Python compilation, workflow YAML,
  published release-manifest verification, and unchanged publication bytes are
  required by the Linux/Windows PR checks.

Live October 14 publication must be checked after the branch is reviewed and
merged and the required observation actually arrives. Offline calendar tests
and synthetic pipeline tests do not assert a live source price or forecast.
