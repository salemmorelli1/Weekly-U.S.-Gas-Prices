# Source identity and sleeve evidence audit — October 11, 2026

Repository: `salemmorelli1/Weekly-U.S.-Gas-Prices`.
Audited main: `46d037d3fb3d8087fe5b444108050034a610c9cb`.
Original tree: `55c9ecb106404c8b619df97e2afa6f0f694163df`.

## Findings

| Finding | Reproduction or source evidence | Correction |
| --- | --- | --- |
| The direct EIA outcome fallback requested diesel. The supplementary price mappings also mislabeled diesel as regular gasoline, used nonexistent diesel-like IDs for midgrade/premium, and mislabeled an all-grade gasoline ID as diesel. | `PET.EMD_EPD2D_PTE_NUS_DPG.W` is No. 2 diesel. The actual forecast target `GASREGCOVW` is regular **conventional** gasoline, whose EIA counterpart is `PET.EMM_EPMRU_PTE_NUS_DPG.W`. | Share exact target IDs between ingestion and backfill. Correct the other price mappings and reject provider responses naming a different series or unit. Clarify the dashboard's target formulation. |
| Missing gas surveys could be promoted to observations. | A FRED fixture with `3.8` on September 7 and a missing September 14 became `3.8` on September 14 after weekly resampling/forward filling. | Retain exact Monday survey labels and missing prices. Reject duplicate/non-Monday labels and nonpositive/nonfinite prices. Keep interior calendar gaps; trim only unobserved edges. Local backfill sources require the exact-observation contract and their producing content hash. |
| H=1 could become H=2 across a missing survey. | With observations September 7 and September 21, positional `shift(-1)` assigned September 21 as September 7's next-week target. Dropping missing weeks similarly compressed lag windows. | Join targets by anchor + seven calendar days, reindex feature inputs to the complete weekly calendar, preserve missing return values, and publish only the latest actually observed source row. |
| Days-of-supply units were wrong by a factor of seven. | Stocks of 210,000 thousand barrels and demand of 9,000 thousand barrels **per day** produced 163.333 days instead of 23.333. | Divide stocks directly by the already daily demand rate in both feature builders. Keep the existing conservative release lag. |
| Optional gate observations influenced early stopping. | XGBoost's scored outer observations were its `eval_set`; LSTM epochs and learning-rate scheduling also used its reported gate window. Their single fixed training origin was not the core's four expanding origins. | Use the same four outer blocks as the core. Select boosting rounds/epoch counts only on inner chronological history, refit on the available outer-origin history, and then score untouched outer blocks. Save row-level optional OOF evidence. Refit final preprocessing and models on all labeled history for live prediction. |
| Core approval alone could approve a different fused operator. | Governance treated the core gate as authorization for the final fusion without a paired test of fusion. Optional tape hashes, feature lineage, live clock and OOF comparability were not checked. | Verify same-run optional hashes, exact live source/target, feature matrix and common scored dates/actuals/baselines/folds. Recompute gate metrics from row-level evidence. Evaluate fusion with weights derived only from earlier scored blocks. A failed fusion retains the validated core; a failed core publishes persistence. |
| Declared dates could qualify a backdated forecast. Invalid outcomes could inflate health counts. | A row declaring September 8 with `run_utc` September 15 qualified for a September 14 target. Attribution admitted `actual=inf` and did not require `actual_date` to equal its target. | Require the V2 ledger schema, aware recorded issue time agreeing with the Eastern decision date, provenance and a finite frozen baseline. Require finite positive exact-target outcome pairs, matured targets and unique realized target dates. |
| Direction scoring could use a revised comparator. A backfill stage could re-sign changed forecast inputs. | Direction was measured against the newly fetched prior price rather than its stored issue-time persistence. Publication re-hashing did not compare immutable ledger fields before replacing data. | Use the recorded baseline for direction; persistence predicts no change. Validate all immutable fields before backfill staging. Changes to realized fields remain allowed. |

## Primary source verification

- [FRED GASREGCOVW](https://fred.stlouisfed.org/series/GASREGCOVW): U.S. regular conventional gasoline, dollars per gallon. Its October 5 observation is 4.157.
- [EIA regular conventional API series](https://www.eia.gov/opendata/browser/petroleum/pri/gnd?data=value%3B&facets=series%3B&frequency=weekly&series=EMM_EPMRU_PTE_NUS_DPG%3B&sortColumn=period%3B&sortDirection=desc): product EPMRU, conventional regular gasoline.
- [EIA old fallback](https://www.eia.gov/opendata/series.php?sdid=PET.EMD_EPD2D_PTE_NUS_DPG.W): U.S. No. 2 diesel retail prices.
- [EIA weekly products supplied](https://www.eia.gov/dnav/pet/pet_cons_wpsup_k_w.htm): thousand barrels per day.
- [EIA gasoline stocks](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?f=W&n=PET&s=WGTSTUS1): thousand barrels.

The current three eligible realized outcomes agree with the published FRED observations (September 21: 4.337; September 28: 4.309; October 5: 4.157). This audit found the incorrect fallback path; it does not establish that those three outcomes were filled from diesel. No existing ledger row or source artifact is rewritten by this PR.

## Current evidence and controls

The preserved main ledger has six rows: two ineligible legacy rows and four eligible prospective forecasts, of which three are realized. October 12 remains pending. The retained production decision publishes persistence at **4.157/gal** for that target; its learned candidate is **4.1469/gal**.

The unchanged 52-row core validation tape reports learned RMSE **0.149664/gal**, persistence RMSE **0.119026/gal**, one-sided HAC/HLN p-value **0.993208**, and **0/4** positive folds. Its gate is correctly closed. These are the retained production snapshot, not results of a corrected live provider replay.

No thresholds are weakened: core/fusion approval still requires lower RMSE than same-date persistence, one-sided HAC/HLN p < 0.10, and at least 75% positive outer folds. General live health still requires **52 eligible realized forecasts**. Optional sleeves additionally require comparable independent outer evidence and improvement over the core. A research override of LSTM computation does not bypass publication gates.

The latest forecast workflow [37683898633](https://github.com/salemmorelli1/Weekly-U.S.-Gas-Prices/actions/runs/37683898633) intentionally skipped its training job because the source week had already been published. The earlier complete run [37526749289](https://github.com/salemmorelli1/Weekly-U.S.-Gas-Prices/actions/runs/37526749289) produced the retained October 12 decision. A green idempotent skip is not a model replay.

## Verification

The suite covers source identities, provider product/unit mismatch, missing latest releases, duplicate/non-Monday dates, exact calendar targets and lags, physical units, aware issue-time provenance, finite exact-target pairs, duplicate cohort entries, frozen direction baselines and immutable backfill staging.

Actual CPU XGBoost/LSTM tests alter the outer outcomes and confirm that the first scored outer block's predictions remain unchanged. They also verify OOF alignment and that weights do not use their own current/future blocks. A disposable 200-week pipeline runs Parts 0/0c/6/1/2/2b/2a/3/9, reloads a saved LSTM, stages and verifies the release manifest, backfills an exact target and verifies every immutable forecast field. External requests in this test are fixtures; synthetic artifacts stay in temporary directories.

CI uses Python 3.12 on Linux and Windows with exact dependency pins and a separately installed CPU-only PyTorch wheel. It verifies dependency compatibility, compilation, fatal lint, all regressions, the retained manifest and an unchanged Git working tree. Normal production still leaves PyTorch optional. Future production releases also commit the new fusion OOF tape; the current 13-file release manifest remains unchanged and verifies successfully.

## Activation and scientific limits

Merge this feature branch, then let the next eligible Tuesday/Wednesday release run. Every normal weekly production run refits its models, so no cached-model migration is required. The published October 12 prediction is immutable and is not regenerated by this audit. No production workflow on main is dispatched by this audit.

The next corrected live run and future outcomes are still required to assess forecast accuracy. Tests establish implementation consistency, not a claimed performance improvement. Previously inspected historical validation is development evidence. FRED/EIA current-vintage histories and adjusted market histories can contain revisions; the weekly lag does not reconstruct a true archived release vintage. Monthly/quarterly non-vintage macro series and full-history regimes remain excluded from predictive features. The existing report is a historical snapshot; this code audit does not turn its old metrics into corrected prospective evidence.
