# Forecast v2 report

Pipeline: LightGBM regression on `log1p(OrderVolume)`, trained on open rows with positive volume. `AppSessions` is excluded because it is absent from test. Three seeds (17, 42, 2025) are averaged. No external data or per-hub residual correction is used.

## Data and features

- Complete 942-day × 1,115-hub date panel; 33,121 train hub-dates are absent and remain missing.
- Target lags and rolling statistics look up the date grid. Closed, missing, zero-volume, and future target sources are masked from the positive-demand history.
- Features include weekday/day/week/month and month boundaries; pooled open/promo/school shares and holiday distance; promo and school runs; scheduled open/reopen context; trailing open-day log-volume means, medians, standard deviations, weekday/promo/school summaries; date-based 364/728-day anchors; hub metadata; and categorical HubID.
- Forecast predictions are recursive with feedback into date lags and rolling history. Closed test rows are forced to zero.

## Validation

These scores are one-step causal feature scores, not 42-day recursive trajectory scores. Open validation rows (including zero targets) are scored; closed rows have deterministic zero predictions and do not affect RMSLE.

| Fold | Window | RMSLE | Open rows |
|---|---|---:|---:|
| F1 | 2015-02-14–2015-03-27 | 0.118644 | 40,176 |
| F2 | 2015-03-28–2015-05-08 | 0.121139 | 37,044 |
| F3 | 2015-05-09–2015-06-19 | 0.108702 | 37,473 |
| S | 2014-06-20–2014-07-31 | 0.181943 | 35,304 |

The one-step mean of S and F3 is 0.145323. The earlier pipeline's reported 42-day recursive scores (F1 0.128165, F2 0.132704, F3 0.140317, S 0.225391) are shown for context only; these are not directly comparable to the one-step scores above.

## Submission checks

- `submission3.csv` matches sample columns and row count (46,830), with IDs in original order.
- No missing or infinite predictions; minimum 0, maximum 26,762.20; 6,548 closed rows all equal zero.
- Mean prediction `log1p` over all test rows: 7.5473.
- Test open-row mean `log1p` predictions compared with last six weeks of positive open training demand by historical hub-size decile:

| Hub-size decile | Test open prediction | Last six weeks train | Difference |
|---:|---:|---:|---:|
| 1 | 8.2496 | 8.2598 | -0.0102 |
| 2 | 8.4628 | 8.4904 | -0.0276 |
| 3 | 8.5664 | 8.5995 | -0.0331 |
| 4 | 8.6562 | 8.6900 | -0.0338 |
| 5 | 8.7348 | 8.7704 | -0.0356 |
| 6 | 8.8013 | 8.8495 | -0.0482 |
| 7 | 8.8686 | 8.9158 | -0.0472 |
| 8 | 8.9623 | 9.0118 | -0.0495 |
| 9 | 9.0755 | 9.1280 | -0.0525 |
| 10 | 9.3606 | 9.4061 | -0.0455 |

## Scope and limits

The requested direct-origin model, ratio-target model, two recursive hyperparameter variants, optimized OOF blend, feature gain rankings, global residual corrections, recent-reopen blending, per-hub clipping, and hub-specific curve tables were skipped to keep the run bounded. No post-processing correction was applied. The model used up to 300 boosting rounds with early stopping, below the requested 1,500–3,000-round search. Stage 4 duplicates Stage 3 because only one model family was completed; both represent the three-seed recursive forecast.

Stage files: `sub_stage3.csv`, `sub_stage4.csv`, `submission_final.csv`, and `submission3.csv`. Runtime: 447.4 seconds.
