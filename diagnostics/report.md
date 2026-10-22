# Demand forecasting diagnostics

Read-only analysis of the supplied CSVs and current forecasting implementation. No model or submission files were changed.

The competition overview mentions RMSPE, while its Evaluation section specifies RMSLE. The CSVs cannot determine the official scoring rule; this report uses RMSLE because the Evaluation section explicitly defines it and the project README does too.

## Section 1: Data facts
1. Train dates: **2013-01-01 to 2015-06-19** (900 days); test dates: **2015-06-20 to 2015-07-31** (42 days). Train/test hubs: 1115/1115. Test-only hubs: 0. Hubs with fewer than 90 observed training dates: 0; test hubs with fewer than 90 days: 0.

2. Column availability and missingness:
| File | Column | Missing | Status |
| --- | --- | --- | --- |
| train | HubID | 0.00% | present |
| train | Weekday | 0.00% | present |
| train | Date | 0.00% | present |
| train | OrderVolume | 0.00% | present |
| train | AppSessions | 0.00% | present |
| train | IsOpen | 0.00% | present |
| train | PromoActive | 0.00% | present |
| train | RegionalHoliday | 0.00% | present |
| train | SchoolClosureFlag | 0.00% | present |
| test | Id | 0.00% | present |
| test | HubID | 0.00% | present |
| test | Weekday | 0.00% | present |
| test | Date | 0.00% | present |
| test | IsOpen | 0.00% | present |
| test | PromoActive | 0.00% | present |
| test | RegionalHoliday | 0.00% | present |
| test | SchoolClosureFlag | 0.00% | present |
| metadata | HubID | 0.00% | present |
| metadata | HubFormat | 0.00% | present |
| metadata | AssortmentTier | 0.00% | present |
| metadata | CompetitorDistance | 0.27% | present |
| metadata | CompetitorOpenSinceMonth | 31.75% | present |
| metadata | CompetitorOpenSinceYear | 31.75% | present |
| metadata | LoyaltyProgram | 0.00% | present |
| metadata | LoyaltyProgramSinceWeek | 48.79% | present |
| metadata | LoyaltyProgramSinceYear | 48.79% | present |
| metadata | LoyaltyProgramInterval | 48.79% | present |
| test | AppSessions | 100.00% | absent |
| train | Id | 100.00% | absent |
| test | OrderVolume | 100.00% | absent |
`IsOpen`, `PromoActive`, `RegionalHoliday`, and `SchoolClosureFlag` are present in test. `AppSessions` is absent from test (not merely null), so it is excluded.

3. Matching same-season windows:
Test season: Jun 20–Jul 31. Exact historical windows available: 2013-06-20 to 2013-07-31 (2013), 2014-06-20 to 2014-07-31 (2014). Count: 2 years.

4. Zero volume and closed status:
Train zero-volume rows: 166,323 (17.14%); zero rows with IsOpen=0: 166,269 (99.97% of zero rows); IsOpen=1 and zero volume: 54. Test IsOpen=0 rows: 6,548/46,830 (13.98%).

5. Closure patterns (contiguous IsOpen=0 periods per hub):
Hubs with any IsOpen=0 day: 1105; closure periods: 161633; length min/median/max: 1/1.0/185; closure periods overlapping the test dates (known scheduled IsOpen): 6493.

6. Growth trend: mean log1p(OrderVolume) by month; HubFormat split:
| Month | MeanLogVolume |
| --- | --- |
| 2013-01 | 7.221639970734742 |
| 2013-02 | 7.412718793962463 |
| 2013-03 | 7.091055144405219 |
| 2013-04 | 7.253569803845748 |
| 2013-05 | 6.654988532778182 |
| 2013-06 | 7.260462242166354 |
| 2013-07 | 7.636563423955487 |
| 2013-08 | 7.562239380650542 |
| 2013-09 | 7.254964373104746 |
| 2013-10 | 7.29065884350474 |
| 2013-11 | 7.440535568865651 |
| 2013-12 | 6.966281512315339 |
| 2014-01 | 7.226062425641668 |
| 2014-02 | 7.484766949766877 |
| 2014-03 | 7.3108888229083595 |
| 2014-04 | 7.078512835715585 |
| 2014-05 | 7.104033460612043 |
| 2014-06 | 6.893963051527044 |
| 2014-07 | 7.619235591118852 |
| 2014-08 | 7.342543689863822 |
| 2014-09 | 7.575639745035866 |
| 2014-10 | 7.3265935233642745 |
| 2014-11 | 7.2520604744352175 |
| 2014-12 | 7.259985014961293 |
| 2015-01 | 7.290219731465441 |
| 2015-02 | 7.480440531424149 |
| 2015-03 | 7.385212868419578 |
| 2015-04 | 7.095628382270314 |
| 2015-05 | 6.604515995480364 |
| 2015-06 | 7.656017076000017 |

By month and HubFormat:
| Month | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 2013-01 | 7.2291617642125505 | 8.069782375818237 | 7.1364543141865955 | 7.2034249685725715 |
| 2013-02 | 7.415272636298202 | 8.143708469180906 | 7.351579676207123 | 7.398593368281318 |
| 2013-03 | 7.091289139908338 | 7.950360550093633 | 7.037619287071913 | 7.07139839065737 |
| 2013-04 | 7.251144459543567 | 8.25288880418062 | 7.21230032950957 | 7.226499448862918 |
| 2013-05 | 6.636911849945169 | 8.574609454407954 | 6.677201188795858 | 6.583037769292983 |
| 2013-06 | 7.247983804047182 | 8.541857765759977 | 7.238259052913524 | 7.2288943337067835 |
| 2013-07 | 7.620608759474584 | 8.950225108380542 | 7.603866659718966 | 7.6138955517982545 |
| 2013-08 | 7.535930448921006 | 8.976405625203027 | 7.554794781901699 | 7.541834068462662 |
| 2013-09 | 7.23316196803918 | 9.008705003814999 | 7.187197129278163 | 7.23582934210972 |
| 2013-10 | 7.270876645936384 | 9.050047946983446 | 7.227632875778147 | 7.265736749828156 |
| 2013-11 | 7.4088333932729125 | 9.096302672806472 | 7.462680816937486 | 7.4050734200875015 |
| 2013-12 | 6.924333531694308 | 9.174037415268083 | 6.955798334431235 | 6.935454915506465 |
| 2014-01 | 7.200792391082785 | 9.023429013175328 | 7.172203688496276 | 7.204879729991938 |
| 2014-02 | 7.449963938761217 | 9.08599361627585 | 7.453077248462402 | 7.480228544846813 |
| 2014-03 | 7.275010508332994 | 9.075143275344985 | 7.259344149882537 | 7.308690636962294 |
| 2014-04 | 7.049561718060774 | 9.130407513302409 | 7.034962056267893 | 7.046880303150866 |
| 2014-05 | 7.06470140732479 | 9.146683601111768 | 7.0547158608565335 | 7.093262910193208 |
| 2014-06 | 6.862862087126348 | 9.164668619378883 | 6.88091552560949 | 6.842387820928009 |
| 2014-07 | 7.588550144161596 | 9.142680481394684 | 7.543318561901906 | 7.625575760386704 |
| 2014-08 | 7.3217048803182445 | 9.095318139619375 | 7.277191267080996 | 7.310735962528316 |
| 2014-09 | 7.547457060578016 | 9.11672624455212 | 7.505958799496251 | 7.572937472598467 |
| 2014-10 | 7.298756524578926 | 9.147965938444154 | 7.261320300299419 | 7.3042123196947 |
| 2014-11 | 7.2037896916687005 | 9.211991208953664 | 7.261736184701127 | 7.223404401570531 |
| 2014-12 | 7.2095720170439845 | 9.263622326703803 | 7.262597631999503 | 7.236410652602361 |
| 2015-01 | 7.2470745642373045 | 9.093520609236597 | 7.289984985471239 | 7.276863461799837 |
| 2015-02 | 7.4374677365992055 | 9.138406663654832 | 7.4521771080136 | 7.485806005278905 |
| 2015-03 | 7.349749026334921 | 9.154703172144865 | 7.3297074149984205 | 7.383726388183846 |
| 2015-04 | 7.0611974593274685 | 9.179844034125905 | 7.0493324226318 | 7.073063875248899 |
| 2015-05 | 6.561537930588619 | 9.206522844392861 | 6.526896389109101 | 6.58476421495239 |
| 2015-06 | 7.63524504576963 | 9.241025367161996 | 7.690166840903612 | 7.599998271526476 |
Last 8 weeks (2015-04-25–2015-06-19) vs same 8 weeks one year earlier (2014-04-26–2014-06-20): median per-hub log-mean change=+0.0301; median percent change in geometric mean=3.06%; hubs compared=1115.

7. Hub age from first nonzero order to train end (calendar days):
| Statistic | Days |
| --- | --- |
| count | 1115.0 |
| mean | 897.5802690582959 |
| std | 8.358096619547412 |
| min | 714.0 |
| 10% | 898.0 |
| 25% | 898.0 |
| 50% | 898.0 |
| 75% | 898.0 |
| 90% | 898.0 |
| max | 899.0 |
Hubs with <365 days between first nonzero order and train end: 0.

## Section 2: Validation and leakage

8. Three folds (model scores before adjustment; final calibrated after online prior-fold per-hub bias adjustment):
| Fold | Start | End | TrainEnd | Model | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2015-02-14 | 2015-03-27 | 2015-02-13 | recursive | 46830 | 0.128165 | -0.022923 |
| 1 | 2015-02-14 | 2015-03-27 | 2015-02-13 | direct | 46830 | 0.138975 | -0.03771 |
| 1 | 2015-02-14 | 2015-03-27 | 2015-02-13 | seasonal-naive | 46830 | 0.676769 | -0.15445 |
| 1 | 2015-02-14 | 2015-03-27 | 2015-02-13 | final-calibrated | 46830 | 0.128165 | -0.022923 |
| 2 | 2015-03-28 | 2015-05-08 | 2015-03-27 | recursive | 46830 | 0.132266 | 0.007677 |
| 2 | 2015-03-28 | 2015-05-08 | 2015-03-27 | direct | 46830 | 0.164362 | 0.005674 |
| 2 | 2015-03-28 | 2015-05-08 | 2015-03-27 | seasonal-naive | 46830 | 3.860912 | -1.830041 |
| 2 | 2015-03-28 | 2015-05-08 | 2015-03-27 | final-calibrated | 46830 | 0.132704 | 0.013665 |
| 3 | 2015-05-09 | 2015-06-19 | 2015-05-08 | recursive | 46830 | 0.141496 | -0.04504 |
| 3 | 2015-05-09 | 2015-06-19 | 2015-05-08 | direct | 46830 | 0.143178 | -0.022932 |
| 3 | 2015-05-09 | 2015-06-19 | 2015-05-08 | seasonal-naive | 46830 | 3.6797 | -1.439973 |
| 3 | 2015-05-09 | 2015-06-19 | 2015-05-08 | final-calibrated | 46830 | 0.140317 | -0.042822 |

9. Same-season validation: no rolling fold covers June 20–July 31. A separate season-matched fold is reported below. Public leaderboard score: unavailable because no leaderboard score was supplied.

10. Leakage audit for the current recursive pipeline:
| Feature(s) | Availability | Assessment |
| --- | --- | --- |
| HubID | Static identifier, available in advance | Numeric feature; order may impose arbitrary split structure. |
| Weekday, Year, Month, Day, DayOfYear, WeekOfYear, IsWeekend, DaysSinceStart, MonthSin/Cos, DoySin/Cos | Calendar-derived from forecast date; known at origin | Safe. |
| IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag | Provided in test/scheduled context | Safe if competition provides planned values as stated. |
| HubFormat, AssortmentTier, CompetitorDistance, CompetitorOpenSinceMonth/Year, LoyaltyProgram, LoyaltyProgramSinceWeek/Year, LoyaltyProgramInterval, LoyaltyRoundActive | Hub metadata plus forecast date | Static fields/derived calendar flag; safe. |
| Lag1/7/14/21/28/56/364, Mean/Std7/28/56 | Prior target observations or prior recursive predictions | Safe in recursive forecast. In training they are row shifts/rolling rows, while inference looks up calendar dates; generally same for complete daily hub series, possible semantic mismatch where dates are absent. |
| HubLogMean | Mean log target strictly before each training row; for forecast rows history before origin | Safe: training uses shift/expanding; validation/test forecast computes from pre-origin history. |
| OrderVolume | Target | Excluded from X; used only as training target and historical lag source. |
| AppSessions | Train-only column | Excluded because absent in test. |

Same-season fold:
Additional season-matched fold: train through 2014-06-19, predict 2014-06-20–2014-07-31, rows=41,250, recursive RMSLE=0.225391. Bias adjustment omitted: no non-leaking previous-fold per-hub residual calibration was available for this 2014 origin without constructing earlier folds. The public score remains unavailable.

## Section 3: Error analysis (OOF final calibrated pipeline; open rows only)

11. By horizon bucket:
| HorizonBucket | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| 1-7 | 17924.0 | 0.193074 | -0.049339 |
| 8-14 | 19053.0 | 0.134824 | -0.044727 |
| 15-21 | 19057.0 | 0.153488 | 0.000307 |
| 22-28 | 19455.0 | 0.131864 | -0.00672 |
| 29-35 | 19056.0 | 0.128814 | -0.009089 |
| 36-42 | 20148.0 | 0.140335 | -0.02488 |

12. By hub size decile (origin history mean log volume):
| Decile | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| 1 | 11471.0 | 0.153757 | -0.024717 |
| 2 | 11350.0 | 0.140465 | -0.015619 |
| 3 | 11450.0 | 0.136825 | -0.020832 |
| 4 | 11355.0 | 0.132942 | -0.021234 |
| 5 | 11463.0 | 0.131509 | -0.023907 |
| 6 | 11352.0 | 0.130608 | -0.020293 |
| 7 | 11447.0 | 0.148523 | -0.02214 |
| 8 | 11329.0 | 0.205212 | -0.023895 |
| 9 | 11479.0 | 0.129708 | -0.022075 |
| 10 | 11997.0 | 0.155462 | -0.026136 |

13. By hub age at validation origin (days since first observed train row):
| AgeBucket | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| <90d | 0.0 | NA | NA |
| 90-365d | 0.0 | NA | NA |
| 1-2y | 0.0 | NA | NA |
| >2y | 114693.0 | 0.148074 | -0.022111 |

14. By day of week, promo, holiday, and school closure:

**Weekday**
| Weekday | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| 1 | 17861.0 | 0.16045 | -0.008454 |
| 2 | 20052.0 | 0.146716 | -0.018966 |
| 3 | 20049.0 | 0.150531 | -0.038492 |
| 4 | 18288.0 | 0.15017 | -0.030293 |
| 5 | 17864.0 | 0.142885 | -0.036188 |
| 6 | 20058.0 | 0.134531 | -0.00123 |
| 7 | 521.0 | 0.235244 | -0.014954 |

**PromoActive**
| PromoActive | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| 0 | 61898.0 | 0.149045 | -0.035934 |
| 1 | 52795.0 | 0.146927 | -0.005904 |

**RegionalHoliday**
| RegionalHoliday | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| 0 | 114472.0 | 0.146946 | -0.022061 |
| 1 | 172.0 | 0.420711 | -0.0582 |
| 2 | 49.0 | 0.504611 | -0.012292 |

**SchoolClosureFlag**
| SchoolClosureFlag | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| 0 | 101148.0 | 0.143581 | -0.017615 |
| 1 | 13545.0 | 0.178077 | -0.055685 |

**ReopenBucket**
| ReopenBucket | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| 0-7 | 111374.0 | 0.145566 | -0.021621 |
| 15-30 | 571.0 | 0.239303 | -0.1217 |
| 31-60 | 280.0 | 0.187743 | -0.124033 |
| 8-14 | 326.0 | 0.30624 | -0.037915 |
| >60 | 882.0 | 0.24822 | -0.024116 |
| never closed | 1260.0 | 0.147154 | 0.007894 |

**NearestHolidayDayOffset**
| NearestHolidayDayOffset | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| -7.0 | 2230.0 | 0.133775 | 0.010712 |
| -6.0 | 2261.0 | 0.111507 | 0.012355 |
| -5.0 | 5573.0 | 0.126048 | -0.005579 |
| -4.0 | 5573.0 | 0.149369 | -0.031491 |
| -3.0 | 3406.0 | 0.117185 | -0.012644 |
| -2.0 | 4493.0 | 0.135854 | -0.035176 |
| -1.0 | 6690.0 | 0.204843 | -0.112518 |
| 0.0 | 566.0 | 0.306442 | -0.020208 |
| 1.0 | 4522.0 | 0.231049 | -0.166567 |
| 2.0 | 5574.0 | 0.142174 | -0.06282 |
| 3.0 | 5573.0 | 0.129735 | -0.001723 |
| 4.0 | 3409.0 | 0.126396 | 0.02621 |
| 5.0 | 2293.0 | 0.124621 | -0.046985 |
| 6.0 | 3344.0 | 0.105167 | 0.019145 |
| 7.0 | 2230.0 | 0.137747 | -0.048849 |
| NA | 56956.0 | 0.142282 | -0.004207 |

**HubFormat**
| HubFormat | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| 1 | 61790.0 | 0.14059 | -0.021711 |
| 2 | 2142.0 | 0.195213 | -0.005286 |
| 3 | 15134.0 | 0.19473 | -0.024881 |
| 4 | 35627.0 | 0.133811 | -0.022638 |

**AssortmentTier**
| AssortmentTier | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| 1 | 60784.0 | 0.155704 | -0.021707 |
| 2 | 1134.0 | 0.13906 | 0.015713 |
| 3 | 52775.0 | 0.138967 | -0.023388 |

**LoyaltyProgram**
| LoyaltyProgram | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| 0 | 56238.0 | 0.140562 | -0.021868 |
| 1 | 58455.0 | 0.154957 | -0.022344 |

**CompetitorDistanceBucket**
| CompetitorDistanceBucket | Rows | RMSLE | MeanSignedLogError |
| --- | --- | --- | --- |
| <1km | 33965.0 | 0.142748 | -0.017537 |
| 1-5km | 44481.0 | 0.157271 | -0.02107 |
| 5-10km | 16555.0 | 0.133659 | -0.025455 |
| >10km | 19384.0 | 0.147453 | -0.029572 |
| NA | 308.0 | 0.130218 | -0.027279 |

17. Top 20 hubs by total squared log error:
| HubID | ShareOfTotalError | MeanSignedLogError | HubSizeDecile | HubFormat | Rows |
| --- | --- | --- | --- | --- | --- |
| 971 | 9.27% | -0.68799 | 7 | 3 | 94 |
| 644 | 2.35% | -0.19047 | 3 | 3 | 80 |
| 674 | 2.15% | -0.03376 | 3 | 1 | 69 |
| 1081 | 1.49% | -0.22237 | 1 | 2 | 126 |
| 530 | 0.49% | -0.22273 | 10 | 1 | 122 |
| 433 | 0.40% | 0.00466 | 10 | 1 | 120 |
| 732 | 0.40% | -0.15036 | 10 | 1 | 120 |
| 524 | 0.37% | 0.03263 | 10 | 1 | 120 |
| 183 | 0.37% | -0.09908 | 1 | 1 | 96 |
| 310 | 0.36% | -0.02931 | 10 | 1 | 120 |
| 198 | 0.33% | -0.05292 | 1 | 1 | 103 |
| 931 | 0.33% | -0.13286 | 9 | 1 | 118 |
| 299 | 0.32% | -0.14055 | 10 | 4 | 120 |
| 415 | 0.30% | 0.02094 | 5 | 4 | 103 |
| 867 | 0.29% | -0.09728 | 10 | 4 | 120 |
| 453 | 0.29% | -0.11224 | 10 | 1 | 120 |
| 39 | 0.27% | -0.02616 | 2 | 1 | 102 |
| 699 | 0.26% | -0.0701 | 6 | 1 | 102 |
| 209 | 0.25% | -0.03254 | 10 | 1 | 120 |
| 254 | 0.23% | -0.08048 | 1 | 4 | 102 |
Worst-hub cumulative shares: top 10: 17.66%, top 50: 25.78%, top 100: 31.96%.

18. Top 20 dates by squared error:
| Date | ShareOfTotalError | Weekday | RegionalHoliday | PromoShare | Rows |
| --- | --- | --- | --- | --- | --- |
| 2015-02-16 | 5.17% | Monday | 0 | 100.00% | 1083 |
| 2015-05-15 | 3.94% | Friday | 0 | 0.00% | 1115 |
| 2015-05-13 | 3.85% | Wednesday | 0 | 0.00% | 1114 |
| 2015-04-02 | 3.16% | Thursday | 0 | 100.00% | 1115 |
| 2015-03-26 | 2.76% | Thursday | 0 | 0.00% | 1115 |
| 2015-04-15 | 2.70% | Wednesday | 0 | 100.00% | 1113 |
| 2015-05-26 | 2.51% | Tuesday | 0 | 0.00% | 1115 |
| 2015-02-17 | 2.10% | Tuesday | 0 | 100.00% | 1111 |
| 2015-04-01 | 2.09% | Wednesday | 0 | 100.00% | 1115 |
| 2015-04-04 | 1.92% | Saturday | 0 | 0.00% | 1115 |
| 2015-04-16 | 1.83% | Thursday | 0 | 100.00% | 1115 |
| 2015-05-29 | 1.79% | Friday | 0 | 0.00% | 1115 |
| 2015-04-18 | 1.78% | Saturday | 0 | 0.00% | 1115 |
| 2015-06-03 | 1.46% | Wednesday | 0 | 100.00% | 1114 |
| 2015-04-14 | 1.36% | Tuesday | 0 | 100.00% | 1115 |
| 2015-04-07 | 1.34% | Tuesday | 0 | 0.00% | 1115 |
| 2015-03-27 | 1.30% | Friday | 0 | 0.00% | 1115 |
| 2015-06-15 | 1.21% | Monday | 0 | 100.00% | 1114 |
| 2015-03-23 | 1.16% | Monday | 0 | 0.00% | 1114 |
| 2015-06-08 | 1.10% | Monday | 0 | 0.00% | 1115 |

19. Residual autocorrelation and bias stability:
Average per-hub residual autocorrelation: lag-1=0.3820 (hubs=1115), lag-7=0.1221. Consecutive-fold per-hub mean residual correlation: fold1/2=-0.1344, fold2/3=0.0756.

## Section 4: Model introspection

20. Permutation importance (last-fold one-step validation proxy, 12k rows, log-target MSE; recursive horizon importance may differ):
Top features:
| Feature | ImportanceMean | ImportanceStd |
| --- | --- | --- |
| PromoActive | 0.03024 | 0.000392 |
| Mean28 | 0.018562 | 0.000398 |
| Lag14 | 0.009118 | 0.000283 |
| Lag364 | 0.008385 | 0.000188 |
| Lag1 | 0.008303 | 6.9e-05 |
| Lag28 | 0.0062 | 0.000325 |
| Lag7 | 0.003982 | 0.000207 |
| Std7 | 0.003647 | 0.000175 |
| Std28 | 0.002637 | 0.000131 |
| Weekday | 0.002534 | 0.000113 |
| Lag21 | 0.002265 | 0.000123 |
| Std56 | 0.001201 | 0.000166 |
| Mean7 | 0.001122 | 0.000145 |
| Day | 0.00093 | 5.5e-05 |
| RegionalHoliday | 0.000775 | 2.8e-05 |
| Lag56 | 0.000643 | 8.4e-05 |
| DoySin | 0.000284 | 5e-06 |
| AssortmentTier | 0.00021 | 2.6e-05 |
| HubFormat | 0.000172 | 2.7e-05 |
| Mean56 | 0.000169 | 1.4e-05 |
| LoyaltyRoundActive | 0.000143 | 4e-06 |
| HubLogMean | 0.000136 | 7e-06 |
| CompetitorDistance | 6.2e-05 | 8e-06 |
| LoyaltyProgramSinceYear | 3.4e-05 | 5e-06 |
| LoyaltyProgramSinceWeek | 1.5e-05 | 9e-06 |

Bottom features:
| Feature | ImportanceMean | ImportanceStd |
| --- | --- | --- |
| DoyCos | -0.000638 | 1.6e-05 |
| DayOfYear | -0.000608 | 1.8e-05 |
| MonthCos | -0.000248 | 1.9e-05 |
| SchoolClosureFlag | -1.6e-05 | 0.0 |
| IsWeekend | -1.6e-05 | 3e-06 |
| LoyaltyProgramInterval | -3e-06 | 1e-06 |
| IsOpen | 0.0 | 0.0 |
| LoyaltyProgram | 0.0 | 0.0 |
| DaysSinceStart | 0.0 | 0.0 |
| Year | 0.0 | 0.0 |
| MonthSin | 0.0 | 0.0 |
| Month | 5e-06 | 1e-06 |
| CompetitorOpenSinceMonth | 9e-06 | 1e-06 |
| HubID | 1e-05 | 3e-06 |
| WeekOfYear | 1.2e-05 | 4e-06 |
| CompetitorOpenSinceYear | 1.4e-05 | 3e-06 |
| LoyaltyProgramSinceWeek | 1.5e-05 | 9e-06 |
| LoyaltyProgramSinceYear | 3.4e-05 | 5e-06 |
| CompetitorDistance | 6.2e-05 | 8e-06 |
| HubLogMean | 0.000136 | 7e-06 |
| LoyaltyRoundActive | 0.000143 | 4e-06 |
| Mean56 | 0.000169 | 1.4e-05 |
| HubFormat | 0.000172 | 2.7e-05 |
| AssortmentTier | 0.00021 | 2.6e-05 |
| DoySin | 0.000284 | 5e-06 |

21. Alternative model checks:
Variant (a), no hub-bias adjustment: already reported as recursive RMSLE in Section 8; compare to final-calibrated per-fold. Variant (b), native HubID categorical: HGBR cannot encode 1,115 categories with its default max_bins=255, and ordinary numeric HubID is already present; native categorical variant therefore not run. Variant (c) and (d) are evaluated below.
| Fold | Variant | RMSLE |
| --- | --- | --- |
| 1 | c: log target minus trailing-56d mean | 0.130522 |
| 1 | d: lags>=42, one-shot | 0.145405 |
| 2 | c: log target minus trailing-56d mean | 0.139355 |
| 2 | d: lags>=42, one-shot | 0.174323 |
| 3 | c: log target minus trailing-56d mean | 0.135394 |
| 3 | d: lags>=42, one-shot | 0.141631 |
| Pooled | c: log target minus trailing-56d mean | 0.135139 |
| Pooled | d: lags>=42, one-shot | 0.154478 |
Variant (c) uses each training row’s prior 56-day mean as its log-level anchor and updates that anchor recursively at inference; it is a one-step residual-target comparison, not a fixed-origin non-recursive model. Variant (d) uses only lags ≥42 days plus context, with feedback disabled.

22. Year-over-year anchors (open rows):
| Fold | Baseline | RMSLE |
| --- | --- | --- |
| 1 | 364d same weekday | 0.520995 |
| 1 | 364d anchor x recent growth ratio | 0.537232 |
| 2 | 364d same weekday | 2.667786 |
| 2 | 364d anchor x recent growth ratio | 2.670547 |
| 3 | 364d same weekday | 2.440984 |
| 3 | 364d anchor x recent growth ratio | 2.442949 |

23. Oracle noise-floor estimates (open rows, use actual fold targets; non-causal by design):
| Fold | Per-hub constant oracle RMSLE | Per-hub weekday oracle RMSLE |
| --- | --- | --- |
| 1 | 0.238593 | 0.180963 |
| 2 | 0.279998 | 0.21853 |
| 3 | 0.242529 | 0.174564 |

## Section 5: Summary and reproducibility

24. Five largest error concentrations (ranked summaries):
| Finding | Evidence |
| --- | --- |
| Largest hub concentration | Top 10 hubs contributed 17.66% of open-row squared log error. |
| Worst dates | Top 10 dates contributed 30.20% of total open-row squared log error. |
| Horizon concentration | Largest bucket: 1-7 days, 26.57% of error. |
| Hub size | Fold-specific size decile 8 had the highest RMSLE: 0.205212 versus overall open-row RMSLE 0.148074. |
| School closures | SchoolClosureFlag=1 RMSLE was 0.178077 versus 0.143581 when flag=0; flag=1 contributed 17.08% of squared log error. |

25. Feature groups and hyperparameters:
- **calendar:** Weekday, Year, Month, Day, DayOfYear, WeekOfYear, IsWeekend, DaysSinceStart, MonthSin, MonthCos, DoySin, DoyCos
- **lag/rolling:** Lag1, Lag7, Lag14, Lag21, Lag28, Lag56, Lag364, Mean7, Std7, Mean28, Std28, Mean56, Std56
- **hub static:** HubID, HubFormat, AssortmentTier
- **promo/operations:** IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag
- **loyalty:** LoyaltyProgram, LoyaltyProgramSinceWeek, LoyaltyProgramSinceYear, LoyaltyProgramInterval, LoyaltyRoundActive
- **competitor:** CompetitorDistance, CompetitorOpenSinceMonth, CompetitorOpenSinceYear
- **other:** HubLogMean
HGBR: loss=squared_error, learning_rate=0.07, max_iter=140, max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=2.0, random_state=42, early_stopping=False.
26. Train rows: 970,379 raw; 564,519 usable after lag/rolling requirements; rows used by full model: 467,281 open rows. One full-data model fit plus 42-day recursive prediction took 32.5 seconds (excluding feature engineering and diagnostics). Random seeds: model/permutation sampling use 42. No external data used.

Limits: no leaderboard score was supplied. Closure dates outside supplied train/test context cannot be reconstructed. Holiday-distance analysis only sees RegionalHoliday dates present in provided data. Same-season fold is uncalibrated for hub residual because using future 2015 validation residuals would leak. Sections 21(c) is explicitly exploratory. Permutation importance is one-step validation and not a substitute for recursive trajectory importance.
