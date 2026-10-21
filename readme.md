Overview
Company_X operates a network of over 800 micro-fulfillment hubs delivering groceries and household essentials within 15–30 minutes of order placement. Because delivery windows are short and hub storage is limited, demand forecasting errors are far more costly here than in traditional retail — overstocking wastes space and spoils perishables, while understocking causes cancelled orders and lost trust.

Your task is to predict daily order volume for each hub over a future time window, using historical order data, hub attributes, and promotional/calendar context. Solari's operations team will use accurate forecasts to drive replenishment schedules, picker/packer staffing, and delivery fleet allocation.

This is a regression problem evaluated on a held-out future time period, not a random data split. Submissions are scored using RMSPE (see the Evaluation section for details).

Start

5 hours ago
Close

19 hours to go
Description
Demand forecasting for a large-scale fulfillment network requires understanding both historical demand patterns and the operational factors that influence them. In this competition, participants will use historical order data along with hub, promotional, calendar, and operational information to predict future daily order volumes for each fulfillment hub.

The dataset reflects real-world business scenarios, requiring careful exploratory data analysis, feature engineering, and robust validation to build models that generalize well to unseen future periods.

This is a time-series regression problem where success depends on both forecasting accuracy and a strong understanding of the underlying data.

Three files are provided:

orders_train.csv — historical daily order volume per hub, along with daily operating context (rider availability, operational status, promotion activity, day of week, holiday indicator).
orders_test.csv — the same structure, covering a future date range for which OrderVolume must be predicted. Same-day actuals that wouldn't be known in advance (e.g. rider headcount) are replaced with planned/scheduled values.
hub_metadata.csv — static and semi-static hub attributes (format, assortment tier, competitor distance and tenure, loyalty program status, launch date), joined via hub identifier.
All three files share a common Store (hub) identifier for joining. No external data may be used to supplement these files.

Evaluation
Submissions are evaluated on Root Mean Squared Logarithmic Error (RMSLE) between predicted and actual daily order volume.


where (p_i) is your predicted order volume, (a_i) is the actual order volume, and (n) is the number of rows in the test set.

## Run the forecasting pipeline

The project includes a reproducible Python pipeline in `forecast.py`. It checks
the train/test fields, adds calendar and hub features (including a time-aware
hub demand mean and loyalty-round activity), and trains a regularized gradient
boosting regressor on log-transformed demand using open training rows. It
compares recursive forecasts with horizon-aware direct forecasts trained from
historical 42-day forecast origins, plus a recursively carried 7-day
seasonal-naive forecast, over three rolling 42-day folds. It tunes the forecast
blend and a shrunken per-hub log-residual correction using those folds.
The resulting CSV follows `sample_submission.csv`'s `Id` order.

Place `orders_train.csv`, `orders_test.csv`, `hub_metadata.csv`, and
`sample_submission.csv` in `data/` (the supplied files are included here), then
run:

```bash
python -m pip install -r requirements.txt
python forecast.py
```

By default this runs three 42-day chronological validation folds and writes
`submission2.csv`. For a faster final fit without validation, run
`python forecast.py --skip-validation`. To use another data directory or output
path, pass `--data-dir PATH` or `--output PATH`.

RMSLE is used because hub sizes vary by an order of magnitude across the network. A plain RMSE would let a handful of large, mature hubs dominate the score; RMSLE penalizes proportional error instead, so accuracy at small and newly launched hubs counts just as much as accuracy at large ones. RMSLE also handles the many zero-order-volume rows (renovation- closed hubs) more gracefully than a percentage-based metric like MAPE, which is undefined at zero.
