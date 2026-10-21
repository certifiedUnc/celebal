#!/usr/bin/env python3
"""Train a daily hub demand model and write a Kaggle submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error


TARGET = "OrderVolume"
LAGS = (1, 7, 14, 21, 28, 56, 364)
ROLLS = (7, 28, 56)


def read_data(data_dir: Path):
    train = pd.read_csv(data_dir / "orders_train.csv", parse_dates=["Date"])
    test = pd.read_csv(data_dir / "orders_test.csv", parse_dates=["Date"])
    metadata = pd.read_csv(data_dir / "hub_metadata.csv")
    return train, test, metadata


def enrich_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    date = out["Date"]
    out["Year"] = date.dt.year
    out["Month"] = date.dt.month
    out["Day"] = date.dt.day
    out["DayOfYear"] = date.dt.dayofyear
    out["WeekOfYear"] = date.dt.isocalendar().week.astype("int16")
    out["IsWeekend"] = (date.dt.dayofweek >= 5).astype("int8")
    out["DaysSinceStart"] = (date - pd.Timestamp("2013-01-01")).dt.days
    out["MonthSin"] = np.sin(2 * np.pi * date.dt.month / 12)
    out["MonthCos"] = np.cos(2 * np.pi * date.dt.month / 12)
    out["DoySin"] = np.sin(2 * np.pi * date.dt.dayofyear / 365.25)
    out["DoyCos"] = np.cos(2 * np.pi * date.dt.dayofyear / 365.25)
    return out


def add_static_features(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(metadata, on="HubID", how="left", validate="many_to_one")
    out["LoyaltyProgramInterval"] = out["LoyaltyProgramInterval"].fillna("none")
    # The interval is a small categorical field; numeric codes are sufficient for
    # the tree model and keep the complete pipeline dependency-light.
    out["LoyaltyProgramInterval"] = pd.Categorical(
        out["LoyaltyProgramInterval"]
    ).codes.astype("int16")
    for col in out.select_dtypes(include=["bool"]).columns:
        out[col] = out[col].astype("int8")
    month_number = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    round_months = {}
    for hub, interval in metadata.set_index("HubID")["LoyaltyProgramInterval"].items():
        names = str(interval).split(",") if pd.notna(interval) else []
        round_months[hub] = {
            month_number[name.strip().title()[:3]]
            for name in names if name.strip().title()[:3] in month_number
        }
    out["LoyaltyRoundActive"] = [
        int(month in round_months.get(hub, set()) and loyalty == 1)
        for hub, month, loyalty in zip(out["HubID"], out["Month"], out["LoyaltyProgram"])
    ]
    return out


def make_training_features(train: pd.DataFrame, metadata: pd.DataFrame):
    frame = add_static_features(enrich_calendar(train), metadata)
    frame = frame.sort_values(["HubID", "Date"]).reset_index(drop=True)
    grouped = frame.groupby("HubID", sort=False)[TARGET]
    for lag in LAGS:
        frame[f"Lag{lag}"] = grouped.shift(lag)
    for window in ROLLS:
        frame[f"Mean{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean()
        )
        frame[f"Std{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=w).std()
        )
    log_target = np.log1p(frame[TARGET].clip(lower=0))
    frame["HubLogMean"] = log_target.groupby(frame["HubID"], sort=False).transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    feature_cols = get_feature_columns(frame)
    usable = frame.dropna(subset=[f"Lag{x}" for x in LAGS] + [f"Mean{x}" for x in ROLLS])
    return usable, feature_cols


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"Date", TARGET, "Id", "AppSessions"}
    return [c for c in frame.columns if c not in excluded]


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.07,
        max_iter=140,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=2.0,
        random_state=42,
        early_stopping=False,
    )


def recursive_forecast(
    history: pd.DataFrame,
    test: pd.DataFrame,
    metadata: pd.DataFrame,
    model: HistGradientBoostingRegressor,
    feature_cols: list[str],
    feedback: bool = True,
) -> pd.DataFrame:
    """Forecast in date order, optionally feeding predictions into future lags."""
    known = {
        hub: group.set_index("Date")[TARGET].astype(float).to_dict()
        for hub, group in history.groupby("HubID", sort=False)
    }
    base = add_static_features(enrich_calendar(test), metadata)
    hub_log_mean = history.assign(_log=np.log1p(history[TARGET].clip(lower=0))).groupby("HubID")["_log"].mean()
    base["HubLogMean"] = base["HubID"].map(hub_log_mean)
    predictions = []
    for day in sorted(base["Date"].unique()):
        batch = base.loc[base["Date"].eq(day)].copy()
        lag_values = {lag: [] for lag in LAGS}
        mean_values = {w: [] for w in ROLLS}
        std_values = {w: [] for w in ROLLS}
        for hub in batch["HubID"].tolist():
            values = known.setdefault(hub, {})
            d = pd.Timestamp(day)
            for lag in LAGS:
                lag_values[lag].append(values.get(d - pd.Timedelta(days=lag), np.nan))
            prior = np.array(
                [values.get(d - pd.Timedelta(days=k), np.nan) for k in range(1, 57)],
                dtype=float,
            )
            for window in ROLLS:
                recent = prior[:window]
                observed = recent[np.isfinite(recent)]
                mean_values[window].append(float(observed.mean()) if observed.size else np.nan)
                std_values[window].append(float(observed.std(ddof=1)) if observed.size > 1 else np.nan)
        for lag, vals in lag_values.items():
            batch[f"Lag{lag}"] = vals
        for window in ROLLS:
            batch[f"Mean{window}"] = mean_values[window]
            batch[f"Std{window}"] = std_values[window]
        x = batch[feature_cols].replace([np.inf, -np.inf], np.nan)
        pred = np.maximum(0, np.expm1(model.predict(x)))
        # Closed hubs are deterministically zero; this also covers known renovation
        # closures and aligns the forecast with the operational definition.
        pred = np.where(batch["IsOpen"].to_numpy() == 0, 0.0, pred)
        if feedback:
            for hub, value in zip(batch["HubID"], pred):
                known[hub][pd.Timestamp(day)] = float(value)
        predictions.append(pd.DataFrame({"Id": batch["Id"].to_numpy(), TARGET: pred}))
    return pd.concat(predictions, ignore_index=True)


def make_direct_training(feature_frame, train, cutoff, horizon_days=42):
    """Build horizon-aware training examples from earlier 42-day forecast origins."""
    first_origin = train["Date"].min() + pd.Timedelta(days=364)
    origins = pd.date_range(first_origin, cutoff - pd.Timedelta(days=1), freq=f"{horizon_days}D")
    blocks = []
    for origin in origins:
        end = origin + pd.Timedelta(days=horizon_days)
        block = feature_frame.loc[
            feature_frame["Date"].ge(origin)
            & feature_frame["Date"].lt(end)
            & feature_frame["Date"].lt(cutoff)
        ].copy()
        if block.empty:
            continue
        block["Horizon"] = (block["Date"] - origin).dt.days + 1
        for lag in LAGS:
            block.loc[block["Horizon"].gt(lag), f"Lag{lag}"] = np.nan
        log_means = train.loc[train["Date"].lt(origin)].assign(
            _log=lambda x: np.log1p(x[TARGET].clip(lower=0))
        ).groupby("HubID")["_log"].mean()
        block["HubLogMean"] = block["HubID"].map(log_means)
        blocks.append(block)
    if not blocks:
        raise ValueError("Not enough history to build direct-forecast examples.")
    return pd.concat(blocks, ignore_index=True)


def direct_forecast(history, test, metadata, model, feature_cols):
    """Horizon-aware direct forecast using only observations before the origin."""
    origin = test["Date"].min()
    known = {
        hub: group.set_index("Date")[TARGET].astype(float).to_dict()
        for hub, group in history.groupby("HubID", sort=False)
    }
    base = add_static_features(enrich_calendar(test), metadata)
    base["Horizon"] = (base["Date"] - origin).dt.days + 1
    hub_log_mean = history.assign(_log=np.log1p(history[TARGET].clip(lower=0))).groupby("HubID")["_log"].mean()
    base["HubLogMean"] = base["HubID"].map(hub_log_mean)
    outputs = []
    for day in sorted(base["Date"].unique()):
        batch = base.loc[base["Date"].eq(day)].copy()
        day_date = pd.Timestamp(day)
        for lag in LAGS:
            batch[f"Lag{lag}"] = [
                known.get(hub, {}).get(day_date - pd.Timedelta(days=lag), np.nan)
                if day_date - pd.Timedelta(days=lag) < origin else np.nan
                for hub in batch["HubID"]
            ]
        pred = np.maximum(0, np.expm1(model.predict(batch[feature_cols].replace([np.inf, -np.inf], np.nan))))
        pred = np.where(batch["IsOpen"].to_numpy() == 0, 0.0, pred)
        outputs.append(pd.DataFrame({"Id": batch["Id"].to_numpy(), TARGET: pred}))
    return pd.concat(outputs, ignore_index=True)


def get_direct_feature_columns(features):
    # Direct examples have known lags and a fixed origin-level hub mean. Rolling
    # windows are omitted because most would overlap unobserved forecast days.
    return [col for col in features if not col.startswith(("Mean", "Std"))] + ["Horizon"]


def seasonal_naive(history: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Same-weekday forecast, recursively carrying the 7-day pattern forward."""
    known = {
        hub: group.set_index("Date")[TARGET].astype(float).to_dict()
        for hub, group in history.groupby("HubID", sort=False)
    }
    ordered = test.sort_values(["Date", "HubID"])
    predictions = []
    for day, batch in ordered.groupby("Date", sort=True):
        values = []
        date = pd.Timestamp(day)
        for row in batch.itertuples(index=False):
            hub_values = known.setdefault(row.HubID, {})
            value = hub_values.get(date - pd.Timedelta(days=7), 0.0)
            value = 0.0 if row.IsOpen == 0 else max(0.0, float(value))
            hub_values[date] = value
            values.append((row.Id, value))
        predictions.append(pd.DataFrame(values, columns=["Id", TARGET]))
    return pd.concat(predictions, ignore_index=True)


def rmsle(actual, predicted) -> float:
    return float(np.sqrt(mean_squared_log_error(
        np.maximum(0, actual), np.maximum(0, predicted)
    )))


def tune_on_rolling_folds(train, metadata, feature_frame, features, days=42):
    """Compare recursive/direct forecasts and tune a seasonal-naive blend OOF."""
    last_day = train["Date"].max()
    first_day = last_day - pd.Timedelta(days=days * 3 - 1)
    fold_starts = [first_day + pd.Timedelta(days=days * i) for i in range(3)]
    fold_results = []
    for fold_no, start in enumerate(fold_starts, 1):
        end = start + pd.Timedelta(days=days - 1)
        history = train[train["Date"] < start]
        valid = train[train["Date"].between(start, end)].copy()
        valid["Id"] = np.arange(len(valid), dtype=np.int64)
        fit = feature_frame[feature_frame["Date"] < start]
        model = make_model()
        open_fit = fit[fit["IsOpen"].eq(1)]
        model.fit(open_fit[features], np.log1p(open_fit[TARGET].clip(lower=0)))
        direct_features = get_direct_feature_columns(features)
        direct_fit = make_direct_training(feature_frame, history, start, days)
        direct_open = direct_fit[direct_fit["IsOpen"].eq(1)]
        direct_model = make_model()
        direct_model.fit(direct_open[direct_features], np.log1p(direct_open[TARGET].clip(lower=0)))
        valid_test = valid.drop(columns=[TARGET]).copy()
        recursive = recursive_forecast(history, valid_test, metadata, model, features, feedback=True)
        direct = direct_forecast(history, valid_test, metadata, direct_model, direct_features)
        naive = seasonal_naive(history, valid_test)
        actual = valid.sort_values("Id")[["HubID", TARGET]].reset_index(drop=True)
        for name, pred in (("recursive", recursive), ("direct", direct), ("naive", naive)):
            ordered_pred = pred.sort_values("Id")[TARGET].to_numpy()
            score = rmsle(actual[TARGET].to_numpy(), ordered_pred)
            print(f"Fold {fold_no} ({start.date()} to {end.date()}) {name} RMSLE: {score:.6f}")
        fold_results.append((valid, recursive, direct, naive, history))

    # Select forecasting mode and GBDT weight from out-of-fold predictions.
    best = (float("inf"), None, None)
    for mode in ("recursive", "direct"):
        for weight in np.linspace(0, 1, 21):
            actuals, preds = [], []
            for valid, recursive, direct, naive, _ in fold_results:
                gbdt = recursive if mode == "recursive" else direct
                pred = weight * gbdt.sort_values("Id")[TARGET].to_numpy() + (1 - weight) * naive.sort_values("Id")[TARGET].to_numpy()
                actuals.extend(valid.sort_values("Id")[TARGET].to_numpy())
                preds.extend(pred)
            score = rmsle(np.asarray(actuals), np.asarray(preds))
            if score < best[0]:
                best = (score, mode, float(weight))
    print(f"Selected OOF strategy: {best[1]} + seasonal naive, GBDT weight={best[2]:.2f}, RMSLE={best[0]:.6f}")

    # Apply hub residual offsets learned only from earlier folds for each fold,
    # then estimate a shrinkage amount for the final test-period correction.
    corrections = []
    prior_bias = {}
    for valid, recursive, direct, naive, history in fold_results:
        gbdt = recursive if best[1] == "recursive" else direct
        ordered_valid = valid.sort_values("Id")
        pred = best[2] * gbdt.sort_values("Id")[TARGET].to_numpy() + (1 - best[2]) * naive.sort_values("Id")[TARGET].to_numpy()
        bias = np.array([np.mean(prior_bias.get(h, [0.0])) for h in ordered_valid["HubID"]])
        actual_values = ordered_valid[TARGET].to_numpy()
        corrections.append((actual_values, pred, bias))
        residual = np.log1p(ordered_valid[TARGET].clip(lower=0).to_numpy()) - np.log1p(np.maximum(0, pred))
        hub_size = history.groupby("HubID")[TARGET].mean()
        terciles = hub_size.rank(pct=True)
        bucket = ordered_valid["HubID"].map(terciles).map(
            lambda p: "small" if p <= 1 / 3 else ("medium" if p <= 2 / 3 else "large")
        )
        bucket_error = pd.DataFrame({"Bucket": bucket.to_numpy(), "LogResidual": residual}).groupby("Bucket")["LogResidual"].mean()
        print("Fold bias by hub size (mean log residual): " + ", ".join(
            f"{name}={value:+.4f}" for name, value in bucket_error.items()
        ))
        by_hub = pd.DataFrame({"HubID": ordered_valid["HubID"].to_numpy(), "Residual": residual}).groupby("HubID")["Residual"].mean()
        for hub, value in by_hub.items():
            prior_bias.setdefault(hub, []).append(float(value))
    best_shrink = 0.0
    best_score = float("inf")
    for shrink in np.linspace(0, 1, 21):
        actuals, preds = [], []
        for actual, pred, bias in corrections:
            fixed = np.maximum(0, np.expm1(np.log1p(np.maximum(0, pred)) + shrink * bias))
            actuals.extend(actual)
            preds.extend(fixed)
        score = rmsle(np.asarray(actuals), np.asarray(preds))
        if score < best_score:
            best_score, best_shrink = score, float(shrink)
    print(f"Selected hub-bias shrinkage={best_shrink:.2f}; calibrated OOF RMSLE={best_score:.6f}")
    return best[1], best[2], best_shrink, prior_bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("submission2.csv"))
    parser.add_argument("--validation-days", type=int, default=42)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    train, test, metadata = read_data(args.data_dir)
    if "AppSessions" in test.columns:
        missing = test["AppSessions"].isna().mean()
        print(f"AppSessions is present in test (missing fraction={missing:.3f}); using it is not enabled in this data pipeline.")
    else:
        print("AppSessions is absent from test; excluded from train/test features.")
    print(f"Test horizon: {test['Date'].nunique()} days ({test['Date'].min().date()} to {test['Date'].max().date()}).")
    feature_frame, features = make_training_features(train, metadata)
    mode, blend_weight, bias_shrink, hub_bias = "recursive", 1.0, 0.0, {}
    if not args.skip_validation:
        mode, blend_weight, bias_shrink, hub_bias = tune_on_rolling_folds(
            train, metadata, feature_frame, features, days=args.validation_days
        )

    if mode == "recursive":
        model = make_model()
        open_rows = feature_frame[feature_frame["IsOpen"].eq(1)]
        model.fit(open_rows[features], np.log1p(open_rows[TARGET].clip(lower=0)))
        gbdt = recursive_forecast(train, test, metadata, model, features, feedback=True)
    else:
        direct_features = get_direct_feature_columns(features)
        direct_fit = make_direct_training(feature_frame, train, test["Date"].min(), args.validation_days)
        direct_open = direct_fit[direct_fit["IsOpen"].eq(1)]
        model = make_model()
        model.fit(direct_open[direct_features], np.log1p(direct_open[TARGET].clip(lower=0)))
        gbdt = direct_forecast(train, test, metadata, model, direct_features)
    naive = seasonal_naive(train, test)
    submission = gbdt.rename(columns={TARGET: "GBDTPrediction"}).merge(
        naive.rename(columns={TARGET: "NaivePrediction"}),
        on="Id", how="left", validate="one_to_one",
    )
    submission[TARGET] = (
        blend_weight * submission["GBDTPrediction"]
        + (1 - blend_weight) * submission["NaivePrediction"]
    )
    if bias_shrink:
        hub_by_id = test.set_index("Id")["HubID"]
        offsets = submission["Id"].map(
            hub_by_id.map(lambda hub: float(np.mean(hub_bias.get(hub, [0.0]))))
        ).fillna(0).to_numpy()
        submission[TARGET] = np.expm1(
            np.log1p(submission[TARGET].clip(lower=0)) + bias_shrink * offsets
        )
    is_open = test.set_index("Id")["IsOpen"]
    submission.loc[submission["Id"].map(is_open).to_numpy() == 0, TARGET] = 0
    submission = submission[["Id", TARGET]]
    sample_path = args.data_dir / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        submission = sample[["Id"]].merge(submission, on="Id", how="left", validate="one_to_one")
    submission[TARGET] = submission[TARGET].fillna(0).clip(lower=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.output} (mode={mode}, GBDT weight={blend_weight:.2f}, hub-bias shrink={bias_shrink:.2f}).")


if __name__ == "__main__":
    main()
