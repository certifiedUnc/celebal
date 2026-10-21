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
) -> pd.DataFrame:
    """Predict future days in order, feeding each prediction into later lags."""
    known = {
        hub: group.set_index("Date")[TARGET].astype(float).to_dict()
        for hub, group in history.groupby("HubID", sort=False)
    }
    base = add_static_features(enrich_calendar(test), metadata)
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
                mean_values[window].append(float(np.nanmean(recent)))
                std_values[window].append(float(np.nanstd(recent, ddof=1)))
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
        for hub, value in zip(batch["HubID"], pred):
            known[hub][pd.Timestamp(day)] = float(value)
        predictions.append(pd.DataFrame({"Id": batch["Id"].to_numpy(), TARGET: pred}))
    return pd.concat(predictions, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--validation-days", type=int, default=42)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    train, test, metadata = read_data(args.data_dir)
    feature_frame, features = make_training_features(train, metadata)
    cutoff = train["Date"].max() - pd.Timedelta(days=args.validation_days - 1)

    if not args.skip_validation:
        fit_rows = feature_frame[feature_frame["Date"] < cutoff]
        valid_rows = train[train["Date"] >= cutoff]
        validation_test = valid_rows.drop(columns=[TARGET]).copy()
        validation_test["Id"] = np.arange(len(validation_test), dtype=np.int64)
        validation_model = make_model()
        validation_model.fit(fit_rows[features], np.log1p(fit_rows[TARGET].clip(lower=0)))
        validation_pred = recursive_forecast(
            train[train["Date"] < cutoff], validation_test, metadata,
            validation_model, features,
        )
        score = np.sqrt(mean_squared_log_error(
            valid_rows[TARGET].to_numpy(),
            validation_pred[TARGET].to_numpy(),
        ))
        print(f"{args.validation_days}-day chronological validation RMSLE: {score:.6f}")

    model = make_model()
    model.fit(feature_frame[features], np.log1p(feature_frame[TARGET].clip(lower=0)))
    submission = recursive_forecast(train, test, metadata, model, features)
    sample_path = args.data_dir / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        submission = sample[["Id"]].merge(submission, on="Id", how="left", validate="one_to_one")
    submission[TARGET] = submission[TARGET].fillna(0).clip(lower=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
