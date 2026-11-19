
from pathlib import Path

import pandas as pd


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset locations
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
LATER_PATH = PROJECT_ROOT / "data" / "raw" / "later.csv"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "raw" / "holdout_features.csv"

def load_datasets():
    """
    Load the original datasets without modifying them.
    """
    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    return train, later, holdout


def check_leaking_column(train, holdout):
    """
    Identify a column that is highly correlated with the target
    but unavailable in the holdout dataset.
    """
    target = "yield_qtl_per_acre"
    suspicious_column = "yield_from_mandi_receipt"

    correlation = train[suspicious_column].corr(train[target])
    missing_from_holdout = suspicious_column not in holdout.columns

    print("\n1. LEAKING COLUMN CHECK")
    print("Column:", suspicious_column)
    print("Target correlation:", correlation)
    print("Missing from holdout:", missing_from_holdout)

    return {
        "present": correlation > 0.95 and missing_from_holdout,
        "column": suspicious_column,
        "correlation": correlation
    }


def check_unit_mixup(train):
    """
    Compare fertilizer values across tehsils to identify
    a group whose values use a different measurement unit.
    """
    summary = train.groupby("tehsil")["fertilizer_kg"].agg(
        ["count", "min", "mean", "median", "max"]
    )

    north_median = summary.loc["Tehsil North", "median"]
    west_median = summary.loc["Tehsil West", "median"]
    ratio = west_median / north_median

    print("\n2. UNIT MIX-UP CHECK")
    print(summary)
    print("West/North median ratio:", ratio)

    return {
        "present": 900 <= ratio <= 1100,
        "column": "fertilizer_kg",
        "group_value": "Tehsil West",
        "factor": 1000,
        "ratio": ratio
    }



def check_sentinel_value(train):
    """
    Check numeric columns for common sentinel values and
    values outside broad physical/domain validity ranges.
    """

    candidates = [-999, -99, -1, 999, 9999, 99999, -9999]

    # Broad validity rules, not dataset-specific official limits.
    valid_ranges = {
        "rainfall_mm": (0, None),
        "fertilizer_kg": (0, None),
        "sowing_delay_days": (0, None),
        "irrigation_count": (0, None),
        "soil_ph": (0, 14),
        "farm_acres": (0, None),
        "yield_qtl_per_acre": (0, None)
    }

    findings = []

    numeric_columns = train.select_dtypes(
        include="number"
    ).columns

    # Check common sentinel candidates.
    for column in numeric_columns:
        counts = train[column].value_counts()

        for value in candidates:
            count = int(counts.get(value, 0))

            if count > 0:
                findings.append({
                    "column": column,
                    "issue": "common_sentinel",
                    "value": value,
                    "count": count
                })

    # Check broad domain validity rules.
    for column, (lower, upper) in valid_ranges.items():
        if column not in train.columns:
            continue

        invalid = pd.Series(False, index=train.index)

        if lower is not None:
            invalid |= train[column] < lower

        if upper is not None:
            invalid |= train[column] > upper

        if invalid.any():
            findings.append({
                "column": column,
                "issue": "outside_valid_range",
                "value": None,
                "count": int(invalid.sum())
            })

    print("\n3. SENTINEL AND VALIDITY CHECK")

    if findings:
        print(pd.DataFrame(findings).to_string(index=False))
    else:
        print("No common sentinel candidates or domain violations found.")

    return {
        "present": len(findings) > 0,
        "findings": findings
    }



def check_duplicate_rows(train):
    """
    Count exact duplicate rows in the training dataset.
    """
    duplicate_count = int(train.duplicated().sum())

    print("\n4. DUPLICATE ROW CHECK")
    print("Extra duplicate rows:", duplicate_count)

    return {
        "present": duplicate_count > 0,
        "count": duplicate_count
    }

def check_spelling_variants(train, later):
    """
    Compare training categories against the clean categories
    available in later.csv.
    """
    column = "seed_variety"

    true_levels = sorted(later[column].dropna().unique().tolist())
    train_levels = sorted(train[column].dropna().unique().tolist())

    print("\n5. SPELLING VARIANTS CHECK")
    print("Distinct training spellings:", len(train_levels))
    print("Clean reference categories:", true_levels)
    print("Number of true categories:", len(true_levels))

    print("\nTraining spellings:")
    for value in train_levels:
        print(repr(value))

    return {
        "present": len(train_levels) > len(true_levels),
        "column": column,
        "true_levels": true_levels,
        "training_levels": train_levels
    }



def check_stuck_sensor(train):
    """
    Screen input measurements for sustained identical readings
    across consecutive calendar days.

    This is a screening method, not proof of sensor failure.
    """

    sensor_columns = [
        "rainfall_mm",
        "fertilizer_kg",
        "sowing_delay_days",
        "irrigation_count",
        "soil_ph"
    ]

    data = train.copy()
    data["date"] = pd.to_datetime(data["date"])

    # Only dates with at least two farm observations are useful
    # for checking whether readings are identical across farms.
    daily = data.groupby("date")[sensor_columns].agg(
        ["mean", "nunique", "count"]
    )

    results = []

    for column in sensor_columns:
        averages = daily[(column, "mean")]
        unique_counts = daily[(column, "nunique")]
        observation_counts = daily[(column, "count")]

        # A day qualifies when it has at least two observations
        # and every observation has the same value.
        same_reading = (
            (observation_counts >= 2) &
            (unique_counts == 1)
        )

        # Find consecutive calendar-day runs.
        best_run = 0
        current_run = 0
        best_start = None
        best_end = None
        current_start = None
        previous_date = None

        for date, qualifies in same_reading.items():
            if qualifies:
                if (
                    current_run == 0 or
                    previous_date is None or
                    (date - previous_date).days != 1
                ):
                    current_run = 1
                    current_start = date
                else:
                    current_run += 1

                if current_run > best_run:
                    best_run = current_run
                    best_start = current_start
                    best_end = date
            else:
                current_run = 0
                current_start = None

            previous_date = date

        # Also screen daily averages for sustained flat patterns.
        # np.isclose avoids exact floating-point comparisons.
        avg_run = 1
        best_avg_run = 1

        for i in range(1, len(averages)):
            consecutive_date = (
                (averages.index[i] - averages.index[i - 1]).days == 1
            )

            if (
                consecutive_date and
                pd.notna(averages.iloc[i]) and
                pd.notna(averages.iloc[i - 1]) and
                abs(averages.iloc[i] - averages.iloc[i - 1]) <= 1e-6
            ):
                avg_run += 1
                best_avg_run = max(best_avg_run, avg_run)
            else:
                avg_run = 1

        results.append({
            "column": column,
            "identical_reading_days": best_run,
            "start_date": best_start,
            "end_date": best_end,
            "longest_repeated_daily_average": best_avg_run,
            "suspicious": best_run >= 3 or best_avg_run >= 3
        })

    result_df = pd.DataFrame(results)

    print("\n6. STUCK SENSOR SCREENING")
    print(result_df.to_string(index=False))
    print(
        "\nNote: Suspicious patterns require further investigation. "
        "They do not independently confirm sensor failure."
    )

    return result_df






def check_target_typos(train):
    """
    Screen for target values that may have been entered
    approximately ten times too large.

    Also inspect statistical outliers using the IQR method.
    Statistical outliers are not automatically typos.
    """
    target = "yield_qtl_per_acre"
    values = train[target].dropna()

    median = values.median()
    threshold = median * 10

    suspicious = train[train[target] >= threshold].copy()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = train[
        (train[target] < lower_bound) |
        (train[target] > upper_bound)
    ]

    print("\n7. TARGET TYPO CHECK")
    print("Target median:", median)
    print("10x median threshold:", threshold)
    print("Potential 10x typo rows:", len(suspicious))
    print("IQR lower bound:", lower_bound)
    print("IQR upper bound:", upper_bound)
    print("Potential statistical outliers:", len(outliers))

    return {
        "present": len(suspicious) > 0,
        "count": len(suspicious),
        "iqr_outlier_count": len(outliers)
    }




if __name__ == "__main__":
    train, later, holdout = load_datasets()

    print("Datasets loaded successfully!")
    print("Training dataset:", train.shape)
    print("Later dataset:", later.shape)
    print("Holdout dataset:", holdout.shape)

    check_leaking_column(train, holdout)
    check_unit_mixup(train)
    check_sentinel_value(train)
    check_duplicate_rows(train)
    check_spelling_variants(train, later)
    check_stuck_sensor(train)
    check_target_typos(train)
