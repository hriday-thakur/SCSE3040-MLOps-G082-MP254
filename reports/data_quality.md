# MP254: Data Quality Verification Report

**Group:** G082  
**Problem ID:** MP254   
**Responsibility:** Data Quality Verification

## 1. Dataset Overview

The following datasets were loaded and investigated:

| Dataset | Rows | Columns |
|---|---:|---:|
| train.csv | 1,044 | 12 |
| later.csv | 300 | 12 |
| holdout_features.csv | 250 | 10 |

The target variable is `yield_qtl_per_acre`.

## 2. Seven Data Quality Checks

### 2.1 Leaking Column

- Column: `yield_from_mandi_receipt`
- Correlation with target: 0.998957
- Absent from holdout: Yes

The column has an extremely high correlation with the target and is absent from the holdout dataset. It is therefore identified as a potential data leakage column and should not be used as a model feature.

### 2.2 Unit Mix-up

- Column: `fertilizer_kg`
- Affected group: `Tehsil West`
- Conversion factor: 1,000

The median fertilizer value for Tehsil West is 81,650, compared with 83.6 for Tehsil North and 85.9 for Tehsil South.

Dividing Tehsil West values by 1,000 makes the distribution comparable to the other tehsils. This indicates a likely grams-to-kilograms unit mix-up.

### 2.3 Sentinel Values

Common sentinel candidates were checked across numeric columns:

`-999, -99, -1, 999, 9999, 99999, -9999`

Additional broad domain checks were performed for negative measurements, invalid pH values, and non-positive farm area.

**Result:** No common sentinel candidates or broad domain violations were detected.

These checks do not establish that every value is valid.

### 2.4 Duplicate Rows

- Extra exact duplicate rows: 44

The check using `DataFrame.duplicated()` identified 44 extra copies of exact duplicate records.

These duplicates should be removed from the working dataset during preprocessing. The original raw dataset should remain unchanged.

### 2.5 Stuck Sensor

Input measurement columns were investigated using daily aggregation.

The improved screening method:
- Excludes the target and leakage columns.
- Checks dates with at least two observations.
- Looks for identical readings across farms.
- Checks consecutive repeated daily averages using numerical tolerance.
- Flags sustained patterns of at least three days for further investigation.

Results:
- `irrigation_count` had one qualifying date with identical readings across farms: 2026-03-04.
- The longest repeated daily average was two consecutive days.
- No sustained suspicious pattern of at least three days was detected.

**Conclusion:** A stuck sensor was not confirmed. Further verification would require sensor identifiers or additional measurement-level information.

### 2.6 Target Typos

The target column `yield_qtl_per_acre` was investigated using two methods.

**10× median screening:**
- Median: 20.495
- Threshold: 204.95
- Potential 10× typo rows: 0

**IQR outlier screening:**
- Lower bound: 8.075
- Upper bound: 33.335
- Statistical outliers: 0

**Conclusion:** No obvious tenfold target-entry errors or statistical outliers were detected.

### 2.7 Spelling Variants

The training dataset contains 15 distinct raw spellings in `seed_variety`.

The later dataset provides three clean reference categories:

- `PBW-343`
- `DBW-187`
- `HD-2967`

The variations include inconsistent capitalization, leading and trailing spaces, and incomplete spellings.

**Conclusion:** 15 raw spellings represent 3 true categories.

## 3. Summary of Findings

| Issue | Result |
|---|---|
| Leaking column | Detected |
| Unit mix-up | Detected |
| Sentinel values | No common candidates detected |
| Duplicate rows | 44 extra duplicates |
| Stuck sensor | Not confirmed |
| Target typos | 0 tenfold errors and 0 IQR outliers |
| Spelling variants | 15 raw spellings, 3 true categories |

## 4. Recommended Data Preparation

1. Exclude `yield_from_mandi_receipt` from model features.
2. Convert Tehsil West fertilizer measurements to kilograms by dividing by 1,000.
3. Remove exact duplicate rows from the working training dataset.
4. Standardize seed-variety spellings using the three clean reference categories.
5. Preserve the original raw datasets.
6. Investigate sensor behaviour further if sensor-level information becomes available.

## 5. Implementation

The data-quality checks are implemented in:

`src/mp254/data_quality.py`

A separate stuck-sensor screening implementation is also present in:

`src/mp254/check_stuck_sensor.py`

The main script loads the training, later, and holdout datasets and executes all seven checks.

Run the implementation using:

```bash
python src/mp254/data_quality.py
```

All seven checks executed successfully on the supplied training dataset. The results in this report reflect the script output and additional exploratory investigations.