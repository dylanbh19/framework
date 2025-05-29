"""
Call Center Volume Prediction – Robust Production-Ready Solution
================================================================
Prepares IVR/Genesys and Contact datasets for call-volume
modelling, with aggressive cleaning, feature engineering and
extensive resilience to messy real-world data.

Requirements
------------
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
holidays >= 0.13
openpyxl >= 3.0.0   (only if you load .xlsx files)
"""

import os
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Global plotting style
# ------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class RobustCallCenterPipeline:
    """
    End-to-end pipeline for call-center data, from raw dumps to
    modelling-ready feature tables and business-grade reports.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        genesys_path: str | None = None,
        contact_path: str | None = None,
        country: str = "US",
        output_dir: str | Path = "output",
    ):
        self.genesys_path = genesys_path
        self.contact_path = contact_path
        self.country = country.upper()
        self.holidays = holidays.US() if self.country == "US" else holidays.UK()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Will be populated later
        self.genesys_df = None
        self.contact_df = None
        self.contact_agg = None
        self.merged_df = None
        self.daily_summary = None
        self.hourly_summary = None
        self.modeling_data = None

        # Flexible-name map
        self.column_mappings = {
            "connection_id": [
                "ConnectionID",
                "CNID",
                "connectionid",
                "connection_id",
                "CallID",
            ],
            "reference_no": [
                "ReferenceNo",
                "Reference_No",
                "referenceno",
                "RefNum",
                "Reference",
            ],
            "timestamp": [
                "ConversationStart",
                "StartTime",
                "CallStartTime",
                "timestamp",
                "CreatedDate",
            ],
            "wait_time": ["WaitTime", "Wait_Time", "waittime", "QueueTime"],
            "talk_time": ["TalkTime", "Talk_Time", "talktime", "ConversationTime"],
            "hold_time": ["HoldTime", "Hold_Time", "holdtime"],
            "call_centre": [
                "CallCentre",
                "Call_Centre",
                "call_centre",
                "Location",
                "Site",
            ],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_column_names(columns, ref_col):
        """Convert agg MultiIndex into plain names."""
        flattened = [ref_col]
        for col in columns[1:]:
            if not isinstance(col, tuple):
                flattened.append(col)
                continue

            col0, col1 = col
            match (col1):
                case "count":
                    flattened.append("activity_count")
                case "nunique":
                    flattened.append("unique_activities")
                case "<lambda_0>":
                    flattened.append("activity_sequence")
                case "<lambda_1>":
                    flattened.append("first_activity")
                case "<lambda_2>":
                    flattened.append("last_activity")
                case "sum":
                    flattened.append("total_duration")
                case "mean":
                    flattened.append("avg_duration")
                case "max":
                    flattened.append("max_duration")
                case "min":
                    flattened.append("min_duration")
                case "std":
                    flattened.append("std_duration")
                case "median":
                    flattened.append("median_duration")
                case "<lambda>":
                    lc = col0.lower()
                    if "callcentre" in lc or "call_centre" in lc:
                        flattened.append("call_centre")
                    elif "callertype" in lc:
                        flattened.append("caller_type")
                    elif "companycode" in lc:
                        flattened.append("company_code")
                    else:
                        flattened.append(lc)
                case _:
                    flattened.append(f"{col0.lower()}_{col1}")
        return flattened

    def find_column(self, df: pd.DataFrame, aliases: list[str]) -> str | None:
        for a in aliases:
            if a in df.columns:
                return a
        return None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self, genesys_df=None, contact_df=None):
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        try:
            # — From DataFrames passed directly —
            if genesys_df is not None and contact_df is not None:
                self.genesys_df = genesys_df.copy()
                self.contact_df = contact_df.copy()
            else:
                # — From files —
                if self.genesys_path:
                    self.genesys_df = (
                        pd.read_excel(self.genesys_path)
                        if self.genesys_path.lower().endswith(".xlsx")
                        else pd.read_csv(self.genesys_path, low_memory=False)
                    )
                if self.contact_path:
                    self.contact_df = (
                        pd.read_excel(self.contact_path)
                        if self.contact_path.lower().endswith(".xlsx")
                        else pd.read_csv(self.contact_path, low_memory=False)
                    )

            print(
                f"✓ Genesys rows: {len(self.genesys_df):,} | "
                f"Contact rows: {len(self.contact_df):,}"
            )

            # Show which columns were detected
            print("\nKey columns detected:")
            for key, aliases in self.column_mappings.items():
                gcol = self.find_column(self.genesys_df, aliases)
                ccol = self.find_column(self.contact_df, aliases)
                print(f"  {key:<15} Genesys→ {gcol} | Contact→ {ccol}")

        except Exception as exc:
            raise RuntimeError(f"Error during load: {exc}") from exc

    # ------------------------------------------------------------------
    # Cleaning: ID normalisation
    # ------------------------------------------------------------------
    def clean_connection_ids(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Return df with *_clean1 … *_clean5 variants."""
        if col_name not in df.columns:
            return df

        print(f"  Cleaning {col_name} …")
        df[col_name] = df[col_name].astype(str).str.strip()

        # -- Strategy 1: strip common prefixes/suffixes -----------------
        df[f"{col_name}_clean1"] = (
            df[col_name]
            .str.replace(r"^00E303907400", "", regex=True)
            .str.replace(r"^[A-Z\-]+", "", regex=True)  # <<< CHANGED
            .str.replace(r"_\d+$", "", regex=True)
            .str.replace(r"[-/]+$", "", regex=True)  # <<< CHANGED
        )

        # -- Strategy 2: longest alphanum block -------------------------
        df[f"{col_name}_clean2"] = df[col_name].str.extract(r"([A-Z0-9]{4,})")

        # -- Strategy 3: split on “_” and take first part ---------------
        df[f"{col_name}_clean3"] = df[col_name].str.split("_").str[0]

        # -- Strategy 4: purge non-alphanum -----------------------------
        df[f"{col_name}_clean4"] = df[col_name].str.replace(r"[^A-Z0-9]", "", regex=True)

        # -- Strategy 5: ensure a leading zero -------------------------- # <<< CHANGED
        def _ensure_leading_zero(token):
            if pd.isna(token):
                return token
            token = str(token)
            token = re.sub(r"^[A-Z\-]+", "", token)  # strip any residual alpha prefix
            return token if token.startswith("0") else "0" + token

        df[f"{col_name}_clean5"] = df[f"{col_name}_clean4"].apply(_ensure_leading_zero)

        return df

    # ------------------------------------------------------------------
    # Cleaning & standardising both datasets
    # ------------------------------------------------------------------
    def clean_and_standardize_data(self):
        print("\n" + "=" * 80)
        print("DATA CLEANING & STANDARDISATION")
        print("=" * 80)

        g_conn = self.find_column(self.genesys_df, self.column_mappings["connection_id"])
        c_ref = self.find_column(self.contact_df, self.column_mappings["reference_no"])
        if not g_conn or not c_ref:
            raise ValueError("Cannot locate connection/reference columns")

        print(f"Using Genesys[{g_conn}]  ⇄  Contact[{c_ref}]")

        # --- Genesys ---
        self.genesys_df.dropna(subset=[g_conn], inplace=True)
        self.genesys_df = self.clean_connection_ids(self.genesys_df, g_conn)

        t_col = self.find_column(self.genesys_df, self.column_mappings["timestamp"])
        if t_col:
            self.genesys_df[t_col] = pd.to_datetime(
                self.genesys_df[t_col], errors="coerce"
            )

        # --- Contact ---
        self.contact_df.dropna(subset=[c_ref], inplace=True)
        self.contact_df = self.clean_connection_ids(self.contact_df, c_ref)

        if "ActivityName" in self.contact_df.columns:
            self.contact_df["ActivityName"] = (
                self.contact_df["ActivityName"].str.strip().str.title()
            )
        if "ActivityDuration" in self.contact_df.columns:
            self.contact_df["ActivityDuration"] = pd.to_numeric(
                self.contact_df["ActivityDuration"], errors="coerce"
            ).fillna(0)

        print("✓ Cleaning done")

    # ------------------------------------------------------------------
    # Contact-side aggregations
    # ------------------------------------------------------------------
    def create_contact_aggregations(self):
        print("\n" + "=" * 80)
        print("CONTACT AGGREGATIONS")
        print("=" * 80)

        ref_col = self.find_column(self.contact_df, self.column_mappings["reference_no"])

        agg_dict = {}
        if "ActivityName" in self.contact_df.columns:
            agg_dict["ActivityName"] = [
                "count",
                "nunique",
                lambda x: "|".join(x.astype(str)),
                lambda x: x.iloc[0],
                lambda x: x.iloc[-1],
            ]
        if "ActivityDuration" in self.contact_df.columns:
            agg_dict["ActivityDuration"] = ["sum", "mean", "max", "min", "std", "median"]

        cc_col = self.find_column(self.contact_df, self.column_mappings["call_centre"])
        for col in [cc_col, "CallerType", "CompanyCode"]:
            if col and col in self.contact_df.columns:
                agg_dict[col] = lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]

        contact_aggs = []
        for clean_col in [ref_col] + [f"{ref_col}_clean{i}" for i in range(1, 6)]:  # <<< CHANGED
            if clean_col in self.contact_df.columns:
                try:
                    tmp = (
                        self.contact_df.groupby(clean_col)
                        .agg(agg_dict)
                        .reset_index()
                    )
                    tmp.columns = self._flatten_column_names(tmp.columns, clean_col)
                    contact_aggs.append(tmp)
                except Exception:
                    continue

        if not contact_aggs:
            raise RuntimeError("Aggregation failed for all cleaned variants")

        self.contact_agg = contact_aggs[0]
        self._derive_contact_features()
        print(f"✓ Aggregated {len(self.contact_agg):,} reference IDs")

    def _derive_contact_features(self):
        df = self.contact_agg
        if "activity_count" in df.columns:
            df["was_transferred_count"] = (df["activity_count"] > 2).astype(int)
            df["complexity_simple"] = df["activity_count"]
        if "unique_activities" in df.columns:
            df["was_transferred_unique"] = (df["unique_activities"] > 1).astype(int)
        if "activity_sequence" in df.columns:
            df["was_transferred_keyword"] = df["activity_sequence"].str.contains(
                "Transfer|Escalat|Specialist", case=False, na=False
            ).astype(int)
        if {"activity_count", "total_duration"}.issubset(df.columns):
            df["complexity_combined"] = (
                (df["activity_count"] - 1) * 2 + df["total_duration"] / 180
            )
            df["is_simple_call"] = (
                (df["activity_count"] <= 2) & (df["total_duration"] < 300)
            ).astype(int)
            df["is_complex_call"] = (
                (df["activity_count"] > 5) | (df["total_duration"] > 900)
            ).astype(int)

    # ------------------------------------------------------------------
    # Merge Genesys ↔ Contact
    # ------------------------------------------------------------------
    def merge_datasets_robust(self):
        print("\n" + "=" * 80)
        print("MERGING DATASETS")
        print("=" * 80)

        g_conn = self.find_column(self.genesys_df, self.column_mappings["connection_id"])
        merge_candidates = []

        for g_suffix in [""] + [f"_clean{i}" for i in range(1, 6)]:  # <<< CHANGED
            g_col = f"{g_conn}{g_suffix}"
            for c_suffix in [""] + [f"_clean{i}" for i in range(1, 6)]:  # <<< CHANGED
                c_col = f"{self.contact_agg.columns[0]}{c_suffix}" if c_suffix else self.contact_agg.columns[0]
                if g_col in self.genesys_df.columns and c_col in self.contact_agg.columns:
                    merge_candidates.append((g_col, c_col))

        best_pair, best_rate = (None, None), 0.0
        for g_col, c_col in merge_candidates:
            try:
                sample = pd.merge(
                    self.genesys_df.sample(min(10000, len(self.genesys_df)), random_state=42),
                    self.contact_agg,
                    left_on=g_col,
                    right_on=c_col,
                    how="left",
                )
                rate = sample[c_col].notna().mean()
                print(f"  {g_col} ⇄ {c_col}: {rate:.1%}")
                if rate > best_rate:
                    best_rate, best_pair = rate, (g_col, c_col)
            except Exception:
                continue

        if best_rate == 0:
            print("⚠️  No matches found; falling back to raw join")
            best_pair = (g_conn, self.contact_agg.columns[0])

        print(f"\n✓ Selected merge: {best_pair[0]} ⇄ {best_pair[1]}  (match {best_rate:.1%})")
        self.merged_df = pd.merge(
            self.genesys_df, self.contact_agg, left_on=best_pair[0], right_on=best_pair[1], how="left"
        )

        total, matched = len(self.merged_df), self.merged_df[best_pair[1]].notna().sum()
        print(f"✓ Final match-rate: {matched / total:.1%}  ({matched:,}/{total:,})")

    # ------------------------------------------------------------------
    # Time-metric conversion
    # ------------------------------------------------------------------
    def calculate_time_metrics(self):
        print("\n" + "=" * 80)
        print("TIME METRICS")
        print("=" * 80)

        df = self.merged_df
        w = self.find_column(df, self.column_mappings["wait_time"])
        t = self.find_column(df, self.column_mappings["talk_time"])
        h = self.find_column(df, self.column_mappings["hold_time"])

        components = []
        for label, col in [
            ("wait_time_seconds", w),
            ("talk_time_seconds", t),
            ("hold_time_seconds", h),
            ("acw_time_seconds", "ACWTime"),
            ("alert_time_seconds", "AlertTime"),
            ("aban_time_seconds", "AbanTime"),
        ]:
            if col in df.columns:
                df[label] = pd.to_numeric(df[col], errors="coerce").fillna(0) / 1000
                components.append(label)
                print(f"  → {col} → {label}")

        df["total_call_time_seconds"] = df[components].sum(axis=1) if components else 0

        if "wait_time_seconds" in df.columns:
            df["wait_time_category"] = pd.cut(
                df["wait_time_seconds"],
                [-1, 30, 60, 120, 300, np.inf],
                labels=["No Wait", "Short", "Medium", "Long", "Very Long"],
            )
        if "talk_time_seconds" in df.columns:
            df["talk_time_category"] = pd.cut(
                df["talk_time_seconds"],
                [-1, 60, 180, 300, 600, 1200, np.inf],
                labels=["Very Short", "Short", "Medium", "Long", "Very Long", "Extremely Long"],
            )

        self.merged_df = df

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------
    def create_temporal_features_robust(self):
        print("\n" + "=" * 80)
        print("TEMPORAL FEATURES")
        print("=" * 80)

        df = self.merged_df
        t_col = self.find_column(df, self.column_mappings["timestamp"])
        if not t_col:
            print("⚠️  No timestamp column; skipping")
            return

        df["timestamp"] = pd.to_datetime(df[t_col], errors="coerce")
        valid_mask = df["timestamp"].notna()  # <<< CHANGED – always defined

        n_bad = (~valid_mask).sum()
        if n_bad:
            print(f"  Warning: {n_bad:,} unparsable timestamps")

        df.loc[valid_mask, "date"] = df.loc[valid_mask, "timestamp"].dt.date
        basics = {
            "year": "year",
            "month": "month",
            "day": "day",
            "hour": "hour",
            "minute": "minute",
            "day_of_week": "dayofweek",
            "day_name": "day_name",
            "quarter": "quarter",
            "day_of_year": "dayofyear",
        }
        for name, attr in basics.items():
            df.loc[valid_mask, name] = getattr(df.loc[valid_mask, "timestamp"].dt, attr)

        try:
            df.loc[valid_mask, "week_of_year"] = df.loc[valid_mask, "timestamp"].dt.isocalendar().week
        except AttributeError:
            pass

        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_holiday"] = df["date"].apply(lambda d: d in self.holidays if pd.notna(d) else False).astype(int)
        df["is_business_hours"] = df["hour"].between(8, 17).astype(int)
        df["time_of_day"] = pd.cut(
            df["hour"],
            [-1, 6, 9, 12, 15, 18, 21, 24],
            labels=["Night", "Early Morning", "Morning", "Lunch", "Afternoon", "Evening", "Late Evening"],
        )

        self.merged_df = df
        print(f"✓ Temporal features for {valid_mask.sum():,} records")

    # ------------------------------------------------------------------
    # Intent augmentation (unchanged)
    # ------------------------------------------------------------------
    def augment_intent_data(self):
        …  # (identical to original – omitted for brevity)

    # ------------------------------------------------------------------
    # Call metrics (unchanged)
    # ------------------------------------------------------------------
    def create_call_metrics(self):
        …  # (identical to original – omitted for brevity)

    # ------------------------------------------------------------------
    # Aggregated views (unchanged)
    # ------------------------------------------------------------------
    def create_aggregated_views(self):
        …  # (identical to original – omitted for brevity)

    # ------------------------------------------------------------------
    # Lag features (unchanged)
    # ------------------------------------------------------------------
    def create_lag_features(self):
        …  # (identical to original – omitted for brevity)

    # ------------------------------------------------------------------
    # Visualisations, saving, reporting – unchanged
    # ------------------------------------------------------------------
    def create_visualizations_safe(self):
        …  # (identical to original – omitted for brevity)

    def save_outputs(self):
        …  # (identical to original – omitted for brevity)

    def generate_business_report(self):
        …  # (identical to original – omitted for brevity)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def run_complete_pipeline(self):
        steps = [
            ("Loading data", self.load_data),
            ("Cleaning & standardising", self.clean_and_standardize_data),
            ("Contact aggregations", self.create_contact_aggregations),
            ("Merging datasets", self.merge_datasets_robust),
            ("Time metrics", self.calculate_time_metrics),
            ("Temporal features", self.create_temporal_features_robust),
            ("Intent augmentation", self.augment_intent_data),
            ("Call metrics", self.create_call_metrics),
            ("Aggregated views", self.create_aggregated_views),
            ("Lag features", self.create_lag_features),
            ("Visualisations", self.create_visualizations_safe),
            ("Saving outputs", self.save_outputs),
            ("Business report", self.generate_business_report),
        ]
        done = []
        print("\n" + "=" * 80)
        print("PIPELINE START")
        print("=" * 80)
        for label, fn in steps:
            try:
                print(f"\n→ {label} …")
                fn()
                done.append(label)
            except Exception as exc:
                print(f"❌  {label} failed: {exc}")
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for label in done:
            print(f"✓ {label}")
        print(f"\nCompleted {len(done)}/{len(steps)} steps.")
        return len(done) == len(steps)


# ----------------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------------
def main():
    import sys

    print("\n" + "=" * 80)
    print("ROBUST CALL-CENTER PIPELINE")
    print("=" * 80)

    if len(sys.argv) >= 3:
        genesys_path, contact_path = sys.argv[1:3]
        country = sys.argv[3] if len(sys.argv) > 3 else "US"
    else:
        genesys_path = input("Genesys/IVR file path: ").strip()
        contact_path = input("Contact-log file path: ").strip()
        country = input("Country for holidays (US/UK) [US]: ").strip() or "US"

    for pth, label in [(genesys_path, "Genesys"), (contact_path, "Contact")]:
        if not os.path.exists(pth):
            print(f"❌ {label} file not found: {pth}")
            return

    pipe = RobustCallCenterPipeline(
        genesys_path=genesys_path,
        contact_path=contact_path,
        country=country,
        output_dir="call_center_analysis_output",
    )
    ok = pipe.run_complete_pipeline()
    print("\n✅ Finished" if ok else "\n⚠️  Finished with errors")


if __name__ == "__main__":
    main()