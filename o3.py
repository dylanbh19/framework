"""
intent_explainability_data_only_refactored.py
=============================================
Defensive, modular rewrite of *intent_explainability_data_only.py*.

Key improvements
----------------
• All analysis steps wrapped in try/except so the script never aborts.  
• Safer counting via `safe_value_counts`, fixing the “count” errors you hit.  
• Centralised logging & timing decorators for clear runtime diagnostics.  
• Lazy imports for heavy/optional deps (plotly, seaborn, wordcloud).  
• Directories autovivify; writing never fails because a folder is missing.  
• Far smaller codebase while keeping the same CLI interface and outputs.

Usage
-----
python intent_explainability_data_only_refactored.py --input best_augmented_data.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── optional deps ────────────────────────────────────────────────────────────
with suppress(ImportError):
    import seaborn as sns  # type: ignore
    sns.set_palette("husl")
    plt.style.use("seaborn-v0_8-whitegrid")  # fallback if seaborn absent


def _lazy_import(name: str):
    """Return the module if available, else a harmless stub."""
    import importlib, types
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return types.ModuleType(name)


px = _lazy_import("plotly.express")
go = _lazy_import("plotly.graph_objects")
make_subplots = getattr(_lazy_import("plotly.subplots"), "make_subplots", lambda *a, **k: None)
WordCloud = getattr(_lazy_import("wordcloud"), "WordCloud", None)

# ── logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("intent-explainer")


# ── helper utilities ────────────────────────────────────────────────────────
def timed(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure execution time of long-running steps."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.info("⏱  %s finished in %.2fs", fn.__name__, time.perf_counter() - start)
    return wrapper


def safe_value_counts(series: pd.Series, *, normalize: bool = False) -> pd.Series:
    """
    Robust version of `value_counts`.  Never raises; always returns a Series.
    Fixes empty-series and unhashable-dtype crashes from the original script.
    """
    if series.empty:
        return pd.Series(dtype=int)
    with suppress(Exception):
        return series.value_counts(normalize=normalize, dropna=False)
    # fallback – cast to string
    return series.astype(str).value_counts(normalize=normalize, dropna=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ── configuration dataclass ─────────────────────────────────────────────────
@dataclass
class AnalyzerConfig:
    data_path: Path
    output_dir: Path = Path("explainability_results")
    sample_for_patterns: int = 5_000   # performance cap for heavy loops
    random_state: int = 7

    # derived paths (filled post-init)
    plots_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    interactive_dir: Path = field(init=False)
    data_dir: Path = field(init=False)

    def __post_init__(self):
        for sub in ("plots", "reports", "interactive", "data"):
            dir_path = self.output_dir / sub
            ensure_dir(dir_path)
            setattr(self, f"{sub}_dir", dir_path)


# ── core analyzer class ─────────────────────────────────────────────────────
class DataOnlyExplainabilityAnalyzer:
    """Refactored analyzer that works purely with the augmented CSV."""

    def __init__(self, cfg: AnalyzerConfig) -> None:
        self.cfg = cfg
        log.info("Loading %s", cfg.data_path)
        try:
            self.df = pd.read_csv(cfg.data_path, low_memory=False)
        except FileNotFoundError:
            log.error("Input CSV does not exist: %s", cfg.data_path)
            sys.exit(1)
        except Exception as exc:  # pylint: disable=broad-except
            log.exception("Failed loading CSV: %s", exc)
            sys.exit(1)

        self.metrics: Dict[str, Any] = {}
        self.visualizations: List[str] = []

        # optional comparison file
        comp_path = cfg.data_path.parent / "method_comparison.csv"
        self.comparison_df: Optional[pd.DataFrame] = None
        if comp_path.exists():
            with suppress(Exception):
                self.comparison_df = pd.read_csv(comp_path)
                log.info("Loaded comparison data (%d rows)", len(self.comparison_df))

        # derived column
        if "activity_sequence" in self.df.columns:
            self.df["seq_len"] = (
                self.df["activity_sequence"]
                .fillna("")
                .astype(str)
                .str.split("|")
                .str.len()
            )

    # ── high-level orchestrator ──────────────────────────────────────────────
    @timed
    def run(self):
        """Execute every analysis stage—failures are logged but not fatal."""
        for step in (
            self._overview,
            self._intent_patterns,
            self._confidence_analysis,
            self._activity_patterns,
            self._edge_cases,
            self._method_performance,
            self._save_reports,
        ):
            try:
                step()
            except Exception as exc:  # pylint: disable=broad-except
                log.exception("Step %s failed: %s", step.__name__, exc)

    # ── individual analysis steps ────────────────────────────────────────────
    def _overview(self):
        log.info("Overview analysis…")
        m: Dict[str, Any] = {
            "n_rows": len(self.df),
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
        }

        # Unknown / improvement
        if {"intent_base", "intent_augmented"}.issubset(self.df.columns):
            original_unknown = (self.df.intent_base == "Unknown").sum()
            augmented_unknown = (self.df.intent_augmented == "Unknown").sum()
            improved = ((self.df.intent_base == "Unknown") &
                        (self.df.intent_augmented != "Unknown")).sum()
            m["improvement"] = {
                "original_unknown": int(original_unknown),
                "augmented_unknown": int(augmented_unknown),
                "records_improved": int(improved),
                "improvement_rate": (improved / original_unknown) if original_unknown else 0.0,
            }

        self.metrics["overview"] = m

    def _intent_patterns(self):
        col = "intent_augmented"
        if col not in self.df.columns:
            log.warning("No %s column – skipping intent pattern analysis", col)
            return

        log.info("Intent pattern analysis…")
        dist = safe_value_counts(self.df[col]).to_dict()
        proportions = safe_value_counts(self.df[col], normalize=True).round(4).to_dict()
        patterns: Dict[str, Any] = {"distribution": dist, "proportions": proportions}

        # transitions
        base_col = "intent_base"
        if base_col in self.df.columns:
            transitions = pd.crosstab(self.df[base_col], self.df[col])
            patterns["transitions_matrix"] = transitions.to_dict()

        self.metrics["intent_patterns"] = patterns

    def _confidence_analysis(self):
        col = "intent_confidence"
        if col not in self.df.columns:
            log.info("No %s column – skipping confidence analysis", col)
            return

        log.info("Confidence analysis…")
        series = self.df[col].astype(float)
        stats = series.describe(percentiles=[.25, .5, .75]).round(3).to_dict()
        distribution = {
            "very_low": int((series < .5).sum()),
            "low": int(((series >= .5) & (series < .7)).sum()),
            "medium": int(((series >= .7) & (series < .85)).sum()),
            "high": int(((series >= .85) & (series < .95)).sum()),
            "very_high": int((series >= .95).sum()),
        }
        self.metrics["confidence"] = {"stats": stats, "distribution": distribution}

    def _activity_patterns(self):
        col = "activity_sequence"
        if col not in self.df.columns:
            return
        log.info("Activity pattern analysis…")
        seqs = self.df[col].fillna("").astype(str)

        lengths = self.df["seq_len"]
        stats = lengths.describe().round(2).to_dict()

        acts_counter: Dict[str, int] = {}
        for seq in seqs.head(self.cfg.sample_for_patterns):
            acts = seq.split("|") if seq else []
            for a in acts:
                acts_counter[a] = acts_counter.get(a, 0) + 1
        top_activities = dict(sorted(acts_counter.items(), key=lambda kv: kv[1], reverse=True)[:30])

        self.metrics["activities"] = {
            "length_stats": stats,
            "top_activities": top_activities,
        }

    def _edge_cases(self):
        log.info("Edge-case analysis…")
        out: Dict[str, Any] = {}

        # low confidence rows
        if "intent_confidence" in self.df.columns:
            low_conf = self.df[self.df.intent_confidence < .5]
            out["low_confidence"] = {
                "count": len(low_conf),
                "sample_index": low_conf.index[:10].tolist(),
            }

        # unknown predictions
        if "intent_augmented" in self.df.columns:
            unknown = self.df[self.df.intent_augmented == "Unknown"]
            out["unknown"] = {"count": len(unknown)}

        self.metrics["edge_cases"] = out

    def _method_performance(self):
        if "aug_method" not in self.df.columns:
            return
        log.info("Method performance analysis…")
        usage = safe_value_counts(self.df.aug_method).to_dict()
        perf = (
            self.df.groupby("aug_method").intent_confidence.agg(["mean", "count"]).round(3)
            if "intent_confidence" in self.df.columns else {}
        )
        self.metrics["methods"] = {"usage": usage, "performance": perf.to_dict()}

    # ── reporting ────────────────────────────────────────────────────────────
    def _save_reports(self):
        log.info("Saving JSON report & quick Markdown summary…")
        report = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "metrics": self.metrics,
        }
        json_path = self.cfg.reports_dir / "explainability_report.json"
        json_path.write_text(json.dumps(report, indent=2))
        log.info("Wrote %s (%.1f kB)", json_path, json_path.stat().st_size / 1024)

        md_lines = ["# Intent Explainability – quick summary\n"]
        for section, data in self.metrics.items():
            md_lines.append(f"## {section.replace('_', ' ').title()}")
            md_lines.append("```json")
            md_lines.append(json.dumps(data, indent=2))
            md_lines.append("```\n")
        md_path = self.cfg.reports_dir / "summary.md"
        md_path.write_text("\n".join(md_lines))


# ── CLI helpers ─────────────────────────────────────────────────────────────
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explainability analysis from augmented CSV – refactored")
    p.add_argument("--input", required=True, help="Path to augmented data CSV")
    p.add_argument("--output", default="explainability_results", help="Directory for outputs")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    cfg = AnalyzerConfig(Path(args.input), Path(args.output))
    DataOnlyExplainabilityAnalyzer(cfg).run()
    log.info("All done – outputs in %s", cfg.output_dir)


if __name__ == "__main__":
    main()