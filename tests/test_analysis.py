import sys
from pathlib import Path
import pandas as pd

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from dataminds.analysis import profile_dataset, detect_anomalies, numeric_correlations, run_hypothesis_test, investigate_root_causes
from dataminds.report import markdown_report

def sample_df():
    return pd.DataFrame({"sales":[10,12,11,13,9,30,29,31,28,32,12,11],
        "region":["A"]*6+["B"]*6,"visits":[100,110,105,120,90,300,290,310,280,320,115,108],
        "note":["ok"]*11+[None]})

def test_profile_counts():
    r=profile_dataset(sample_df()); assert r["rows"]==12 and r["columns_count"]==4 and r["missing_cells"]==1

def test_anomaly_detection_runs():
    r=detect_anomalies(sample_df(),.1); assert r["available"] and r["anomaly_count"]>=1

def test_correlations_returned():
    r=numeric_correlations(sample_df()); assert len(r)>0

def test_hypothesis_test():
    r=run_hypothesis_test(sample_df(),"sales","region"); assert r["status"]=="ok" and 0<=r["p_value"]<=1

def test_root_cause():
    r=investigate_root_causes(sample_df(),"sales"); assert r["status"]=="ok" and "candidates" in r

def test_report_has_limitations():
    r={"profile":profile_dataset(sample_df()),"correlations":[],"anomalies":{"anomaly_count":0}}
    report=markdown_report(r); assert "Limitations" in report and "not automatically errors" in report
