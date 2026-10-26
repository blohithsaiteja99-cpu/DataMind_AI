"""Core statistical analysis for DataMind AI."""
from __future__ import annotations
from typing import Any
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def _json_safe(value: Any) -> Any:
    if value is None: return None
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (pd.Timestamp,)): return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value

def profile_dataset(df: pd.DataFrame) -> dict:
    if df.empty: raise ValueError("The dataset has no rows.")
    rows, cols = df.shape
    missing = df.isna().sum()
    numeric = df.select_dtypes(include=np.number)
    categorical = df.select_dtypes(exclude=np.number)
    columns = []
    for name in df.columns:
        s = df[name]
        item = {"name": str(name), "dtype": str(s.dtype),
                "missing_count": int(s.isna().sum()),
                "missing_percent": round(float(s.isna().mean()*100), 2),
                "unique_count": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            item["statistics"] = {
                "mean": _json_safe(clean.mean()) if len(clean) else None,
                "median": _json_safe(clean.median()) if len(clean) else None,
                "std": _json_safe(clean.std()) if len(clean)>1 else None,
                "min": _json_safe(clean.min()) if len(clean) else None,
                "max": _json_safe(clean.max()) if len(clean) else None,
                "q1": _json_safe(clean.quantile(.25)) if len(clean) else None,
                "q3": _json_safe(clean.quantile(.75)) if len(clean) else None}
        else:
            item["top_values"] = {str(k): int(v) for k,v in s.value_counts(dropna=False).head(5).items()}
        columns.append(item)
    high_missing = [str(c) for c in df.columns if df[c].isna().mean() >= .30]
    constant = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    duplicate_rows = int(df.duplicated().sum())
    notes = []
    if duplicate_rows: notes.append(f"Found {duplicate_rows} duplicate rows.")
    if high_missing: notes.append("Columns with at least 30% missing values: "+", ".join(high_missing)+".")
    if constant: notes.append("Columns with one or zero distinct non-null values: "+", ".join(constant)+".")
    if not notes: notes.append("No major issues were found by the basic automated quality checks.")
    notes.append("Automated checks flag issues for review; they do not automatically delete or repair data.")
    return {"rows": int(rows), "columns_count": int(cols), "duplicate_rows": duplicate_rows,
            "numeric_columns": [str(c) for c in numeric.columns],
            "categorical_columns": [str(c) for c in categorical.columns],
            "missing_cells": int(missing.sum()),
            "missing_percent_overall": round(float(missing.sum()/(rows*cols)*100),2),
            "high_missing_columns": high_missing, "constant_columns": constant,
            "columns": columns, "quality_notes": notes}

def numeric_correlations(df: pd.DataFrame) -> list[dict]:
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2: return []
    corr = numeric.corr(numeric_only=True)
    result=[]; cols=list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            val=corr.iloc[i,j]
            if pd.notna(val):
                result.append({"feature_a":str(cols[i]),"feature_b":str(cols[j]),
                    "correlation":round(float(val),4),
                    "strength":"strong" if abs(val)>=.7 else "moderate" if abs(val)>=.4 else "weak"})
    return sorted(result,key=lambda x:abs(x["correlation"]),reverse=True)

def detect_anomalies(df: pd.DataFrame, contamination: float=.05) -> dict:
    numeric=df.select_dtypes(include=np.number)
    if numeric.shape[1]==0:
        return {"available":False,"message":"No numeric columns are available for anomaly detection.","anomaly_count":0,"rows":[]}
    if len(df)<5:
        return {"available":False,"message":"At least 5 rows are needed for anomaly detection.","anomaly_count":0,"rows":[]}
    arr=SimpleImputer(strategy="median").fit_transform(numeric.replace([np.inf,-np.inf],np.nan))
    scaled=StandardScaler().fit_transform(arr)
    model=IsolationForest(n_estimators=150,contamination=float(min(max(contamination,.01),.20)),random_state=42)
    labels=model.fit_predict(scaled); scores=-model.score_samples(scaled)
    indices=np.where(labels==-1)[0]
    ranked=sorted(indices,key=lambda i:scores[i],reverse=True); rows=[]
    for i in ranked[:100]:
        record={"row_index":int(i),"anomaly_score":round(float(scores[i]),5)}
        for c in df.columns: record[str(c)]=_json_safe(df.iloc[i][c])
        rows.append(record)
    return {"available":True,"method":"Isolation Forest",
            "contamination":float(min(max(contamination,.01),.20)),
            "anomaly_count":int(len(indices)),
            "anomaly_percent":round(float(len(indices)/len(df)*100),2),
            "rows":rows,"note":"Anomaly scores indicate unusual patterns, not proof of errors or fraud."}

def run_hypothesis_test(df: pd.DataFrame, target: str, group: str) -> dict:
    if target not in df.columns or group not in df.columns: raise ValueError("Choose columns that exist in the dataset.")
    if target==group: raise ValueError("The target and grouping columns must be different.")
    if not pd.api.types.is_numeric_dtype(df[target]): raise ValueError("The target column must be numeric.")
    data=df[[target,group]].dropna()
    groups=[]
    for label,sub in data.groupby(group,observed=True):
        vals=pd.to_numeric(sub[target],errors="coerce").dropna().to_numpy()
        if len(vals)>=2: groups.append((str(label),vals))
    if len(groups)<2: raise ValueError("At least two groups with two numeric observations each are required.")
    arrays=[g[1] for g in groups]
    if len(groups)==2:
        a,b=arrays; test=stats.ttest_ind(a,b,equal_var=False,nan_policy="omit")
        pooled=math.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2)) if len(a)+len(b)>2 else np.nan
        effect=float((np.mean(a)-np.mean(b))/pooled) if pooled and np.isfinite(pooled) else None
        return {"status":"ok","test":"Welch's independent two-sample t-test","target":target,"group_by":group,
            "null_hypothesis":"The two group population means are equal.",
            "groups":[{"name":groups[0][0],"n":int(len(a)),"mean":round(float(np.mean(a)),4),"std":round(float(np.std(a,ddof=1)),4)},
                      {"name":groups[1][0],"n":int(len(b)),"mean":round(float(np.mean(b)),4),"std":round(float(np.std(b,ddof=1)),4)}],
            "statistic":_json_safe(test.statistic),"p_value":_json_safe(test.pvalue),
            "effect_size_cohens_d_approx":_json_safe(effect),
            "interpretation":"Evidence against equal means at alpha=0.05." if test.pvalue<.05 else "Insufficient evidence to reject equal means at alpha=0.05.",
            "caution":"Statistical significance does not establish causality. Check independence, sampling, and practical significance."}
    test=stats.f_oneway(*arrays); all_values=np.concatenate(arrays); grand=float(np.mean(all_values))
    ss_between=sum(len(x)*(float(np.mean(x))-grand)**2 for x in arrays)
    ss_total=float(np.sum((all_values-grand)**2)); eta=ss_between/ss_total if ss_total else None
    return {"status":"ok","test":"One-way ANOVA","target":target,"group_by":group,
        "null_hypothesis":"All group population means are equal.",
        "groups":[{"name":name,"n":int(len(vals)),"mean":round(float(np.mean(vals)),4),"std":round(float(np.std(vals,ddof=1)),4)} for name,vals in groups],
        "statistic":_json_safe(test.statistic),"p_value":_json_safe(test.pvalue),
        "effect_size_eta_squared":_json_safe(eta),
        "interpretation":"Evidence that at least one group mean differs at alpha=0.05." if test.pvalue<.05 else "Insufficient evidence that group means differ at alpha=0.05.",
        "caution":"ANOVA does not identify which groups differ; assumptions and post-hoc testing may be needed. Association does not establish causality."}

def investigate_root_causes(df: pd.DataFrame, target: str) -> dict:
    if target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError("Choose an existing numeric target column.")
    y=pd.to_numeric(df[target],errors="coerce"); valid=y.notna()
    if valid.sum()<10: return {"status":"insufficient_data","message":"At least 10 non-missing target values are recommended."}
    q20,q80=y[valid].quantile([.2,.8]); low_mask=valid&(y<=q20); high_mask=valid&(y>=q80)
    if low_mask.sum()<3 or high_mask.sum()<3: return {"status":"insufficient_data","message":"Not enough observations in low/high outcome groups."}
    results=[]
    for col in df.select_dtypes(include=np.number).columns:
        if col==target: continue
        x=pd.to_numeric(df[col],errors="coerce"); low=x[low_mask].dropna(); high=x[high_mask].dropna()
        if len(low)<3 or len(high)<3 or x.nunique(dropna=True)<2: continue
        test=stats.mannwhitneyu(low,high,alternative="two-sided"); delta=float(high.median()-low.median()); sd=float(x.std())
        results.append({"feature":str(col),"low_outcome_median":round(float(low.median()),4),
            "high_outcome_median":round(float(high.median()),4),"median_difference_high_minus_low":round(delta,4),
            "standardized_median_difference":round(delta/sd,4) if sd else None,"p_value":round(float(test.pvalue),6),
            "low_group_n":int(len(low)),"high_group_n":int(len(high))})
    results.sort(key=lambda r:(r["p_value"],-abs(r["standardized_median_difference"] or 0)))
    return {"status":"ok","target":target,"low_group_definition":f"target <= {q20:.4g} (lowest 20%)",
        "high_group_definition":f"target >= {q80:.4g} (highest 20%)","candidates":results[:15],
        "interpretation":"These are associations between feature values and low/high target groups, not confirmed causal explanations. Investigate confounding, data quality, and temporal ordering.",
        "method":"Low/high quantile comparison with two-sided Mann–Whitney U tests"}

def build_investigation(df: pd.DataFrame, target: str|None=None) -> dict:
    result={"profile":profile_dataset(df),"correlations":numeric_correlations(df)[:20],"anomalies":detect_anomalies(df)}
    if target: result["root_cause_investigation"]=investigate_root_causes(df,target)
    return result
