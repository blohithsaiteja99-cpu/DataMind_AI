from datetime import datetime, timezone

def markdown_report(findings: dict, dataset_name="uploaded dataset") -> str:
    p=findings.get("profile",{}); a=findings.get("anomalies",{})
    lines=["# DataMind AI — Investigation Report","",f"**Dataset:** {dataset_name}",
        f"**Generated (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}","",
        "## Executive summary",f"- Rows: {p.get('rows','N/A')}",f"- Columns: {p.get('columns_count','N/A')}",
        f"- Duplicate rows: {p.get('duplicate_rows','N/A')}",f"- Missing cells: {p.get('missing_cells','N/A')} ({p.get('missing_percent_overall','N/A')}% overall)",
        f"- Detected anomalies: {a.get('anomaly_count','N/A')}","","## Data quality observations"]
    lines += ["- "+x for x in p.get("quality_notes",[])]
    lines += ["","## Strongest numeric correlations"]
    corrs=findings.get("correlations",[])
    lines += [f"- **{x['feature_a']} ↔ {x['feature_b']}**: r = {x['correlation']} ({x['strength']}). Correlation is not causation." for x in corrs[:10]] or ["- Not enough numeric columns."]
    lines += ["","## Anomaly detection",f"- Method: {a.get('method','Not available')}",f"- Count: {a.get('anomaly_count',0)}",
        f"- Note: {a.get('note',a.get('message','No additional notes.'))}"]
    root=findings.get("root_cause_investigation")
    if root:
        lines += ["","## Root-cause investigation (associations, not causal proof)",f"- Target: {root.get('target','N/A')}",
            f"- Method: {root.get('method','N/A')}",f"- Interpretation: {root.get('interpretation',root.get('message','N/A'))}"]
        lines += [f"- **{x['feature']}**: low-group median {x['low_outcome_median']}; high-group median {x['high_outcome_median']}; p = {x['p_value']}." for x in root.get("candidates",[])[:10]]
    h=findings.get("hypothesis_test")
    if h: lines += ["","## Hypothesis test",f"- Test: {h.get('test','N/A')}",f"- Null hypothesis: {h.get('null_hypothesis','N/A')}",f"- p-value: {h.get('p_value','N/A')}",f"- Interpretation: {h.get('interpretation','N/A')}",f"- Caution: {h.get('caution','Check assumptions and sampling.')}"]
    lines += ["","## Limitations","- Anomalies are unusual observations, not automatically errors or fraud.",
        "- Correlation and group differences do not prove causation.","- Statistical conclusions depend on data quality, sampling, and test assumptions.",
        "- Review underlying records and domain context before taking action.",""]
    return "\n".join(lines)
