"""Build thesis-ready figures and tables from frozen offline evaluation artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/thesis_final_results"
ROI_RUN = ROOT / "analysis/roi_direct_vs_checklist/run_20260809_preflight"
BASELINE_JSON = ROOT / "analysis/affected_part_baseline/affected_part_baseline_metrics.json"
PROMPT_JSON = ROOT / "analysis/vision_prompt_ab/results/affected_part_prompt_ab_metrics.json"
TARGETED_JSON = ROOT / "analysis/vision_prompt_ab/targeted_run_20260809_111248/evaluation/targeted_ab_metrics.json"
ROI_CSV = ROOT / "analysis/roi_identity_poc/candidate_reduction.csv"
FINAL_JSON = ROI_RUN / "evaluation/final_evaluation_report.json"
CREATED_AT = datetime.now(timezone.utc).isoformat()

COLORS = {"blue": "#356AA0", "orange": "#D97732", "green": "#4D8B5C", "red": "#B54A4A", "gray": "#6B7280", "light": "#D7E3F1"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax: plt.Axes, title: str, ylabel: str = "Rate (%)") -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D1D5DB", linewidth=.7, alpha=.7)
    ax.spines[["top", "right"]].set_visible(False)


def percent_bars(path: Path, title: str, labels: list[str], values: list[float | None], *, colors=None, note="", ylabel="Rate (%)") -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    numeric = [0 if value is None else value * 100 for value in values]
    bars = ax.bar(range(len(labels)), numeric, color=colors or COLORS["blue"], width=.68)
    ax.set_xticks(range(len(labels)), labels, rotation=20 if len(labels) > 4 else 0, ha="right" if len(labels) > 4 else "center")
    ax.set_ylim(0, 108)
    style(ax, title, ylabel)
    for bar, value in zip(bars, values):
        label = "N/A" if value is None else f"{value:.1%}"
        ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 1) + 2, label, ha="center", fontsize=10, fontweight="bold")
    if note:
        fig.text(.5, .01, note, ha="center", fontsize=9, color=COLORS["gray"])
        fig.subplots_adjust(bottom=.18)
    save(fig, path)


def grouped(path: Path, title: str, metrics: list[str], series: dict[str, list[float | None]], *, note="") -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.arange(len(metrics)); width = .8 / len(series)
    for idx, (name, values) in enumerate(series.items()):
        positions = x - .4 + width/2 + idx*width
        nums = [0 if v is None else v*100 for v in values]
        bars = ax.bar(positions, nums, width, label=name, color=[COLORS["blue"], COLORS["orange"], COLORS["green"]][idx % 3])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, max(bar.get_height(), 1)+1.2, "N/A" if value is None else f"{value:.0%}", ha="center", fontsize=8, rotation=90 if len(metrics) > 5 else 0)
    ax.set_xticks(x, metrics, rotation=18, ha="right")
    ax.set_ylim(0, 112); ax.legend(frameon=False, ncol=len(series), loc="upper center")
    style(ax, title)
    if note:
        fig.text(.5, .01, note, ha="center", fontsize=9, color=COLORS["gray"]); fig.subplots_adjust(bottom=.2)
    save(fig, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_thesis_png(source: Path, target: Path, *, max_width: int = 3200) -> None:
    """Copy a source figure into a thesis-sized, 300-dpi PNG without altering it."""
    with Image.open(source) as image:
        image.load()
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        image.save(target, format="PNG", optimize=True, dpi=(300, 300))


def main() -> None:
    baseline = load_json(BASELINE_JSON)
    prompt = load_json(PROMPT_JSON)
    targeted = load_json(TARGETED_JSON)
    roi = read_csv(ROI_CSV)
    final = load_json(FINAL_JSON)
    for folder in ("01_freeform_baseline", "02_prompt_candidate", "03_roi_candidate_reduction", "04_roi_direct_vs_checklist", "05_final_case_results", "06_research_evolution", "thesis_tables"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)

    # Stage 1
    b = baseline["summary"]
    percent_bars(OUT/"01_freeform_baseline/baseline_identity_metrics.png", "Free-form VLM affected-part identity", ["Exact match", "At least one", "All parts", "Precision", "Recall", "F1"], [b["exact_set_match_accuracy"], b["at_least_one_part_recall"], b["all_parts_recall"], b["part_level_precision"], b["part_level_recall"], b["part_level_f1"]], note="Baseline evaluation: n=25 cases; metric-specific part denominators apply.")
    false = baseline["false_confident"]
    percent_bars(OUT/"01_freeform_baseline/baseline_false_confident_rate.png", "False-confident identity rate by threshold", ["≥0.70", "≥0.80", "≥0.90"], [false[key]["false_confident_identity_rate"] for key in ("0.70", "0.80", "0.90")], colors=COLORS["red"], note="Each threshold has 25 high-confidence predictions; 22 are incorrect identities.")
    bins = baseline["confidence_bins"]
    fig, ax = plt.subplots(figsize=(10, 5.8)); values = [item["empirical_accuracy"] for item in bins]
    bars = ax.bar([item["bin"] for item in bins], [0 if v is None else v*100 for v in values], color=COLORS["blue"])
    style(ax, "Baseline confidence calibration", "Empirical accuracy (%)"); ax.set_ylim(0, 105)
    for bar, item, value in zip(bars, bins, values): ax.text(bar.get_x()+bar.get_width()/2, max(bar.get_height(),1)+2, "N/A\n(n=0)" if value is None else f"{value:.1%}\n(n={item['prediction_count']})", ha="center", fontsize=9)
    save(fig, OUT/"01_freeform_baseline/baseline_confidence_calibration.png")
    a01 = [row for row in read_csv(ROOT/"analysis/affected_part_baseline_predictions.csv") if row["case_id"] == "missingpart-A01"]
    fig, ax = plt.subplots(figsize=(11, 6)); y=np.arange(len(a01)); correct=[row["predicted_part_ids"]=="PIN_RED_SHORT" for row in a01]
    ax.barh(y, [float(row["predicted_confidence"])*100 for row in a01], color=[COLORS["green"] if ok else COLORS["red"] for ok in correct])
    ax.set_yticks(y, [row["view_angle"] for row in a01]); ax.invert_yaxis(); ax.set_xlim(0,110); style(ax,"missingpart-A01 multiview predictions","Confidence (%)")
    for idx,row in enumerate(a01): ax.text(2,idx,f"{row['predicted_part_ids']}  {'CORRECT' if correct[idx] else 'INCORRECT'}",va="center",color="white",fontsize=9,fontweight="bold")
    fig.text(.5,.01,"Ground Truth: PIN_RED_SHORT; eight frozen views.",ha="center",fontsize=9,color=COLORS["gray"]); fig.subplots_adjust(bottom=.12)
    save(fig,OUT/"01_freeform_baseline/missingpart_A01_multiview_predictions.png")

    # Stage 2
    metrics = ["Exact", "At least one", "All parts", "F1", "False-conf. @.80"]
    def variant_values(data):
        s=data["summary"]; return [s["exact_set_match_accuracy"],s["at_least_one_part_recall"],s["all_parts_recall"],s["part_level_f1"],data["false_confident"]["0.80"]["false_confident_identity_rate"]]
    grouped(OUT/"02_prompt_candidate/prompt_ab_metrics.png","Prompt A/B affected-part metrics",metrics,{"Baseline":variant_values(prompt["baseline"]),"Reference":variant_values(prompt["reference"]),"Reference+Candidate":variant_values(prompt["reference_candidate"])},note="Reference has no schema-valid samples (n=0); N/A is not plotted as 0%. Other denominators: Baseline n=2, Reference+Candidate n=3.")
    tv=targeted["variants"]
    grouped(OUT/"02_prompt_candidate/targeted_candidate_metrics.png","Targeted prompt/candidate experiment",metrics,{"Reference":variant_values(tv["reference"]),"Reference+Candidate":variant_values(tv["reference_candidate"])},note="Reference n=0 schema-valid; Reference+Candidate n=3. Directional comparison only.")
    fig,ax=plt.subplots(figsize=(10,5.8)); ax.axis("off"); ax.set_title("Why the A01 candidate constraint failed",fontsize=15,fontweight="bold",pad=16)
    items=[("Candidate inventory","15 / 15 parts"),("Expected identity","PIN_RED_SHORT present"),("Distractor","EYE_BALL present"),("Model prediction","EYE_BALL (0.95)"),("Conclusion","Membership was valid, but the set was too broad")]
    for i,(label,value) in enumerate(items):
        y=.82-i*.16; ax.add_patch(FancyBboxPatch((.08,y-.06),.84,.11,boxstyle="round,pad=.01",facecolor="#EEF3F8",edgecolor="#AAB7C4")); ax.text(.12,y,label,va="center",fontweight="bold"); ax.text(.42,y,value,va="center",color=COLORS["red"] if i==3 else "#243447")
    save(fig,OUT/"02_prompt_candidate/candidate_constraint_failure_A01.png")

    # Stage 3
    cases=[row["case_id"] for row in roi]; full=[int(row["full_candidate_count"]) for row in roi]; reduced=[int(row["roi_candidate_count"]) for row in roi]
    fig,ax=plt.subplots(figsize=(10,5.8)); x=np.arange(3); w=.36
    ax.bar(x-w/2,full,w,label="Full inventory",color=COLORS["gray"]); ax.bar(x+w/2,reduced,w,label="ROI candidates",color=COLORS["blue"])
    ax.set_xticks(x,cases); ax.set_ylim(0,17); ax.legend(frameon=False); style(ax,"ROI candidate reduction","Candidate count")
    for i,row in enumerate(roi): ax.text(i,float(row["roi_candidate_count"])+.5,f"{float(row['reduction_ratio']):.1%} reduced",ha="center",fontsize=9)
    save(fig,OUT/"03_roi_candidate_reduction/roi_candidate_reduction.png")
    fig,ax=plt.subplots(figsize=(10,5.8)); ax.bar(cases,[100,100,100],color=COLORS["green"]); ax.set_ylim(0,110); style(ax,"Confirmed Ground Truth coverage after ROI reduction")
    for i in range(3): ax.text(i,102,"covered",ha="center",fontweight="bold"); fig.text(.5,.02,"Coverage 3/3 = 100%; EYE_BALL retained in 0/3 candidate sets.",ha="center",fontsize=10)
    save(fig,OUT/"03_roi_candidate_reduction/roi_gt_coverage.png")
    percent_bars(OUT/"03_roi_candidate_reduction/roi_localization_scores.png","Frozen ROI localization scores",cases,[float(row["localization_score"]) for row in roi],note="All localization packages remain unverified and require manual review.",ylabel="Localization score (%)")
    fig,axes=plt.subplots(2,2,figsize=(11,8)); axes=axes.ravel()
    for ax,values,title,ylabel in zip(axes,[reduced,[float(r["reduction_ratio"])*100 for r in roi],[float(r["localization_score"])*100 for r in roi],[100,100,100]],["ROI candidate count","Candidate reduction","Localization score","GT coverage"],["Count","Percent","Percent","Percent"]):
        bars=ax.bar(cases,values,color=COLORS["blue"]); ax.set_title(title,fontweight="bold"); ax.set_ylabel(ylabel); ax.tick_params(axis="x",rotation=15); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.3)
        for bar,v in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,v+max(values)*.03,f"{v:.1f}",ha="center",fontsize=8)
    fig.suptitle("Localization-guided ROI method summary",fontsize=16,fontweight="bold"); fig.tight_layout(rect=(0,0,1,.96)); save(fig,OUT/"03_roi_candidate_reduction/roi_method_summary.png")

    # Stage 4
    dm=final["semantic_metrics"]["roi_direct"]; cm=final["semantic_metrics"]["roi_checklist"]
    stage4_metrics=["Exact","At least one","All parts","Precision","Recall","F1"]
    grouped(OUT/"04_roi_direct_vs_checklist/direct_vs_checklist_metrics.png","ROI Direct vs normalized Checklist",stage4_metrics,{"ROI Direct":[dm["exact_set_match"],dm["at_least_one_recall"],dm["all_parts_recall"],dm["part_precision"],dm["part_recall"],dm["part_f1"]],"ROI Checklist":[cm["exact_set_match"],cm["at_least_one_recall"],cm["all_parts_recall"],cm["part_precision"],cm["part_recall"],cm["part_f1"]]},note="n=3 per method. Checklist values are post-hoc normalized analysis; original Checklist schema validity was 0/3.")
    shutil.copy2(ROI_RUN/"figures/checklist_confusion_matrix.png",OUT/"04_roi_direct_vs_checklist/checklist_confusion_matrix.png")
    grouped(OUT/"04_roi_direct_vs_checklist/false_confident_comparison.png","False-confident identity rate @0.80",["False-confident @0.80"],{"ROI Direct":[dm["false_confident_identity"]["0.80"]],"ROI Checklist":[cm["false_confident_identity"]["0.80"]]},note="Lower is safer; n=3 cases per method, identity-level denominators differ.")
    grouped(OUT/"04_roi_direct_vs_checklist/manual_review_and_uncertainty.png","Manual review and uncertainty",["Manual review","Uncertain checks"],{"ROI Direct":[dm["manual_review_rate"],None],"ROI Checklist":[cm["manual_review_rate"],final["checklist_component_metrics"]["uncertain_rate"]]},note="Case-level manual review uses n=3; Checklist uncertainty uses 16 candidate checks.")

    # Stage 5 copies
    source_fig=ROI_RUN/"figures"
    for case in ("missingpart_A01","missingpart_B01","wrongpart_B01"):
        copy_thesis_png(source_fig/"cases"/f"{case}_result.png",OUT/"05_final_case_results"/f"{case}_result.png")
        copy_thesis_png(source_fig/f"thesis_case_{case}.png",OUT/"05_final_case_results"/f"thesis_case_{case}.png")

    # Stage 6
    stages=["Free-form\nVLM","Prompt +\nCandidate","ROI candidate\nreduction","ROI Direct","ROI Checklist"]
    exact=[.08,0,None,dm["exact_set_match"],cm["exact_set_match"]]; f1=[.105263,.285714,None,dm["part_f1"],cm["part_f1"]]; false80=[.88,.666667,None,dm["false_confident_identity"]["0.80"],cm["false_confident_identity"]["0.80"]]
    grouped(OUT/"06_research_evolution/research_method_evolution.png","Directional evolution of affected-part methods",[s.replace("\n"," ") for s in stages],{"Exact match":exact,"Part F1":f1,"False-confident @.80":false80},note="Different stages use different evaluation denominators; values indicate directional method evolution rather than statistical significance.")
    fig,ax=plt.subplots(figsize=(12,4.8)); ax.axis("off")
    nodes=["Free-form Vision","Prompt Constraint","ROI Grounding","Checklist Verification","Rule Engine","Identity Verifier","Deterministic Annotation"]
    xs=np.linspace(.06,.94,len(nodes))
    for i,(x,label) in enumerate(zip(xs,nodes)):
        ax.add_patch(FancyBboxPatch((x-.06,.39),.12,.22,boxstyle="round,pad=.012",facecolor="#EAF1F7",edgecolor=COLORS["blue"],linewidth=1.3)); ax.text(x,.5,label,ha="center",va="center",fontsize=9,wrap=True)
        if i<len(nodes)-1: ax.add_patch(FancyArrowPatch((x+.062,.5),(xs[i+1]-.062,.5),arrowstyle="->",mutation_scale=14,color=COLORS["gray"]))
    ax.set_title("Research pipeline evolution",fontsize=16,fontweight="bold",pad=10); fig.text(.5,.08,"Offline evidence flow; final output is deterministic annotation, not an image-generation API.",ha="center",fontsize=10,color=COLORS["gray"])
    save(fig,OUT/"06_research_evolution/research_pipeline_evolution.png")

    # Thesis tables
    tables=OUT/"thesis_tables"
    base_rows=[{"metric":k,"value":v,"sample_size":25,"notes":"Baseline frozen evaluation; metric-specific part denominator may differ."} for k,v in [("exact_match",b["exact_set_match_accuracy"]),("at_least_one_recall",b["at_least_one_part_recall"]),("all_parts_recall",b["all_parts_recall"]),("precision",b["part_level_precision"]),("recall",b["part_level_recall"]),("f1",b["part_level_f1"]),("false_confident_080",false["0.80"]["false_confident_identity_rate"])]]
    write_csv(tables/"01_baseline_metrics.csv",["metric","value","sample_size","notes"],base_rows)
    prompt_rows=[]
    for name,key in (("Baseline","baseline"),("Reference","reference"),("Reference+Candidate","reference_candidate")):
        vals=variant_values(prompt[key]); prompt_rows.append({"method":name,"sample_size":prompt[key]["summary"]["evaluated_case_count"],**dict(zip(["exact_match","at_least_one_recall","all_parts_recall","f1","false_confident_080"],vals)),"notes":"N/A fields have no schema-valid denominator."})
    write_csv(tables/"02_prompt_candidate_metrics.csv",["method","sample_size","exact_match","at_least_one_recall","all_parts_recall","f1","false_confident_080","notes"],prompt_rows)
    shutil.copy2(ROI_CSV,tables/"03_roi_candidate_reduction.csv")
    shutil.copy2(ROI_RUN/"thesis_tables/roi_direct_vs_checklist_metrics.csv",tables/"04_direct_vs_checklist_metrics.csv")
    shutil.copy2(ROI_RUN/"thesis_tables/checklist_component_results.csv",tables/"05_checklist_component_results.csv")
    shutil.copy2(ROI_RUN/"thesis_tables/roi_direct_vs_checklist_cases.csv",tables/"06_case_results.csv")
    shutil.copy2(ROI_RUN/"thesis_tables/research_method_evolution.csv",tables/"07_research_method_evolution.csv")
    shutil.copy2(ROI_RUN/"thesis_tables/request_efficiency.csv",tables/"08_request_efficiency.csv")
    master=[]
    def add(stage,method,n,vals,notes,**extra): master.append({"stage":stage,"method":method,"sample_size":n,"exact_match":vals[0],"at_least_one_recall":vals[1],"all_parts_recall":vals[2],"precision":vals[3],"recall":vals[4],"f1":vals[5],"false_confident_080":vals[6],"candidate_reduction":extra.get("candidate_reduction"),"gt_coverage":extra.get("gt_coverage"),"manual_review_rate":extra.get("manual_review_rate"),"notes":notes})
    add("Stage 1","Free-form VLM",25,[b["exact_set_match_accuracy"],b["at_least_one_part_recall"],b["all_parts_recall"],b["part_level_precision"],b["part_level_recall"],b["part_level_f1"],false["0.80"]["false_confident_identity_rate"]],"Historical baseline; denominators differ from targeted ROI stages.")
    rc=prompt["reference_candidate"]; s=rc["summary"]; add("Stage 2","Reference+Candidate",3,[s["exact_set_match_accuracy"],s["at_least_one_part_recall"],s["all_parts_recall"],s["part_level_precision"],s["part_level_recall"],s["part_level_f1"],rc["false_confident"]["0.80"]["false_confident_identity_rate"]],"Targeted n=3; full-inventory candidate set remained weak.")
    add("Stage 3","ROI candidate reduction",3,[None,None,None,None,None,None,None],"Identity inference not performed in this stage.",candidate_reduction=np.mean([float(r["reduction_ratio"]) for r in roi]),gt_coverage=1.0,manual_review_rate=1.0)
    add("Stage 4","ROI Direct",3,[dm["exact_set_match"],dm["at_least_one_recall"],dm["all_parts_recall"],dm["part_precision"],dm["part_recall"],dm["part_f1"],dm["false_confident_identity"]["0.80"]],"Frozen ROI experiment n=3.",candidate_reduction=.6444444444444445,gt_coverage=1.0,manual_review_rate=dm["manual_review_rate"])
    add("Stage 5","ROI Checklist (normalized)",3,[cm["exact_set_match"],cm["at_least_one_recall"],cm["all_parts_recall"],cm["part_precision"],cm["part_recall"],cm["part_f1"],cm["false_confident_identity"]["0.80"]],"Post-hoc normalized analysis; original schema validity 0/3.",candidate_reduction=.6444444444444445,gt_coverage=1.0,manual_review_rate=cm["manual_review_rate"])
    write_csv(tables/"master_experiment_summary.csv",["stage","method","sample_size","exact_match","at_least_one_recall","all_parts_recall","precision","recall","f1","false_confident_080","candidate_reduction","gt_coverage","manual_review_rate","notes"],master)

    # Figure validation
    validation=[]
    for path in sorted(OUT.rglob("*.png")):
        try:
            with Image.open(path) as image: image.verify()
            with Image.open(path) as image: width,height=image.size
            ok=width>0 and height>0 and path.stat().st_size>0
        except Exception: width=height=0; ok=False
        validation.append({"file":path.relative_to(OUT).as_posix(),"exists":path.exists(),"readable":ok,"width":width,"height":height,"file_size":path.stat().st_size if path.exists() else 0,"sha256":sha256(path) if path.exists() else "","status":"PASS" if ok else "FAIL"})
    write_csv(OUT/"figure_validation.csv",["file","exists","readable","width","height","file_size","sha256","status"],validation)

    readme = """# Thesis Final Results\n\nThis directory consolidates thesis-ready figures and tables derived entirely from frozen offline evaluation artifacts. No inference or image-generation API is used.\n\n## Reading the results\n\nStages 1–5 document the progression from free-form identity inference to prompt constraints, ROI candidate reduction, Direct/Checklist comparison, and fail-closed deterministic annotations. `06_research_evolution/` is directional: stage denominators differ and no statistical-significance claim is made.\n\nChecklist values labeled normalized are experiment-only post-hoc compatibility analysis; original Checklist schema validity remains 0/3. Missing denominators are shown as N/A, never 0%.\n\n`figure_validation.csv` verifies PNG readability and hashes. `artifact_manifest.json` records provenance. Ground Truth is used only in evaluation tables, never inference or normalization.\n"""
    (OUT/"README.md").write_text(readme,encoding="utf-8")

    index = """# Thesis Figure Index\n\n| Figure | Caption | Key observation | Source |\n|---|---|---|---|\n| Figure 4-1 | Free-form VLM affected-part baseline | Identity metrics remain low despite broad error recognition. | `analysis/thesis_final_results/01_freeform_baseline/baseline_identity_metrics.png` |\n| Figure 4-2 | False-confident identity rate | Incorrect identities persist at all high-confidence thresholds. | `analysis/thesis_final_results/01_freeform_baseline/baseline_false_confident_rate.png` |\n| Figure 4-3 | ROI candidate reduction | Candidate sets shrink from 15 to 5/5/6 while retaining confirmed targets. | `analysis/thesis_final_results/03_roi_candidate_reduction/roi_candidate_reduction.png` |\n| Figure 4-4 | ROI Direct versus Checklist | Normalized Checklist improves recall/F1 descriptively, but n=3. | `analysis/thesis_final_results/04_roi_direct_vs_checklist/direct_vs_checklist_metrics.png` |\n| Figure 4-5 | Checklist confusion matrix | Resolved checks have zero FN but four FP; six checks remain uncertain. | `analysis/thesis_final_results/04_roi_direct_vs_checklist/checklist_confusion_matrix.png` |\n| Figure 4-6 | missingpart-A01 qualitative result | Red short pin is identified, but verifier conflict keeps manual review. | `analysis/thesis_final_results/05_final_case_results/thesis_case_missingpart_A01.png` |\n| Figure 4-7 | missingpart-B01 qualitative result | Checklist fails closed instead of accepting the Direct red-pin error. | `analysis/thesis_final_results/05_final_case_results/thesis_case_missingpart_B01.png` |\n| Figure 4-8 | wrongpart-B01 qualitative result | Both swap identities are retained, with unresolved false positives. | `analysis/thesis_final_results/05_final_case_results/thesis_case_wrongpart_B01.png` |\n| Figure 4-9 | Research method evolution | Results show directional method evolution across differing denominators. | `analysis/thesis_final_results/06_research_evolution/research_method_evolution.png` |\n"""
    (ROOT/"docs/thesis_figure_index.md").write_text(index,encoding="utf-8")

    source_map={"01_freeform_baseline":[str(BASELINE_JSON.relative_to(ROOT)),"analysis/affected_part_baseline_predictions.csv"],"02_prompt_candidate":[str(PROMPT_JSON.relative_to(ROOT)),str(TARGETED_JSON.relative_to(ROOT))],"03_roi_candidate_reduction":[str(ROI_CSV.relative_to(ROOT))],"04_roi_direct_vs_checklist":[str(FINAL_JSON.relative_to(ROOT))],"05_final_case_results":["analysis/roi_direct_vs_checklist/run_20260809_preflight/figures"],"06_research_evolution":[str(BASELINE_JSON.relative_to(ROOT)),str(PROMPT_JSON.relative_to(ROOT)),str(FINAL_JSON.relative_to(ROOT))],"thesis_tables":["frozen evaluation CSV/JSON artifacts"]}
    manifest=[]
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        rel=path.relative_to(OUT); stage=rel.parts[0] if len(rel.parts)>1 else "root"
        manifest.append({"stage":stage,"artifact_type":path.suffix.lower().lstrip(".") or "file","file_path":rel.as_posix(),"source_artifacts":source_map.get(stage,["derived consolidation metadata"]),"sha256":sha256(path),"created_at":CREATED_AT,"thesis_usage":"figure" if path.suffix.lower()==".png" else "table or audit metadata","notes":"Derived offline; no API. Different-stage comparisons are directional."})
    (OUT/"artifact_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS" if all(r["status"]=="PASS" for r in validation) else "FAIL","figure_count":len(validation),"artifact_count":len(manifest),"output":str(OUT)},indent=2))


if __name__ == "__main__":
    main()
