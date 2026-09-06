"""Render actual manual point marks and summarize them without granting verification."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
MODULE = HERE.parents[1] / "03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1"
sys.path.insert(0, str(MODULE))
from geometry_contract import correspondence_summary

batch_argument = next((arg for arg in sys.argv if arg.startswith("--batch")), "")
batch_suffix = "_"+batch_argument[2:] if batch_argument else ""
roster = json.loads((HERE / "frozen_roster.json").read_text())["roster"]
by_id = {r["audit_id"]: r for r in roster}
raw = json.loads((HERE / ("primary_visual_observations"+batch_suffix+".json")).read_text())
summaries = []
entries = []
for obs in raw["observations"]:
    row = by_id[obs["audit_id"]]
    summary = dict(audit_id=obs["audit_id"], dataset=row["dataset"], group=row["group"],
        status=obs["status"], independent_review_required=row["independent_review_required"],
        review_status="pending", point_count=len(obs["points"]), formal_geometry_verified=False)
    if obs["points"]:
        rgb = [p["rgb"] for p in obs["points"]]
        ir = [p["ir"] for p in obs["points"]]
        stats = correspondence_summary(rgb, ir, obs["original_shape"], obs["uncertainty_raw_px"])
        summary.update(stats)
        scale = obs["panel_to_raw_scale"]
        summary["p95_at_unaugmented_640_input_px"] = stats["p95_error_raw_px"]/scale
        summary["max_at_unaugmented_640_input_px"] = stats["max_error_raw_px"]/scale
        image = Image.open(HERE / "panels" / (obs["audit_id"]+"_unannotated.jpg")).convert("RGB")
        draw = ImageDraw.Draw(image)
        for p in obs["points"]:
            for key, offset in [("rgb", 0), ("ir", 640)]:
                x, y = p[key][0]/scale+offset, p[key][1]/scale+36
                draw.ellipse((x-3, y-3, x+3, y+3), outline=(255, 30, 30), width=2)
                draw.text((x+5, y-6), p["id"], fill=(255, 30, 30))
        image_suffix = batch_suffix if batch_suffix not in ("", "_batch2") else ""
        image.save(HERE / "panels" / (obs["audit_id"]+"_primary_points"+image_suffix+".png"))
        entries.append(dict(status="pending_independent_review", audit_id=obs["audit_id"],
            source="independent_physical_correspondences", annotator=raw["annotator"],
            image_paths=sorted(set([row["rgb_path"], row["rgb_source"]])),
            original_shape=obs["original_shape"], rgb_points=rgb, ir_points=ir,
            uncertainty_raw_px=obs["uncertainty_raw_px"],
            independent_review_required=row["independent_review_required"], review_status="pending"))
    summaries.append(summary)
summary = dict(protocol=raw["protocol"], frozen_pairs=len(roster), viewed_pairs=len(summaries),
    primary_point_pairs=sum(r["point_count"] > 0 for r in summaries), accepted_pairs=0,
    pending_unviewed_pairs=len(roster)-len(summaries), geometry_verified=False,
    scientific_scope="Approximate primary visual marks only; numerical residuals are pending independent validation.",
    observations=summaries)
(HERE / ("primary_batch_summary"+batch_suffix+".json")).write_text(json.dumps(summary, indent=2)+"\n")
contract = dict(protocol="tc_geometry_exact_image_v1", mode="verified_identity_grid", verified=False,
    scope="exact_images_and_convex_hull_intersection_only", entries=entries,
    note="Pending approximate marks do not admit any object to formal DFL KD.")
(HERE / ("geometry_contract_unverified"+batch_suffix+".json")).write_text(json.dumps(contract, indent=2)+"\n")
print(json.dumps(summary, indent=2))
