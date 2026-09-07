"""Train-only roster and unannotated panels for independent RGB/IR geometry review.

CPU only. Never uses detection annotations, predictions, or image registration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def freeze_roster(campaign_root):
    root = Path(campaign_root)
    drone = root / "data/processed/dronevehicle/yolo/hbb_v1"
    groups = defaultdict(list)
    for line in (drone / "rgb_train_source_groups.tsv").read_text().splitlines():
        stem, group = line.split("\t")
        groups[group].append(stem)
    rows = []
    for group in sorted(groups):
        stems = sorted(set(groups[group]))
        for rank, ix in zip(("first", "middle", "last"), (0, len(stems)//2, len(stems)-1)):
            stem = stems[ix]
            rgb = next((drone / "rgb/images/train").glob(stem + ".*"))
            ir = next((drone / "infrared/images/train").glob(stem + ".*"))
            rows.append(dict(dataset="dronevehicle", split="train", group=group,
                sample_position=rank, stem=stem, rgb_path=str(rgb), ir_path=str(ir),
                rgb_source=str(rgb.resolve()), ir_source=str(ir.resolve())))
    llvip = root / "data/processed/llvip/splits/grouped_v1/fit.tsv"
    groups = defaultdict(list)
    for row in csv.DictReader(llvip.open(), delimiter="\t"):
        groups[row["sequence_prefix"]].append(row)
    for group in sorted(groups):
        items = sorted(groups[group], key=lambda row: row["stem"])
        for rank, ix in zip(("first", "middle", "last"), (0, len(items)//2, len(items)-1)):
            row = items[ix]
            rows.append(dict(dataset="llvip", split="fit", group=group,
                sample_position=rank, stem=row["stem"], rgb_path=row["visible"],
                ir_path=row["infrared"], rgb_source=str(Path(row["visible"]).resolve()),
                ir_source=str(Path(row["infrared"]).resolve())))
    for dataset in ("dronevehicle", "llvip"):
        selected = [r for r in rows if r["dataset"] == dataset]
        # Exactly ceil(20%) fixed and spread over the full sorted roster.
        n = math.ceil(len(selected) * 0.2)
        review = set(i * len(selected) // n for i in range(n))
        for i, row in enumerate(selected):
            row["independent_review_required"] = i in review
            row["audit_id"] = dataset + "_" + row["stem"]
    return dict(protocol="tc_geometry_train_roster_v1", scope="exact_images_only",
                point_source="independent_visible_physical_structures_only",
                annotation_status="pending", roster=rows)


def make_panel(row, output_dir, width=640):
    from PIL import Image, ImageDraw
    panels = []
    sizes = []
    for key in ("rgb_path", "ir_path"):
        with Image.open(row[key]) as im:
            sizes.append(im.size)
            image = im.convert("RGB")
            h = round(image.height * width / image.width)
            panels.append(image.resize((width, h)))
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (width * 2, h+36), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 7), row["audit_id"] + " RGB | raw " + str(sizes[0]), fill="black")
    draw.text((width+5, 7), "IR | raw " + str(sizes[1]), fill="black")
    canvas.paste(panels[0], (0, 36))
    canvas.paste(panels[1], (width, 36))
    out = Path(output_dir) / (row["audit_id"] + "_unannotated.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=96)
    return dict(audit_id=row["audit_id"], output=str(out), original_sizes=sizes,
                panel_width=width, top_margin=36, point_annotations=False)


def make_overview_pages(roster, output_dir, dataset="llvip"):
    """Six pairs/page, two columns: visual triage only, never point measurement."""
    from PIL import Image
    rows = [row for row in roster["roster"] if row["dataset"] == dataset]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pages = []
    for first in range(0, len(rows), 6):
        batch = rows[first:first+6]
        canvas = Image.new("RGB", (1280, 822), "white")
        for i, row in enumerate(batch):
            info = make_panel(row, out / "individual_panels")
            with Image.open(info["output"]) as im:
                canvas.paste(im.resize((640, 274)), ((i%2)*640, (i//2)*274))
        target = out / (dataset+"_overview_%02d.jpg" % (first//6+1))
        canvas.save(target, quality=95)
        pages.append(dict(path=str(target), audit_ids=[r["audit_id"] for r in batch],
                          purpose="visual_triage_only_not_coordinate_measurement"))
    return pages


def build_exact_contract(roster, observations, accepted_ids, review_artifact, reviewer):
    """Materialize explicitly reviewed IDs; never infer acceptance from low residuals."""
    if not Path(review_artifact).is_file():
        raise FileNotFoundError("The independent review artifact must already exist.")
    rows = {row["audit_id"]: row for row in roster["roster"]}
    entries = []
    for item in observations["observations"]:
        if item["audit_id"] not in accepted_ids:
            continue
        row = rows[item["audit_id"]]
        entries.append(dict(status="accepted", audit_id=item["audit_id"],
            source="independent_physical_correspondences", annotator=observations["annotator"],
            reviewer=reviewer, review_status="accepted", review_artifact=str(review_artifact),
            independent_review_required=row["independent_review_required"],
            image_paths=sorted(set([row["rgb_path"], row["rgb_source"]])),
            original_shape=item["original_shape"],
            rgb_points=[p["rgb"] for p in item["points"]],
            ir_points=[p["ir"] for p in item["points"]],
            uncertainty_raw_px=item["uncertainty_raw_px"]))
    if set(accepted_ids) != {entry["audit_id"] for entry in entries}:
        raise ValueError("A requested accepted ID was not present in the actual annotations.")
    result = dict(protocol="tc_geometry_exact_image_v1", mode="verified_identity_grid",
        verified=bool(entries), scope="exact_images_and_convex_hull_intersection_only",
        evidence_type="empirical_correspondence_bound_not_pixelwise_calibration",
        entries=entries)
    try:
        from .geometry_contract import GeometryContract
    except ImportError:
        from geometry_contract import GeometryContract
    GeometryContract(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--panel-count-per-dataset", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    roster = freeze_roster(args.campaign_root)
    target = out / "frozen_roster.json"
    if target.exists():
        if json.loads(target.read_text()) != roster:
            raise RuntimeError("Existing frozen roster differs; create a new protocol/attempt.")
    else:
        target.write_text(json.dumps(roster, indent=2) + "\n")
    panels = []
    for dataset in ("dronevehicle", "llvip"):
        rows = [row for row in roster["roster"] if row["dataset"] == dataset]
        # Initial genuine review is first/middle/last of the frozen sample universe.
        indexes = sorted(set(i * (len(rows)-1) // max(args.panel_count_per_dataset-1, 1)
                             for i in range(args.panel_count_per_dataset)))
        for ix in indexes:
            panels.append(make_panel(rows[ix], out / "panels"))
    summary = dict(protocol=roster["protocol"], geometry_verified=False,
        total_rows=len(roster["roster"]),
        counts={dataset: sum(r["dataset"] == dataset for r in roster["roster"])
                for dataset in ("dronevehicle", "llvip")},
        review_counts={dataset: sum(r["dataset"] == dataset and r["independent_review_required"]
                                    for r in roster["roster"])
                       for dataset in ("dronevehicle", "llvip")},
        panels=panels, annotation_count=0,
        note="Panels have been decoded; generating panels is not correspondence annotation.")
    (out / "roster_receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
