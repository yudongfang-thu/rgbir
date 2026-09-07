"""Read-only CPU intersection of frozen geometry roster and existing train D2.

Dataset/stem keys reconcile known raw/processed aliases for this coverage
diagnostic only. They do not authorize GeometryContract entries or prove that
an unobserved image is registered. No model, teacher, GPU or training is used.
"""
from pathlib import Path, PurePosixPath
import json


def main():
    here = Path(__file__).resolve().parent
    logs = here.parent
    roster_path = logs / "2026-09-07_probe_TaskConditional几何审计/frozen_roster.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))["roster"]
    result = {
        "scope": "read-only CPU roster/D2 intersection; no geometry approval or model execution",
        "key": "dataset, stem; reconciles raw/processed path aliases only for this diagnostic",
        "natural_stream_approximation": "64*(1-(1-M/N)^32); independent uniform image approximation, not actual fixed-loader result",
        "datasets": {},
    }
    for folder, dataset, train_images in (
        ("drone_train", "dronevehicle", 17990),
        ("llvip_train", "llvip", 9619),
    ):
        path = logs / "2026-09-07_probe_TaskConditional机会诊断" / folder / "d2_anchors.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        base_images = {PurePosixPath(row["image_id"]).stem for row in rows}
        selected_images = {PurePosixPath(row["image_id"]).stem for row in rows if row["selected"]}
        entries = [row for row in roster if row["dataset"] == dataset]
        image_ids = {row["stem"] for row in entries}
        result["datasets"][dataset] = {
            "assumed_train_image_count_from_existing_recipe": train_images,
            "roster_rows": len(entries),
            "roster_unique_stems": len(image_ids),
            "d2_base_images": len(base_images),
            "d2_selected_images": len(selected_images),
            "roster_with_existing_d2_base": len(image_ids & base_images),
            "roster_with_existing_d2_selected": len(image_ids & selected_images),
            "roster_selected_stems": sorted(image_ids & selected_images),
            "nonoverlap_is_not_a_negative_geometry_or_selection_result": True,
            "expected_hit_batches_if_every_roster_image_effective": 64 * (1 - (1 - len(image_ids) / train_images) ** 32),
            "expected_hit_batches_if_only_known_selected_roster_images_effective": 64 * (1 - (1 - len(image_ids & selected_images) / train_images) ** 32),
        }
    output = here / "l1_geometry_coverage_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
