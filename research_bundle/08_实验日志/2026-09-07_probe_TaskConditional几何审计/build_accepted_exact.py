"""Build the one actually independently reviewed exact-image entry."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parents[1] / "03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1"
sys.path.insert(0, str(MODULE))
from geometry_audit_tools import build_exact_contract

contract = build_exact_contract(
    json.loads((HERE/"frozen_roster.json").read_text()),
    json.loads((HERE/"primary_visual_observations_batch2.json").read_text()),
    ["llvip_050001"], HERE/"INDEPENDENT_REVIEW_BATCH2.md", "Codex root independent image review")
target = HERE/"geometry_contract_accepted_exact_v1.json"
if target.exists():
    if json.loads(target.read_text()) != contract:
        raise RuntimeError("Existing accepted contract differs; use a new version.")
else:
    target.write_text(json.dumps(contract, indent=2)+"\n")
print(target)
