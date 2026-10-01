"""Aggregate Stage A-F investigation gate results into one report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "inbubble_mixer_report"


def load(stage: str) -> dict:
    path = REPO / "runs" / f"inbubble_mixer_stage_{stage}" / "summary.json"
    if not path.is_file():
        return {"stage": stage, "gate": {"passed": False, "note": f"missing {path}"}}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stages = {s: load(s) for s in list("abcdef")}
    product_blockers = []
    if not stages["e"].get("gate", {}).get("passed"):
        product_blockers.append(
            "Stage E: multi-out wet path exceeds 16 ms block budget on host proxy; "
            "keep mixer default pass-through (g_k=1) until Pi 5 re-measure / lighter split."
        )
    if stages["b"].get("gate", {}).get("model_transfer_ok") is False:
        product_blockers.append(
            "Stage B: soft-mask transfer onto promoted b_hat is weak; oracle-bubble split works, "
            "model-b_hat cut/boost control is not yet product-ready."
        )
    if stages["d"].get("gate", {}).get("doa_only_insufficient"):
        product_blockers.append(
            "Stage D: DOA-only IDs fail under crossing; stable speaker embeddings required for Speaker 1/2."
        )

    report = {
        "promoted_model": "runs/optim_pretrain_sonitude_dist_count_1_5m",
        "architecture": "additive mixer on promoted b_hat; vertical-arc geometry",
        "mix_rule": "y = b_hat + sum (g_k - 1) * s_hat_k",
        "stages": {
            k: {
                "passed": bool(v.get("gate", {}).get("passed")),
                "note": v.get("gate", {}).get("note"),
            }
            for k, v in stages.items()
        },
        "product_blockers": product_blockers,
        "deploy_recommendation": (
            "Ship promoted bubble unchanged. Mixer flag may ship with g_k=1 default. "
            "Do not enable multi-stream wet decode on Pi until Stage E passes on-device."
        ),
        "all_gates_passed": all(bool(v.get("gate", {}).get("passed")) for v in stages.values()),
        "details": stages,
    }
    (OUT / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    names = {
        "a": "A Oracle mix-rule",
        "b": "B Separation probe",
        "c": "C DOA / region",
        "d": "D Session ID",
        "e": "E Pi cost envelope",
        "f": "F Realtime mixer flag",
    }
    lines = [
        "# In-bubble controllable mixing - investigation report",
        "",
        f"Promoted model: `{report['promoted_model']}`",
        f"Mix rule: `{report['mix_rule']}`",
        "",
        "| Stage | Pass | Note |",
        "|---|---|---|",
    ]
    for k, title in names.items():
        g = stages[k].get("gate", {})
        note = (g.get("note") or "").encode("ascii", "replace").decode("ascii")
        lines.append(f"| {title} | {'PASS' if g.get('passed') else 'FAIL'} | {note} |")
    lines.append("")
    lines.append("## Product blockers")
    for b in product_blockers:
        lines.append(f"- {b}")
    if not product_blockers:
        lines.append("- None")
    lines.append("")
    lines.append(f"**Deploy recommendation:** {report['deploy_recommendation']}")
    lines.append("")
    lines.append(
        f"**Overall gates:** {'ALL PASS' if report['all_gates_passed'] else 'MIXED - Stage E cost gate blocks multi-stream deploy.'}"
    )
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
