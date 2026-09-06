from __future__ import annotations


def parse_experiment_id(experiment_id: str) -> dict:
    """
    Parse experiment_id like 'HMF3A_DMSO_R1' into:
    - cell='HMF3A'
    - condition='DMSO'
    - batch='R1'
    """
    exp = str(experiment_id or "")
    parts = exp.split("_")
    cell = parts[0] if len(parts) >= 1 else ""
    condition = parts[1] if len(parts) >= 2 else ""
    batch = parts[2] if len(parts) >= 3 else ""
    return {
        "cell": cell,
        "condition": condition,
        "batch": batch,
    }


def ensure_meta_fields(meta: dict) -> dict:
    """
    Ensure metadata contains cell/condition/batch inferred from experiment_id.
    Explicitly provided values in ``meta`` take precedence.
    """
    out = dict(meta or {})
    inferred = parse_experiment_id(out.get("experiment_id", ""))
    for key in ("cell", "condition", "batch"):
        if out.get(key) in (None, ""):
            out[key] = inferred[key]
    return out
