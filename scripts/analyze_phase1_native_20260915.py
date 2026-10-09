"""Audit and summarize the transferred native GPT Phase 1 seed blocks.

This reads immutable transferred artifacts and writes derived review data only.
The five-seed study remains incomplete when only blocks 0--2 are present.
"""

import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = "20260915T180331Z"
BATCHES = {
    0: ("phase1_native_20260914_v1_r0", "927025"),
    1: ("phase1_native_20260914_v1_r1", "927026"),
    2: ("phase1_native_20260914_v1_r2", "927027"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_lines(path):
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]


def normalized_auc(points, key, start=None):
    points = [p for p in points if start is None or p["global_step"] >= start]
    if len(points) < 2:
        raise ValueError("AUC needs at least two saved evaluations")
    x = np.array([p["global_step"] for p in points], dtype=float)
    y = np.array([p[key] for p in points], dtype=float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def interval(values):
    values = [float(x) for x in values]
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    mean = float(np.mean(values))
    if len(values) > 1:
        half = float(t.ppf(.975, len(values) - 1)) * float(
            np.std(values, ddof=1)) / math.sqrt(len(values))
        bounds = [mean - half, mean + half]
    else:
        bounds = None
    return {"n": len(values), "mean": float(np.mean(values)),
            "ci95": bounds}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def audit_settings(cell, summary):
    expected = dict(cell["args"])
    size = expected["num_envs"] * expected["num_steps"]
    expected.update(batch_size=size,
                    minibatch_size=size // expected["num_minibatches"],
                    num_iterations=expected["total_timesteps"] // size)
    expected["max_cost_dollars"] = cell["worst_case_usd"]
    actual = summary["args"]
    path_keys = {"budget_ledger", "embed_cache", "price_table"}
    differing = [key for key, value in expected.items()
                 if key not in path_keys and actual.get(key) != value]
    if differing:
        raise ValueError(f"resolved settings differ: {differing}")
    if cell["args"]["guidance"]:
        if not actual["budget_ledger"].endswith("/budget.json"):
            raise ValueError("paid run has no resolved budget ledger")
        if Path(actual["price_table"]).name != Path(expected["price_table"]).name:
            raise ValueError("price table path changed")


def audit_curve(points, summary, seed):
    if len(points) < 3:
        raise ValueError("too few evaluation points")
    previous = -1
    for point in points:
        if (point.get("teacher_on") is not False
                or point.get("episodes") != 50
                or point.get("seed_base") != seed + 50_000
                or point["global_step"] != point["iteration"] * 1024
                or point["global_step"] <= previous):
            raise ValueError("evaluation identity/ordering differs")
        previous = point["global_step"]
        for key in ("success_rate", "sampled_success_rate"):
            value = point[key]
            if (not math.isfinite(value) or not 0 <= value <= 1
                    or not math.isclose(value * 50, round(value * 50),
                                        abs_tol=1e-8)):
                raise ValueError("invalid evaluation success count")
    if points[-1]["global_step"] != summary["global_step"]:
        raise ValueError("final evaluation does not reach final checkpoint")
    if (points[-1]["success_rate"] != summary["latest"]["eval_success_rate"]
            or points[-1]["sampled_success_rate"]
            != summary["latest"]["eval_sampled_success_rate"]):
        raise ValueError("summary endpoint differs from evaluation journal")


def audit_journal(run, summary):
    advising = summary["latest"]["advising"]
    counts = summary["latest"]["consultations"]
    path = run / "consultations.jsonl"
    if not path.is_file():
        if advising["num_asked"] or counts["records"]:
            raise ValueError("missing nonempty consultation journal")
        return {"queries": 0, "consultations": 0, "http_attempts": 0,
                "cache_hits": 0, "cached_deliveries": 0,
                "labels": 0, "withheld": 0, "failures": 0,
                "known_cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0,
                "last_query_transition": None, "first_rollout_queries": 0,
                "queried_env_episodes": 0, "query_quartiles": [0, 0, 0, 0]}
    rows = json_lines(path)
    # A consultation may return from the response cache without an HTTP
    # request. Require recorded attempt counts rather than inferring billing.
    attempts = []
    cached = []
    for row in rows:
        metadata = row.get("metadata", {})
        count = metadata.get("attempt_count")
        hit = bool(metadata.get("cache_hit"))
        if (type(count) is not int or count < 0
                or (hit and count != 0) or (not hit and count == 0)):
            raise ValueError("invalid or missing HTTP/cache accounting")
        attempts.append(count)
        cached.append(hit)
    if len(rows) != advising["num_asked"] or len(rows) != counts["records"]:
        raise ValueError("journal/query aggregates differ")
    delivered = sum(bool(r["delivered"]) for r in rows)
    failures = sum(bool(r.get("metadata", {}).get("failed")) for r in rows)
    if delivered != advising["num_delivered"] or failures != counts["failures"]:
        raise ValueError("journal delivery/failure aggregates differ")
    positions = [r["rollout"] * 1024 + r["step"] * 8 + r["env"] for r in rows]
    quartiles = [0, 0, 0, 0]
    for position in positions:
        quartiles[min(3, int(4 * position / summary["global_step"]))] += 1
    known = sum(float(r.get("known_response_dollars") or 0) for r in rows)
    recorded = summary["latest"]["teacher_cost_dollars"]
    if not math.isclose(known, recorded, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("journal known cost differs from run summary")
    return {
        "queries": len(rows), "consultations": len(rows),
        "http_attempts": sum(attempts), "cache_hits": sum(cached),
        "cached_deliveries": sum(hit and bool(r["delivered"])
                                 for hit, r in zip(cached, rows)),
        "labels": delivered,
        "withheld": len(rows) - delivered, "failures": failures,
        "unknown_cost_rows": sum(r.get("known_response_dollars") is None
                                 for r in rows),
        "known_cost_usd": known,
        "tokens_in": sum(int(r.get("tokens_in") or 0) for r in rows),
        "tokens_out": sum(int(r.get("tokens_out") or 0) for r in rows),
        "last_query_transition": max(positions),
        "first_rollout_queries": sum(r["rollout"] == 0 for r in rows),
        "queried_env_episodes": len({(r["env"], r["episode"]) for r in rows}),
        "query_quartiles": quartiles,
    }


def audit_episodes(path, summary):
    rows = np.loadtxt(io.BytesIO(path.read_bytes()), delimiter=",", skiprows=1,
                      usecols=(0, 1, 5), ndmin=2)
    if (not len(rows) or not np.isfinite(rows).all()
            or not np.array_equal(rows[:, 0], np.arange(1, len(rows) + 1))
            or np.any(np.diff(rows[:, 1]) < 0)):
        raise ValueError("invalid episode journal")
    if len(rows) != summary["total_episodes"]:
        raise ValueError("episode total differs from summary")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    data = ROOT / "results/vulcan_sync/data/advising_strength"
    accounting_path = (ROOT / "results/vulcan_sync/receipts" / args.receipt
                       / "accounting_after.psv")
    accounting = {r["JobID"]: r for r in csv.DictReader(
        accounting_path.open(encoding="utf-8-sig"), delimiter="|")
                  if "." not in r["JobID"]}
    rows = []
    curves = []
    source_hashes = {}
    for replicate, (name, job) in BATCHES.items():
        batch = data / name
        manifest_path = batch / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        expected_hash = (batch / "manifest.sha256").read_text().strip()
        if hashlib.sha256(manifest_bytes).hexdigest() != expected_hash:
            raise ValueError(f"manifest hash mismatch: {name}")
        manifest = json.loads(manifest_bytes)
        if len(manifest["cells"]) != 8 or manifest["replicate"] != replicate:
            raise ValueError(f"unexpected block shape: {name}")
        source_hashes[name + "/manifest.json"] = expected_hash
        block_hashes = set()
        for cell in manifest["cells"]:
            directory = batch / "cells" / str(cell["index"])
            dispatch = read_json(directory / "dispatch.json")
            if (dispatch.get("commit") != manifest["commit"]
                    or any(dispatch.get(key) != cell[key]
                           for key in ("index", "provider", "bonus", "strategy"))):
                raise ValueError(f"dispatch differs: {name}/{cell['index']}")
            dispatch_args = dispatch.get("args", {})
            allowed_dispatch_changes = {
                "budget_ledger", "embed_cache", "price_table",
                "max_cost_dollars",
            }
            changed = [key for key, value in cell["args"].items()
                       if key not in allowed_dispatch_changes
                       and dispatch_args.get(key) != value]
            if changed:
                raise ValueError(f"dispatch args differ: {changed}")
            exit_row = read_json(directory / "exit.json")
            if exit_row.get("returncode") != 0:
                raise ValueError(f"nonzero worker exit: {name}/{cell['index']}")
            scheduler = accounting.get(f"{job}_{cell['index']}")
            if not scheduler or scheduler["State"] != "COMPLETED" \
                    or scheduler["ExitCode"] != "0:0":
                raise ValueError(f"scheduler incomplete: {name}/{cell['index']}")
            candidates = []
            for candidate in (batch / "code/results/runs").iterdir():
                summary_path = candidate / "run_summary.json"
                if not summary_path.is_file():
                    continue
                candidate_summary = read_json(summary_path)
                if (candidate_summary.get("args", {}).get("experiment_id")
                        == cell["args"]["experiment_id"]
                        and candidate_summary.get("args", {}).get("seed")
                        == cell["seed"]):
                    candidates.append(candidate)
            if len(candidates) != 1:
                raise ValueError(f"run join is not unique: {name}/{cell['index']}")
            run = candidates[0]
            summary = read_json(run / "run_summary.json")
            if summary["status"] != "completed" or summary["global_step"] != 9_999_360:
                raise ValueError(f"training incomplete: {run.name}")
            audit_settings(cell, summary)
            points = json_lines(run / "evaluations.jsonl")
            audit_curve(points, summary, cell["seed"])
            initial_hash = (run / "initial_policy.sha256").read_text().strip()
            if len(initial_hash) != 64 or any(c not in "0123456789abcdef"
                                              for c in initial_hash):
                raise ValueError("invalid initial policy hash")
            block_hashes.add(initial_hash)
            journal = audit_journal(run, summary)
            episode_rows = audit_episodes(run / "episodes.csv", summary)
            cutoff = int(.75 * summary["global_step"])
            row = {
                "replicate": replicate, "seed": cell["seed"],
                "bonus": cell["bonus"], "strategy": cell["strategy"],
                "job_id": f"{job}_{cell['index']}", "run": run.name,
                "initial_policy_sha256": initial_hash,
                "global_step": summary["global_step"],
                "evaluation_points": len(points), "episode_rows": episode_rows,
                "auc_greedy": normalized_auc(points, "success_rate"),
                "tail_auc_greedy": normalized_auc(points, "success_rate", cutoff),
                "final_greedy": points[-1]["success_rate"],
                "success_2m_greedy": next(
                    (p["success_rate"] for p in points
                     if p["global_step"] == 2_000_000 // 1024 * 1024), None),
                "success_5m_greedy": next(
                    (p["success_rate"] for p in points
                     if p["global_step"] == 5_000_000 // 1024 * 1024), None),
                "wall_hours": summary["wall_time_sec"] / 3600,
                **journal,
            }
            rows.append(row)
            curves.extend({
                "replicate": replicate, "seed": cell["seed"],
                "bonus": cell["bonus"], "strategy": cell["strategy"],
                **point,
            } for point in points)
        if len(block_hashes) != 1:
            raise ValueError(f"initial policies differ within seed block {replicate}")

    grouped = []
    by_group = collections.defaultdict(list)
    for row in rows:
        by_group[row["bonus"], row["strategy"]].append(row)
    for (bonus, strategy), members in sorted(by_group.items()):
        grouped.append({
            "bonus": bonus, "strategy": strategy, "n": len(members),
            "seeds": [r["seed"] for r in members],
            **{metric: interval([r[metric] for r in members]) for metric in (
                "auc_greedy", "tail_auc_greedy", "final_greedy",
                "success_2m_greedy", "success_5m_greedy", "queries", "labels",
                "consultations", "http_attempts", "cache_hits", "cached_deliveries",
                "failures", "known_cost_usd", "wall_hours")},
            "query_quartiles_total": np.sum(
                [r["query_quartiles"] for r in members], axis=0).tolist(),
        })

    lookup = {(r["seed"], r["bonus"], r["strategy"]): r for r in rows}
    contrasts = []
    for bonus in ("none", "count"):
        for strategy in ("entropy", "probability20", "exact"):
            for metric in ("auc_greedy", "tail_auc_greedy", "final_greedy",
                           "success_2m_greedy", "success_5m_greedy"):
                differences = [lookup[seed, bonus, strategy][metric]
                               - lookup[seed, bonus, "none"][metric]
                               for seed in sorted({r["seed"] for r in rows})]
                contrasts.append({
                    "bonus": bonus, "contrast": f"{strategy}-none",
                    "metric": metric, "differences": differences,
                    **interval(differences), "planned_n": 5,
                    "complete_planned_batch": False,
                })
        for left, right in (("probability20", "entropy"),
                            ("probability20", "exact"),
                            ("exact", "entropy")):
            differences = [lookup[seed, bonus, left]["auc_greedy"]
                           - lookup[seed, bonus, right]["auc_greedy"]
                           for seed in sorted({r["seed"] for r in rows})]
            contrasts.append({
                "bonus": bonus, "contrast": f"{left}-{right}",
                "metric": "auc_greedy", "differences": differences,
                **interval(differences), "planned_n": 5,
                "complete_planned_batch": False,
            })

    controls = {g["bonus"]: g for g in grouped if g["strategy"] == "none"}
    decisions = {
        "submitted_blocks_complete": True,
        "submitted_cells_validated": len(rows),
        "planned_cells": 40,
        "planned_seed_blocks_complete": False,
        "missing_replicates": [3, 4],
        "no_count_control_final_mean": controls["none"]["final_greedy"]["mean"],
        "count_control_final_mean": controls["count"]["final_greedy"]["mean"],
        "weak_strong_labels_supported_on_three_seeds": (
            controls["none"]["final_greedy"]["mean"] <= .2
            and controls["count"]["final_greedy"]["mean"] >= .8),
        "confirmatory_decisions_withheld": True,
        "independent_review": "pending",
    }
    out = ROOT / "results/reviews" / f"phase1_native_{args.receipt}"
    save(out / "cells.json", rows)
    save(out / "curves.json", curves)
    save(out / "groups.json", grouped)
    save(out / "contrasts.json", contrasts)
    save(out / "decisions.json", decisions)
    save(out / "source_hashes.json", source_hashes)
    print(json.dumps(decisions, indent=2))
    for group in grouped:
        print(group["bonus"], group["strategy"],
              "AUC", round(group["auc_greedy"]["mean"], 4),
              "final", round(group["final_greedy"]["mean"], 4),
              "2M", round(group["success_2m_greedy"]["mean"], 4),
              "labels", round(group["labels"]["mean"], 1),
              "cost", round(group["known_cost_usd"]["mean"], 4))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
