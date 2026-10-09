"""Saved-artifact diagnostics for executable teaching.

This module never starts a simulator, trainer or API client. It reuses the
production rule matcher, validates persisted runs, and makes descriptive
figures. See research/executable_teaching_analysis_protocol_2026-10-02.md.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("dk_confirm", "mr_confirm", "kc_confirm", "kc_mem_confirm",
          "dk16", "kc_s4")
PANELS = {
    "doorkey_8x8": ("conditional_rules_v3_20260928", "v3_20260928"),
    "multiroom_n6": ("conditional_rules_multiroom_20260928", "multiroom_20260928"),
    "keycorridor_s3r3": ("conditional_rules_keycorridor_mem_20260929",
                        "keycorridor_20260928"),
}
NAMES = {"doorkey_8x8": "DoorKey-8", "multiroom_n6": "MultiRoom-N6",
         "keycorridor_s3r3": "KeyCorridor-S3", "doorkey_16x16": "DoorKey-16",
         "keycorridor_s4r3": "KeyCorridor-S4"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha(path):
    """Git's Windows checkout may change only line endings from Linux runs."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_bank(path, expected):
    if not path.exists():
        raise ValueError(f"Trained bank unavailable: {path}")
    if sha(path) == expected:
        return "exact_bytes"
    if lf_sha(path) == expected:
        return "CRLF_to_LF_only"
    raise ValueError(f"Current bank differs from trained bank: {path}")


def rows_csv(path, rows):
    """Keep heterogeneous inventory fields rather than dropping failures."""
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parsed_rules(bank, strip_progress=False):
    """Remove only progress predicates for the label-level counterfactual."""
    result = []
    for item in bank.get("rules", []):
        cond = dict(item["condition"])
        exc = [tuple(x) for x in item["exceptions"]]
        if strip_progress:
            cond = {k: v for k, v in cond.items()
                    if k != "door_unlocked" and not k.startswith("done_")}
            exc = [(k, v) for k, v in exc
                   if k != "door_unlocked" and not k.startswith("done_")]
        if bank["mode"] == "direct":
            exc = []
        result.append((cond, int(item["action"]), tuple(exc)))
    return result


def bank_advice(bank, state, strip_progress=False):
    """Return native matcher output plus the exact rules licensing it."""
    from scripts.conditional_rules_v3 import advise, executable

    if bank["mode"] == "replay":
        action = bank.get("replay", {}).get(state["image"])
        return action, "advised" if action is not None else "no_rule", []
    pred = state.get("pred", state.get("v3"))
    rules = parsed_rules(bank, strip_progress)
    active = list(enumerate(rules))
    if bank.get("filter") == "crafter_preconditions_v1":
        from scripts.crafter_rules_v2_20260930 import applicable
        active = [(i, r) for i, r in active if applicable(r[1], pred)]
    matching = [i for i, r in active if executable(r, pred)]
    action, status = advise([r for _, r in active], pred)
    return action, status, matching


def score_bank(bank, states):
    statuses = Counter()
    correct = 0
    for state in states:
        action, status, _ = bank_advice(bank, state)
        statuses[status] += 1
        correct += int(action is not None and action in state["optimal"])
    n = len(states)
    labels = statuses["advised"]
    return dict(states=n, labels=labels, correct=correct,
                conflicts=statuses["conflict"], no_rule=statuses["no_rule"],
                coverage=labels / n if n else None,
                precision=correct / labels if labels else None)


def paired_estimate(a, b):
    """Training seeds are the replication unit; intervals are descriptive."""
    if len(a) != len(b):
        raise ValueError("Unpaired arrays")
    delta = np.asarray(a, float) - np.asarray(b, float)
    if not np.isfinite(delta).all():
        raise ValueError("Non-finite outcomes")
    n = len(delta)
    if not n:
        return dict(n=0, gain=None, ci_low=None, ci_high=None)
    mean = float(delta.mean())
    half = float(t.ppf(.975, n - 1) * delta.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return dict(n=n, gain=mean,
                ci_low=mean-half if half is not None else None,
                ci_high=mean+half if half is not None else None)


def progress_debug(repo, provenance):
    cases = [
        ("KeyCorridor", "conditional_rules_keycorridor_mem_20260929", "confirm",
         "keycorridor_20260928/scoped_valid.json",
         "keycorridor_mem_20260929/self_checked_pooled_valid.json",
         "door_unlocked", 3),
        ("Crafter", "crafter_rules_v3_20261001", "dev",
         "crafter_v2_20260930/self_checked.json",
         "crafter_v3_20261001/v3b_preconditions.json", "done_place_table", 8),
    ]
    summaries, witnesses = [], []
    for domain, panel_dir, split, old_path, new_path, flag, action_id in cases:
        # Crafter's action names come from the native task mapping.
        if domain == "Crafter":
            from scripts.crafter_rules_pilot_20260928 import ACTIONS
            action_id = ACTIONS.index("place_table")
        panel = repo / "results" / panel_dir / "panels.json"
        states = read(panel)[split]
        provenance[str(panel)] = sha(panel)
        paths = [repo / "research/rule_banks" / p for p in (old_path, new_path)]
        for path in paths:
            provenance[str(path)] = sha(path)
        banks = [read(p) for p in paths]
        variants = [("base", banks[0], False), ("progress", banks[1], False),
                    ("progress_stripped", banks[1], True)]
        all_outputs = {}
        post = [s for s in states if s["pred"][flag] == "yes"]
        for label, bank, stripped in variants:
            outputs = [bank_advice(bank, s, stripped) for s in states]
            all_outputs[label] = outputs
            for subset_name, indices in [
                ("all", range(len(states))),
                ("after_progress", [i for i, s in enumerate(states)
                                    if s["pred"][flag] == "yes"]),
            ]:
                indices = list(indices)
                statuses = Counter(outputs[i][1] for i in indices)
                repeated = sum(
                    states[i]["pred"][flag] == "yes" and outputs[i][0] == action_id
                    and (domain != "KeyCorridor" or states[i]["pred"]["front"] == "key")
                    for i in indices)
                summaries.append(dict(
                    domain=domain, panel=split, subset=subset_name, variant=label,
                    states=len(indices), labelled=statuses["advised"],
                    conflict=statuses["conflict"], no_rule=statuses["no_rule"],
                    repeated_subgoal_labels=repeated,
                    rate_per_state=repeated/len(indices) if indices else None))
        # Deterministic diagnostic witness; it is illustrative, never a random
        # sample or a causal test of downstream learning.
        candidates = [i for i, s in enumerate(states)
                      if s["pred"][flag] == "yes"
                      and all_outputs["base"][i][0] == action_id
                      and all_outputs["progress"][i][0] != action_id
                      and (domain != "KeyCorridor" or s["pred"]["front"] == "key")]
        if not candidates:
            witnesses.append(dict(domain=domain, status="no_witness",
                                  post_progress_states=len(post)))
            continue
        i = candidates[0]
        details = {}
        for label, bank, stripped in variants:
            action, status, matching = all_outputs[label][i]
            rules = parsed_rules(bank, stripped)
            details[label] = dict(action=action, status=status,
                                 matching_rule_indices=matching,
                                 matching_rules=[rules[j] for j in matching])
        # Show the progress-fact rules that would fire without their progress clauses,
        # so an abstention can be inspected as concretely as an emitted label.
        progress_rules = parsed_rules(banks[1])
        blocked = set(details["progress_stripped"]["matching_rule_indices"]) - set(
            details["progress"]["matching_rule_indices"])
        details["progress"]["progress_blocked_rules"] = [
            dict(index=j, rule=progress_rules[j]) for j in sorted(blocked)]
        witnesses.append(dict(domain=domain, panel=split, index=i,
                              source=str(panel), state=states[i], outputs=details,
                              selection="first base-bank repeated-subgoal label suppressed by progress clauses",
                              banks={str(p): sha(p) for p in paths}))
    return summaries, witnesses


def collect_runs(sync, repo, provenance):
    """Only completed, contract-consistent cells contribute to estimates."""
    from scripts.run_rule_bank_pilot_20260928 import validate_run
    inventory, runs = [], []
    for suite in SUITES:
        batch = sync / suite
        manifest_path = batch / "manifest.json"
        if not manifest_path.exists():
            inventory.append(dict(suite=suite, status="missing_manifest"))
            continue
        provenance[str(manifest_path)] = sha(manifest_path)
        sidecar = batch / "manifest.sha256"
        if sidecar.exists() and sidecar.read_text().strip().split()[0] != sha(manifest_path):
            raise ValueError(f"Manifest hash mismatch: {batch}")
        manifest = read(manifest_path)
        for cell in manifest["cells"]:
            status_row = dict(suite=suite, index=cell["index"], task=cell["task"],
                              bonus=cell["bonus"], arm=cell["arm"], seed=cell["seed"])
            exit_path = batch / "cells" / str(cell["index"]) / "exit.json"
            if not exit_path.exists():
                inventory.append(dict(status_row, status="no_terminal_record"))
                continue
            saved = read(exit_path)
            provenance[str(exit_path)] = sha(exit_path)
            if saved.get("returncode") != 0 or saved.get("artifact_status") != "terminal_contract_validated":
                inventory.append(dict(status_row, status="failed_or_unvalidated"))
                continue
            if len(saved["runs"]) != 1:
                raise ValueError(f"Ambiguous run identity: {exit_path}")
            run = (batch / saved["runs"][0]).resolve()
            if not run.is_relative_to(batch.resolve()):
                raise ValueError(f"Run escaped batch: {run}")
            metrics = validate_run(run, cell["args"])
            if not np.isclose(metrics["auc"], saved["metrics"]["auc"], atol=1e-12, rtol=0):
                raise ValueError(f"Worker AUC mismatch: {run}")
            if metrics["initial_sha256"] != saved["metrics"]["initial_sha256"]:
                raise ValueError(f"Initial identity mismatch: {run}")
            for name in ("run_summary.json", "evaluations.jsonl", "initial_policy.sha256"):
                provenance[str(run / name)] = sha(run / name)
            summary = read(run / "run_summary.json")
            args = summary["args"]
            bank = args.get("rule_bank", "")
            bank_hash = args.get("rule_bank_sha256", "")
            bank_identity = "not_applicable"
            if bank:
                current = repo / bank
                bank_identity = verify_bank(current, bank_hash)
                provenance[str(current)] = sha(current)
            latest = summary["latest"]
            advising = latest["advising"]
            total = summary["global_step"]
            checks, labels = advising["num_asked"], advising["num_delivered"]
            if not 0 <= labels <= checks <= total:
                raise ValueError(f"Impossible exposure counts: {run}")
            evaluations = [json.loads(line) for line in
                           (run / "evaluations.jsonl").read_text().splitlines()]
            runs.append(dict(
                **status_row, status="validated", bank=bank, bank_sha256=bank_hash,
                bank_identity=bank_identity,
                weight=args.get("distill_coef_start"), total_steps=total,
                labels=labels, checks=checks, labels_per_step=labels/total,
                labels_per_check=labels/checks if checks else None,
                first_label=latest.get("advisor_control", {}).get("first_label_global_step"),
                auc=metrics["auc"], final=metrics["final"],
                initial_sha256=metrics["initial_sha256"], run=str(run),
                args=args, evaluations=evaluations))
            inventory.append(dict(status_row, status="validated"))
    return inventory, runs


def learning_contrasts(runs, inventory):
    controls = {}
    for run in runs:
        if run["arm"] == "none":
            key = run["task"], run["bonus"], run["seed"]
            if key in controls:
                raise ValueError(f"Ambiguous control: {key}")
            controls[key] = run
    grouped = defaultdict(list)
    for run in runs:
        if run["bank"]:
            grouped[run["suite"], run["task"], run["bonus"],
                    run["arm"], run["bank"]].append(run)
    contrasts = []
    # Paired cells must share the actual learner, environment and evaluation
    # configuration. Only treatment/identity fields may intentionally differ.
    learner_fields = ("learning_rate", "anneal_lr", "gamma", "gae_lambda",
                      "num_envs", "num_steps", "update_epochs", "num_minibatches",
                      "ent_coef", "vf_coef", "clip_coef", "clip_vloss",
                      "max_grad_norm", "recurrent", "agent_view_size", "bonus",
                      "count_observation", "eval_episodes", "eval_sampled",
                      "obs_mode", "obs_target_size", "obs_tile_size", "norm_adv",
                      "int_coef", "int_gamma", "norm_int_reward", "dual_value",
                      "target_kl", "total_timesteps")
    for (suite, task, bonus, arm, bank), group in sorted(grouped.items()):
        if len({r["seed"] for r in group}) != len(group):
            raise ValueError("Duplicate taught seed within comparison")
        if len({r["weight"] for r in group}) != 1 or len({
                r["bank_sha256"] for r in group}) != 1:
            raise ValueError("Mixed teaching weights or bank identities within comparison")
        left, right, seeds = [], [], []
        for run in sorted(group, key=lambda r: r["seed"]):
            control = controls.get((task, bonus, run["seed"]))
            if control is None:
                continue
            if run["initial_sha256"] != control["initial_sha256"]:
                raise ValueError("Paired initial weights differ")
            if any(run["args"].get(k) != control["args"].get(k) for k in learner_fields):
                raise ValueError("Paired learner configuration differs")
            if [e["global_step"] for e in run["evaluations"]] != [
                    e["global_step"] for e in control["evaluations"]]:
                raise ValueError("Paired evaluation grids differ")
            left.append(run["auc"])
            right.append(control["auc"])
            seeds.append(run["seed"])
        expected = sum(r.get("suite") == suite and r.get("task") == task
                       and r.get("bonus") == bonus and r.get("arm") == arm
                       for r in inventory)
        contrasts.append(dict(suite=suite, task=task, bonus=bonus, arm=arm,
                              bank=bank, bank_sha256=group[0]["bank_sha256"],
                              weight=group[0]["weight"], expected=expected,
                              interim=len(seeds) != expected,
                              seeds=",".join(map(str, seeds)),
                              **paired_estimate(left, right)))
    return contrasts


def bank_quality(repo, provenance):
    rows = []
    for task, (panel_dir, bank_dir) in PANELS.items():
        panel = repo / "results" / panel_dir / "panels.json"
        provenance[str(panel)] = sha(panel)
        states = read(panel)["confirm"]
        folders = [bank_dir]
        if task == "keycorridor_s3r3":
            folders.append("keycorridor_mem_20260929")
        for folder in folders:
            for path in sorted((repo / "research/rule_banks" / folder).glob("*.json")):
                bank = read(path)
                provenance[str(path)] = sha(path)
                rows.append(dict(task=task, bank=path.relative_to(repo).as_posix(),
                                 bank_sha256=lf_sha(path), local_sha256=sha(path), panel=str(panel),
                                 rules=len(bank.get("rules", [])),
                                 **score_bank(bank, states)))
    return rows


def write_figures(out, debug, witnesses, runs, contrasts, quality):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    figures = []
    def rule_text(rule, domain):
        condition, action, exceptions = rule
        if domain == "Crafter":
            from scripts.crafter_rules_pilot_20260928 import ACTIONS
            action_name = ACTIONS[action]
        else:
            from minigrid.core.actions import Actions
            action_name = Actions(action).name
        return ("WHEN " + ", ".join(f"{k}={v}" for k, v in condition.items())
                + f" -> {action_name}" + (" UNLESS " + ", ".join(
                    f"{k}={v}" for k, v in exceptions) if exceptions else ""))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, domain in zip(axes, ("KeyCorridor", "Crafter")):
        data = [r for r in debug if r["domain"] == domain and r["subset"] == "after_progress"]
        ax.bar(range(3), [r["rate_per_state"] or 0 for r in data],
               color=["#b04e46", "#297f78", "#8a8e91"])
        ax.set_xticks(range(3), ["Without progress\nfact", "With progress\nfact", "Progress clauses\nomitted"])
        ax.set_title(f"{domain}: {data[0]['states']} saved post-progress states")
        ax.set_ylabel("Repeated-subgoal labels / state")
        for i, row in enumerate(data):
            ax.text(i, row["rate_per_state"] or 0,
                    f" {row['repeated_subgoal_labels']}/{row['states']}", ha="center", va="bottom")
        ax.margins(y=.3)
    fig.suptitle("Progress facts in executable teaching explanations\nLabel counterfactuals on fixed saved panels; no new learning experiment", fontsize=12)
    figures.append(("progress_labels", fig))

    # Each witness shows the unchanged recorded predicates and exact matched
    # rule text, rather than an invented before/after environment rendering.
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    for ax, witness in zip(axes, witnesses):
        ax.axis("off")
        ax.set_title(f"{witness['domain']}: an inspectable teaching decision", loc="left")
        if "outputs" not in witness:
            ax.text(0, .9, "No qualifying saved witness.", va="top")
            continue
        parts = [f"Saved {witness['panel']} panel, state index {witness['index']}."]
        keys = set()
        for record in witness["outputs"].values():
            for cond, _, exc in record["matching_rules"]:
                keys.update(cond)
                keys.update(k for k, _ in exc)
        keys.update(k for k in witness["state"]["pred"]
                    if k == "door_unlocked" or k == "done_place_table")
        parts.append("Observed predicates / recorded progress: " + ", ".join(
            f"{k}={witness['state']['pred'][k]}" for k in sorted(keys)))
        for label, record in witness["outputs"].items():
            text = "; ".join(dict.fromkeys(rule_text(r, witness["domain"])
                                         for r in record["matching_rules"]))
            parts.append(f"{label}: {record['status']}; action={record['action']}. {text}")
            for blocked in record.get("progress_blocked_rules", []):
                parts.append("Clause blocked by recorded progress: "
                             + rule_text(blocked["rule"], witness["domain"]))
        ax.text(0, .94, "\n\n".join(textwrap.fill(s, 116) for s in parts),
                va="top", fontsize=9, transform=ax.transAxes)
    figures.append(("progress_witnesses", fig))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    tasks = list(NAMES)
    for ax, task in zip(axes.flat, tasks):
        selected = [r for r in runs if r["task"] == task
                    and r["arm"] in ("rules_weak", "rules_mem_weak")]
        if task == "keycorridor_s3r3":
            selected = [r for r in selected if r["arm"] == "rules_mem_weak"]
        for x, bonus in enumerate(("none", "count")):
            data = [r for r in selected if r["bonus"] == bonus]
            if not data:
                continue
            y = np.array([r["labels_per_step"] for r in data])
            ax.bar(x, y.mean(), color=["#4778a8", "#297f78"][x], alpha=.5)
            ax.scatter(x + np.linspace(-.08, .08, len(y)), y, s=18, color="black")
            ax.text(x, max(y)+.015, f"n={len(y)}", ha="center")
        ax.set_title(NAMES[task] + (" (different banks)" if task == "multiroom_n6" else ""))
        ax.set_xticks([0, 1], ["Plain PPO", "Count PPO"])
        ax.set_ylabel("Accepted labels / training transition")
        ax.set_ylim(0, .85)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, .9, "Endpoint counts; weight 0.1.\nDots: completed training seeds.\nDoorKey-16 plain: interim (9/10).\nNo advice time series was synced.\nKeyCorridor uses the progress-fact bank.\nMultiRoom: plain=scoped, count=strict.\nThese are not causal exposure tests.", va="top", fontsize=9)
    fig.suptitle("How much teaching reaches each student?", fontsize=13)
    figures.append(("advice_exposure", fig))

    # Keep learning and exposure on the same documented cohort; never add an
    # unmeasured step-zero point. These are greedy, teacher-free evaluations.
    fig, axes = plt.subplots(5, 2, figsize=(11, 13), layout="constrained")
    for axes_row, task in zip(axes, tasks):
        arm = "rules_mem_weak" if task.startswith("keycorridor") else "rules_weak"
        for ax, bonus in zip(axes_row, ("none", "count")):
            contrast = next((c for c in contrasts if c["task"] == task
                             and c["bonus"] == bonus and c["arm"] == arm), None)
            if contrast is None or not contrast["n"]:
                ax.set_title(f"{NAMES[task]}: no complete pairs")
                continue
            seeds = {int(s) for s in contrast["seeds"].split(",")}
            for method, color, label in (("none", "#72777d", "No teaching"),
                                          (arm, "#297f78", "Executable advice")):
                selected = [r for r in runs if r["task"] == task and r["bonus"] == bonus
                            and r["arm"] == method and r["seed"] in seeds]
                x = np.array([e["global_step"] for e in selected[0]["evaluations"]]) / 1e6
                y = np.array([[e["success_rate"] for e in r["evaluations"]] for r in selected])
                mean = y.mean(axis=0)
                ax.plot(x, mean, color=color, label=label)
                if len(y) > 1:
                    half = t.ppf(.975, len(y)-1) * y.std(axis=0, ddof=1) / np.sqrt(len(y))
                    ax.fill_between(x, np.maximum(0, mean-half), np.minimum(1, mean+half),
                                    color=color, alpha=.15)
            student = "Plain PPO" if bonus == "none" else "Count PPO"
            interim = "; interim" if contrast["interim"] else ""
            ax.set_title(f"{NAMES[task]}, {student}: {len(seeds)} paired seeds{interim}")
            ax.set_ylim(-.03, 1.06)
            ax.set_xlim(0, 5)
            ax.set_ylabel("Greedy success rate")
            ax.set_xlabel("Training transitions (millions)")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Teacher-free learning for the exposure cohorts\nMeans and pointwise 95% seed-level t intervals (clipped to [0, 1]); first evaluation after training", fontsize=11)
    figures.append(("teacher_free_learning", fig))

    # Quality is measured on source confirmation panels, so only source-task
    # learning comparisons can be joined without inventing target quality.
    qmap = {(r["task"], r["bank_sha256"]): r for r in quality}
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    bank_styles = {
        "v3_20260928": ("#4778a8", "o", "DoorKey"),
        "multiroom_20260928": ("#a06615", "^", "MultiRoom"),
        "keycorridor_20260928": ("#b04e46", "s", "KeyCorridor (no progress fact)"),
        "keycorridor_mem_20260929": ("#297f78", "D", "KeyCorridor (progress fact)"),
    }
    for row in contrasts:
        q = qmap.get((row["task"], row["bank_sha256"]))
        if q is None or row["gain"] is None:
            continue
        axes_row = axes[0 if row["bonus"] == "none" else 1]
        color, marker, _ = bank_styles[Path(row["bank"]).parent.name]
        for ax, metric in zip(axes_row, ("precision", "coverage")):
            if q[metric] is None:
                continue
            ax.errorbar(q[metric], row["gain"],
                        yerr=[[row["gain"]-row["ci_low"]], [row["ci_high"]-row["gain"]]]
                        if row["ci_low"] is not None else None,
                        fmt=marker, color=color,
                        markerfacecolor=color if row["weight"] == .1 else "white",
                        alpha=.65, capsize=2)
    for axes_row, student in zip(axes, ("Plain PPO", "Count PPO")):
        for ax, metric in zip(axes_row, ("precision", "coverage")):
            ax.axhline(0, color="gray", lw=.7)
            ax.set_xlabel(f"Saved source-panel {metric}")
            ax.set_ylabel("Teacher-free AUC gain")
            ax.set_title(student)
            ax.set_xlim(.70 if metric == "precision" else 0, 1.02)
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", label=n)
               for c, m, n in bank_styles.values()]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, fontsize=8)
    fig.suptitle("Bank quality and learning: descriptive comparisons\nFilled: imitation weight 0.1; hollow: 1.0. Paired 95% t intervals; reused banks/controls.", fontsize=11)
    figures.append(("bank_quality_gain", fig))
    with PdfPages(out / "teaching_diagnostics.pdf") as pdf:
        for name, fig in figures:
            fig.savefig(out / f"{name}.png", dpi=170)
            fig.savefig(out / f"{name}.pdf")
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--sync", type=Path, default=ROOT.parent /
                        "vlm-rl-bench/results/vulcan_sync/data/fix_wave/fix_wave_20260929_v1")
    parser.add_argument("--out", type=Path, default=ROOT /
                        "docs/assets/executable_teaching_2026-10-02")
    args = parser.parse_args()
    repo, sync, out = args.repo.resolve(), args.sync.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance = {str(Path(__file__).resolve()): sha(__file__)}
    for name in ("scripts/conditional_rules_v3.py", "scripts/crafter_rules_v2_20260930.py",
                 "scripts/crafter_rules_pilot_20260928.py", "teachers/minigrid/rule_bank.py",
                 "scripts/run_rule_bank_pilot_20260928.py",
                 "research/executable_teaching_analysis_protocol_2026-10-02.md"):
        provenance[str(repo / name)] = sha(repo / name)
    debug, witnesses = progress_debug(repo, provenance)
    inventory, runs = collect_runs(sync, repo, provenance)
    contrasts = learning_contrasts(runs, inventory)
    quality = bank_quality(repo, provenance)
    public_runs = [{k: v for k, v in r.items() if k not in ("args", "evaluations")}
                   for r in runs]
    for filename, rows in [
        ("progress_labels.csv", debug), ("run_inventory.csv", inventory),
        ("advice_exposure.csv", public_runs), ("paired_learning.csv", contrasts),
        ("bank_quality.csv", quality),
    ]:
        rows_csv(out / filename, rows)
    result = dict(
        author="mahdiehmn", date="2026-10-02", protocol="retrospective descriptive",
        source_hashes=provenance, inventory_counts=dict(Counter(r["status"] for r in inventory)),
        progress_counterfactuals=debug, witnesses=witnesses, contrasts=contrasts,
        bank_count=len(quality), advice_time_series_available=False,
        independent_review="pending",
        limitations=["Fixed saved state panels; no label-to-learning causal attribution.",
                     "MultiRoom students use different banks; exposure differences do not isolate exploration.",
                     "Source-panel quality is not target-map or on-policy quality.",
                     "Paired intervals are descriptive, not new confirmatory tests.",
                     "Incomplete cells are retained in inventory; partial comparisons are interim.",
                     "Crafter effectiveness is not MiniGrid optimal-action precision."])
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    write_figures(out, debug, witnesses, runs, contrasts, quality)
    print(json.dumps(dict(output=str(out), inventory=result["inventory_counts"],
                          progress_counterfactuals=debug, bank_count=len(quality)), indent=2))


if __name__ == "__main__":
    main()
