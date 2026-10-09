"""Aleph-only action/consequence check on saved student states; no RL training.

Simulator answers are scoring evidence only.
"""

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import shlex
import subprocess
import time
import zipfile
from importlib.metadata import version

import jsonschema

from scripts import explanation_screen_panel as physics
from scripts import run_explanation_screen as common
from teachers.minigrid.llm_general import render_ascii_map
from envs.state import extract_generic_state

ROOT = Path(__file__).resolve().parents[1]
STUDY = "consequence_readiness_20260918_v1"
RUNNER = "scripts/run_consequence_readiness.py"
WORKER = "scripts/submit_consequence_readiness.sh"
MODELS = ("qwen38-27b", "gpt-oss-120b")
EFFECTS = {
    "movement": ["stay", "forward"],
    "inventory": ["same", "picked_up", "dropped"],
    "front_door": ["same", "opened", "closed"],
    "goal_reached": [False, True],
}
CONTRACT = dict(
    study=STUDY,
    models=list(MODELS),
    cases=64,
    max_attempts=1,
    sdk_retries=0,
    timeout_seconds=180,
    max_output_tokens=8192,
    deadline_seconds=3 * 3600,
    reasoning_effort="low",
    endpoint=common.ENDPOINT,
    minimum_valid=61,
    minimum_exact_pairs=58,
    minimum_recommended_informative_states=16,
)


def response_schema():
    """Use relative local effects; never predict the state of a distant door."""
    effect = physics.response_schema("contrastive")["properties"]["reference"]
    effect = dict(
        effect,
        properties={
            k: {
                "type": "boolean" if k == "goal_reached" else "string",
                "enum": v,
            }
            for k, v in EFFECTS.items()
        },
        required=list(EFFECTS),
    )
    fields = dict(
        action={"type": "integer", "minimum": 0, "maximum": 5},
        recommended=effect,
        student=effect,
        why_recommended={"type": "string", "minLength": 1, "maxLength": 500},
        why_student={"type": "string", "minLength": 1, "maxLength": 500},
    )
    return dict(
        type="object",
        properties=fields,
        required=list(fields),
        additionalProperties=False,
    )


def effect_truth(state, action, env_id=None):
    """One copied environment step scores outcomes without choosing actions.

    `env_id` selects the generic restorer for tasks other than DoorKey
    8x8. Left None, the DoorKey screen's own restorer runs exactly as
    before, including its stricter layout and inventory checks.
    """
    if env_id is None:
        env = physics.restore(state)
    else:
        from envs.physics import restore as restore_any
        env = restore_any(env_id, state)
    try:
        old_pos, old_inventory = tuple(env.agent_pos), env.carrying
        front = env.grid.get(*env.front_pos)
        # Grid.decode restores NumPy booleans; normalize before identity tests.
        door_open = (
            bool(front.is_open) if front and front.type == "door" else None
        )
        _, reward, _, _, _ = env.step(action)
        return dict(
            movement="forward" if tuple(env.agent_pos) != old_pos else "stay",
            inventory=(
                "picked_up"
                if old_inventory is None and env.carrying is not None
                else "dropped"
                if old_inventory is not None and env.carrying is None
                else "same"
            ),
            front_door=(
                "opened"
                if door_open is False and front.is_open
                else "closed"
                if door_open is True and not front.is_open
                else "same"
            ),
            goal_reached=bool(reward > 0),
        )
    finally:
        env.close()


def component_mask(action):
    """Exclude action-determined rotations/no-ops from semantic supervision."""
    return dict(
        movement=action == 2,
        inventory=action in (3, 4),
        front_door=action == 5,
        goal_reached=action == 2,
    )


def make_request(case):
    """Same prompt for both models; no reference action or solver output."""
    env = physics.restore(case["state"])
    try:
        state = case["state"]
        prompt = (
            "In MiniGrid-DoorKey-8x8-v0 choose your own next action to solve the mission. "
            "Then explain only immediate effects of that action and of the supplied "
            "student action. These are independent one-step branches, not a sequence. "
            "Do not write future routes, unseen facts, or assume your action is optimal. "
            "If you agree with the student, report both identical branches honestly.\n"
            "The student sees a local7x7 symbolic view; you see the full map. "
            "Effects concern only the current agent, carried object and door immediately "
            "in front. A distant door must give front_door=same. "
            "Coordinates x east/right,y south/down; direction0east,1south,2west,3north.\n"
            f"Mission: {env.mission}\nPosition: {state['agent_pos']}; "
            f"direction: {state['agent_dir']}; carried: {state['carrying']}\n"
            f"{render_ascii_map(env)}\n"
            f"Current objects (type,color,x,y,state): {extract_generic_state(env)[-1]}\n"
            "Actions0LEFT,1RIGHT rotate in place;2FORWARD moves one cell only onto "
            "floor, open door or goal;3PICKUP needs an empty hand and key in front; "
            "4DROP needs a held key and empty floor in front;5TOGGLE affects only "
            "a front door: matching key unlocks AND opens a locked door; an unlocked "
            "door switches open/closed.6DONE has no effect (student only). "
            "Keys/walls/closed doors block forward. Goal success requires entering "
            "the goal tile. Ignore episode time limits. "
            "Movement is stay/forward; inventory reports the immediate change; "
            "front_door is same/opened/closed, not the absolute door state.\n"
            f"Student action: {case['student_action']}. Choose an action0..5. "
            "Return JSON with action,recommended,student,why_recommended,why_student."
        )
    finally:
        env.close()
    return dict(
        input=[dict(role="user", content=prompt)],
        text=dict(
            format=dict(
                type="json_schema",
                name="action_consequences",
                strict=True,
                schema=response_schema(),
            )
        ),
    )


def panel(corpus, screens):
    """Freeze fresh worlds, balanced backgrounds, at most two per episode."""
    candidates, sources = physics.read_candidates(corpus)
    excluded = set()
    for name in (
        "aleph_explanation_screen_20260915_v1",
        "aleph_explanation_screen_20260915_v2",
    ):
        if not (Path(screens) / name / "panel.json").is_file():
            raise ValueError(
                f"Missing previous-panel exclusion evidence: {name}"
            )
    for path in sorted(Path(screens).glob("*/panel.json")):
        if path.parent.name == STUDY:
            continue
        old = json.loads(path.read_text("utf-8"))
        excluded.update(c["world_id"] for c in old["cases"])
        sources[str(path)] = physics.file_hash(path)
    selected, episodes, counts = [], Counter(), Counter()
    # Include rare real interactions before filling quotas by state hash.
    ranked = sorted(
        candidates,
        key=lambda r: (
            physics.object_hash([STUDY, r["world_id"]]),
            r["sample_id"],
        ),
    )
    priority = []
    for stratum in (
        "closed_unlocked_front",
        "goal_front",
        "unlock_front",
        "key_front",
    ):
        worlds = set()
        for row in ranked:
            if (
                row["stratum"] == stratum
                and row["world_id"] not in excluded | worlds
            ):
                priority.append(row)
                worlds.add(row["world_id"])
                if len(worlds) == 2:
                    break
        if not worlds:
            raise ValueError(f"No unseen saved critical state: {stratum}")
    for row in priority + ranked:
        background = "count" if "_g1_count_" in row["sample_id"] else "none"
        episode = tuple(row["episode_group"])
        if (
            row["world_id"] in excluded
            or counts[background] >= 32
            or episodes[episode] >= 2
        ):
            continue
        source = Path(corpus) / row["source"]
        raw = json.loads(
            source.read_text("utf-8").splitlines()[row["source_line"] - 1]
        )
        if raw["student_action"] != raw["executed_action"]:
            raise ValueError(
                "Recorded alternative was not the executed student action"
            )
        row = {
            k: row[k]
            for k in (
                "world_id",
                "state",
                "phase",
                "episode_group",
                "sample_id",
                "source",
                "source_line",
                "student_action",
                "stratum",
            )
        }
        row.update(case_id=f"C{len(selected):02}", background=background)
        row["truth"] = {
            str(a): effect_truth(row["state"], a) for a in range(7)
        }
        selected.append(row)
        excluded.add(row["world_id"])
        episodes[episode] += 1
        counts[background] += 1
    if len(selected) != 64 or counts != {"none": 32, "count": 32}:
        raise ValueError(
            "Insufficient fresh saved states; do not silently rebalance"
        )
    if not {
        "closed_unlocked_front",
        "goal_front",
        "unlock_front",
        "key_front",
    } <= {c["stratum"] for c in selected}:
        raise ValueError(
            "Episode constraints excluded a required critical situation"
        )
    coverage = Counter(
        f"{c['background']}/{c['phase']}/{c['stratum']}" for c in selected
    )
    return dict(
        cases=selected,
        coverage=dict(coverage),
        sources=sources,
        description="Saved treatment-policy query states; descriptive development panel",
    )


def prepare(root, corpus, credentials):
    """Write a committed snapshot and frozen requests without calling a model."""
    batch = root / "results/explanation_screens" / STUDY
    if batch.exists():
        raise FileExistsError(f"Already prepared: {batch}; do not resubmit")
    common.credentials(credentials)
    frozen = panel(corpus, root / "results/explanation_screens")
    requests = [
        dict(case_id=c["case_id"], request=make_request(c))
        for c in frozen["cases"]
    ]
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=root, check=True
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    archive = subprocess.check_output(
        ["git", "archive", "--format=zip", commit], cwd=root
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        if not {RUNNER, WORKER} <= set(z.namelist()):
            raise ValueError("Commit the readiness code before preparing")
        batch.mkdir(parents=True)
        z.extractall(batch / "code")
    common.write_json(batch / "panel.json", frozen)
    common.write_json(batch / "requests.json", requests)
    manifest = dict(
        **CONTRACT,
        commit=commit,
        credential_file=str(credentials.resolve()),
        packages={
            name: version(name)
            for name in ("minigrid", "gymnasium", "jsonschema", "openai")
        },
        artifact_sha256={
            name: physics.file_hash(batch / name)
            for name in ("panel.json", "requests.json")
        },
        source_sha256={
            p.relative_to(batch / "code").as_posix(): physics.file_hash(p)
            for p in (batch / "code").rglob("*")
            if p.is_file()
        },
    )
    common.write_json(batch / "manifest.json", manifest)
    (batch / "manifest.sha256").write_text(
        physics.file_hash(batch / "manifest.json") + "\n"
    )
    (batch / "slurm").mkdir()
    (batch / "workers").mkdir()
    return batch


def verify(batch):
    """Fail before access if the contract, questions or source changed."""
    m = json.loads((batch / "manifest.json").read_text("utf-8"))
    if any(m.get(k) != v for k, v in CONTRACT.items()):
        raise ValueError("Readiness contract changed")
    if any(version(k) != v for k, v in m["packages"].items()):
        raise ValueError("Packages changed since frozen preparation")
    if (
        physics.file_hash(batch / "manifest.json")
        != (batch / "manifest.sha256").read_text().strip()
    ):
        raise ValueError("Manifest changed")
    for field, root in (
        ("artifact_sha256", batch),
        ("source_sha256", batch / "code"),
    ):
        for name, expected in m[field].items():
            if physics.file_hash(root / name) != expected:
                raise ValueError(f"Changed artifact: {name}")
    frozen = json.loads((batch / "panel.json").read_text("utf-8"))
    requests = json.loads((batch / "requests.json").read_text("utf-8"))
    if requests != [
        dict(case_id=c["case_id"], request=make_request(c))
        for c in frozen["cases"]
    ]:
        raise ValueError("Requests do not match saved states")
    return m, frozen, requests


def identity(item, index):
    """Bind a record to the complete requested condition, not just its prompt."""
    envelope = dict(
        request=item["request"],
        model=MODELS[index],
        effort=CONTRACT["reasoning_effort"],
        endpoint=CONTRACT["endpoint"],
        max_output_tokens=CONTRACT["max_output_tokens"],
        timeout_seconds=CONTRACT["timeout_seconds"],
        max_attempts=1,
        sdk_retries=0,
    )
    return dict(
        case_id=item["case_id"],
        requested_model=MODELS[index],
        request_sha256=physics.object_hash(envelope),
    )


def collect(batch, index):
    """One attempt per item; persist starts so interruption is not hidden."""
    from teachers.aleph import AlephClient
    from teachers.reliability import failure_details

    m, _, requests = verify(batch)
    if index not in range(2):
        raise ValueError("Worker must be 0 or 1")
    destination = batch / "workers" / str(index)
    destination.mkdir()  # An existing worker is never silently retried.
    client = AlephClient(
        api_key=common.credentials(Path(m["credential_file"])),
        base_url=m["endpoint"],
        timeout=180,
        max_retries=0,
    )
    started = time.monotonic()
    consecutive = 0
    try:
        for item in requests:
            if time.monotonic() - started >= m["deadline_seconds"]:
                break
            identified = identity(item, index)
            common.append_row(
                destination / "attempts.jsonl",
                dict(event="start", **identified, attempt=1),
            )
            row = dict(**identified, tokens_in=None, tokens_out=None)
            t0 = time.monotonic()
            try:
                response = client.teacher_response(
                    model=MODELS[index],
                    **item["request"],
                    reasoning={"effort": "low"},
                    max_output_tokens=8192,
                )
                row.update(
                    raw_output=response.output_text,
                    served_model=getattr(response, "model", None),
                    response_id=getattr(response, "response_id", None),
                )
                usage = response.usage
                if usage is not None:
                    row.update(
                        tokens_in=usage.input_tokens,
                        tokens_out=usage.output_tokens,
                    )
                answer = json.loads(response.output_text)
                jsonschema.validate(answer, response_schema())
                row.update(status="valid", answer=answer)
                consecutive = 0
            except Exception as exc:
                error = failure_details(exc)
                error.pop("message", None)
                row.update(status="failed", error=error)
                consecutive += 1
            row["seconds"] = time.monotonic() - t0
            common.append_row(destination / "responses.jsonl", row)
            common.append_row(
                destination / "attempts.jsonl",
                dict(
                    event="end", **identified, attempt=1, status=row["status"]
                ),
            )
            print(
                f"{MODELS[index]} {item['case_id']}: {row['status']}",
                flush=True,
            )
            if consecutive >= 3 or row.get("error", {}).get("http_status") in (
                400,
                401,
                403,
                404,
            ):
                break
    finally:
        client.close()
    (destination / "DONE").write_text(
        "collection ended; inspect completeness\n"
    )
    rows = common.load_rows(destination / "responses.jsonl")
    return int(len(rows) != 64 or any(r["status"] != "valid" for r in rows))


def score(frozen, rows):
    """Report both branches and variation; model-selected easy actions cannot hide."""
    cases = {c["case_id"]: c for c in frozen["cases"]}
    seen = set()
    n = Counter()
    actions = Counter()
    errors = Counter()
    branch_coverage = Counter()
    donors = {}
    for r in rows:
        cid = r["case_id"]
        if cid in seen or cid not in cases:
            raise ValueError("Repeated/unknown case")
        seen.add(cid)
        if r["status"] != "valid":
            continue
        c = cases[cid]
        a = json.loads(r["raw_output"])
        jsonschema.validate(a, response_schema())
        if a != r["answer"]:
            raise ValueError("Raw and parsed answers disagree")
        n["valid"] += 1
        actions[a["action"]] += 1
        n["recommended_informative_states"] += any(
            component_mask(a["action"]).values()
        )
        n["agreement"] += a["action"] == c["student_action"]
        exact = []
        informative = False
        for branch, act in [
            ("recommended", a["action"]),
            ("student", c["student_action"]),
        ]:
            branch_coverage[
                f"{branch}/{c.get('stratum', 'unknown')}/action{act}"
            ] += 1
            truth = effect_truth(c["state"], act)
            if truth != c["truth"][str(act)]:
                raise ValueError("Simulator truth changed")
            ok = a[branch] == truth
            exact.append(ok)
            n[f"{branch}_exact"] += ok
            mask = component_mask(act)
            informative |= any(mask.values())
            if any(mask.values()):
                group = (c["phase"], act)
                encoded = tuple(a[branch][k] for k in EFFECTS if mask[k])
                donors.setdefault(group, []).append((cid, encoded))
            for field in EFFECTS:
                if mask[field]:
                    n["components"] += 1
                    n["correct_components"] += a[branch][field] == truth[field]
                errors[f"{branch}/{field}"] += a[branch][field] != truth[field]
        n["exact_pairs"] += all(exact)
        n["informative_states"] += informative
    # These are opportunities across the panel, not measured rollout donors.
    eligible = changed = total = 0
    for members in donors.values():
        for cid, target in members:
            others = [t for other, t in members if other != cid]
            total += 1
            eligible += bool(others)
            changed += any(t != target for t in others)
    return dict(
        **n,
        returned=len(rows),
        planned=64,
        chosen_actions=dict(actions),
        branch_action_coverage=dict(branch_coverage),
        panel_donor_components=dict(
            samples=total,
            eligible=eligible,
            with_different_target=changed,
            warning="Panel opportunity only; training rollout donor availability unmeasured",
        ),
        component_errors=dict(errors),
        candidate_for_semantic_review=(
            n["valid"] >= CONTRACT["minimum_valid"]
            and n["exact_pairs"] >= CONTRACT["minimum_exact_pairs"]
            and n["recommended_informative_states"]
            >= CONTRACT["minimum_recommended_informative_states"]
        ),
        warning="A candidate is not cleared for training; manual semantics and donor coverage remain required.",
    )


def report(batch):
    """Reconcile one-to-one attempts and stored requests before scoring."""
    _, frozen, requests = verify(batch)
    result = dict(coverage=frozen["coverage"], workers={})
    for index, model in enumerate(MODELS):
        destination = batch / "workers" / str(index)
        rows = common.load_rows(destination / "responses.jsonl")
        events = common.load_rows(destination / "attempts.jsonl")
        expected = {r["case_id"]: identity(r, index) for r in requests}
        for row in rows:
            identified = expected.get(row["case_id"])
            if identified is None or any(
                row.get(k) != v for k, v in identified.items()
            ):
                raise ValueError("Response/request identity changed")
            linked = [e for e in events if e["case_id"] == row["case_id"]]
            if linked != [
                dict(event="start", **identified, attempt=1),
                dict(
                    event="end", **identified, attempt=1, status=row["status"]
                ),
            ]:
                raise ValueError("Invalid request accounting")
        metrics = score(frozen, rows)
        complete = (
            (destination / "DONE").exists()
            and len(events) == 128
            and len(rows) == 64
        )
        metrics.update(
            complete=complete,
            request_starts=sum(e["event"] == "start" for e in events),
            unknown_usage_attempts=sum(
                r.get("tokens_in") is None or r.get("tokens_out") is None
                for r in rows
            ),
            unclosed_attempts=sum(e["event"] == "start" for e in events)
            - sum(e["event"] == "end" for e in events),
            tokens_in=sum(r.get("tokens_in") or 0 for r in rows),
            tokens_out=sum(r.get("tokens_out") or 0 for r in rows),
        )
        metrics["candidate_for_semantic_review"] &= complete
        result["workers"][model] = metrics
    common.write_json(batch / "report.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--report", action="store_true")
    modes.add_argument("--run-worker", type=int)
    parser.add_argument(
        "--batch",
        type=Path,
        default=ROOT / "results/explanation_screens" / STUDY,
    )
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "results/advising_strength"
    )
    parser.add_argument(
        "--credential-file", type=Path, default=Path.home() / ".aleph_tyk.env"
    )
    args = parser.parse_args()
    if args.run_worker is not None:
        return collect(args.batch, args.run_worker)
    if args.report:
        report(args.batch)
        return 0
    print(
        "64 saved states x2 Aleph models =128 requests, one attempt each; $0 OpenAI; no training."
    )
    if args.prepare:
        batch = prepare(ROOT, args.corpus, args.credential_file)
        print(
            shlex.join(
                [
                    "sbatch",
                    "--parsable",
                    "--job-name=effect_ready",
                    "--array=0-1",
                    f"--output={batch}/slurm/%x_%A_%a.out",
                    str(batch / "code" / WORKER),
                    str(batch),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
