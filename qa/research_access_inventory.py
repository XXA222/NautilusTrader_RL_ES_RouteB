"""Read legacy WFO ledgers and configurations without loading market data.

The inventory is conservative: an old query reservation counts even when its
report is absent. Unknown Holdout end boundaries quarantine the remaining future
interval until the evidence is repaired. This command never edits legacy DBs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import subprocess

from rles_data.core import MAX_NS, digest
from rles_data.json_utils import loads_unique_json


def _contained_path(root: Path, path: Path, label: str, *, must_exist: bool = True) -> Path:
    root = root.resolve()
    if not path.is_absolute() and ".." in path.parts:
        raise ValueError(f"{label} contains parent traversal")
    candidate = path if path.is_absolute() else root / path
    try:
        relative = Path(os.path.abspath(candidate)).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the inventory root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} traverses a symlink")
    resolved = current.resolve(strict=must_exist)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes the inventory root")
    return resolved


def _read_json_object(path: Path) -> dict:
    try:
        value = loads_unique_json(path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read unambiguous JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON inventory input must be an object: {path}")
    return value


def inventory(root: Path) -> dict:
    root = root.resolve()
    scan_roots = [root / ".tmp" / "real-ethusdt-scenario",
                  root / ".tmp" / "wfo-validation-recovery-20261013"]
    unresolved: list[dict] = []
    configs = []
    for folder in scan_roots:
        _contained_path(root, folder.relative_to(root), "legacy scan root", must_exist=False)
        if not folder.is_dir():
            unresolved.append({
                "path": folder.relative_to(root).as_posix(),
                "reason": "legacy research root missing; historical access coverage unresolved",
            })
        for path in sorted(folder.glob("config*.json")):
            path = _contained_path(root, path.relative_to(root), "legacy config")
            value = _read_json_object(path)
            configs.append((path, value))
    discovery = subprocess.run([
        "rg", "--files", "--hidden", "--no-ignore", "--no-messages",
        *[str(path) for path in scan_roots], "-g", "wfo-run.json",
    ], capture_output=True, text=True, check=False)
    entries: list[dict] = []
    databases: list[dict] = []
    if discovery.returncode not in (0, 1):
        unresolved.append({
            "reason": "legacy WFO manifest discovery failed; historical access coverage unresolved",
            "discovery_exit_code": discovery.returncode,
        })
    for name in sorted(discovery.stdout.splitlines()):
        manifest_path = _contained_path(root, Path(name), "legacy WFO manifest")
        manifest = _read_json_object(manifest_path)
        db = manifest_path.parent / "research.sqlite"
        try:
            db = _contained_path(root, db.relative_to(root), "legacy research database")
        except ValueError as exc:
            unresolved.append({"manifest": str(manifest_path.relative_to(root)),
                               "reason": f"ledger path rejected: {exc}"})
            continue
        if not db.is_file():
            unresolved.append({"manifest": str(manifest_path.relative_to(root)), "reason": "missing ledger"})
            continue
        with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            states = [dict(row) for row in connection.execute("SELECT * FROM protocol")]
            queries = [dict(row) for row in connection.execute("SELECT * FROM query")]
        db_rel = str(db.relative_to(root)).replace("\\", "/")
        databases.append({"path": db_rel, "sha256": sha256(db.read_bytes()).hexdigest(),
                          "protocols": states, "queries": queries})
        for state in states:
            pid = state["id"]
            matching = [value for _, value in configs
                        if value.get("wfo", {}).get("protocol_id") == pid]
            instrument = matching[0]["environment"]["instrument"] if matching else "ETHUSDT-PERP.SIM"
            market = instrument.split(".")[0].removesuffix("-PERP").upper()
            folds = manifest["fold_manifests"]

            def add(purpose, start, end, evidence):
                identity = {"database": db_rel, "protocol_id": pid,
                            "source_sha": manifest["dataset_hash"], "evidence": evidence}
                entries.append({"key": digest([db_rel, purpose, start, end, evidence]),
                                "market": market, "family": "perpetual-market-v1",
                                "start_ns": start, "end_ns": end, "purpose": purpose,
                                "identity": identity})

            for query in queries:
                if query["protocol_id"] != pid:
                    continue
                if query["kind"] not in {"validation", "test"}:
                    unresolved.append({"database": db_rel, "query": query, "reason": "unknown query scope"})
                    continue
                index = int(query["fold_id"].rsplit("-", 1)[-1])
                fold = folds[index]
                scope = fold[query["kind"]]
                add("test" if query["kind"] == "test" else "development",
                    scope["score_start_ns"], scope["score_end_ns"], query)
                train = fold["train"]
                add("development", train["score_start_ns"], train["score_end_ns"],
                    {"training_for": query["query_id"]})
            if state.get("holdout_sha"):
                start = state.get("holdout_start_ns")
                if start is None and matching:
                    start = matching[0]["wfo"]["holdout_start_ns"]
                end = matching[0]["wfo"]["holdout_end_ns"] if matching else None
                if start is None or end is None:
                    unresolved.append({"database": db_rel, "reason": "unknown Holdout boundary; quarantined"})
                    start, end = start or 1, end or MAX_NS
                add("holdout", start, end, {"holdout_sha": state["holdout_sha"],
                                             "state": state["state"], "conservative_end": not bool(matching)})
    standalone_reports = []
    for config_path, config in configs:
        evaluation = config.get("evaluation")
        if config.get("wfo") or not isinstance(evaluation, dict):
            continue
        output = Path(config["output"])
        output = output if output.is_absolute() else config_path.parent / output
        try:
            report_path = _contained_path(
                root, output / "report.json", "standalone report", must_exist=False,
            )
        except ValueError as exc:
            unresolved.append({"config": str(config_path.relative_to(root)),
                               "reason": f"standalone report path rejected: {exc}"})
            continue
        if not report_path.is_file():
            continue
        split = evaluation.get("split", "validation")
        purpose = split if split in {"test", "holdout"} else "development"
        identity = {"config": str(config_path.relative_to(root)).replace("\\", "/"),
                    "config_sha": sha256(config_path.read_bytes()).hexdigest(),
                    "report_sha": sha256(report_path.read_bytes()).hexdigest(),
                    "split": split, "scope": "completed standalone report; conservatively used"}
        standalone_reports.append(identity)
        entries.append({"key": digest(identity),
                        "market": config["environment"]["instrument"].split(".")[0].removesuffix("-PERP").upper(),
                        "family": "perpetual-market-v1", "start_ns": evaluation["start_ns"],
                        "end_ns": evaluation["end_ns"], "purpose": purpose, "identity": identity})
    return {"schema": "rles.legacy-research-access.v1", "date": "2026-10-16",
            "scope": [str(path.relative_to(root)).replace("\\", "/") for path in scan_roots],
            "market_data_loaded": False, "legacy_databases_modified": False,
            "discovery_exit_code": discovery.returncode,
            "limitations": "Known WFO roots only; unledgered historical accesses cannot be inferred.",
            "databases": databases, "standalone_reports": standalone_reports,
            "entries": entries, "unresolved": unresolved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, ensure_ascii=False, allow_nan=False)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        sha256(args.output.read_bytes()).hexdigest(), encoding="ascii")
    print(json.dumps({"databases": len(result["databases"]), "entries": len(result["entries"]),
                      "unresolved": len(result["unresolved"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
