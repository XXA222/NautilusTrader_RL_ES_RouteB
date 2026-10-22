"""Freeze the sealed DOGEUSDT Fold config before registered Train/Validation access."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

from rles_data.core import MINUTE_NS, FoldManifest, Window
from rles_data.io import curated_source_identity
from rles_data.json_utils import loads_unique_json


def _ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path,
                        default=Path("configs/routeb/r4_dogeusdt_e2_e3_baseline_protocol.json"))
    parser.add_argument("--preregistration", type=Path,
                        default=Path("docs/world_model/R4_DOGEUSDT_E2_E3_BASELINE_PREREGISTRATION_20261022.md"))
    parser.add_argument("--run-id", default="fresh-dogeusdt-r4-e2-e3-20261022")
    parser.add_argument("--fold-output", type=Path,
                        default=Path("configs/routeb/fresh_dogeusdt_r4_e2_e3_fold.json"))
    args = parser.parse_args()
    if args.run_id.startswith(".") or "/" in args.run_id or "\\" in args.run_id:
        raise ValueError("run id must be a single artifact directory name")
    protocol_path = args.protocol_json if args.protocol_json.is_absolute() else root / args.protocol_json
    prereg_path = args.preregistration if args.preregistration.is_absolute() else root / args.preregistration
    artifact = root / ".artifacts/world_model" / args.run_id / "source"
    source_dir = artifact / "curated"
    source_path = source_dir / "market.parquet"
    prep_path = source_dir / "source_preparation.json"
    archive_receipt = artifact / "official_archive_receipt.json"
    output = args.fold_output if args.fold_output.is_absolute() else root / args.fold_output
    if output.exists():
        raise FileExistsError(f"refusing to replace frozen DOGEUSDT Fold config: {output}")
    protocol = loads_unique_json(protocol_path.read_bytes())
    prereg = prereg_path.read_text(encoding="utf-8")
    prep = loads_unique_json(prep_path.read_bytes())
    archives = loads_unique_json(archive_receipt.read_bytes())
    identity = curated_source_identity(source_path)
    cutoff = _ns(protocol["validation_score_utc"][1])
    event_end = _ns(protocol["test_score_utc"][1]) - MINUTE_NS
    test_start = datetime.fromisoformat(
        protocol["test_score_utc"][0].replace("Z", "+00:00"))
    test_warmup = _ns((test_start - timedelta(days=protocol["warmup_days"]))
                      .isoformat().replace("+00:00", "Z"))
    if ("Status: protocol and date scope frozen before downloading or decoding these market rows." not in prereg
            or protocol.get("source_symbol") != "DOGEUSDT"
            or len(archives.get("archives", ())) != 14
            or archives.get("schema") != protocol.get("archive_receipt_schema")
            or archives.get("market_rows_decoded") is not False
            or prep.get("test_ohlcv_values_excluded") is not True
            or prep.get("market_payload_cutoff_ns") != cutoff
            or prep.get("last_market_event_ns") != event_end
            or prep.get("dataset_hash") != identity["snapshot_hash"]
            or prep.get("marker_sha256") != identity["marker_sha256"]
            or identity["schema_version"] != "0.4.1"):
        raise ValueError("verified DOGEUSDT source identities do not match the frozen protocol")
    fold = FoldManifest(
        schema_version="0.4.0", dataset_hash=identity["snapshot_hash"],
        feature_hash="fc7c4373d2a4e5d802f9c0663d37ecf89d900e73afc7742d2d6ea76258540890",
        train=Window(_ns(protocol["train_warmup_start_utc"]),
                     _ns(protocol["train_score_utc"][0]), _ns(protocol["train_score_utc"][1])),
        validation=Window(_ns(protocol["validation_warmup_start_utc"]),
                          _ns(protocol["validation_score_utc"][0]), cutoff),
        test=Window(test_warmup, _ns(protocol["test_score_utc"][0]),
                    _ns(protocol["test_score_utc"][1])),
        purge_ns=60 * MINUTE_NS, embargo_ns=23 * 60 * MINUTE_NS, scaler_hash=None)
    source_posix = source_path.resolve().as_posix()
    if source_posix.startswith("/mnt/"):
        wsl_source_path = source_posix
    else:
        drive = source_path.drive.rstrip(":").lower()
        if len(drive) != 1 or ":/" not in source_posix:
            raise ValueError("cannot translate the sealed source path to WSL")
        wsl_source_path = "/mnt/" + drive + "/" + source_posix.split(":/", 1)[1]
    config = {
        "schema_version": "rles.world_model.registered_fold_baseline.v0",
        "allow_legacy_isolated_snapshot": False,
        "label": protocol.get(
            "fold_label", "routeb-fold-dogeusdt-r4-e2-e3-20261022-v1"),
        "instrument": protocol["instrument"], "marker_sha256": identity["marker_sha256"],
        "config": {"environment": {"instrument": protocol["instrument"],
                                    "market_path": wsl_source_path}},
        "fold": asdict(fold),
    }
    output.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "SEALED_FOLD_FROZEN_BEFORE_REGISTERED_READ",
        "fold_hash": fold.hash, "dataset_hash": fold.dataset_hash,
        "marker_sha256": identity["marker_sha256"], "fold_config": str(output),
        "fold_config_sha256": sha256(output.read_bytes()).hexdigest(),
        "protocol_sha256": sha256(protocol_path.read_bytes()).hexdigest(),
        "preregistration_sha256": sha256(prereg_path.read_bytes()).hexdigest(),
        "source_preparation_sha256": sha256(prep_path.read_bytes()).hexdigest(),
        "archive_receipt_sha256": sha256(archive_receipt.read_bytes()).hexdigest(),
        "sealed_test_rows": prep["sealed_market_rows"],
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
