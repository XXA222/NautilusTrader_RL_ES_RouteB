"""Freeze the sealed XRP Fold config before registered Train/Validation access."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path,
                        default=Path("configs/routeb/r4_xrpusdt_e2_e3_baseline_protocol.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol_json if args.protocol_json.is_absolute() else root / args.protocol_json
    protocol = loads_unique_json(protocol_path.read_bytes())
    prereg_path = root / "docs/world_model/R4_XRPUSDT_E2_E3_BASELINE_PREREGISTRATION_20261021.md"
    prereg = prereg_path.read_text(encoding="utf-8")
    run_id = "fresh-xrpusdt-r4-e2-e3-20261021"
    source_dir = root / ".artifacts/world_model" / run_id / "source/curated"
    source_path = source_dir / "market.parquet"
    prep_path = source_dir / "source_preparation.json"
    archive_receipt = source_dir.parent / "official_archive_receipt.json"
    output = root / "configs/routeb/fresh_xrpusdt_r4_e2_e3_fold.json"
    if output.exists():
        raise FileExistsError(f"refusing to replace frozen XRP Fold config: {output}")
    prep = loads_unique_json(prep_path.read_bytes())
    archives = loads_unique_json(archive_receipt.read_bytes())
    identity = curated_source_identity(source_path)
    cutoff = _ns(protocol["validation_score_utc"][1])
    event_end = _ns(protocol["test_score_utc"][1]) - MINUTE_NS
    if ("Status: protocol and date scope frozen before downloading or decoding these market rows." not in prereg
            or protocol.get("source_symbol") != "XRPUSDT"
            or len(archives.get("archives", ())) != 14
            or archives.get("schema") != protocol.get("archive_receipt_schema")
            or archives.get("market_rows_decoded") is not False
            or prep.get("test_ohlcv_values_excluded") is not True
            or prep.get("market_payload_cutoff_ns") != cutoff
            or prep.get("last_market_event_ns") != event_end
            or prep.get("dataset_hash") != identity["snapshot_hash"]
            or prep.get("marker_sha256") != identity["marker_sha256"]
            or identity["schema_version"] != "0.4.1"):
        raise ValueError("verified XRP source identities do not match the frozen protocol")
    fold = FoldManifest(
        schema_version="0.4.0", dataset_hash=identity["snapshot_hash"],
        feature_hash="fc7c4373d2a4e5d802f9c0663d37ecf89d900e73afc7742d2d6ea76258540890",
        train=Window(_ns(protocol["train_warmup_start_utc"]),
                     _ns(protocol["train_score_utc"][0]), _ns(protocol["train_score_utc"][1])),
        validation=Window(_ns(protocol["validation_warmup_start_utc"]),
                          _ns(protocol["validation_score_utc"][0]), cutoff),
        test=Window(_ns("2024-08-10T00:01:00Z"),
                    _ns(protocol["test_score_utc"][0]), _ns(protocol["test_score_utc"][1])),
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
        "label": "routeb-fold-xrpusdt-r4-e2-e3-20261021-v1",
        "instrument": protocol["instrument"], "marker_sha256": identity["marker_sha256"],
        "config": {"environment": {"instrument": protocol["instrument"],
                                    "market_path": wsl_source_path}},
        "fold": asdict(fold),
    }
    output.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "SEALED_FOLD_FROZEN_BEFORE_REGISTERED_READ",
        "fold_hash": fold.hash, "dataset_hash": fold.dataset_hash,
        "marker_sha256": identity["marker_sha256"],
        "fold_config": str(output),
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

