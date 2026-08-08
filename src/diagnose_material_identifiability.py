"""Run the B0.2 material identifiability audit over HDF5 record directories."""

import argparse
import math
from pathlib import Path

from utils.material_identifiability import (
    MATERIAL_NAMES,
    AuditSettings,
    RecordValidationError,
    analyze_confounding,
    analyze_responses,
    build_coverage_rows,
    build_support_rows,
    classify_identifiability,
    read_h5_record,
    write_audit_outputs,
)


class _AuditArgumentParser(argparse.ArgumentParser):
    """Validate CLI sampling settings while preserving argparse error handling."""

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        try:
            _validate_settings(_settings_from_args(parsed))
        except ValueError as error:
            self.error(str(error))
        return parsed


def _settings_from_args(args: argparse.Namespace) -> AuditSettings:
    return AuditSettings(
        seed=args.seed,
        folds=args.folds,
        permutations=args.permutations,
        bootstrap_samples=args.bootstrap_samples,
        contact_band_raw=args.contact_band_raw,
    )


def _validate_settings(settings: AuditSettings) -> None:
    if (
        not isinstance(settings.seed, int)
        or isinstance(settings.seed, bool)
        or settings.seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    if (
        not isinstance(settings.folds, int)
        or isinstance(settings.folds, bool)
        or settings.folds < 2
    ):
        raise ValueError("folds must be at least 2")
    if (
        not isinstance(settings.permutations, int)
        or isinstance(settings.permutations, bool)
        or settings.permutations <= 0
    ):
        raise ValueError("permutations must be positive")
    if (
        not isinstance(settings.bootstrap_samples, int)
        or isinstance(settings.bootstrap_samples, bool)
        or settings.bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap-samples must be positive")
    if (
        not isinstance(settings.contact_band_raw, (int, float))
        or isinstance(settings.contact_band_raw, bool)
        or not math.isfinite(settings.contact_band_raw)
        or settings.contact_band_raw <= 0
    ):
        raise ValueError("contact-band-raw must be positive and finite")


def _resolve_directory(path: Path, *, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{name} directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{name} path is not a directory: {resolved}")
    return resolved


def _validate_input_paths(paths: list[Path], *, split: str) -> None:
    if not paths:
        raise ValueError(f"{split} directory contains no *.h5 files")


def _read_split_records(
    paths: list[Path],
    *,
    split: str,
    settings: AuditSettings,
    invalid_records: list[dict[str, str]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        print(f"{split} {index}/{len(paths)}")
        try:
            records.append(read_h5_record(path, split=split, settings=settings))
        except RecordValidationError as error:
            invalid_records.append(
                {"path": str(path), "split": split, "error": str(error)}
            )
    return records


def _validate_train_material_counts(
    train_records: list[dict[str, object]],
    *,
    folds: int,
) -> None:
    for material in MATERIAL_NAMES.values():
        count = sum(record["material"] == material for record in train_records)
        if count < folds:
            raise ValueError(
                f"material {material} has {count} valid train records; "
                f"at least {folds} are required"
            )


def build_parser() -> argparse.ArgumentParser:
    """Build the B0.2 audit command-line parser."""
    parser = _AuditArgumentParser(
        description="Diagnose B0.2 material identifiability."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/material_identifiability_b02"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--contact-band-raw", type=float, default=0.08)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_material_identifiability_audit(
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    settings: AuditSettings,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Stream records, run frozen statistics, and atomically write audit artifacts."""
    _validate_settings(settings)
    train_dir = _resolve_directory(train_dir, name="train")
    test_dir = _resolve_directory(test_dir, name="test")
    train_paths = sorted(train_dir.glob("*.h5"))
    test_paths = sorted(test_dir.glob("*.h5"))
    _validate_input_paths(train_paths, split="train")
    _validate_input_paths(test_paths, split="test")
    invalid_records: list[dict[str, str]] = []

    train_records = _read_split_records(
        train_paths,
        split="train",
        settings=settings,
        invalid_records=invalid_records,
    )
    test_records = _read_split_records(
        test_paths,
        split="test",
        settings=settings,
        invalid_records=invalid_records,
    )
    _validate_train_material_counts(train_records, folds=settings.folds)

    print("coverage")
    coverage_rows = build_coverage_rows(train_records, test_records)
    print("support")
    support_rows = build_support_rows(train_records, test_records)
    records = [*train_records, *test_records]
    print("confounding")
    confounding_rows = analyze_confounding(records, settings)
    print("response")
    response_rows = analyze_responses(records, settings)
    print("summary")
    summary_rows = classify_identifiability(
        response_rows,
        confounding_rows,
        support_rows,
    )
    metadata = {
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "seed": settings.seed,
        "folds": settings.folds,
        "permutations": settings.permutations,
        "bootstrap_samples": settings.bootstrap_samples,
        "contact_band_raw": settings.contact_band_raw,
        "train_file_count": len(train_paths),
        "test_file_count": len(test_paths),
        "train_valid_count": len(train_records),
        "test_valid_count": len(test_records),
        "invalid_records": invalid_records,
    }
    print("write outputs")
    return write_audit_outputs(
        Path(output_dir),
        records=records,
        coverage_rows=coverage_rows,
        support_rows=support_rows,
        confounding_rows=confounding_rows,
        response_rows=response_rows,
        summary_rows=summary_rows,
        metadata=metadata,
        overwrite=overwrite,
    )


def main() -> None:
    args = build_parser().parse_args()
    paths = run_material_identifiability_audit(
        args.train_dir,
        args.test_dir,
        args.output_dir,
        _settings_from_args(args),
        overwrite=args.overwrite,
    )
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
