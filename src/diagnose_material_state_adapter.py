"""Inspect a trained B3a adapter without running rollout inference."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from safetensors.torch import load_file

try:
    from diagnose_material_condition import _load_checkpoint_strict, _validate_b0_identity
    from model.spacetime import MDM_ST
    from options import TestingConfig
    from utils.material_state_diagnostics import summarize_material_state_adapter
except ModuleNotFoundError:
    from src.diagnose_material_condition import _load_checkpoint_strict, _validate_b0_identity
    from src.model.spacetime import MDM_ST
    from src.options import TestingConfig
    from src.utils.material_state_diagnostics import summarize_material_state_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a frozen B3a material-state adapter checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output path prefix.")
    parser.add_argument(
        "--profile",
        choices=("b3a45", "b3a90"),
        default="b3a45",
        help="Strict B3a checkpoint/config identity profile.",
    )
    return parser


def _output_paths(prefix: Path) -> tuple[Path, Path]:
    prefix = Path(prefix)
    return prefix.with_suffix(".json"), prefix.with_suffix(".md")


def write_summary(
    prefix: Path,
    summary: dict,
    config: Path,
    checkpoint: Path,
) -> tuple[Path, Path]:
    json_path, markdown_path = _output_paths(prefix)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(config),
        "checkpoint": str(checkpoint),
        **summary,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# B3a Material-State Adapter 诊断",
        "",
        f"- config: `{config}`",
        f"- checkpoint: `{checkpoint}`",
        f"- 参数量: {summary['parameter_count']:,}",
        f"- rank / interval / stages: {summary['rank']} / {summary['interval']} / {summary['num_stages']}",
        f"- runtime scale: {summary['runtime_scale']}",
        f"- stage scales: {summary['stage_scales']}",
        f"- state projection norm: {summary['state_projection_norm']:.8g}",
        f"- material projection norm: {summary['material_projection_norm']:.8g}",
        f"- output projection norm: {summary['output_projection_norm']:.8g}",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    cli = build_parser().parse_args()
    config_path = Path(cli.config)
    checkpoint = Path(cli.checkpoint)
    args = OmegaConf.merge(
        OmegaConf.structured(TestingConfig), OmegaConf.load(config_path)
    )
    args.resume = str(checkpoint)
    _validate_b0_identity(args, profile=cli.profile)
    args.model_config.cond_frames = int(args.input_frames)
    model = MDM_ST(
        args.pc_size,
        int(args.output_frames),
        n_feats=3,
        model_config=args.model_config,
    )
    _load_checkpoint_strict(model, checkpoint, load_file)
    paths = write_summary(
        Path(cli.output),
        summarize_material_state_adapter(model),
        config_path,
        checkpoint,
    )
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
