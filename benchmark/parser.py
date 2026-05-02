from __future__ import annotations

import argparse
from pathlib import Path


def collect_provided_flags(argv: list[str]) -> set[str]:
    return {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.cli",
        description="Image/video augmentation benchmark suite",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run benchmarks")
    run_p.add_argument("--config", type=Path, help="YAML benchmark run config")
    run_p.add_argument("--resolved-config", type=Path, help=argparse.SUPPRESS)
    run_p.add_argument("--dry-run", action="store_true", help="Print the resolved config and exit without running")
    run_p.add_argument("--data-dir", "-d", help="Directory with images or videos")
    run_p.add_argument("--output", "-o", help="Directory to write result JSON files")
    run_p.add_argument("--media", choices=["image", "video"], default="image", help="Media type (default: image)")
    run_p.add_argument(
        "--libraries",
        nargs="+",
        metavar="LIB",
        help="Which built-in libraries to run (default: all for the selected media type)",
    )
    run_p.add_argument(
        "--transforms",
        nargs="+",
        metavar="TRANSFORM",
        help="Run only these transforms (by name). Default: all.",
    )
    run_p.add_argument(
        "--transform-set",
        choices=["paper"],
        help="Use a named transform set. 'paper' selects transforms present in 2+ selected libraries for the scenario.",
    )
    run_p.add_argument(
        "--spec",
        "-s",
        metavar="FILE",
        help="Custom spec file (overrides --libraries; library inferred from LIBRARY variable)",
    )
    run_p.add_argument(
        "--scenario",
        choices=["image-rgb", "image-9ch", "video-decode-16f", "video-16f"],
        help="Run a benchmark scenario such as image-rgb, image-9ch, video-decode-16f, or video-16f.",
    )
    run_p.add_argument(
        "--mode",
        choices=["micro", "pipeline", "decode"],
        help="Scenario benchmark mode. Defaults depend on --scenario.",
    )
    run_p.add_argument("--batch-size", type=int, default=32, help="Pipeline dataloader batch size")
    run_p.add_argument("--workers", type=int, default=0, help="Pipeline dataloader worker count")
    run_p.add_argument("--min-time", type=float, default=0.0, help="Minimum measured seconds per run")
    run_p.add_argument("--min-batches", type=int, default=1, help="Minimum measured dataloader batches per run")
    run_p.add_argument(
        "--pipeline-scope",
        choices=["memory_dataloader_augment", "decode_dataloader_augment", "decode_dataloader_augment_batch_copy"],
        default="decode_dataloader_augment",
        help="Pipeline measurement scope for --mode pipeline",
    )
    run_p.add_argument(
        "--device",
        choices=["none", "cuda", "mps", "auto"],
        default="none",
        help="Device copy target for batch-copy pipeline scope",
    )
    run_p.add_argument(
        "--thread-policy",
        choices=["micro-single", "pipeline-default", "pipeline-single-worker"],
        help="Thread policy. Defaults to micro-single for micro and pipeline-default for pipeline.",
    )
    run_p.add_argument("--clip-length", type=int, help="Video frames per clip for scenario benchmarks")
    run_p.add_argument(
        "--decoders",
        nargs="+",
        metavar="DECODER",
        help="Video decoders for --scenario video-decode-16f",
    )

    run_p.add_argument("--cloud", choices=["gcp"], default=None, help="Run on cloud (currently: gcp)")
    run_p.add_argument("--gcp-project", metavar="PROJECT", help="GCP project ID (required with --cloud gcp)")
    run_p.add_argument("--gcp-zone", metavar="ZONE", default="us-central1-a", help="GCP zone (default: us-central1-a)")
    run_p.add_argument("--gcp-machine-type", metavar="TYPE", default="n1-standard-8")
    run_p.add_argument(
        "--gcp-gpu-type",
        metavar="TYPE",
        default=None,
        help="GPU accelerator type (e.g. nvidia-tesla-t4)",
    )
    run_p.add_argument(
        "--gcp-remote-data-dir",
        metavar="PATH",
        help="Attached mode: dataset path on the VM (required with --gcp-attached)",
    )
    run_p.add_argument(
        "--gcp-remote-repo-dir",
        metavar="PATH",
        default="~/benchmark",
        help="Remote directory for the repo extract and results (default: ~/benchmark)",
    )
    run_p.add_argument(
        "--gcp-gcs-data-uri",
        metavar="GS_URI",
        help=(
            "Detached mode: gs:// URI of the dataset archive/object to download to the VM "
            "(required unless --gcp-attached)"
        ),
    )
    run_p.add_argument(
        "--gcp-gcs-results-uri",
        metavar="GS_URI",
        help="Detached mode: gs:// URI prefix for run artifacts (required unless --gcp-attached)",
    )
    run_p.add_argument(
        "--gcp-attached",
        action="store_true",
        help="Use blocking SSH workflow (upload repo, run, download results) instead of detached startup-script",
    )
    run_p.add_argument(
        "--gcp-dry-run",
        action="store_true",
        help="Detached mode: print job.json and exit without uploading or creating a VM",
    )
    run_p.add_argument(
        "--gcp-disk-size-gb",
        type=int,
        default=100,
        metavar="N",
        help="Boot disk size in GB (default: 100)",
    )
    run_p.add_argument("--gcp-keep-instance", action="store_true", help="Do not delete instance after run (debug)")
    run_p.add_argument(
        "--gcp-keep-on-failure",
        action="store_true",
        help="Detached mode: keep the VM alive only when the startup script or benchmark fails.",
    )
    run_p.add_argument(
        "--gcp-preemptible",
        action="store_true",
        help="Use a preemptible GCP VM. Default is a regular VM for benchmark stability and quota compatibility.",
    )
    run_p.add_argument(
        "--gcp-venv-cache-uri",
        metavar="GS_URI",
        help="GCS prefix for reusable VM venv cache (default: sibling augmentation-cache bucket prefix).",
    )
    run_p.add_argument("--gcp-no-venv-cache", action="store_true", help="Disable GCS venv cache restore/populate.")
    run_p.add_argument(
        "--gcp-force-venv-cache-rebuild",
        action="store_true",
        help="Bypass venv cache lookup and upload a fresh cache after a successful run.",
    )

    run_p.add_argument("--num-items", "-n", type=int, help="Number of images/videos (default depends on media type)")
    run_p.add_argument("--num-runs", "-r", type=int, default=5, help="Benchmark runs per transform (default: 5)")
    run_p.add_argument(
        "--slow-threshold-sec-per-item",
        type=float,
        default=None,
        help="Skip micro/pipeline transforms slower than this many seconds per image/video in preflight.",
    )
    run_p.add_argument(
        "--slow-preflight-items",
        type=int,
        default=None,
        help="Number of images/videos used for slow-transform preflight.",
    )
    run_p.add_argument(
        "--disable-slow-skip",
        action="store_true",
        help="Run exhaustive measurements even when preflight says a transform is slow.",
    )
    run_p.set_defaults(refresh_requirements=True)
    run_p.add_argument(
        "--no-refresh-requirements",
        action="store_false",
        dest="refresh_requirements",
        help="Skip regenerating requirements/*.txt from requirements/*.in before checking the venv dependency cache",
    )
    run_p.add_argument(
        "--num-channels",
        type=int,
        default=3,
        help=(
            "Number of image channels (must be multiple of 3). Values > 3 stack the RGB source image "
            "to synthesize multi-channel data, e.g. 9 for 3x stacked RGB (default: 3)"
        ),
    )
    run_p.add_argument(
        "--multichannel",
        action="store_true",
        help=(
            "Use multi-channel specs (9ch) and output to <output>/multichannel/. "
            "Implies --num-channels 9 for image mode."
        ),
    )

    plan_p = subparsers.add_parser("plan", help="Print resolved config, generated jobs, and expected outputs")
    plan_p.add_argument("--config", type=Path, required=True, help="YAML benchmark run config")
    plan_p.add_argument("--resolved-config", type=Path, help=argparse.SUPPRESS)
    plan_p.add_argument("--output", "-o", help="Override output.output_dir")
    plan_p.add_argument("--num-items", "-n", type=int, help="Override data.num_items")
    plan_p.add_argument("--num-runs", "-r", type=int, help="Override execution.num_runs")
    plan_p.add_argument("--device", choices=["none", "cuda", "mps", "auto"], help="Override execution.device")
    plan_p.add_argument("--workers", type=int, help="Override execution.workers")
    plan_p.add_argument("--batch-size", type=int, help="Override execution.batch_size")
    plan_p.add_argument("--gcp-zone", metavar="ZONE", help="Override cloud.zone")
    plan_p.add_argument("--gcp-machine-type", metavar="TYPE", help="Override cloud.machine_type")
    plan_p.add_argument("--gcp-dry-run", action="store_true", help="Override cloud.dry_run")

    cmp_p = subparsers.add_parser("compare", help="Compare two result directories")
    cmp_p.add_argument("--baseline", required=True, help="Baseline results directory")
    cmp_p.add_argument("--current", required=True, help="Current results directory")
    cmp_p.add_argument("--libraries", nargs="+", metavar="LIB", help="Filter to specific libraries")
    cmp_p.add_argument("--transforms", nargs="+", metavar="TRANSFORM", help="Filter to specific transforms")
    cmp_p.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Minimum delta fraction to consider a change significant (default: 0.05 = 5%%)",
    )
    cmp_p.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if any regression exceeds --threshold",
    )

    doctor_p = subparsers.add_parser("doctor", help="Check benchmark environment reliability")
    doctor_p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    doctor_p.add_argument("--fail-on-warning", action="store_true", help="Exit 1 when doctor reports warnings")

    validate_p = subparsers.add_parser("validate-results", help="Audit benchmark result JSON files")
    validate_p.add_argument("path", help="Result JSON file or directory")
    validate_p.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    return parser
