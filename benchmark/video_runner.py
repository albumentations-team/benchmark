"""Compatibility shim — video runner is now part of benchmark.runner."""

from .runner import BenchmarkRunner, MediaType, load_from_python_file

# Alias used by run_video_single.sh
VideoBenchmarkRunner = BenchmarkRunner


def main() -> None:
    """CLI entry point for video benchmarks (delegates to runner.main with --media video)."""
    import sys

    if "--media" not in sys.argv:
        sys.argv.extend(["--media", "video"])

    from .runner import main as runner_main

    runner_main()


__all__ = ["BenchmarkRunner", "MediaType", "VideoBenchmarkRunner", "load_from_python_file", "main"]

if __name__ == "__main__":
    main()
