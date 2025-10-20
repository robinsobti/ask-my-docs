from pathlib import Path
from datetime import datetime


def init_run_log() -> tuple[str, Path]:
    """
    Initializes a run log directory and queries file.

    Returns:
        tuple[str, Path]: 
            - run_identifier: A string representing the UTC timestamp used as the unique run directory name.
            - queries_path: The absolute Path to the created 'queries.jsonl' file inside the run directory.
    """
    base_dir = Path("runs")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Run identifier uses UTC timestamp in 'YYYY-MM-DD_HHMM-SS' format for unique run directories
    run_identifier = datetime.now().strftime("%Y-%m-%d_%H%M-%S")
    run_dir = base_dir / run_identifier
    run_dir.mkdir(parents=True, exist_ok=True)

    queries_path = run_dir / "queries.jsonl"
    queries_path.touch(exist_ok=True)

    return run_identifier, queries_path.resolve()
