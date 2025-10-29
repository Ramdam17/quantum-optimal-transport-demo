#!/usr/bin/env python3
"""
CLI script to generate data for all scenarios.

Usage:
    python scripts/generate_data.py --scenario hyperscanning --seed 42 --output data/
    python scripts/generate_data.py --scenario llm_alignment --seed 123
    python scripts/generate_data.py --scenario genetics --config config/scenario_genetics.yaml
    python scripts/generate_data.py --all  # Generate all scenarios

Examples:
    # Generate hyperscanning data with specific seed
    python scripts/generate_data.py --scenario hyperscanning --seed 42

    # Generate all scenarios with default settings
    python scripts/generate_data.py --all

    # Generate with custom output directory
    python scripts/generate_data.py --scenario genetics --output outputs/custom/
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.data.loaders import get_simulator
from src.utils.logger import setup_logger
from src.utils.helpers import format_size

# Setup logger
logger = setup_logger(__name__)

# Available scenarios
SCENARIOS = ["hyperscanning", "llm_alignment", "genetics"]


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate simulated data for Quantum Optimal Transport scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        choices=SCENARIOS,
        help="Scenario name (hyperscanning, llm_alignment, or genetics)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate data for all scenarios",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: use config value)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory for generated data (default: data/)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom configuration file (default: use scenario config)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validation: must specify either --scenario or --all
    if not args.scenario and not args.all:
        parser.error("Must specify either --scenario or --all")

    if args.scenario and args.all:
        parser.error("Cannot specify both --scenario and --all")

    return args


def print_progress(message: str, step: int = 0, total: int = 0):
    """
    Print progress message with optional step indicator.

    Parameters
    ----------
    message : str
        Progress message
    step : int, optional
        Current step number
    total : int, optional
        Total number of steps
    """
    if total > 0:
        progress = f"[{step}/{total}]"
        print(f"{progress} {message}")
    else:
        print(f"► {message}")


def print_summary(data: Dict[str, Any], elapsed_time: float):
    """
    Print summary statistics for generated data.

    Parameters
    ----------
    data : Dict[str, Any]
        Generated data dictionary
    elapsed_time : float
        Time taken to generate data (seconds)
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Time taken
    print(f"Generation time: {elapsed_time:.2f}s")

    # Data keys and shapes
    print(f"\nGenerated data contains {len(data)} keys:")
    total_size = 0

    for key, value in data.items():
        if hasattr(value, "shape"):
            # NumPy array
            shape_str = " × ".join(map(str, value.shape))
            size_bytes = value.nbytes
            total_size += size_bytes
            print(f"  • {key}: shape=({shape_str}), size={format_size(size_bytes)}")
        elif isinstance(value, dict):
            # Nested dictionary
            print(f"  • {key}: {len(value)} items (dictionary)")
        elif isinstance(value, list):
            # List
            print(f"  • {key}: {len(value)} items (list)")
        else:
            # Other types
            print(f"  • {key}: {type(value).__name__}")

    if total_size > 0:
        print(f"\nTotal data size: {format_size(total_size)}")

    print("=" * 60 + "\n")


def generate_scenario(
    scenario: str,
    output_dir: Path,
    seed: Optional[int] = None,
    config_path: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """
    Generate data for a single scenario.

    Parameters
    ----------
    scenario : str
        Scenario name
    output_dir : Path
        Output directory
    seed : Optional[int], optional
        Random seed
    config_path : Optional[str], optional
        Custom config path
    verbose : bool, optional
        Enable verbose output

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        print_progress(f"Starting data generation for scenario: {scenario}")

        # Load configuration
        if config_path:
            print_progress(f"Loading custom config: {config_path}", 1, 4)
            config = ConfigLoader(config_path)
        else:
            config_file = f"config/scenario_{scenario}.yaml"
            print_progress(f"Loading config: {config_file}", 1, 4)
            config = ConfigLoader(config_file)

        # Get data parameters
        data_config = config.get_section("data")

        # Override seed if provided
        if seed is not None:
            data_config["seed"] = seed
            print_progress(f"Using seed: {seed}")

        # Create simulator
        print_progress(f"Creating {scenario} simulator", 2, 4)
        simulator_class = get_simulator(scenario)
        simulator = simulator_class(seed=data_config.get("seed"))

        if verbose:
            logger.info(f"Simulator config: {data_config}")

        # Generate data
        print_progress("Generating data (this may take a moment)...", 3, 4)
        start_time = time.time()

        data = simulator.generate(**data_config)

        elapsed_time = time.time() - start_time

        # Save data
        print_progress("Saving data to disk", 4, 4)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{scenario}_data.npz"

        simulator.save(str(output_file))

        print(f"✓ Data saved to: {output_file}")

        # Print summary
        print_summary(data, elapsed_time)

        return True

    except Exception as e:
        logger.error(f"Failed to generate {scenario} data: {e}", exc_info=verbose)
        print(f"✗ Error: {e}")
        return False


def main():
    """Main entry point for the script."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("QUANTUM OPTIMAL TRANSPORT - DATA GENERATION")
    print("=" * 60 + "\n")

    # Setup output directory
    output_dir = Path(args.output)

    # Determine which scenarios to generate
    if args.all:
        scenarios_to_generate = SCENARIOS
        print(f"Generating data for all {len(SCENARIOS)} scenarios\n")
    else:
        scenarios_to_generate = [args.scenario]

    # Generate data for each scenario
    success_count = 0
    fail_count = 0

    for idx, scenario in enumerate(scenarios_to_generate, 1):
        if len(scenarios_to_generate) > 1:
            print(f"\n{'─' * 60}")
            print(f"SCENARIO {idx}/{len(scenarios_to_generate)}: {scenario.upper()}")
            print(f"{'─' * 60}\n")

        success = generate_scenario(
            scenario=scenario,
            output_dir=output_dir,
            seed=args.seed,
            config_path=args.config,
            verbose=args.verbose,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Final summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"✓ Successful: {success_count}")
    if fail_count > 0:
        print(f"✗ Failed: {fail_count}")
    print(f"Output directory: {output_dir.resolve()}")
    print("=" * 60 + "\n")

    # Exit with appropriate code
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
