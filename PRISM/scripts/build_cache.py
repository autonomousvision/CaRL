"""
Standalone nuPlan → CaRL cache builder.

Bypasses Hydra entirely — no YAML configs, no pkg:// search paths needed.
Reads scenarios directly from nuPlan SQLite .db files and writes them to
the GymScenarioCache format (.gz) that CacheScenarioSampler can load.

Usage
-----
    python scripts/build_cache.py \
        --data_root    $NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini \
        --map_root     $NUPLAN_MAPS_ROOT \
        --cache_path   $NUPLAN_EXP_ROOT/mini_cache \
        --map_version  nuplan-maps-v1.0

    # Limit to 500 scenarios for a quick smoke-test:
    python scripts/build_cache.py ... --max_scenarios 500
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_PRISM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PRISM_ROOT))
sys.path.insert(0, str(_PRISM_ROOT.parent / "nuPlan"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("prism.build_cache")


def main():
    parser = argparse.ArgumentParser(
        description="Build CaRL scenario cache from nuPlan .db files (no Hydra)"
    )
    parser.add_argument(
        "--data_root", required=True,
        help="Path to the mini (or trainval) split directory containing *.db files",
    )
    parser.add_argument(
        "--map_root", required=True,
        help="Path to the nuPlan maps root (contains nuplan-maps-v1.0.json)",
    )
    parser.add_argument(
        "--cache_path", required=True,
        help="Output directory for the .gz scenario cache",
    )
    parser.add_argument(
        "--map_version", default="nuplan-maps-v1.0",
        help="nuPlan map version string (default: nuplan-maps-v1.0)",
    )
    parser.add_argument(
        "--max_scenarios", type=int, default=None,
        help="Cap the total number of scenarios to cache (None = all)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for scenario extraction (default: 1)",
    )
    args = parser.parse_args()

    # ── Imports ───────────────────────────────────────────────────────────────
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    from carl_nuplan.planning.gym.cache.gym_scenario_cache import GymScenarioCache

    # ── Build scenario list ───────────────────────────────────────────────────
    logger.info(f"Data root  : {args.data_root}")
    logger.info(f"Map root   : {args.map_root}")
    logger.info(f"Map version: {args.map_version}")
    logger.info(f"Cache path : {args.cache_path}")

    builder = NuPlanScenarioBuilder(
        data_root=args.data_root,
        map_root=args.map_root,
        sensor_root=args.data_root,   # sensor blobs not used; any valid path
        db_files=None,                 # scan data_root for all .db files
        map_version=args.map_version,
        include_cameras=False,
        max_workers=args.workers,
        verbose=True,
    )

    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=args.max_scenarios,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
        expand_scenarios=False,
        remove_invalid_goals=True,
        shuffle=False,   # deterministic order
    )

    worker = Sequential()
    logger.info("Extracting scenarios from .db files …")
    scenarios = builder.get_scenarios(scenario_filter, worker)
    logger.info(f"Found {len(scenarios)} scenarios.")

    if len(scenarios) == 0:
        logger.error("No scenarios found. Check --data_root points to a directory with *.db files.")
        sys.exit(1)

    # ── Cache each scenario ────────────────────────────────────────────────────
    cache = GymScenarioCache(cache_path=args.cache_path, format="gz", compression_level=1)
    Path(args.cache_path).mkdir(parents=True, exist_ok=True)

    failed = 0
    for i, scenario in enumerate(scenarios):
        try:
            cache.save_scenario(scenario)
            if (i + 1) % 50 == 0 or (i + 1) == len(scenarios):
                logger.info(f"  Cached {i+1}/{len(scenarios)} scenarios …")
        except Exception as e:
            logger.warning(f"  Skipping scenario {scenario.token}: {e}")
            failed += 1

    logger.info(
        f"\nDone. {len(scenarios) - failed} scenarios cached to {args.cache_path}/"
        + (f"  ({failed} failed)" if failed else "")
    )
    logger.info("Cache structure: <cache_path>/<log_name>/<scenario_type>/<token>.gz")


if __name__ == "__main__":
    main()
