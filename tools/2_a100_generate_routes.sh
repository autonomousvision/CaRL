#!/bin/bash
echo "START TIME: $(date)"
export SCENARIO_RUNNER_ROOT=/mnt/bernhard/code/ad_planning/2_carla/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/mnt/bernhard/code/ad_planning/2_carla/custom_leaderboard/leaderboard
export CARLA_ROOT=/mnt/bernhard/carla_0_9_15
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

for i in $(seq 16 31); do
  python -u generate_long_routes_with_scenarios.py --save_folder /mnt/bernhard/code/ad_planning/2_carla/custom_leaderboard/leaderboard/data/roach_preprocessed_routes6 --carla_root /mnt/bernhard/carla_0_9_15 --start_repetition ${i} --scenario_dilation 150 --scenario_runner_root /mnt/bernhard/code/ad_planning/2_carla/custom_leaderboard/scenario_runner --generate_scenarios 0 &
done
wait


echo "END TIME: $(date)"
