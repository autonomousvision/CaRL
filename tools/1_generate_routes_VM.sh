#!/bin/bash

# --- Configuration ---
START_VAL=0
END_VAL=127

# --- Environment Setup ---
export SCENARIO_RUNNER_ROOT=/data${1}/bernhard/code/ad_planning/2_carla/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/data${1}/bernhard/code/ad_planning/2_carla/custom_leaderboard/leaderboard
export CARLA_ROOT=/data${1}/bernhard/custom_carla_0_9_15/carla/Dist/CARLA_Shipping_0.9.15-431-g23eac0359/LinuxNoEditor

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

echo "GLOBAL START TIME: $(date)"

# --- Execution Loop ---
for i in $(seq $START_VAL $END_VAL); do
    echo "------------------------------------------------"
    echo "STARTING REPETITION: $i at $(date)"
    echo "------------------------------------------------"

    python -u generate_long_routes_with_scenarios.py \
        --save_folder /data${1}/bernhard/code/ad_planning/2_carla/custom_leaderboard/leaderboard/data/1000_meters_old_scenarios_05 \
        --carla_root /data${1}/bernhard/custom_carla_0_9_15/carla/Dist/CARLA_Shipping_0.9.15-431-g23eac0359/LinuxNoEditor \
        --start_repetition "$i" \
        --scenario_dilation 100 \
        --scenario_runner_root /data${1}/bernhard/code/ad_planning/2_carla/custom_leaderboard/scenario_runner \
        --generate_scenarios 1 \
        --only_leaderboard_1 1 \
        --route_length 1000 \
        --num_routes 1000 > "logs/log_rep_$i.txt" 2>&1 &

done

wait
echo "GLOBAL END TIME: $(date)"