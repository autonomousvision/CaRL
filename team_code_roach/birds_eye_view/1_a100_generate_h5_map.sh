#!/bin/bash

start=`date +%s`
echo "START TIME: $(date)"
export SCENARIO_RUNNER_ROOT=/mnt/bernhard/code/ad_planning/2_carla/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/mnt/bernhard/code/ad_planning/2_carla/custom_leaderboard/leaderboard
export CARLA_ROOT=/mnt/bernhard/carla_0_9_15
export CODE_ROOT=/mnt/bernhard/code/ad_planning/2_carla/team_code_roach
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:${CODE_ROOT}

python -u birdview_map_opencv.py --save_dir /mnt/bernhard/code/ad_planning/2_carla/team_code_roach/birds_eye_view/maps_8ppm_cv --carla_root /mnt/bernhard/carla_0_9_15 --gpu_id 0 --carla_sh_path /mnt/bernhard/carla_0_9_15/CarlaUE4.sh --pixels_per_meter 8.0


echo "END TIME: $(date)"
