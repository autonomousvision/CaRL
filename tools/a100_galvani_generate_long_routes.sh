#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/generate_%a_%A.out
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/generate_%a_%A.err
#SBATCH --partition=2080-galvani

# print info about current job
scontrol show job $SLURM_JOB_ID

echo "START TIME: $(date)"

export SCENARIO_RUNNER_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/leaderboard
export CARLA_ROOT=/mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

python -u generate_long_routes.py --save_folder /mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/leaderboard/data/roach_preprocessed_routes4/ --carla_root /mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15 --start_repetition $SLURM_ARRAY_TASK_ID


echo "END TIME: $(date)"
