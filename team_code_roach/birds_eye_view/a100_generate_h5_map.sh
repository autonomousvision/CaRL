#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=900G
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/generate_%a_%A.out
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/generate_%a_%A.err
#SBATCH --partition=a100-galvani

# print info about current job
scontrol show job $SLURM_JOB_ID

echo "START TIME: $(date)"

export SCENARIO_RUNNER_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/leaderboard
export CARLA_ROOT=/mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15
export CODE_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/team_code_roach
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:${CODE_ROOT}

python -u birdview_map_opencv.py --save_dir /mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/team_code_roach/birds_eye_view/maps_5ppm_cv --carla_root /mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15 --gpu_id 0 --carla_sh_path /mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15/CarlaUE4.sh --pixels_per_meter 5.0


echo "END TIME: $(date)"
