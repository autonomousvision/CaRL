#!/bin/bash
#SBATCH --job-name=C_009
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:8
#SBATCH --mem=1800G
#SBATCH --cpus-per-task=256
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/carl_2_0_009_%a_%A.out
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/carl_2_0_009_%a_%A.err
#SBATCH --partition=a100-fat-galvani

# Define the source folder (change this to your actual folder path)
#ORG_CARLA_ROOT="/mnt/lustre/work/geiger/bjaeger25/custom_carla/CARLA_Shipping_0.9.15-422-gf0e332a38/LinuxNoEditor"
#
## Define the destination folder in scratch
#NEW_CARLA_ROOT="$SCRATCH/$(basename "$ORG_CARLA_ROOT")"
#
#mkdir -p "$NEW_CARLA_ROOT"
#rsync -a --info=progress2 --ignore-existing "$ORG_CARLA_ROOT/" "$NEW_CARLA_ROOT/"


NEW_CARLA_ROOT="/mnt/lustre/work/geiger/bjaeger25/custom_carla/CARLA_Shipping_0.9.15-422-gf0e332a38/LinuxNoEditor"

# Echo the new path
echo "Folder copied to: $NEW_CARLA_ROOT"
ls $NEW_CARLA_ROOT
ulimit -n 65535

start=`date +%s`
echo "START TIME: $(date)"
export CODE_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla
export SCENARIO_RUNNER_ROOT=${CODE_ROOT}/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=${CODE_ROOT}/custom_leaderboard/leaderboard
export CARLA_ROOT=$NEW_CARLA_ROOT
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}


repetition=2 #$(($SLURM_ARRAY_TASK_ID))
program_seed=$((000 + 100 * repetition))
start_port=$((1024 + 1000 * repetition))
ex_name=$(printf "CaRL_2_0_009_%02d" ${repetition})
python -u train_parallel.py --debug 0 --train_cpp 0 --team_code_folder ${CODE_ROOT}/team_code_roach --ml_cloud 1 --num_nodes 1 --node_id 0 --rdzv_addr 127.0.0.1 --rdzv_port 0 --collect_device gpu --train_device gpu --PYTORCH_KERNEL_CACHE_PATH /home/rocky/.cache --ppo_cpp_install_path /scratch_local/rocky/code/ppo.cpp/install/bin --cpp_singularity_file_path /scratch_local/rocky/code/ppo.cpp/tools/ppo_cpp.sif --git_root ${CODE_ROOT} --cpp_system_lib_path_1 ${CODE_ROOT} --cpp_system_lib_path_2 ${CODE_ROOT}/results --carla_root "${CARLA_ROOT}" --exp_name "${ex_name}" --use_dd_ppo_preempt 0 --num_envs_per_gpu 32 --seed ${program_seed} --start_port ${start_port} --gpu_ids 0 1 2 3 4 5 6 7 --train_towns 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 --num_envs_per_node 256 --total_batch_size 65536 --total_minibatch_size 16384 --total_timesteps 300000000 --consider_tl 1 --use_green_wave 0 --routes_folder roach_preprocessed_routes6 --route_repetitions 20 --track 1 --use_sensorimotor 1 --image_architecture convnext_zepto_rms_ols.ra4_e3600_r224_in1k --lidar_architecture convnext_zepto_rms_ols.ra4_e3600_r224_in1k --matmul_precision high --allow_tf32 1 --torch_deterministic 0 --benchmark 1 &
wait

end=`date +%s`
runtime=$((end-start))

echo "END TIME: $(date)"
printf 'Runtime: %dd:%dh:%dm:%ds\n' $((${runtime}/86400)) $((${runtime}%86400/3600)) $((${runtime}%3600/60)) $((${runtime}%60)) 2>&1 | tee ${CODE_ROOT}/results/"${ex_name}"/train_time.txt
