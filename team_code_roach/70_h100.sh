#!/bin/bash
#SBATCH --job-name=TF_PPO_022
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=88G
#SBATCH --cpus-per-task=32
#SBATCH --output=/weka/geiger/bjaeger25/ad_planning/2_carla/results/logs/DD_PPO_025_%a_%A.out
#SBATCH --error=/weka/geiger/bjaeger25/ad_planning/2_carla/results/logs/DD_PPO_025_%a_%A.err
#SBATCH --partition=h100-ferranti

ulimit -n 65535
ulimit -u 8096

start=`date +%s`
echo "START TIME: $(date)"
export CODE_ROOT=/weka/geiger/bjaeger25/ad_planning/2_carla
export SCENARIO_RUNNER_ROOT=${CODE_ROOT}/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=${CODE_ROOT}/custom_leaderboard/leaderboard
export CARLA_ROOT=/weka/geiger/bjaeger25/custom_carla/carla/Dist/CARLA_Shipping_0.9.15-429-g090538917/LinuxNoEditor
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}


repetition=0
program_seed=$((000 + 100 * repetition))
start_port=$((1024 + 1000 * repetition))
ex_name=$(printf "Test_TF_PPO_026_%02d" ${repetition})
python -u train_parallel.py --train_cpp 0  --train_mode rl_sensorimotor --team_code_folder ${CODE_ROOT}/team_code_roach --ml_cloud 1 --num_nodes 1 --node_id 0 --rdzv_addr 127.0.0.1 --rdzv_port 0 --collect_device gpu --train_device gpu --PYTORCH_KERNEL_CACHE_PATH /weka/geiger/bjaeger25/.cache --ppo_cpp_install_path /weka/geiger/bjaeger25/ppo.cpp/install/bin --cpp_singularity_file_path /weka/geiger/bjaeger25/ppo.cpp/tools/ppo_cpp.sif --git_root ${CODE_ROOT} --cpp_system_lib_path_1 ${CODE_ROOT} --cpp_system_lib_path_2 ${CODE_ROOT}/results --carla_root "${CARLA_ROOT}" --exp_name "${ex_name}" --use_dd_ppo_preempt 0 --num_envs_per_gpu 1 --seed ${program_seed} --start_port ${start_port} --gpu_ids 0 --train_towns 1 --num_envs_per_node 1 --total_batch_size 512 --total_minibatch_size 128 --total_timesteps 1000000 --consider_tl 1 --use_green_wave 0 --routes_folder roach_preprocessed_routes6 --route_repetitions 20 --track 1 --use_sensorimotor 1 --image_architecture convnext_zepto_rms_ols.ra4_e3600_r224_in1k --lidar_architecture convnext_zepto_rms_ols.ra4_e3600_r224_in1k --matmul_precision high --allow_tf32 1 --torch_deterministic 0 --benchmark 1 &
wait

end=`date +%s`
runtime=$((end-start))

echo "END TIME: $(date)"
printf 'Runtime: %dd:%dh:%dm:%ds\n' $((${runtime}/86400)) $((${runtime}%86400/3600)) $((${runtime}%3600/60)) $((${runtime}%60)) 2>&1 | tee ${CODE_ROOT}/results/"${ex_name}"/train_time.txt
