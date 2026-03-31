#!/bin/bash
#SBATCH --job-name=Test_TF_H100
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
ulimit -u 65535

start=`date +%s`
echo "START TIME: $(date)"
export CODE_ROOT=/weka/geiger/bjaeger25/ad_planning/2_carla
export SCENARIO_RUNNER_ROOT=${CODE_ROOT}/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=${CODE_ROOT}/custom_leaderboard/leaderboard
export CARLA_ROOT=/weka/geiger/bjaeger25/custom_carla/carla/Dist/CARLA_Shipping_0.9.15-431-g23eac0359/LinuxNoEditor
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export RECORD=0
export SAVE_PATH=/weka/geiger/bjaeger25/ad_planning/2_carla/results/debug_vids

#export NCCL_IB_HCA=ibp185s0
#export NCCL_IB_DISABLE=0
#export NCCL_IB_GID_INDEX=0
#export NCCL_NET_GDR_LEVEL=SYS
# export NCCL_DEBUG=INFO

#65536 , 16384
repetition=1
program_seed=$((000 + 100 * repetition))
start_port=$((1024 + 1000 * repetition))
ex_name=$(printf "Test_H100_TF_PPO_067_%02d" ${repetition})
python -u train_parallel.py --train_cpp 0 --NO_CARS 0 --train_mode rl_sensorimotor --team_code_folder ${CODE_ROOT}/team_code_roach --ml_cloud 0 --num_nodes 1 --node_id 0 --rdzv_addr 127.0.0.1 --rdzv_port 49151 --collect_device gpu --train_device gpu --PYTORCH_KERNEL_CACHE_PATH /weka/geiger/bjaeger25/.cache --ppo_cpp_install_path /weka/geiger/bjaeger25/ppo.cpp/build --cpp_singularity_file_path /weka/geiger/bjaeger25/ppo.cpp/tools/ppo_cpp.sif --git_root ${CODE_ROOT} --cpp_system_lib_path_1 ${CODE_ROOT} --cpp_system_lib_path_2 ${CODE_ROOT}/results --carla_root "${CARLA_ROOT}" --exp_name "${ex_name}" --use_dd_ppo_preempt 0 --num_envs_per_gpu 8 --seed ${program_seed} --start_port ${start_port} --gpu_ids 0 --train_towns 1 2 3 4 5 6 7 10 --num_envs_per_node 8 --total_batch_size 1024 --total_minibatch_size 128 --num_grad_acc_steps 2 --total_timesteps 30000000 --learning_rate=0.00025 --consider_tl 1 --use_occlusion_check 1 --use_green_wave 0 --routes_folder 1000_meters_old_scenarios_04 --route_repetitions 1 --use_sensorimotor 1 --image_architecture rl_resnet34 --lidar_architecture rl_resnet18 --matmul_precision high --allow_tf32 1 --torch_deterministic 0 --benchmark 1 --use_rl_termination_hint 1 --use_termination_hint 1 --track 1 --teacher_path /weka/geiger/bjaeger25/ad_planning/2_carla/results/CaRL_1_3_012_01/model_final.pth --forward_kl 1 --start_kl_percentage 1.0 --end_kl_percentage 1.0 --use_lstm 1 --use_speed_limit 0 --features_dim 512 &
wait

end=`date +%s`
runtime=$((end-start))

echo "END TIME: $(date)"
printf 'Runtime: %dd:%dh:%dm:%ds\n' $((${runtime}/86400)) $((${runtime}%86400/3600)) $((${runtime}%3600/60)) $((${runtime}%60)) 2>&1 | tee ${CODE_ROOT}/results/"${ex_name}"/train_time.txt