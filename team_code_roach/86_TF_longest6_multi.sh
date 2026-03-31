#!/bin/bash
ulimit -n 65535
ulimit -u 65535

start=`date +%s`
echo "START TIME: $(date)"
export CODE_ROOT=/data${1}/bernhard/code/ad_planning/2_carla
export SCENARIO_RUNNER_ROOT=${CODE_ROOT}/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=${CODE_ROOT}/custom_leaderboard/leaderboard
export CARLA_ROOT=/data${1}/bernhard/custom_carla_0_9_15/carla/Dist/CARLA_Shipping_0.9.15-431-g23eac0359/LinuxNoEditor
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export RECORD=0
export SAVE_PATH=/data${1}/bernhard/code/ad_planning/2_carla/results/debug_vids

export NCCL_IB_HCA=ibp185s0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=0
export NCCL_NET_GDR_LEVEL=SYS
# export NCCL_DEBUG=INFO

#65536 , 16384
repetition=1
program_seed=$((000 + 100 * repetition))
start_port=$((1024 + 1000 * repetition))
ex_name=$(printf "TF_PPO_067_%02d" ${repetition})
python -u train_parallel.py --train_cpp 0 --NO_CARS 0 --train_mode rl_sensorimotor --team_code_folder ${CODE_ROOT}/team_code_roach --ml_cloud 0 --num_nodes 2 --node_id ${1} --rdzv_addr 134.2.12.60 --rdzv_port 49151 --collect_device gpu --train_device gpu --PYTORCH_KERNEL_CACHE_PATH /home/bernhard/.cache --ppo_cpp_install_path /data${1}/bernhard/code/ppo.cpp/build --cpp_singularity_file_path /data${1}/bernhard/code/ppo.cpp/tools/ppo_cpp.sif --git_root ${CODE_ROOT} --cpp_system_lib_path_1 ${CODE_ROOT} --cpp_system_lib_path_2 ${CODE_ROOT}/results --carla_root "${CARLA_ROOT}" --exp_name "${ex_name}" --use_dd_ppo_preempt 1 --num_envs_per_gpu 8 --seed ${program_seed} --start_port ${start_port} --gpu_ids 0 1 2 3 4 5 6 7 --train_towns 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 10 10 10 10 10 10 10 10 --num_envs_per_node 64 --total_batch_size 16384 --total_minibatch_size 2048 --num_grad_acc_steps 2 --total_timesteps 30000000 --learning_rate=0.00025 --consider_tl 1 --use_occlusion_check 1 --use_green_wave 0 --routes_folder 1000_meters_old_scenarios_04 --route_repetitions 1 --use_sensorimotor 1 --image_architecture rl_resnet34 --lidar_architecture rl_resnet18 --matmul_precision high --allow_tf32 1 --torch_deterministic 0 --benchmark 1 --use_rl_termination_hint 1 --use_termination_hint 1 --track 1 --teacher_path /data${1}/bernhard/code/ad_planning/2_carla/results/CaRL_1_3_012_01/model_final.pth --forward_kl 1 --start_kl_percentage 1.0 --end_kl_percentage 1.0 --use_lstm 1 --use_speed_limit 0 --features_dim 512 &
wait

end=`date +%s`
runtime=$((end-start))

echo "END TIME: $(date)"
printf 'Runtime: %dd:%dh:%dm:%ds\n' $((${runtime}/86400)) $((${runtime}%86400/3600)) $((${runtime}%3600/60)) $((${runtime}%60)) 2>&1 | tee ${CODE_ROOT}/results/"${ex_name}"/train_time.txt