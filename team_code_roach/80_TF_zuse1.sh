#!/bin/bash
ulimit -n 65535
ulimit -u 65535

start=`date +%s`
echo "START TIME: $(date)"
export CODE_ROOT=/data1/bernhard/code/ad_planning/2_carla
export SCENARIO_RUNNER_ROOT=${CODE_ROOT}/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=${CODE_ROOT}/custom_leaderboard/leaderboard
export CARLA_ROOT=/data1/bernhard/custom_carla_0_9_15/carla/Dist/CARLA_Shipping_0.9.15-429-g090538917/LinuxNoEditor
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export RECORD=0
export SAVE_PATH=/data1/bernhard/code/ad_planning/2_carla/results/debug_vids

#65536 , 16384
repetition=1
program_seed=$((000 + 100 * repetition))
start_port=$((1024 + 1000 * repetition))
ex_name=$(printf "Test_TF_PPO_038_%02d" ${repetition})
python -u train_parallel.py --train_cpp 0 --NO_CARS 1 --train_mode rl_sensorimotor --team_code_folder ${CODE_ROOT}/team_code_roach --ml_cloud 0 --num_nodes 1 --node_id 0 --rdzv_addr 127.0.0.1 --rdzv_port 0 --cpu_collect 1 --collect_device gpu --train_device gpu --PYTORCH_KERNEL_CACHE_PATH /home/bernhard/.cache --ppo_cpp_install_path /data1/bernhard/code/ppo.cpp/install/bin --cpp_singularity_file_path /data1/bernhard/code/ppo.cpp/tools/ppo_cpp.sif --git_root ${CODE_ROOT} --cpp_system_lib_path_1 ${CODE_ROOT} --cpp_system_lib_path_2 ${CODE_ROOT}/results --carla_root "${CARLA_ROOT}" --exp_name "${ex_name}" --use_dd_ppo_preempt 0 --num_envs_per_gpu 8 --seed ${program_seed} --start_port ${start_port} --gpu_ids 2 3 --train_towns 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 --num_envs_per_node 16 --total_batch_size 8192 --total_minibatch_size 2048 --total_timesteps 40000000 --learning_rate=0.001 --consider_tl 1 --use_occlusion_check 0 --use_green_wave 0 --routes_folder 1000_meters_no_scenarios_weathers_01 --route_repetitions 10 --track 1 --use_sensorimotor 1 --image_architecture convnext_zepto_rms_ols.ra4_e3600_r224_in1k --lidar_architecture convnext_zepto_rms_ols.ra4_e3600_r224_in1k --matmul_precision high --allow_tf32 1 --torch_deterministic 0 --benchmark 1 --use_rl_termination_hint 0 --use_termination_hint 0 &
wait

end=`date +%s`
runtime=$((end-start))

echo "END TIME: $(date)"
printf 'Runtime: %dd:%dh:%dm:%ds\n' $((${runtime}/86400)) $((${runtime}%86400/3600)) $((${runtime}%3600/60)) $((${runtime}%60)) 2>&1 | tee ${CODE_ROOT}/results/"${ex_name}"/train_time.txt