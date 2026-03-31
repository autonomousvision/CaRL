#!/bin/bash
#SBATCH --job-name=PPO_188
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/PPO_188_%a_%A.out
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/PPO_188_%a_%A.err
#SBATCH --partition=a100-fat-galvani

start=`date +%s`
echo "START TIME: $(date)"
export SCENARIO_RUNNER_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/leaderboard
export CARLA_ROOT=/mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

for i in $(seq 1 4); do
    repetition=$((i+0))
    program_seed=$((000 + 100 * repetition))
    start_port=$((1024 + 1000 * repetition))
    gpu=$((-1 + 1 * repetition)) # ${gpu}
    ex_name=$(printf "PPO_188_%02d" ${repetition})
    python -u train_parallel.py --num_nodes 1 --node_id 0 --rdzv_addr localhost --rdzv_port 0  --git_root /mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla --carla_root /mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15 --exp_name "${ex_name}" --use_dd_ppo_preempt False --num_envs_per_gpu 8 --seed ${program_seed}  --start_port ${start_port} --gpu_ids ${gpu} --train_towns 6 6 6 6 6 6 6 6 --num_envs_per_node 8 --total_batch_size 1024 --norm_adv True --clip_vloss True --update_epochs 3 --ent_coef 0.0 --vf_coef 0.5 --gamma 0.995 --gae_lambda 0.95 --clip_coef 0.1 --max_grad_norm 0.5 --total_minibatch_size 512 --learning_rate 2.5e-4 --total_timesteps 10000000 --lr_schedule linear --use_exploration_suggest False --track True --use_speed_limit_as_max_speed False --beta_min_a_b_value 1.0 --use_new_bev_obs True --obs_num_channels 9 --compile_model False --cpu_collect False --reward_type simple_reward --consider_tl False --eval_time 1200 --terminal_reward 0.0 --normalize_rewards False --speeding_infraction True --min_thresh_lat_dist 2.0 --map_folder maps_2ppm_cv --pixels_per_meter 2 --route_width 8 --num_route_points_rendered 150 --use_green_wave False --image_encoder roach_ln --use_comfort_infraction False --use_layer_norm True --use_vehicle_close_penalty False --routes_folder roach_preprocessed_routes6 --render_green_tl True --distribution beta --use_rpo False --rpo_alpha 0.5 --use_termination_hint True --use_perc_progress False --use_min_speed_infraction False --use_leave_route_done False --use_temperature False --use_layer_norm_policy_head False --obs_num_measurements 8 --use_extra_control_inputs False --use_hl_gauss_value_loss False --condition_outside_junction False --use_outside_route_lanes True --use_max_change_penalty False --terminal_hint 3.0 --use_lstm False --penalize_yellow_light True --use_target_point False --speeding_multiplier 0.0 --use_value_measurements True --torch_deterministic True --allow_tf32 False --benchmark False --matmul_precision highest --bev_semantics_width 192 --bev_semantics_height 192 --pixels_ev_to_bottom 40 &
done
wait

end=`date +%s`
runtime=$((end-start))

echo "END TIME: $(date)"
printf 'Runtime: %dd:%dh:%dm:%ds\n' $((${runtime}/86400)) $((${runtime}%86400/3600)) $((${runtime}%3600/60)) $((${runtime}%60))
