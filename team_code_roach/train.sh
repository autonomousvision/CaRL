#!/bin/bash
#SBATCH --job-name=PPO_191
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:9
#SBATCH --cpus-per-task=216
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/PPO_191_%a_%A.out
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/PPO_191_%a_%A.err
#SBATCH --partition=a100-fat-galvani

# print info about current job
scontrol show job $SLURM_JOB_ID

start=`date +%s`
echo "START TIME: $(date)"
export SCENARIO_RUNNER_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/leaderboard
export CARLA_ROOT=/mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export ex_base="PPO_191"
export start_rep=1

for i in $(seq 1 1); do
    repetition=$((i+start_rep))
    program_seed=$((000 + 100 * repetition))
    start_port=$((1024 + 1000 * repetition))
    gpu=$((0 + 1 * i))
    ex_name=$(printf "%s_%02d" ${ex_base} ${repetition})
    echo $ex_name
    python -u train_parallel.py --ml_cloud 1 --num_nodes 1 --node_id 0 --rdzv_addr localhost --rdzv_port 0  --git_root /mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla --carla_root /mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15 --exp_name "${ex_name}" --use_dd_ppo_preempt True --num_envs_per_gpu 16 --seed ${program_seed}  --start_port ${start_port} --gpu_ids 0 1 2 3 4 5 6 7 8 --train_towns 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 --num_envs_per_node 144 --total_batch_size 18432 --norm_adv True --clip_vloss True --update_epochs 3 --ent_coef 0.0 --vf_coef 0.5 --gamma 0.995 --gae_lambda 0.95 --clip_coef 0.1 --max_grad_norm 0.5 --total_minibatch_size 4608 --learning_rate 2.5e-4 --total_timesteps 100000000 --lr_schedule linear --use_exploration_suggest False --track True --use_speed_limit_as_max_speed False --beta_min_a_b_value 1.0 --use_new_bev_obs True --obs_num_channels 9 --compile_model False --cpu_collect False --reward_type simple_reward --consider_tl True --eval_time 1200 --terminal_reward 0.0 --normalize_rewards False --speeding_infraction True --min_thresh_lat_dist 2.0 --map_folder maps_2ppm_cv --pixels_per_meter 2 --route_width 8 --num_route_points_rendered 150 --use_green_wave False --image_encoder roach_ln --use_comfort_infraction False --use_layer_norm True --use_vehicle_close_penalty False --routes_folder preprocessed_routes_with_old_scenarios --render_green_tl True --distribution beta --use_rpo False --rpo_alpha 0.5 --use_termination_hint True --use_perc_progress True --use_min_speed_infraction False --use_leave_route_done False --use_temperature False --use_layer_norm_policy_head True --obs_num_measurements 8 --use_extra_control_inputs False --use_hl_gauss_value_loss False --condition_outside_junction False --use_outside_route_lanes True --use_max_change_penalty True --terminal_hint 3.0 --use_lstm False --penalize_yellow_light True --use_target_point False --speeding_multiplier 0.0 --use_value_measurements True --torch_deterministic True --allow_tf32 False --benchmark False --matmul_precision highest --bev_semantics_width 192 --bev_semantics_height 192 --pixels_ev_to_bottom 40 --use_history True &
done
wait

cd ..

for i in $(seq ${repetition} ${repetition}); do
  ex_name=$(printf "%s_%02d" ${ex_base}  ${i})
  python -u evaluate_routes_slurm.py --experiment "${ex_name}" --benchmark longest6_no_scenarios --team_code team_code_roach --epochs model_best model_final --num_repetitions 3 &
done
wait


end=`date +%s`
runtime=$((end-start))

echo "END TIME: $(date)"
printf 'Runtime: %dd:%dh:%dm:%ds\n' $((${runtime}/86400)) $((${runtime}%86400/3600)) $((${runtime}%3600/60)) $((${runtime}%60))