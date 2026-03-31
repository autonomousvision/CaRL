#!/bin/bash
#SBATCH --job-name=PPO_167
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=18
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/PPO_167_%a_%A.out
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/PPO_167_%a_%A.err
#SBATCH --partition=2080-galvani

start=`date +%s`
echo "START TIME: $(date)"

# print info about current job
scontrol show job $SLURM_JOB_ID

# define the node 0 hostname:port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo ${MASTER_ADDR}
MASTER_IP=$(getent hosts "${MASTER_ADDR}" | awk '{ print $1 }')
# The master node seems not to be able to resolve its own address. In this case we check our own adress.
if [ -z "${MASTER_IP}" ]
then
  MASTER_IP=$(hostname -I | awk '{ print $1 }')
fi
echo ${MASTER_IP}
MASTER_PORT=6000

export SCENARIO_RUNNER_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/custom_leaderboard/leaderboard
export CARLA_ROOT=/mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

repetition=$((0))
program_seed=$((000 + 100 * repetition))
start_port=$((1024 + 1000 * repetition))
ex_name=$(printf "PPO_167_%02d" ${repetition})
LAUNCHER="python -u train_parallel.py --num_envs_per_node 8 --num_nodes ${SLURM_NNODES} --node_id \$SLURM_PROCID --rdzv_addr ${MASTER_IP} --rdzv_port ${MASTER_PORT} --git_root /mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla --carla_root /mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15 --exp_name "${ex_name}" --use_dd_ppo_preempt True --num_envs_per_gpu 8 --seed ${program_seed}  --start_port ${start_port} --gpu_ids 0 1 0 1 --train_towns 1 1 1 1 2 2 2 2 --total_batch_size 2048 --norm_adv True --clip_vloss True --update_epochs 3 --ent_coef 0.0 --vf_coef 0.5 --gamma 0.995 --gae_lambda 0.95 --clip_coef 0.1 --max_grad_norm 0.5 --total_minibatch_size 512 --learning_rate 2.5e-4 --total_timesteps 10000000 --lr_schedule linear --use_exploration_suggest False --track True --use_speed_limit_as_max_speed False --beta_min_a_b_value 1.0 --use_new_bev_obs True --obs_num_channels 8 --compile_model False --cpu_collect False --reward_type simple_reward --consider_tl True --eval_time 600 --terminal_reward 0.0 --normalize_rewards False --speeding_infraction True --min_thresh_lat_dist 2.0 --map_folder maps_2ppm_cv --pixels_per_meter 2 --route_width 8 --num_route_points_rendered 150 --use_green_wave False --image_encoder roach_ln --use_comfort_infraction False --use_layer_norm True --use_vehicle_close_penalty False --routes_folder roach_preprocessed_routes6 --render_green_tl True --distribution beta --use_rpo False --rpo_alpha 0.5 --use_termination_hint True --use_perc_progress True --use_min_speed_infraction True --use_leave_route_done False --use_temperature False --use_layer_norm_policy_head True --obs_num_measurements 8 --use_extra_control_inputs False --use_hl_gauss_value_loss False --condition_outside_junction False --use_outside_route_lanes True --use_max_change_penalty True --terminal_hint 3.0 --use_lstm False --penalize_yellow_light True --use_target_point False "

echo ${LAUNCHER}

SRUN_ARGS=" \
    --wait=0 \
    --kill-on-bad-exit=0 \
    --jobid $SLURM_JOB_ID \
    "

#while true; do foo; sleep 2; done
srun $SRUN_ARGS bash -c "${LAUNCHER}"

end=`date +%s`
runtime=$((end-start))


echo "END TIME: $(date)"
printf 'Runtime: %dd:%dh:%dm:%ds\n' $((${runtime}/86400)) $((${runtime}%86400/3600)) $((${runtime}%3600/60)) $((${runtime}%60))
