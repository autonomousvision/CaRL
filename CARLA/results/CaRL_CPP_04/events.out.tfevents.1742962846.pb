�      �YW�	  �'���A*�
�
hyperparametersB�B�{
    "py/object" : "rl_config.GlobalConfig",
    "seed" : 1,
    "total_timesteps" : 300000000,
    "learning_rate" : 0.00025000001187436283,
    "num_envs" : 128,
    "num_steps" : 512,
    "gamma" : 0.99000000953674316,
    "gae_lambda" : 0.94999998807907104,
    "num_minibatches" : 4,
    "update_epochs" : 3,
    "norm_adv" : true,
    "clip_coef" : 0.10000000149011612,
    "clip_vloss" : true,
    "ent_coef" : 0.0099999997764825821,
    "vf_coef" : 0.5,
    "max_grad_norm" : 0.5,
    "adam_eps" : 9.9999997473787516e-06,
    "lr_schedule" : "linear",
    "num_eval_runs" : 10,
    "clip_actions" : true,
    "torch_deterministic" : true,
    "collect_device" : "gpu",
    "train_device" : "gpu",
    "rdzv_addr" : "127.0.0.1",
    "tcp_store_port" : 7000,
    "use_dd_ppo_preempt" : 0,
    "dd_ppo_min_perc" : 0.25,
    "dd_ppo_preempt_threshold" : 0.60000002384185791,
    "num_devices" : 8,
    "exp_name" : "AC_PPO_025_04",
    "batch_size" : 65536,
    "minibatch_size" : 16384,
    "num_iterations" : 4577,
    "num_envs_per_proc" : 16,
    "batch_size_per_device" : 8192,
    "minibatch_per_device" : 2048,
    "use_exploration_suggest" : false,
    "use_speed_limit_as_max_speed" : false,
    "beta_min_a_b_value" : 1,
    "use_new_bev_obs" : true,
    "obs_num_channels" : 10,
    "map_folder" : "maps_2ppm_cv",
    "pixels_per_meter" : 2,
    "route_width" : 6,
    "reward_type" : "simple_reward",
    "consider_tl" : true,
    "eval_time" : 1200,
    "terminal_reward" : 0,
    "normalize_rewards" : false,
    "speeding_infraction" : true,
    "min_thresh_lat_dist" : 2,
    "num_route_points_rendered" : 150,
    "use_green_wave" : false,
    "image_encoder" : "roach_ln2",
    "use_comfort_infraction" : true,
    "comfort_penalty_factor" : 0.5,
    "use_layer_norm" : true,
    "use_vehicle_close_penalty" : false,
    "render_green_tl" : true,
    "distribution" : "beta",
    "weight_decay" : 0,
    "use_termination_hint" : true,
    "use_perc_progress" : true,
    "lane_distance_violation_threshold" : 0,
    "lane_dist_penalty_softener" : 1,
    "use_min_speed_infraction" : false,
    "use_leave_route_done" : false,
    "obs_num_measurements" : 8,
    "use_extra_control_inputs" : false,
    "condition_outside_junction" : false,
    "use_layer_norm_policy_head" : true,
    "use_outside_route_lanes" : true,
    "use_max_change_penalty" : false,
    "terminal_hint" : 1,
    "penalize_yellow_light" : false,
    "use_target_point" : false,
    "speeding_multiplier" : 0,
    "use_value_measurements" : true,
    "bev_semantics_width" : 256,
    "bev_semantics_height" : 256,
    "num_value_measurements" : 10,
    "pixels_ev_to_bottom" : 100,
    "use_history" : false,
    "team_code_folder" : "/mnt/bernhard/code/ad_planning/2_carla/team_code_roach",
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000002314.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 151715840,
    "max_training_score" : 35.654960632324219,
    "best_iteration" : 2207,
    "latest_iteration" : 2314,
    "use_off_road_term" : true,
    "off_road_term_perc" : 0.94999998807907104,
    "beta_1" : 0.89999997615814209,
    "beta_2" : 0.99900001287460327,
    "render_speed_lines" : true,
    "use_new_stop_sign_detector" : true,
    "use_positional_encoding" : false,
    "use_ttc" : true,
    "ttc_resolution" : 2,
    "ttc_penalty_ticks" : 100,
    "render_yellow_time" : true,
    "use_single_reward" : true,
    "use_rl_termination_hint" : true,
    "render_shoulder" : false,
    "use_shoulder_channel" : true,
    "use_survival_reward" : false,
    "survival_reward_magnitude" : 9.9999997473787516e-05
}J

text�$Z'       ��F	  @*���A���H*

charts/restart  �? et�/       m]P	  �=���A���H*

charts/episodic_returnG� AwGe53       � V	  �=���A���H*#
!
charts/windowed_avg_returnG� A�ru�/       m]P	  �=���A���H*

charts/episodic_length�eC	�7	-       <A��	  �>���A���H*

charts/learning_rate�9<l��*       ����	  �>���A���H*

losses/value_loss��=���L+       ��K	  �>���A���H*

losses/policy_loss*P���]r'       ��F	  �>���A���H*

losses/entropy����9{.�-       <A��	  �>���A���H*

losses/old_approx_kld�:p$�)       7�_ 	  �>���A���H*

losses/approx_kl}^;rl(       �pJ	  �>���A���H*

losses/clipfrac ��=[�2       $V�	  �>���A���H*"
 
losses/discounted_returnsT{#@0ÇZ#       ��wC	  �>���A���H*


charts/SPSLD�q�\'       ��F	  �>���A���H*

charts/restart    �7/       m]P	   S���A���H*

charts/episodic_return)�A�|�3       � V	   S���A���H*#
!
charts/windowed_avg_return�gCA@�/       m]P	   S���A���H*

charts/episodic_length�"DMv-       <A��	  �S���A���H*

charts/learning_rate99F��*       ����	  �S���A���H*

losses/value_lossr�=t'��+       ��K	  �S���A���H*

losses/policy_loss�i���Q�'       ��F	  �S���A���H*

losses/entropy�[���q�-       <A��	  �S���A���H*

losses/old_approx_kl>%�:,L�)       7�_ 	  �S���A���H*

losses/approx_kl��>;�a(       �pJ	  �S���A���H*

losses/clipfrac��=,l��2       $V�	  �S���A���H*"
 
losses/discounted_returnsU@yS�#       ��wC	  �S���A���H*


charts/SPS�GDpEU'       ��F	  �S���A���H*

charts/restart    �W�a/       m]P	   j���A���H*

charts/episodic_return�Bx��}3       � V	   j���A���H*#
!
charts/windowed_avg_return�ΛA��Z8/       m]P	   j���A���H*

charts/episodic_length˂DR�Y-       <A��	  �j���A���H*

charts/learning_rate�p9S_�4*       ����	  �j���A���H*

losses/value_loss�͋=`�S�+       ��K	  �j���A���H*

losses/policy_loss�؋�!��'       ��F	  �j���A���H*

losses/entropy�/����yx-       <A��	  �j���A���H*

losses/old_approx_kl�-;:�£)       7�_ 	  �j���A���H*

losses/approx_kl���:ŽZ�(       �pJ	  �j���A���H*

losses/clipfrac P�=9��2       $V�	  �j���A���H*"
 
losses/discounted_returnsC�@��#       ��wC	  �j���A���H*


charts/SPS!?DYm�z'       ��F	  �j���A���H*

charts/restart    ��eV