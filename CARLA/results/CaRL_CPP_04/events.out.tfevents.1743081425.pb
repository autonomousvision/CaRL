�      �YW�	  @�S��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000003215.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 210763776,
    "max_training_score" : 39.693675994873047,
    "best_iteration" : 3081,
    "latest_iteration" : 3215,
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
text}rM'       ��F	  ��S��A���d*

charts/restart  �?d3�/       m]P	  �T��A���d*

charts/episodic_return1<�@F��}3       � V	  �T��A���d*#
!
charts/windowed_avg_return1<�@E��/       m]P	  �T��A���d*

charts/episodic_length  pC�u��-       <A��	  �	T��A���d*

charts/learning_rate�8Oo=�*       ����	  �	T��A���d*

losses/value_lossf�=�c�+       ��K	  �	T��A���d*

losses/policy_loss"����iP'       ��F	  �	T��A���d*

losses/entropy1X����=�-       <A��	  �	T��A���d*

losses/old_approx_klZ&�:���7)       7�_ 	  �	T��A���d*

losses/approx_kl�;ơ��(       �pJ	  �	T��A���d*

losses/clipfrac�z�=8�Q�2       $V�	  �	T��A���d*"
 
losses/discounted_returnsI�6@��v#       ��wC	  �	T��A���d*


charts/SPS��\D����'       ��F	  �	T��A���d*

charts/restart    {�I/       m]P	  �T��A���d*

charts/episodic_returnZ �A����3       � V	  �T��A���d*#
!
charts/windowed_avg_returnf}AH��/       m]P	  �T��A���d*

charts/episodic_length��;D�]D-       <A��	  @T��A���d*

charts/learning_rate+ɛ8k�*       ����	  @T��A���d*

losses/value_loss�ԙ=$�2�+       ��K	  @T��A���d*

losses/policy_loss�2<�S*o�'       ��F	  @T��A���d*

losses/entropyN�����f-       <A��	  @T��A���d*

losses/old_approx_kl&y;���)       7�_ 	  @T��A���d*

losses/approx_kl`';�5*(       �pJ	  @T��A���d*

losses/clipfracV�=��f2       $V�	  @T��A���d*"
 
losses/discounted_returnsD=+@�[�#       ��wC	  @T��A���d*


charts/SPS�@VDz�-'       ��F	  @T��A���d*

charts/restart    +�W/       m]P	  �0T��A���d*

charts/episodic_returnȆB��3h3       � V	  �0T��A���d*#
!
charts/windowed_avg_return�	�A�'-H/       m]P	  �0T��A���d*

charts/episodic_lengthX��D�}X-       <A��	  @1T��A���d*

charts/learning_rateګ�8h�*       ����	  @1T��A���d*

losses/value_lossSI�=f"�+       ��K	  @1T��A���d*

losses/policy_lossaS�@['       ��F	  @1T��A���d*

losses/entropy=�������-       <A��	  @1T��A���d*

losses/old_approx_klދ;;�{�)       7�_ 	  @1T��A���d*

losses/approx_kl};q�G(       �pJ	  @1T��A���d*

losses/clipfracV��=)}�2       $V�	  @1T��A���d*"
 
losses/discounted_returns`�@G�/x#       ��wC	  @1T��A���d*


charts/SPS"uRD�H'       ��F	  @1T��A���d*

charts/restart    �n/       m]P	   ET��A���d*

charts/episodic_return��&BWF%�3       � V	   ET��A���d*#
!
charts/windowed_avg_return�u�A���/       m]P	   ET��A���d*

charts/episodic_length颩DfX�	-       <A��	  �ET��A���d*

charts/learning_rate���8���*       ����	  �ET��A���d*

losses/value_loss���=��7h+       ��K	  �ET��A���d*

losses/policy_loss�ɘ�#l5`'       ��F	  �ET��A���d*

losses/entropy�̘��e�-       <A��	  �ET��A���d*

losses/old_approx_kl��H;O� )       7�_ 	  �ET��A���d*

losses/approx_klf%;-j~J(       �pJ	  �ET��A���d*

losses/clipfrac ��=kt�2       $V�	  �ET��A���d*"
 
losses/discounted_returns �@
D�#       ��wC	  �ET��A���d*


charts/SPS��OD��NO'       ��F	  �ET��A���d*

charts/restart    ��