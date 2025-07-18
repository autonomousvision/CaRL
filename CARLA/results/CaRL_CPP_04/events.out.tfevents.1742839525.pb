�      <��	  @�g��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000001384.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 90767360,
    "max_training_score" : 32.861976623535156,
    "best_iteration" : 1374,
    "latest_iteration" : 1384,
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
textQ'       ��F	  @�g��A���+*

charts/restart  �?��V/       m]P	  @�g��A���+*

charts/episodic_returnL�@��
�3       � V	  @�g��A���+*#
!
charts/windowed_avg_returnL�@.p�/       m]P	  @�g��A���+*

charts/episodic_length�N�CP&��-       <A��	  @�g��A���+*

charts/learning_rate��69ہ�*       ����	  @�g��A���+*

losses/value_loss��=ˍc�+       ��K	  @�g��A���+*

losses/policy_lossR�»����'       ��F	  @�g��A���+*

losses/entropyvo�$p�-       <A��	  @�g��A���+*

losses/old_approx_kll�i;6ɜ)       7�_ 	  @�g��A���+*

losses/approx_kl��P;��Z�(       �pJ	  @�g��A���+*

losses/clipfrac���=$Om�2       $V�	  @�g��A���+*"
 
losses/discounted_returns:@N]�#       ��wC	  @�g��A���+*


charts/SPSc�ZDa�j
'       ��F	  @�g��A���+*

charts/restart    ^�ɉ/       m]P	  @�g��A���+*

charts/episodic_returnT�A5�3       � V	  @�g��A���+*#
!
charts/windowed_avg_return�>AYƫ!/       m]P	  @�g��A���+*

charts/episodic_length��D��`-       <A��	   �g��A���+*

charts/learning_rate�69�;Iy*       ����	   �g��A���+*

losses/value_loss��=z��+       ��K	   �g��A���+*

losses/policy_loss�1���6?'       ��F	   �g��A���+*

losses/entropyomf�/�@-       <A��	   �g��A���+*

losses/old_approx_kl�P�:;>~�)       7�_ 	   �g��A���+*

losses/approx_klx:K;wv�x(       �pJ	   �g��A���+*

losses/clipfrac ��=�Y�o2       $V�	   �g��A���+*"
 
losses/discounted_returnsT�@�TA#       ��wC	   �g��A���+*


charts/SPS�_D`=#�'       ��F	   �g��A���+*

charts/restart    ���b/       m]P	  ��g��A���+*

charts/episodic_return�B]�k3       � V	  ��g��A���+*#
!
charts/windowed_avg_returnA��A��"�/       m]P	  ��g��A���+*

charts/episodic_length�ƗD��-       <A��	   �g��A���+*

charts/learning_rateg�69�E�*       ����	   �g��A���+*

losses/value_lossH�o=L0)7+       ��K	   �g��A���+*

losses/policy_lossS���a�N'       ��F	   �g��A���+*

losses/entropy�b� ֙�-       <A��	   �g��A���+*

losses/old_approx_klL=c;���)       7�_ 	   �g��A���+*

losses/approx_kl��S;Z�N(       �pJ	   �g��A���+*

losses/clipfracU=�=Cm�2       $V�	   �g��A���+*"
 
losses/discounted_returns�@�.�-#       ��wC	   �g��A���+*


charts/SPS��\D�~�'       ��F	   �g��A���+*

charts/restart    ���/       m]P	   h��A���+*

charts/episodic_return�BR�s3       � V	   h��A���+*#
!
charts/windowed_avg_return���A���]/       m]P	   h��A���+*

charts/episodic_length�)�D��-       <A��	  �h��A���+*

charts/learning_rate��69����*       ����	  �h��A���+*

losses/value_loss9�g='��+       ��K	  �h��A���+*

losses/policy_loss�������'       ��F	  �h��A���+*

losses/entropy4\�T���-       <A��	  �h��A���+*

losses/old_approx_klh�\;����)       7�_ 	  �h��A���+*

losses/approx_kl��;��Z�(       �pJ	  �h��A���+*

losses/clipfracV-�=M7�A2       $V�	  �h��A���+*"
 
losses/discounted_returns�1@vd�#       ��wC	  �h��A���+*


charts/SPSU�YD+��'       ��F	  �h��A���+*

charts/restart    ʪ�.