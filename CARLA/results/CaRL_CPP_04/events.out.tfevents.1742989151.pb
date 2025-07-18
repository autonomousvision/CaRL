�      �YW�	  �����A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000002523.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 165412864,
    "max_training_score" : 35.654960632324219,
    "best_iteration" : 2207,
    "latest_iteration" : 2523,
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
text��
P'       ��F	  �����A���N*

charts/restart  �?u�y*/       m]P	  �����A���N*

charts/episodic_return��@۰��3       � V	  �����A���N*#
!
charts/windowed_avg_return��@ҥ��/       m]P	  �����A���N*

charts/episodic_length]�|Ce<�-       <A��	  �����A���N*

charts/learning_rate�*�8�?*       ����	  �����A���N*

losses/value_lossD?�=��'7+       ��K	  �����A���N*

losses/policy_loss�.���c��'       ��F	  �����A���N*

losses/entropy������-       <A��	  �����A���N*

losses/old_approx_kln�9;V��)       7�_ 	  �����A���N*

losses/approx_kl��;컥�(       �pJ	  �����A���N*

losses/clipfracU�=x��2       $V�	  �����A���N*"
 
losses/discounted_returnsf�@�7�#       ��wC	  �����A���N*


charts/SPSX�CDE4='       ��F	  �����A���N*

charts/restart    �j�/       m]P	  @���A���N*

charts/episodic_return�o�ANǦ�3       � V	  @���A���N*#
!
charts/windowed_avg_returnv�>Aִ</       m]P	  @���A���N*

charts/episodic_lengtht�,D@u�-       <A��	   ���A���N*

charts/learning_rate��8�B)�*       ����	   ���A���N*

losses/value_loss�g�=�[�l+       ��K	   ���A���N*

losses/policy_loss�E���g�'       ��F	   ���A���N*

losses/entropy^ؐ�[bT�-       <A��	   ���A���N*

losses/old_approx_kl�];u�KS)       7�_ 	   ���A���N*

losses/approx_kln�;(3�~(       �pJ	   ���A���N*

losses/clipfracUe�=�'�2       $V�	   ���A���N*"
 
losses/discounted_returns?e"@��31#       ��wC	   ���A���N*


charts/SPSX#=D�-��'       ��F	   ���A���N*

charts/restart    ����/       m]P	   ���A���N*

charts/episodic_return.QB�ս�3       � V	   ���A���N*#
!
charts/windowed_avg_return�۝At��b/       m]P	   ���A���N*

charts/episodic_lengthv�De*)S-       <A��	  ����A���N*

charts/learning_rateS��8>`�*       ����	  ����A���N*

losses/value_loss�=���L+       ��K	  ����A���N*

losses/policy_loss1܌�(���'       ��F	  ����A���N*

losses/entropy@Ց�j�-       <A��	  ����A���N*

losses/old_approx_kl�W1;*��)       7�_ 	  ����A���N*

losses/approx_klo;3�G\(       �pJ	  ����A���N*

losses/clipfrac ��=GT �2       $V�	  ����A���N*"
 
losses/discounted_returnsk,@oN�#       ��wC	  ����A���N*


charts/SPSU:DE�~�'       ��F	  ����A���N*

charts/restart    �4Η/       m]P	   1���A���O*

charts/episodic_return%B2t!~3       � V	   1���A���O*#
!
charts/windowed_avg_returnzw�A��/       m]P	   1���A���O*

charts/episodic_length���DH�M�-       <A��	  �1���A���O*

charts/learning_rate��8U�*       ����	  �1���A���O*

losses/value_losse�=�`n/+       ��K	  �1���A���O*

losses/policy_loss������`�'       ��F	  �1���A���O*

losses/entropy�N��WJH�-       <A��	  �1���A���O*

losses/old_approx_kl�:+;��	�)       7�_ 	  �1���A���O*

losses/approx_klr�::�(       �pJ	  �1���A���O*

losses/clipfrac�Z�=�$6J2       $V�	  �1���A���O*"
 
losses/discounted_returnsC�@�Jq#       ��wC	  �1���A���O*


charts/SPS�@8D~�'       ��F	  �1���A���O*

charts/restart    }t�/       m]P	  �F���A���O*

charts/episodic_return��	B`�ݥ3       � V	  �F���A���O*#
!
charts/windowed_avg_return���A�K��/       m]P	  �F���A���O*

charts/episodic_length�3�DQd�-       <A��	  �G���A���O*

charts/learning_rate���8./��*       ����	  �G���A���O*

losses/value_loss�
�=�]�F+       ��K	  �G���A���O*

losses/policy_loss.-��$�h'       ��F	  �G���A���O*

losses/entropy�������-       <A��	  �G���A���O*

losses/old_approx_kl��:�.)       7�_ 	  �G���A���O*

losses/approx_kl?�;��s(       �pJ	  �G���A���O*

losses/clipfrac ��=mLL�2       $V�	  �G���A���O*"
 
losses/discounted_returns��@Y�8#       ��wC	  �G���A���O*


charts/SPS�8D��S'       ��F	  �G���A���O*

charts/restart    O��+/       m]P	   \���A���O*

charts/episodic_return�pB�vq3       � V	   \���A���O*#
!
charts/windowed_avg_return�A�f�/       m]P	   \���A���O*

charts/episodic_lengthO�Dܼ�7-       <A��	  �\���A���O*

charts/learning_rate[��8r�-�*       ����	  �\���A���O*

losses/value_loss�Q�=��Uh+       ��K	  �\���A���O*

losses/policy_loss���'       ��F	  �\���A���O*

losses/entropyӑ�!y9-       <A��	  �\���A���O*

losses/old_approx_kl�0;�v	�)       7�_ 	  �\���A���O*

losses/approx_kl;).;��(       �pJ	  �\���A���O*

losses/clipfracUՑ=;A��2       $V�	  �\���A���O*"
 
losses/discounted_returns��@���#       ��wC	  �\���A���O*


charts/SPS�*:D,�u�'       ��F	  �\���A���O*

charts/restart    �N�