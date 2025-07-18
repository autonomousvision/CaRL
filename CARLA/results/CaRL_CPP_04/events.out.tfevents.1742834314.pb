�      <��	  ��b��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000001347.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 88342528,
    "max_training_score" : 30.766647338867188,
    "best_iteration" : 1245,
    "latest_iteration" : 1347,
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
text�A;'       ��F	   �b��A���**

charts/restart  �?�Q�5/       m]P	  ��b��A���**

charts/episodic_return�A��3       � V	  ��b��A���**#
!
charts/windowed_avg_return�A���	/       m]P	  ��b��A���**

charts/episodic_length ��C��P-       <A��	  ��b��A���**

charts/learning_rate:�89_j�*       ����	  ��b��A���**

losses/value_loss�h�=�V]�+       ��K	  ��b��A���**

losses/policy_lossW	�b1�'       ��F	  ��b��A���**

losses/entropy�0o����-       <A��	  ��b��A���**

losses/old_approx_kl��;����)       7�_ 	  ��b��A���**

losses/approx_klT�;���(       �pJ	  ��b��A���**

losses/clipfrac�zE>�T2       $V�	  ��b��A���**"
 
losses/discounted_returns�=
@���#       ��wC	  ��b��A���**


charts/SPSR�aDwv'       ��F	  ��b��A���**

charts/restart    e�/       m]P	  ��b��A���**

charts/episodic_returnU�sA񪷑3       � V	  ��b��A���**#
!
charts/windowed_avg_return�6;A���/       m]P	  ��b��A���**

charts/episodic_lengthz�DJ"��-       <A��	  ��b��A���**

charts/learning_rate��89�F�d*       ����	  ��b��A���**

losses/value_lossc`�=�.9+       ��K	  ��b��A���**

losses/policy_loss:�@�1��$'       ��F	  ��b��A���**

losses/entropy�z\���^�-       <A��	  ��b��A���**

losses/old_approx_kl51;�{1n)       7�_ 	  ��b��A���**

losses/approx_kl�\G;�lrI(       �pJ	  ��b��A���**

losses/clipfrac `�=�T�2       $V�	  ��b��A���**"
 
losses/discounted_returnsv=@��#       ��wC	  ��b��A���**


charts/SPSQDX�y�'       ��F	  ��b��A���**

charts/restart    A���/       m]P	  @�b��A���**

charts/episodic_return��B�ٚ�3       � V	  @�b��A���**#
!
charts/windowed_avg_return�6�A�3��/       m]P	  @�b��A���**

charts/episodic_length��D:D��-       <A��	   �b��A���**

charts/learning_rate��89��r*       ����	   �b��A���**

losses/value_loss�]h=����+       ��K	   �b��A���**

losses/policy_loss�@���Rd'       ��F	   �b��A���**

losses/entropy��S��\�-       <A��	   �b��A���**

losses/old_approx_kl4M&;vS��)       7�_ 	   �b��A���**

losses/approx_kl�";c|�3(       �pJ	   �b��A���**

losses/clipfracU��=w9V42       $V�	   �b��A���**"
 
losses/discounted_returns�Y@?�1#       ��wC	   �b��A���**


charts/SPSb�JD�B�'       ��F	   �b��A���**

charts/restart    ���/       m]P	  @�b��A���**

charts/episodic_return�DB�@D3       � V	  @�b��A���**#
!
charts/windowed_avg_returnRK�A�Ĭ�/       m]P	  @�b��A���**

charts/episodic_length�
�D6/3-       <A��	   �b��A���**

charts/learning_rate>�89�.qr*       ����	   �b��A���**

losses/value_loss���=z��+       ��K	   �b��A���**

losses/policy_loss�暻<���'       ��F	   �b��A���**

losses/entropy�,U���E-       <A��	   �b��A���**

losses/old_approx_klh<8;"˭�)       7�_ 	   �b��A���**

losses/approx_klfOT;=T�*(       �pJ	   �b��A���**

losses/clipfracV=�=I��2       $V�	   �b��A���**"
 
losses/discounted_returns�@�!֣#       ��wC	   �b��A���**


charts/SPS;~KDtWk'       ��F	   �b��A���**

charts/restart    N�8�/       m]P	   
c��A���**

charts/episodic_return�IBm��[3       � V	   
c��A���**#
!
charts/windowed_avg_return�Y�A�`a/       m]P	   
c��A���**

charts/episodic_length���D(�j-       <A��	  �
c��A���**

charts/learning_rate��89� �1*       ����	  �
c��A���**

losses/value_loss}�=pu�+       ��K	  �
c��A���**

losses/policy_loss�z�:2��'       ��F	  �
c��A���**

losses/entropy#	V��:�z-       <A��	  �
c��A���**

losses/old_approx_kl
��;t�$V)       7�_ 	  �
c��A���**

losses/approx_kl�=�;\Ղ�(       �pJ	  �
c��A���**

losses/clipfracV1F>��2       $V�	  �
c��A���**"
 
losses/discounted_returnsX�@�C[�#       ��wC	  �
c��A���**


charts/SPS�;LD ,N�'       ��F	  �
c��A���**

charts/restart    1��/       m]P	   c��A���**

charts/episodic_return���AhN�3       � V	   c��A���**#
!
charts/windowed_avg_return��A���a/       m]P	   c��A���**

charts/episodic_lengthð�D���-       <A��	  �c��A���**

charts/learning_rate�89�$�>*       ����	  �c��A���**

losses/value_loss��=)Ts�+       ��K	  �c��A���**

losses/policy_loss׼�}���'       ��F	  �c��A���**

losses/entropy]�S��_B�-       <A��	  �c��A���**

losses/old_approx_kl*d�:@�)       7�_ 	  �c��A���**

losses/approx_kl�1;�M}&(       �pJ	  �c��A���**

losses/clipfrac (�=#�s�2       $V�	  �c��A���**"
 
losses/discounted_returns[�	@�--#       ��wC	  �c��A���**


charts/SPS��MDƱ��'       ��F	  �c��A���**

charts/restart    ����/       m]P	   1c��A���**

charts/episodic_returnT�A�L�3       � V	   1c��A���**#
!
charts/windowed_avg_returnˈ�A��p/       m]P	   1c��A���**

charts/episodic_length �D(Q��-       <A��	  �1c��A���**

charts/learning_rateA�89��(�*       ����	  �1c��A���**

losses/value_lossJo=�`j�+       ��K	  �1c��A���**

losses/policy_loss�&$�4��N'       ��F	  �1c��A���**

losses/entropyG��2LK-       <A��	  �1c��A���**

losses/old_approx_kl���;��!�)       7�_ 	  �1c��A���**

losses/approx_kl��;�p0(       �pJ	  �1c��A���**

losses/clipfracV�=4KEu2       $V�	  �1c��A���**"
 
losses/discounted_returns�o@��
#       ��wC	  �1c��A���**


charts/SPS��MD��'       ��F	  �1c��A���**

charts/restart    SJ�