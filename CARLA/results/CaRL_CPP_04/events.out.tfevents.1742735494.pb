�      ��`�	  �!��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000000488.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 32047104,
    "max_training_score" : 21.303253173828125,
    "best_iteration" : 488,
    "latest_iteration" : 488,
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
text�߉ '       ��F	  @&��A���*

charts/restart  �?<G��/       m]P	  �8��A���*

charts/episodic_return/��@giRL3       � V	  �8��A���*#
!
charts/windowed_avg_return/��@���A/       m]P	  �8��A���*

charts/episodic_length��fCtP�-       <A��	  �9��A���*

charts/learning_rate#j9^�0�*       ����	  �9��A���*

losses/value_loss��=?�6+       ��K	  �9��A���*

losses/policy_loss��;WI� '       ��F	  �9��A���*

losses/entropy�5C�A���-       <A��	  �9��A���*

losses/old_approx_klL��;���)       7�_ 	  �9��A���*

losses/approx_kl9�;�}��(       �pJ	  �9��A���*

losses/clipfrac��<>T�2       $V�	  �9��A���*"
 
losses/discounted_returns2�@-i�#       ��wC	  �9��A���*


charts/SPSKZD�'h�'       ��F	  �9��A���*

charts/restart    �$�/       m]P	  �J��A���*

charts/episodic_returnWDA���3       � V	  �J��A���*#
!
charts/windowed_avg_return��A�8�/       m]P	  �J��A���*

charts/episodic_length `D��/7-       <A��	  �K��A���*

charts/learning_ratedj9�?�k*       ����	  �K��A���*

losses/value_loss��=�c�+       ��K	  �K��A���*

losses/policy_lossD.X���'       ��F	  �K��A���*

losses/entropyS]C�+{�1-       <A��	  �K��A���*

losses/old_approx_kl^��;��_)       7�_ 	  �K��A���*

losses/approx_kl�>;;'s2(       �pJ	  �K��A���*

losses/clipfracU��=o�2       $V�	  �K��A���*"
 
losses/discounted_returns@�@p�#       ��wC	  �K��A���*


charts/SPSn�]D/HaN'       ��F	  �K��A���*

charts/restart    �s�/       m]P	  �]��A���*

charts/episodic_return-��A��[�3       � V	  �]��A���*#
!
charts/windowed_avg_return},hAoo�/       m]P	  �]��A���*

charts/episodic_length>jD ��-       <A��	  �^��A���*

charts/learning_rate�j9a�x*       ����	  �^��A���*

losses/value_lossI��=&�eB+       ��K	  �^��A���*

losses/policy_loss��Թ&ޞ�'       ��F	  �^��A���*

losses/entropyZ�9�7�`-       <A��	  �^��A���*

losses/old_approx_kl,<�pk)       7�_ 	  �^��A���*

losses/approx_kl4~�;�&�(       �pJ	  �^��A���*

losses/clipfracUI
>��m�2       $V�	  �^��A���*"
 
losses/discounted_returns�"�?���O#       ��wC	  �^��A���*


charts/SPSo�[D��'       ��F	  �^��A���*

charts/restart    ��^�/       m]P	  �r��A���*

charts/episodic_return�1�A�h�g3       � V	  �r��A���*#
!
charts/windowed_avg_return*]�A6o�/       m]P	  �r��A���*

charts/episodic_lengthg��D|<�]-       <A��	  �s��A���*

charts/learning_rate�i9Q[�*       ����	  �s��A���*

losses/value_loss^��=��$�+       ��K	  �s��A���*

losses/policy_loss� r�}��'       ��F	  �s��A���*

losses/entropy-�-�;�S-       <A��	  �s��A���*

losses/old_approx_kll�;a���)       7�_ 	  �s��A���*

losses/approx_kl��U;����(       �pJ	  �s��A���*

losses/clipfrac ��=~n�2       $V�	  �s��A���*"
 
losses/discounted_returns�P�?e�#       ��wC	  �s��A���*


charts/SPSQ%UDO�rn'       ��F	  �s��A���*

charts/restart    ��~�/       m]P	  ����A���*

charts/episodic_return���A��ݛ3       � V	  ����A���*#
!
charts/windowed_avg_returnR��A'��W/       m]P	  ����A���*

charts/episodic_length��D\�Ȧ-       <A��	  @���A���*

charts/learning_rateg�i9�z�A*       ����	  @���A���*

losses/value_loss�=}��_+       ��K	  @���A���*

losses/policy_loss��n��\�'       ��F	  @���A���*

losses/entropyH0�H+/-       <A��	  @���A���*

losses/old_approx_kln�:S���)       7�_ 	  @���A���*

losses/approx_klR'J;.�V�(       �pJ	  @���A���*

losses/clipfrac��=ԝB�2       $V�	  @���A���*"
 
losses/discounted_returnsp��?g�IK#       ��wC	  @���A���*


charts/SPS��OD5��'       ��F	  @���A���*

charts/restart    �N�