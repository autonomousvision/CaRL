�      �YW�	   S��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000003208.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 210305024,
    "max_training_score" : 39.693675994873047,
    "best_iteration" : 3081,
    "latest_iteration" : 3208,
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
text[�(�'       ��F	   S��A���d*

charts/restart  �?�U��/       m]P	    S��A���d*

charts/episodic_return�Am\�y3       � V	    S��A���d*#
!
charts/windowed_avg_return�A��Y�/       m]P	    S��A���d*

charts/episodic_length�6|CI|�-       <A��	   !S��A���d*

charts/learning_rateų�8^��R*       ����	   !S��A���d*

losses/value_loss�U�=���t+       ��K	   !S��A���d*

losses/policy_losso.����DH'       ��F	   !S��A���d*

losses/entropy�ܢ� ���-       <A��	   !S��A���d*

losses/old_approx_kl��-;r@�)       7�_ 	   !S��A���d*

losses/approx_kl�;;�N(       �pJ	   !S��A���d*

losses/clipfrac 0�=_�[�2       $V�	   !S��A���d*"
 
losses/discounted_returns�%@'G#       ��wC	   !S��A���d*


charts/SPS$y[DI�5�'       ��F	   !S��A���d*

charts/restart    � {/       m]P	  @4S��A���d*

charts/episodic_return6�A5�3       � V	  @4S��A���d*#
!
charts/windowed_avg_return��]Ao�/       m]P	  @4S��A���d*

charts/episodic_length�D"D���-       <A��	   5S��A���d*

charts/learning_rateq��8�(<�*       ����	   5S��A���d*

losses/value_loss ��=�O+       ��K	   5S��A���d*

losses/policy_lossK��7z��'       ��F	   5S��A���d*

losses/entropy���(B7�-       <A��	   5S��A���d*

losses/old_approx_klhi;(o	R)       7�_ 	   5S��A���d*

losses/approx_kl;�5;�G�(       �pJ	   5S��A���d*

losses/clipfrac px=�{�b2       $V�	   5S��A���d*"
 
losses/discounted_returns�b@��#       ��wC	   5S��A���d*


charts/SPS�*TD�Iq�'       ��F	   5S��A���d*

charts/restart    IE�/       m]P	  �LS��A���d*

charts/episodic_returnRB?b(3       � V	  �LS��A���d*#
!
charts/windowed_avg_return>��A։�0/       m]P	  �LS��A���d*

charts/episodic_length�ԑD0\-       <A��	  @MS��A���d*

charts/learning_ratey�8~��*       ����	  @MS��A���d*

losses/value_lossd'�=�԰[+       ��K	  @MS��A���d*

losses/policy_loss������u'       ��F	  @MS��A���d*

losses/entropy϶��DB�-       <A��	  @MS��A���d*

losses/old_approx_kl��
;.)       7�_ 	  @MS��A���d*

losses/approx_klq�;��8f(       �pJ	  @MS��A���d*

losses/clipfracV�l=�Z�2       $V�	  @MS��A���d*"
 
losses/discounted_returns��@D���#       ��wC	  @MS��A���d*


charts/SPSy�CDBnk�'       ��F	  @MS��A���d*

charts/restart    �</       m]P	  @aS��A���d*

charts/episodic_return�CB�ڡ3       � V	  @aS��A���d*#
!
charts/windowed_avg_return^+�AN��/       m]P	  @aS��A���d*

charts/episodic_length�{�D@���-       <A��	  �aS��A���d*

charts/learning_rate�[�89�M(*       ����	  �aS��A���d*

losses/value_loss4�=gO�+       ��K	  �aS��A���d*

losses/policy_loss�U^��[�4'       ��F	  �aS��A���d*

losses/entropy�[����U-       <A��	  �aS��A���d*

losses/old_approx_kl���:��e�)       7�_ 	  �aS��A���d*

losses/approx_kle��:�Ĥ�(       �pJ	  �aS��A���d*

losses/clipfrac c=����2       $V�	  �aS��A���d*"
 
losses/discounted_returns��@��x�#       ��wC	  �aS��A���d*


charts/SPS�ADDS�'       ��F	  �aS��A���d*

charts/restart     �i/       m]P	  @wS��A���d*

charts/episodic_returnp&)B1Y��3       � V	  @wS��A���d*#
!
charts/windowed_avg_returnEe�A@��/       m]P	  @wS��A���d*

charts/episodic_length �DF+�E-       <A��	   xS��A���d*

charts/learning_ratew>�80;>#*       ����	   xS��A���d*

losses/value_lossDt�=O���+       ��K	   xS��A���d*

losses/policy_loss���F�?�'       ��F	   xS��A���d*

losses/entropy���&�ϛ-       <A��	   xS��A���d*

losses/old_approx_kl!?B;�m�)       7�_ 	   xS��A���d*

losses/approx_kl�;G-A�(       �pJ	   xS��A���d*

losses/clipfrac�Zn=�P�2       $V�	   xS��A���d*"
 
losses/discounted_returns&z@�*��#       ��wC	   xS��A���d*


charts/SPS�BDPW��'       ��F	   xS��A���d*

charts/restart    ��ٶ/       m]P	  ��S��A���d*

charts/episodic_return��B�A33       � V	  ��S��A���d*#
!
charts/windowed_avg_returnM�A�E�/       m]P	  ��S��A���d*

charts/episodic_length�ƺDD�Z�-       <A��	  @�S��A���d*

charts/learning_rate%!�8</|p*       ����	  @�S��A���d*

losses/value_loss���=S��+       ��K	  @�S��A���d*

losses/policy_lossr���W�C�'       ��F	  @�S��A���d*

losses/entropy����x\	�-       <A��	  @�S��A���d*

losses/old_approx_kl�s{;�/·)       7�_ 	  @�S��A���d*

losses/approx_kl�A�::�3(       �pJ	  @�S��A���d*

losses/clipfrac��u=R�x�2       $V�	  @�S��A���d*"
 
losses/discounted_returns�@���C#       ��wC	  @�S��A���d*


charts/SPS��AD�Vb'       ��F	  @�S��A���d*

charts/restart    �j��/       m]P	  ��S��A���d*

charts/episodic_return� BF���3       � V	  ��S��A���d*#
!
charts/windowed_avg_return���AF�P�/       m]P	  ��S��A���d*

charts/episodic_length�E�D5"pz-       <A��	  ��S��A���d*

charts/learning_rate��8��*       ����	  ��S��A���d*

losses/value_loss��=Kƭ+       ��K	  ��S��A���d*

losses/policy_loss�_����U�'       ��F	  ��S��A���d*

losses/entropyY��,�.�-       <A��	  ��S��A���d*

losses/old_approx_kl���;H��)       7�_ 	  ��S��A���d*

losses/approx_kl8:*;>�^(       �pJ	  ��S��A���d*

losses/clipfrac �u=|��2       $V�	  ��S��A���d*"
 
losses/discounted_returns,�@�6��#       ��wC	  ��S��A���d*


charts/SPS@K@DQP}�'       ��F	  ��S��A���d*

charts/restart    ��!5