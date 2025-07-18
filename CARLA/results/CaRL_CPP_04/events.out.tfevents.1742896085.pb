�      �YW�	  @����A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000001792.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 117506048,
    "max_training_score" : 33.135311126708984,
    "best_iteration" : 1620,
    "latest_iteration" : 1792,
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
text�$�^'       ��F	  @����A���8*

charts/restart  �?:�7/       m]P	   ���A���8*

charts/episodic_return��@ȡE]3       � V	   ���A���8*#
!
charts/windowed_avg_return��@�MN,/       m]P	   ���A���8*

charts/episodic_length��C���(-       <A��	   	���A���8*

charts/learning_rate�s9wxk�*       ����	   	���A���8*

losses/value_loss�=؍�H+       ��K	   	���A���8*

losses/policy_lossxbI����b'       ��F	   	���A���8*

losses/entropy䝒�ӡ#B-       <A��	   	���A���8*

losses/old_approx_kly^�:c�N�)       7�_ 	   	���A���8*

losses/approx_klx�+;2V�-(       �pJ	   	���A���8*

losses/clipfrac���=}�E�2       $V�	   	���A���8*"
 
losses/discounted_returns�G@�׮ #       ��wC	   	���A���8*


charts/SPS�F]DrNR'       ��F	   	���A���8*

charts/restart    w��/       m]P	  @���A���8*

charts/episodic_returnJ��A�m>�3       � V	  @���A���8*#
!
charts/windowed_avg_returny=Al�%�/       m]P	  @���A���8*

charts/episodic_length[�D4�n-       <A��	  ����A���8*

charts/learning_rate�d9��*       ����	  ����A���8*

losses/value_lossB�=�~-�+       ��K	  ����A���8*

losses/policy_loss������_b'       ��F	  ����A���8*

losses/entropy����9-       <A��	  ����A���8*

losses/old_approx_kl��^;0ϱ)       7�_ 	  ����A���8*

losses/approx_kl@�v;�$�(       �pJ	  ����A���8*

losses/clipfrac�Z�=)J*�2       $V�	  ����A���8*"
 
losses/discounted_returnsr3@�s�%#       ��wC	  ����A���8*


charts/SPS��UD�9��'       ��F	  ����A���8*

charts/restart    ?�܉/       m]P	  �/���A���8*

charts/episodic_return9I�A���3       � V	  �/���A���8*#
!
charts/windowed_avg_return��A��&/       m]P	  �/���A���8*

charts/episodic_length�҂DT��o-       <A��	  @0���A���8*

charts/learning_rate:V9V�]*       ����	  @0���A���8*

losses/value_lossQ�t=�1��+       ��K	  @0���A���8*

losses/policy_loss̀��͵�'       ��F	  @0���A���8*

losses/entropy+����f�-       <A��	  @0���A���8*

losses/old_approx_klY�;�7�C)       7�_ 	  @0���A���8*

losses/approx_kl�x;x�BP(       �pJ	  @0���A���8*

losses/clipfrac Ȳ=��2       $V�	  @0���A���8*"
 
losses/discounted_returnsO@d�[�#       ��wC	  @0���A���8*


charts/SPS �TD�:�'       ��F	  @0���A���8*

charts/restart    ��#/       m]P	  @C���A���8*

charts/episodic_returnԞBT��3       � V	  @C���A���8*#
!
charts/windowed_avg_return��A� �Y/       m]P	  @C���A���8*

charts/episodic_length���Dk��-       <A��	   D���A���8*

charts/learning_rate�G9R��*       ����	   D���A���8*

losses/value_lossq<�= ���+       ��K	   D���A���8*

losses/policy_lossྤ��@'       ��F	   D���A���8*

losses/entropy��j�Y-       <A��	   D���A���8*

losses/old_approx_kl�4�;X�+)       7�_ 	   D���A���8*

losses/approx_kl�yF;A��(       �pJ	   D���A���8*

losses/clipfrac�r�=��n2       $V�	   D���A���8*"
 
losses/discounted_returns�@�2#       ��wC	   D���A���8*


charts/SPS��SDH��'       ��F	   D���A���8*

charts/restart    �q޲/       m]P	  �X���A���8*

charts/episodic_return� B/�{Q3       � V	  �X���A���8*#
!
charts/windowed_avg_return���A�lq/       m]P	  �X���A���8*

charts/episodic_length���D�EZ1-       <A��	  @Y���A���8*

charts/learning_rate�89���C*       ����	  @Y���A���8*

losses/value_loss���=�;��+       ��K	  @Y���A���8*

losses/policy_loss�B��w�'       ��F	  @Y���A���8*

losses/entropy�م�����-       <A��	  @Y���A���8*

losses/old_approx_kl
��;��)       7�_ 	  @Y���A���8*

losses/approx_kl���;aʿ&(       �pJ	  @Y���A���8*

losses/clipfrac X�=Հ��2       $V�	  @Y���A���8*"
 
losses/discounted_returnsL�@?��e#       ��wC	  @Y���A���8*


charts/SPS�NOD���'       ��F	  @Y���A���8*

charts/restart    UO5/       m]P	  �n���A���8*

charts/episodic_return��Bɦ3       � V	  �n���A���8*#
!
charts/windowed_avg_return�w�A^��]/       m]P	  �n���A���8*

charts/episodic_length�`�D���-       <A��	   o���A���8*

charts/learning_rate>*9�|R�*       ����	   o���A���8*

losses/value_loss�=���+       ��K	   o���A���8*

losses/policy_loss �����S'       ��F	   o���A���8*

losses/entropy�5}��V*-       <A��	   o���A���8*

losses/old_approx_kl��;J��)       7�_ 	   o���A���8*

losses/approx_kl���;���X(       �pJ	   o���A���8*

losses/clipfrac���=�i
�2       $V�	   o���A���8*"
 
losses/discounted_returns���?�#V#       ��wC	   o���A���8*


charts/SPS��KD�T�V'       ��F	   o���A���8*

charts/restart    ;�Q�/       m]P	  @����A���8*

charts/episodic_return��B�an3       � V	  @����A���8*#
!
charts/windowed_avg_return8�A�q�/       m]P	  @����A���8*

charts/episodic_length h�DN��m-       <A��	   ����A���8*

charts/learning_rate�9���*       ����	   ����A���8*

losses/value_loss�1o=��+       ��K	   ����A���8*

losses/policy_loss�v�q7="'       ��F	   ����A���8*

losses/entropy����px0�-       <A��	   ����A���8*

losses/old_approx_kl��`;��Ԭ)       7�_ 	   ����A���8*

losses/approx_klh3;%%@�(       �pJ	   ����A���8*

losses/clipfracU�>/"�W2       $V�	   ����A���8*"
 
losses/discounted_returns%
@G��#       ��wC	   ����A���8*


charts/SPS4]ID:��Y'       ��F	   ����A���8*

charts/restart    e�