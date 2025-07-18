�      <��	  @_J��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000001126.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 73859072,
    "max_training_score" : 29.604303359985352,
    "best_iteration" : 1098,
    "latest_iteration" : 1126,
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
text�\
#'       ��F	   `J��A���#*

charts/restart  �?Uz�/       m]P	  �qJ��A���#*

charts/episodic_returnI�@_��e3       � V	  �qJ��A���#*#
!
charts/windowed_avg_returnI�@Tz/       m]P	  �qJ��A���#*

charts/episodic_length  rC����-       <A��	  �rJ��A���#*

charts/learning_rate��E9�N�*       ����	  �rJ��A���#*

losses/value_lossu	�=����+       ��K	  �rJ��A���#*

losses/policy_loss͗�>���'       ��F	  �rJ��A���#*

losses/entropy͆[� ��<-       <A��	  �rJ��A���#*

losses/old_approx_kl61�:�dHS)       7�_ 	  �rJ��A���#*

losses/approx_klC�;g)��(       �pJ	  �rJ��A���#*

losses/clipfrac ��=Q^/h2       $V�	  �rJ��A���#*"
 
losses/discounted_returns��@&d��#       ��wC	  �rJ��A���#*


charts/SPS'�^D��k'       ��F	  �rJ��A���#*

charts/restart    ��//       m]P	  ��J��A���#*

charts/episodic_returnb�eA�㵧3       � V	  ��J��A���#*#
!
charts/windowed_avg_returnv"*A����/       m]P	  ��J��A���#*

charts/episodic_length�D,���-       <A��	  ��J��A���#*

charts/learning_rate�E9���*       ����	  ��J��A���#*

losses/value_lossd�=NyQZ+       ��K	  ��J��A���#*

losses/policy_loss ��8�S0q'       ��F	  ��J��A���#*

losses/entropy��E�̰%�-       <A��	  ��J��A���#*

losses/old_approx_kl�r�;ί�)       7�_ 	  ��J��A���#*

losses/approx_kl���;Q��%(       �pJ	  ��J��A���#*

losses/clipfrac�B�=6�#2       $V�	  ��J��A���#*"
 
losses/discounted_returns�=@&�q�#       ��wC	  ��J��A���#*


charts/SPSTYRDX��'       ��F	  ��J��A���#*

charts/restart    ���o/       m]P	  ��J��A���#*

charts/episodic_returnF��A9f�3       � V	  ��J��A���#*#
!
charts/windowed_avg_return(	zAb*"�/       m]P	  ��J��A���#*

charts/episodic_lengthYh`D�GB-       <A��	   �J��A���#*

charts/learning_rate@{E9�R25*       ����	   �J��A���#*

losses/value_lossZn=�d�+       ��K	   �J��A���#*

losses/policy_loss~R��+7�z'       ��F	   �J��A���#*

losses/entropyم7�D�-       <A��	   �J��A���#*

losses/old_approx_klD��:��n)       7�_ 	   �J��A���#*

losses/approx_kl��;t�[8(       �pJ	   �J��A���#*

losses/clipfracU�=4�2       $V�	   �J��A���#*"
 
losses/discounted_returns4� @z��#       ��wC	   �J��A���#*


charts/SPS`�JD,x7�'       ��F	   �J��A���#*

charts/restart    D��/       m]P	  ��J��A���#*

charts/episodic_return+�AL�83       � V	  ��J��A���#*#
!
charts/windowed_avg_return3�A\/       m]P	  ��J��A���#*

charts/episodic_length 0�D��y-       <A��	  ��J��A���#*

charts/learning_rate�lE94,�$*       ����	  ��J��A���#*

losses/value_lossF��=bn�a+       ��K	  ��J��A���#*

losses/policy_loss����2��'       ��F	  ��J��A���#*

losses/entropy<�5����-       <A��	  ��J��A���#*

losses/old_approx_kl?�;�}#)       7�_ 	  ��J��A���#*

losses/approx_klV�M;��_}(       �pJ	  ��J��A���#*

losses/clipfracU5�=��2       $V�	  ��J��A���#*"
 
losses/discounted_returns���?���"#       ��wC	  ��J��A���#*


charts/SPSY�LD�$'       ��F	  ��J��A���#*

charts/restart    |1I|/       m]P	  ��J��A���#*

charts/episodic_return���Aܾ�?3       � V	  ��J��A���#*#
!
charts/windowed_avg_return�/�A��w/       m]P	  ��J��A���#*

charts/episodic_length&W�D�\V�-       <A��	  @�J��A���#*

charts/learning_rate�]E9���*       ����	  @�J��A���#*

losses/value_loss�o=iD%\+       ��K	  @�J��A���#*

losses/policy_loss�^6�w��'       ��F	  @�J��A���#*

losses/entropy<
C��M�n-       <A��	  @�J��A���#*

losses/old_approx_kl	�&;I���)       7�_ 	  @�J��A���#*

losses/approx_klT}A;��F�(       �pJ	  @�J��A���#*

losses/clipfrac�Z�=O���2       $V�	  @�J��A���#*"
 
losses/discounted_returns)��?'#       ��wC	  @�J��A���#*


charts/SPS��LD('$�'       ��F	  @�J��A���#*

charts/restart    O�(�/       m]P	  @�J��A���#*

charts/episodic_returnc+�A�,,3       � V	  @�J��A���#*#
!
charts/windowed_avg_returnK��A9�3/       m]P	  @�J��A���#*

charts/episodic_length�H�D\�`�-       <A��	  ��J��A���#*

charts/learning_rateCOE9O$&*       ����	  ��J��A���#*

losses/value_loss��x=�13+       ��K	  ��J��A���#*

losses/policy_loss�������'       ��F	  ��J��A���#*

losses/entropy��;�k���-       <A��	  ��J��A���#*

losses/old_approx_kl�.;~��~)       7�_ 	  ��J��A���#*

losses/approx_kl�+;�-�(       �pJ	  ��J��A���#*

losses/clipfrac h�=���\2       $V�	  ��J��A���#*"
 
losses/discounted_returns�� @%<�^#       ��wC	  ��J��A���#*


charts/SPS"nJD��'       ��F	  ��J��A���#*

charts/restart    �[@e/       m]P	  @�J��A���#*

charts/episodic_return�_�A״�3       � V	  @�J��A���#*#
!
charts/windowed_avg_return�5�AK�[/       m]P	  @�J��A���#*

charts/episodic_lengthy
�D��&d-       <A��	  ��J��A���#*

charts/learning_rate�@E9��Y*       ����	  ��J��A���#*

losses/value_loss
֛=�p��+       ��K	  ��J��A���#*

losses/policy_loss>R5�T�ϡ'       ��F	  ��J��A���#*

losses/entropy�D�x,;�-       <A��	  ��J��A���#*

losses/old_approx_klF~;��)       7�_ 	  ��J��A���#*

losses/approx_kl��;�*ve(       �pJ	  ��J��A���#*

losses/clipfrac H�=xx�2       $V�	  ��J��A���#*"
 
losses/discounted_returns�>@��#       ��wC	  ��J��A���#*


charts/SPS�QID�-��'       ��F	  ��J��A���#*

charts/restart    @�(Q