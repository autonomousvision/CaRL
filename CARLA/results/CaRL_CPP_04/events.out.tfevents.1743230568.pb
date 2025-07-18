�      �YW�	   ����A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000004325.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 283508736,
    "max_training_score" : 42.957332611083984,
    "best_iteration" : 4193,
    "latest_iteration" : 4325,
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
text�N6%(       �pJ	  @����A����*

charts/restart  �?F�U�0       ���_	   ����A����*

charts/episodic_returnp��@ig&�4       ^3\	   ����A����*#
!
charts/windowed_avg_returnp��@�M,C0       ���_	   ����A����*

charts/episodic_length/:aC��j�.       ��W�	   ����A����*

charts/learning_rateef7b�7+       ��K	   ����A����*

losses/value_loss�N�=���,       ���E	   ����A����*

losses/policy_loss_1��A(       �pJ	   ����A����*

losses/entropyJH��z"�.       ��W�	   ����A����*

losses/old_approx_kl�չ:��*       ����	   ����A����*

losses/approx_kl$�`:K��)       7�_ 	   ����A����*

losses/clipfrac �<5X}�3       � V	   ����A����*"
 
losses/discounted_returns�'@�L7E$       B+�M	   ����A����*


charts/SPSU�RD�")�(       �pJ	   ����A����*

charts/restart    m�>m0       ���_	  @����A����*

charts/episodic_return]۱A+*�4       ^3\	  @����A����*#
!
charts/windowed_avg_return�GdAw���0       ���_	  @����A����*

charts/episodic_length�3"D��F(.       ��W�	   ����A����*

charts/learning_rate�e72̸o+       ��K	   ����A����*

losses/value_lossظ�=h���,       ���E	   ����A����*

losses/policy_loss�/C�q��(       �pJ	   ����A����*

losses/entropyC����p��.       ��W�	   ����A����*

losses/old_approx_kly �:�]��*       ����	   ����A����*

losses/approx_kl'�::7���)       7�_ 	   ����A����*

losses/clipfrac  �<��3       � V	   ����A����*"
 
losses/discounted_returns��*@�"t$       B+�M	   ����A����*


charts/SPS\VD���p(       �pJ	   ����A����*

charts/restart    �`�P0       ���_	   ����A����*

charts/episodic_return�B���4       ^3\	   ����A����*#
!
charts/windowed_avg_return�$�AN�_0       ���_	   ����A����*

charts/episodic_length @|D����.       ��W�	  �����A����*

charts/learning_rate6.d7�+       ��K	  �����A����*

losses/value_lossZf�=�B�,       ���E	  �����A����*

losses/policy_loss�fJ�w�ݕ(       �pJ	  �����A����*

losses/entropy떿%\ݏ.       ��W�	  �����A����*

losses/old_approx_kl�u�9`w0*       ����	  �����A����*

losses/approx_kl�KW:*t��)       7�_ 	  �����A����*

losses/clipfracU��<T�W�3       � V	  �����A����*"
 
losses/discounted_returnsG�@���q$       B+�M	  �����A����*


charts/SPS�PDSylD(       �pJ	  �����A����*

charts/restart    [�|�0       ���_	  �����A����*

charts/episodic_return�0B1�4       ^3\	  �����A����*#
!
charts/windowed_avg_return�(�A4)�0       ���_	  �����A����*

charts/episodic_lengthE��Dz��.       ��W�	  @����A����*

charts/learning_rate�Cc7��S+       ��K	  @����A����*

losses/value_loss|�=9��,       ���E	  @����A����*

losses/policy_lossr�.���v6(       �pJ	  @����A����*

losses/entropy���j��.       ��W�	  @����A����*

losses/old_approx_kl�{I:_���*       ����	  @����A����*

losses/approx_kl��(:�� �)       7�_ 	  @����A����*

losses/clipfrac�J�<UX�3       � V	  @����A����*"
 
losses/discounted_returns,:@�M�B$       B+�M	  @����A����*


charts/SPS��MDv���(       �pJ	  @����A����*

charts/restart    ��[