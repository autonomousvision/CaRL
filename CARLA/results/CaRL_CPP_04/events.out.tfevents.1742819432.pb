�      <��	   T��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000001208.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 79233024,
    "max_training_score" : 29.604303359985352,
    "best_iteration" : 1098,
    "latest_iteration" : 1208,
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
text���z'       ��F	  �T��A���%*

charts/restart  �?�⧻/       m]P	  @/T��A���%*

charts/episodic_return�(�@^h�83       � V	  @/T��A���%*#
!
charts/windowed_avg_return�(�@Q\�!/       m]P	  @/T��A���%*

charts/episodic_length�b�C�*�-       <A��	  @0T��A���%*

charts/learning_rateF�@9{�cD*       ����	  @0T��A���%*

losses/value_loss�]n=m��+       ��K	  @0T��A���%*

losses/policy_loss��@�'D'       ��F	  @0T��A���%*

losses/entropy SU��욡-       <A��	  @0T��A���%*

losses/old_approx_kl�G�:��א)       7�_ 	  @0T��A���%*

losses/approx_kl��5;B;z�(       �pJ	  @0T��A���%*

losses/clipfracV�=_��T2       $V�	  @0T��A���%*"
 
losses/discounted_returns��	@�=�A#       ��wC	  @0T��A���%*


charts/SPS�qQD�|�/'       ��F	  @0T��A���%*

charts/restart    ,�/       m]P	   DT��A���%*

charts/episodic_return�scA�v3       � V	   DT��A���%*#
!
charts/windowed_avg_return�.A='~�/       m]P	   DT��A���%*

charts/episodic_length�W+Du@��-       <A��	  �DT��A���%*

charts/learning_rate��@9���5*       ����	  �DT��A���%*

losses/value_loss�6�=,$t�+       ��K	  �DT��A���%*

losses/policy_loss8����%W0'       ��F	  �DT��A���%*

losses/entropy��O�B�	-       <A��	  �DT��A���%*

losses/old_approx_klI;�f�f)       7�_ 	  �DT��A���%*

losses/approx_kl�zy;|�˩(       �pJ	  �DT��A���%*

losses/clipfracUu�=NZ2       $V�	  �DT��A���%*"
 
losses/discounted_returns=�@�#       ��wC	  �DT��A���%*


charts/SPS��MD���'       ��F	  �DT��A���%*

charts/restart    ��a�/       m]P	  @YT��A���%*

charts/episodic_return�S�A�Tr3       � V	  @YT��A���%*#
!
charts/windowed_avg_returnh:}A�zt/       m]P	  @YT��A���%*

charts/episodic_length��~D�Ť-       <A��	   ZT��A���%*

charts/learning_rate��@9Q��e*       ����	   ZT��A���%*

losses/value_loss��f=�]+       ��K	   ZT��A���%*

losses/policy_lossd�P��ϓ'       ��F	   ZT��A���%*

losses/entropy0|O�hɞ?-       <A��	   ZT��A���%*

losses/old_approx_kl<�;��j)       7�_ 	   ZT��A���%*

losses/approx_kl�lm;�YP(       �pJ	   ZT��A���%*

losses/clipfrac �=|KN2       $V�	   ZT��A���%*"
 
losses/discounted_returns�%@Ox�O#       ��wC	   ZT��A���%*


charts/SPS��HD����'       ��F	   ZT��A���%*

charts/restart    U�%�/       m]P	  �lT��A���%*

charts/episodic_returnI�BC�3       � V	  �lT��A���%*#
!
charts/windowed_avg_return�[�A�dU/       m]P	  �lT��A���%*

charts/episodic_length ��D�2�!-       <A��	  @mT��A���%*

charts/learning_rateI�@9N��B*       ����	  @mT��A���%*

losses/value_lossԇu=F̯�+       ��K	  @mT��A���%*

losses/policy_loss�D���9X'       ��F	  @mT��A���%*

losses/entropy~+E����0-       <A��	  @mT��A���%*

losses/old_approx_kl�W�;�uV)       7�_ 	  @mT��A���%*

losses/approx_kl,+;�5��(       �pJ	  @mT��A���%*

losses/clipfrac  �=���2       $V�	  @mT��A���%*"
 
losses/discounted_returns�@Q��#       ��wC	  @mT��A���%*


charts/SPS��KD�F�'       ��F	  @mT��A���%*

charts/restart    ��h@/       m]P	  @�T��A���%*

charts/episodic_return�B��Y3       � V	  @�T��A���%*#
!
charts/windowed_avg_returnh��A��>/       m]P	  @�T��A���%*

charts/episodic_lengthL�D����-       <A��	   �T��A���%*

charts/learning_rate��@9��%�*       ����	   �T��A���%*

losses/value_loss�9e=�@�+       ��K	   �T��A���%*

losses/policy_loss܊4���Np'       ��F	   �T��A���%*

losses/entropy|@�2�'-       <A��	   �T��A���%*

losses/old_approx_kl`8�;��`F)       7�_ 	   �T��A���%*

losses/approx_klψ;���r(       �pJ	   �T��A���%*

losses/clipfrac ��=�/2       $V�	   �T��A���%*"
 
losses/discounted_returnsj��?��Jp#       ��wC	   �T��A���%*


charts/SPS�gLDVa�'       ��F	   �T��A���%*

charts/restart    ��2/       m]P	   �T��A���%*

charts/episodic_returnU��A�sF�3       � V	   �T��A���%*#
!
charts/windowed_avg_return�q�As��B/       m]P	   �T��A���%*

charts/episodic_length���DH���-       <A��	  ��T��A���%*

charts/learning_rate��@9*k�*       ����	  ��T��A���%*

losses/value_lossO�w=���C+       ��K	  ��T��A���%*

losses/policy_loss`ܺ^*��'       ��F	  ��T��A���%*

losses/entropy�tC�����-       <A��	  ��T��A���%*

losses/old_approx_kl��?;�G��)       7�_ 	  ��T��A���%*

losses/approx_kl0�;(��(       �pJ	  ��T��A���%*

losses/clipfracV->Q߸�2       $V�	  ��T��A���%*"
 
losses/discounted_returns��@� d�#       ��wC	  ��T��A���%*


charts/SPS@�LD�3��'       ��F	  ��T��A���%*

charts/restart    ��./       m]P	  ��T��A���&*

charts/episodic_return�� BE�/�3       � V	  ��T��A���&*#
!
charts/windowed_avg_return`9�AbK/       m]P	  ��T��A���&*

charts/episodic_lengthQ~�D�n�l-       <A��	  @�T��A���&*

charts/learning_rateL�@9m-O�*       ����	  @�T��A���&*

losses/value_loss��Y=gw�+       ��K	  @�T��A���&*

losses/policy_loss�OS�f0,]'       ��F	  @�T��A���&*

losses/entropy��<��r��-       <A��	  @�T��A���&*

losses/old_approx_kl�c1;�=o)       7�_ 	  @�T��A���&*

losses/approx_kl�!+;	8((       �pJ	  @�T��A���&*

losses/clipfrac���=\�72       $V�	  @�T��A���&*"
 
losses/discounted_returns ' @���/#       ��wC	  @�T��A���&*


charts/SPSBLD䦖h'       ��F	  @�T��A���&*

charts/restart    CM��/       m]P	   �T��A���&*

charts/episodic_return�8�A2�mh3       � V	   �T��A���&*#
!
charts/windowed_avg_returnL��AC�U�/       m]P	   �T��A���&*

charts/episodic_lengthi�D�q-       <A��	  ��T��A���&*

charts/learning_rate�@9�q��*       ����	  ��T��A���&*

losses/value_loss4[l=����+       ��K	  ��T��A���&*

losses/policy_loss��,�r��{'       ��F	  ��T��A���&*

losses/entropy� A�J��-       <A��	  ��T��A���&*

losses/old_approx_kl�\<��9n)       7�_ 	  ��T��A���&*

losses/approx_kln�;,�R�(       �pJ	  ��T��A���&*

losses/clipfrac�
�=�ը	2       $V�	  ��T��A���&*"
 
losses/discounted_returns5�@zzH&#       ��wC	  ��T��A���&*


charts/SPS`�LDEaK5'       ��F	  ��T��A���&*

charts/restart    ���/       m]P	   �T��A���&*

charts/episodic_returnaQ�A�h�3       � V	   �T��A���&*#
!
charts/windowed_avg_return���Aـ�T/       m]P	   �T��A���&*

charts/episodic_lengthQ��D:#j-       <A��	  ��T��A���&*

charts/learning_rate�p@9���i*       ����	  ��T��A���&*

losses/value_loss��o=�ԋ+       ��K	  ��T��A���&*

losses/policy_loss��E��J��'       ��F	  ��T��A���&*

losses/entropyyK�旼Z-       <A��	  ��T��A���&*

losses/old_approx_klb��;9�Y.)       7�_ 	  ��T��A���&*

losses/approx_kl`�;�(       �pJ	  ��T��A���&*

losses/clipfrac p>�522       $V�	  ��T��A���&*"
 
losses/discounted_returns�t
@8�#       ��wC	  ��T��A���&*


charts/SPS�.ND�h�'       ��F	  ��T��A���&*

charts/restart    \:"#/       m]P	   �T��A���&*

charts/episodic_returncBZ�n3       � V	   �T��A���&*#
!
charts/windowed_avg_return@��A�}�/       m]P	   �T��A���&*

charts/episodic_length���Dφ�-       <A��	  ��T��A���&*

charts/learning_ratePb@9���*       ����	  ��T��A���&*

losses/value_loss�{7=���+       ��K	  ��T��A���&*

losses/policy_loss��ƻF"'       ��F	  ��T��A���&*

losses/entropy*"H��Rf"-       <A��	  ��T��A���&*

losses/old_approx_kl��H;m5��)       7�_ 	  ��T��A���&*

losses/approx_kl��;��z(       �pJ	  ��T��A���&*

losses/clipfrac ��=�%��2       $V�	  ��T��A���&*"
 
losses/discounted_returns&@k��#       ��wC	  ��T��A���&*


charts/SPS�MD� b�'       ��F	  ��T��A���&*

charts/restart    �O�/       m]P	  ��T��A���&*

charts/episodic_return�;B�6�3       � V	  ��T��A���&*#
!
charts/windowed_avg_return
(�A/�s�/       m]P	  ��T��A���&*

charts/episodic_length��D��-       <A��	  @�T��A���&*

charts/learning_rate�S@9��[*       ����	  @�T��A���&*

losses/value_lossO`P= ��+       ��K	  @�T��A���&*

losses/policy_loss�́�jG�]'       ��F	  @�T��A���&*

losses/entropy�j=�����-       <A��	  @�T��A���&*

losses/old_approx_kl�e:�R��)       7�_ 	  @�T��A���&*

losses/approx_kl�\;�Qi�(       �pJ	  @�T��A���&*

losses/clipfracVթ=�N�U2       $V�	  @�T��A���&*"
 
losses/discounted_returns�Z@�2��#       ��wC	  @�T��A���&*


charts/SPS{�JD�6��'       ��F	  @�T��A���&*

charts/restart    ��/       m]P	   U��A���&*

charts/episodic_return�� B����3       � V	   U��A���&*#
!
charts/windowed_avg_return�0�A�0н/       m]P	   U��A���&*

charts/episodic_lengthA��D:�f�-       <A��	  �U��A���&*

charts/learning_rate�D@9ձe*       ����	  �U��A���&*

losses/value_loss/�X=��AZ+       ��K	  �U��A���&*

losses/policy_loss�೻�Xa�'       ��F	  �U��A���&*

losses/entropyY�J�cOU�-       <A��	  �U��A���&*

losses/old_approx_klL-;�IB�)       7�_ 	  �U��A���&*

losses/approx_kl�TF;-�1(       �pJ	  �U��A���&*

losses/clipfracV��=Pޗn2       $V�	  �U��A���&*"
 
losses/discounted_returnsO&@Adʸ#       ��wC	  �U��A���&*


charts/SPS�GD .2'       ��F	  �U��A���&*

charts/restart    g�lA/       m]P	   &U��A���&*

charts/episodic_return��A=��3       � V	   &U��A���&*#
!
charts/windowed_avg_return���AMn T/       m]P	   &U��A���&*

charts/episodic_length,(�D~��-       <A��	  �&U��A���&*

charts/learning_rateT6@9R���*       ����	  �&U��A���&*

losses/value_loss.�j=:>�+       ��K	  �&U��A���&*

losses/policy_loss�ZT�L�]g'       ��F	  �&U��A���&*

losses/entropy��7��s�-       <A��	  �&U��A���&*

losses/old_approx_kls�*;w��)       7�_ 	  �&U��A���&*

losses/approx_kl~(P;K���(       �pJ	  �&U��A���&*

losses/clipfracV��=���2       $V�	  �&U��A���&*"
 
losses/discounted_returns��?gc߬#       ��wC	  �&U��A���&*


charts/SPS�DHD]'8'       ��F	  �&U��A���&*

charts/restart    k�$#