�      ��`�	  @V
��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000000544.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 35717120,
    "max_training_score" : 26.681295394897461,
    "best_iteration" : 523,
    "latest_iteration" : 544,
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
text����'       ��F	  �W
��A���*

charts/restart  �?6Lv/       m]P	   i
��A���*

charts/episodic_return#�A_S�3       � V	   i
��A���*#
!
charts/windowed_avg_return#�A{f��/       m]P	   i
��A���*

charts/episodic_length/z�C����-       <A��	   j
��A���*

charts/learning_rate��f9��Q�*       ����	   j
��A���*

losses/value_loss���=k��+       ��K	   j
��A���*

losses/policy_lossm�����i'       ��F	   j
��A���*

losses/entropy�E�����-       <A��	   j
��A���*

losses/old_approx_kl�P�:�Y�8)       7�_ 	   j
��A���*

losses/approx_kl��Q;,�p(       �pJ	   j
��A���*

losses/clipfrac H�=C &�2       $V�	   j
��A���*"
 
losses/discounted_returnse�@��#       ��wC	   j
��A���*


charts/SPSM�bD;$�8'       ��F	   j
��A���*

charts/restart    ȩ�/       m]P	  �{
��A���*

charts/episodic_return�pA?3       � V	  �{
��A���*#
!
charts/windowed_avg_return�:A>��+/       m]P	  �{
��A���*

charts/episodic_length�9D�ҙ-       <A��	  @|
��A���*

charts/learning_rateO�f9D��*       ����	  @|
��A���*

losses/value_loss��=��w+       ��K	  @|
��A���*

losses/policy_loss2�z�!��A'       ��F	  @|
��A���*

losses/entropy�o4�� ��-       <A��	  @|
��A���*

losses/old_approx_kl��;����)       7�_ 	  @|
��A���*

losses/approx_kl"	�;AuS�(       �pJ	  @|
��A���*

losses/clipfrac  �={)6�2       $V�	  @|
��A���*"
 
losses/discounted_returns?� @�n�7#       ��wC	  @|
��A���*


charts/SPSчaD��6�'       ��F	  @|
��A���*

charts/restart    ��Z/       m]P	  @�
��A���*

charts/episodic_return�̾A�ƶ3       � V	  @�
��A���*#
!
charts/windowed_avg_return��{A3aEp/       m]P	  @�
��A���*

charts/episodic_length�VbDtVh-       <A��	  ��
��A���*

charts/learning_rate��f9�¿�*       ����	  ��
��A���*

losses/value_lossok�=�2��+       ��K	  ��
��A���*

losses/policy_loss0�G<���'       ��F	  ��
��A���*

losses/entropy�0�bd-       <A��	  ��
��A���*

losses/old_approx_kl�߻<����)       7�_ 	  ��
��A���*

losses/approx_kl��r<�x(       �pJ	  ��
��A���*

losses/clipfrac �[>bw@�2       $V�	  ��
��A���*"
 
losses/discounted_returns@H @�4��#       ��wC	  ��
��A���*


charts/SPS�_TD�T��'       ��F	  ��
��A���*

charts/restart    �
/       m]P	  ��
��A���*

charts/episodic_return�A*u�3       � V	  ��
��A���*#
!
charts/windowed_avg_return�4�A���-/       m]P	  ��
��A���*

charts/episodic_length�ޤD���q-       <A��	  ��
��A���*

charts/learning_rate��f9�u�*       ����	  ��
��A���*

losses/value_losss��=�[{b+       ��K	  ��
��A���*

losses/policy_loss@슻� �('       ��F	  ��
��A���*

losses/entropy*]"�s�!-       <A��	  ��
��A���*

losses/old_approx_kl�1;0sb)       7�_ 	  ��
��A���*

losses/approx_kl��*;���(       �pJ	  ��
��A���*

losses/clipfracV]�=7� v2       $V�	  ��
��A���*"
 
losses/discounted_returnsE��?�v��#       ��wC	  ��
��A���*


charts/SPS��PD�O s'       ��F	  ��
��A���*

charts/restart    ��F�/       m]P	   �
��A���*

charts/episodic_return:��A���3       � V	   �
��A���*#
!
charts/windowed_avg_return��AR�.?/       m]P	   �
��A���*

charts/episodic_length�,�DP��-       <A��	  ��
��A���*

charts/learning_rateS�f9�c^�*       ����	  ��
��A���*

losses/value_loss�i�=%��$+       ��K	  ��
��A���*

losses/policy_loss�+!��%�'       ��F	  ��
��A���*

losses/entropypc,���`�-       <A��	  ��
��A���*

losses/old_approx_kl�j;-4')       7�_ 	  ��
��A���*

losses/approx_kl��^;�$��(       �pJ	  ��
��A���*

losses/clipfrac�J�=���2       $V�	  ��
��A���*"
 
losses/discounted_returns���?��L�#       ��wC	  ��
��A���*


charts/SPS�QD"+m�'       ��F	  ��
��A���*

charts/restart    ��[�/       m]P	   �
��A���*

charts/episodic_return_��A�0��3       � V	   �
��A���*#
!
charts/windowed_avg_return�خAV{�7/       m]P	   �
��A���*

charts/episodic_length��DÛ�-       <A��	  ��
��A���*

charts/learning_rate��f9����*       ����	  ��
��A���*

losses/value_loss='�=�8�^+       ��K	  ��
��A���*

losses/policy_loss(��$��'       ��F	  ��
��A���*

losses/entropy�a�����-       <A��	  ��
��A���*

losses/old_approx_kl$��;�Ld�)       7�_ 	  ��
��A���*

losses/approx_kl�d�;?��4(       �pJ	  ��
��A���*

losses/clipfrac `�=K��92       $V�	  ��
��A���*"
 
losses/discounted_returns
]�?f�8#       ��wC	  ��
��A���*


charts/SPS��PD�Whp'       ��F	  ��
��A���*

charts/restart    ���I/       m]P	  ��
��A���*

charts/episodic_return=�ALt�n3       � V	  ��
��A���*#
!
charts/windowed_avg_return�I�A��L /       m]P	  ��
��A���*

charts/episodic_length���D��?-       <A��	  @�
��A���*

charts/learning_rate �f9g�I�*       ����	  @�
��A���*

losses/value_loss��=+Y2+       ��K	  @�
��A���*

losses/policy_lossl�s�¬�~'       ��F	  @�
��A���*

losses/entropy��4� 8-       <A��	  @�
��A���*

losses/old_approx_klQ�K;BR0)       7�_ 	  @�
��A���*

losses/approx_kl�~�;��(       �pJ	  @�
��A���*

losses/clipfrac 8�=E��2       $V�	  @�
��A���*"
 
losses/discounted_returnsP��?&��Q#       ��wC	  @�
��A���*


charts/SPS)YOD6���'       ��F	  @�
��A���*

charts/restart    	
A�/       m]P	  @�
��A���*

charts/episodic_return���A(�Օ3       � V	  @�
��A���*#
!
charts/windowed_avg_returnu޴AP�/       m]P	  @�
��A���*

charts/episodic_length��D����-       <A��	  ��
��A���*

charts/learning_rateV�f9|�
*       ����	  ��
��A���*

losses/value_loss��=)���+       ��K	  ��
��A���*

losses/policy_lossH�����*'       ��F	  ��
��A���*

losses/entropy��Q��-       <A��	  ��
��A���*

losses/old_approx_kl&�|;�v��)       7�_ 	  ��
��A���*

losses/approx_kl�i9;�j��(       �pJ	  ��
��A���*

losses/clipfrac ��=�F��2       $V�	  ��
��A���*"
 
losses/discounted_returnsH>�?&���#       ��wC	  ��
��A���*


charts/SPS�uJD�T+�'       ��F	  ��
��A���*

charts/restart    J(��/       m]P	  ���A���*

charts/episodic_return2 �A�
�3       � V	  ���A���*#
!
charts/windowed_avg_return�S�A��y�/       m]P	  ���A���*

charts/episodic_lengthߤ�DXMz�-       <A��	  @��A���*

charts/learning_rate�xf9fO+�*       ����	  @��A���*

losses/value_lossw>�=��+       ��K	  @��A���*

losses/policy_loss�e��<�I�'       ��F	  @��A���*

losses/entropyز(�7��-       <A��	  @��A���*

losses/old_approx_kl�5�:��&z)       7�_ 	  @��A���*

losses/approx_kl/c);킢�(       �pJ	  @��A���*

losses/clipfrac���=n7�2       $V�	  @��A���*"
 
losses/discounted_returns���?��P�#       ��wC	  @��A���*


charts/SPS�:IDE�d'       ��F	  @��A���*

charts/restart    ��0�/       m]P	  @%��A���*

charts/episodic_return�t�A��3       � V	  @%��A���*#
!
charts/windowed_avg_returnF׶A\J/       m]P	  @%��A���*

charts/episodic_length���DN1�-       <A��	   &��A���*

charts/learning_ratejf9˺�*       ����	   &��A���*

losses/value_lossׇ=�Xc�+       ��K	   &��A���*

losses/policy_loss�;sn�'       ��F	   &��A���*

losses/entropy���0�Y-       <A��	   &��A���*

losses/old_approx_kl��;��!�)       7�_ 	   &��A���*

losses/approx_klp�<���(       �pJ	   &��A���*

losses/clipfrac $>���B2       $V�	   &��A���*"
 
losses/discounted_returnso��?�r��#       ��wC	   &��A���*


charts/SPS��FD�`�'       ��F	   &��A���*

charts/restart    ͞��/       m]P	  @<��A���*

charts/episodic_return���A|3¼3       � V	  @<��A���*#
!
charts/windowed_avg_return�{�A~��/       m]P	  @<��A���*

charts/episodic_lengthUաD�w�W-       <A��	  �<��A���*

charts/learning_rateY[f9�Z��*       ����	  �<��A���*

losses/value_lossl�=���+       ��K	  �<��A���*

losses/policy_loss�Z\��q�'       ��F	  �<��A���*

losses/entropy���-       <A��	  �<��A���*

losses/old_approx_kl5�;a"�)       7�_ 	  �<��A���*

losses/approx_kl��n;)��1(       �pJ	  �<��A���*

losses/clipfrac��=�Iea2       $V�	  �<��A���*"
 
losses/discounted_returns���?�0@2#       ��wC	  �<��A���*


charts/SPS��DDs�\'       ��F	  �<��A���*

charts/restart    �=y�/       m]P	  �P��A���*

charts/episodic_return�N�A����3       � V	  �P��A���*#
!
charts/windowed_avg_returnǢ�A �'{/       m]P	  �P��A���*

charts/episodic_length9.�DA�=�-       <A��	  @Q��A���*

charts/learning_rate�Lf9��*       ����	  @Q��A���*

losses/value_loss��c=�_Yb+       ��K	  @Q��A���*

losses/policy_lossULy�f� '       ��F	  @Q��A���*

losses/entropy���Ϥ-       <A��	  @Q��A���*

losses/old_approx_klUSu;Q�Z2)       7�_ 	  @Q��A���*

losses/approx_kl�7�;k	�(       �pJ	  @Q��A���*

losses/clipfrac ��=b_��2       $V�	  @Q��A���*"
 
losses/discounted_returns	��?%_W�#       ��wC	  @Q��A���*


charts/SPS�1EDDꅩ'       ��F	  @Q��A���*

charts/restart    � //       m]P	  �e��A���*

charts/episodic_return@��Aܿ�3       � V	  �e��A���*#
!
charts/windowed_avg_return�V�A�v,�/       m]P	  �e��A���*

charts/episodic_length�¶D��--       <A��	  @f��A���*

charts/learning_rate>f9g��*       ����	  @f��A���*

losses/value_lossKD�=_}�9+       ��K	  @f��A���*

losses/policy_loss��кh4	�'       ��F	  @f��A���*

losses/entropy
J��㳟-       <A��	  @f��A���*

losses/old_approx_kl{z=;h@�_)       7�_ 	  @f��A���*

losses/approx_kl��;�'Z((       �pJ	  @f��A���*

losses/clipfracUu�=�̂�2       $V�	  @f��A���*"
 
losses/discounted_returns��??&�L#       ��wC	  @f��A���*


charts/SPS��DD7�'       ��F	  @f��A���*

charts/restart    �k�/       m]P	  @{��A���*

charts/episodic_return@l�A��P#3       � V	  @{��A���*#
!
charts/windowed_avg_returnX��A�y�>/       m]P	  @{��A���*

charts/episodic_length� �D�:��-       <A��	   |��A���*

charts/learning_rate]/f9_�*       ����	   |��A���*

losses/value_loss`|x=��&Y+       ��K	   |��A���*

losses/policy_loss�?��âq�'       ��F	   |��A���*

losses/entropy���Gy
�-       <A��	   |��A���*

losses/old_approx_kl�;�vM)       7�_ 	   |��A���*

losses/approx_kl|3;��m;(       �pJ	   |��A���*

losses/clipfrac�:�=u:��2       $V�	   |��A���*"
 
losses/discounted_returns���?���a#       ��wC	   |��A���*


charts/SPSbDD��'       ��F	   |��A���*

charts/restart    �=��/       m]P	  @���A���*

charts/episodic_return��A���{3       � V	  @���A���*#
!
charts/windowed_avg_return�v�AK���/       m]P	  @���A���*

charts/episodic_length�.�D�O|-       <A��	  ����A���*

charts/learning_rate� f9�*       ����	  ����A���*

losses/value_loss�Pu=��c+       ��K	  ����A���*

losses/policy_loss�

�5k�W'       ��F	  ����A���*

losses/entropy����)Wy-       <A��	  ����A���*

losses/old_approx_kl�ڏ;����)       7�_ 	  ����A���*

losses/approx_klo(�;>�n�(       �pJ	  ����A���*

losses/clipfrac�
�=9���2       $V�	  ����A���*"
 
losses/discounted_returns�o�?��2#       ��wC	  ����A���*


charts/SPS�CDxƫ'       ��F	  ����A���*

charts/restart    �@��/       m]P	  ����A���*

charts/episodic_return}�A��h�3       � V	  ����A���*#
!
charts/windowed_avg_return���A�u��/       m]P	  ����A���*

charts/episodic_length� �Dn���-       <A��	  @���A���*

charts/learning_rate
f9���*       ����	  @���A���*

losses/value_loss��=�+       ��K	  @���A���*

losses/policy_loss��r��
� '       ��F	  @���A���*

losses/entropy�� ��<��-       <A��	  @���A���*

losses/old_approx_kl�2;��)       7�_ 	  @���A���*

losses/approx_kl��!;�F�(       �pJ	  @���A���*

losses/clipfracV��=,��L2       $V�	  @���A���*"
 
losses/discounted_returnsjH�?��^�#       ��wC	  @���A���*


charts/SPS��BD�'�'       ��F	  @���A���*

charts/restart    �S/       m]P	  ����A���*

charts/episodic_return1�A�u	�3       � V	  ����A���*#
!
charts/windowed_avg_return>��AlCf�/       m]P	  ����A���*

charts/episodic_lengthU��Dv��-       <A��	  @���A���*

charts/learning_rate`f9X5��*       ����	  @���A���*

losses/value_loss�/�=�|�k+       ��K	  @���A���*

losses/policy_loss8S�$X'       ��F	  @���A���*

losses/entropy�M��g�R-       <A��	  @���A���*

losses/old_approx_kl���;����)       7�_ 	  @���A���*

losses/approx_kl}�8;&��(       �pJ	  @���A���*

losses/clipfracVE�=5�9A2       $V�	  @���A���*"
 
losses/discounted_returns�<�?O�t9#       ��wC	  @���A���*


charts/SPS�gBD�,�2'       ��F	  @���A���*

charts/restart    �!�/       m]P	  @���A���*

charts/episodic_returnH��AO��(3       � V	  @���A���*#
!
charts/windowed_avg_return�+�A�)��/       m]P	  @���A���*

charts/episodic_length�]�D_��-       <A��	   ���A���*

charts/learning_rate��e9��4�*       ����	   ���A���*

losses/value_lossj�`=pC!>+       ��K	   ���A���*

losses/policy_loss��Q��l'       ��F	   ���A���*

losses/entropy�U�2�rJ-       <A��	   ���A���*

losses/old_approx_kl�	 ;��')       7�_ 	   ���A���*

losses/approx_kl���;͖i(       �pJ	   ���A���*

losses/clipfrac���=dL92       $V�	   ���A���*"
 
losses/discounted_returns���?�UZm#       ��wC	   ���A���*


charts/SPS��BD�k�'       ��F	   ���A���*

charts/restart    � 3�/       m]P	  @���A���*

charts/episodic_return�A�APRnp3       � V	  @���A���*#
!
charts/windowed_avg_returnD�A��̣/       m]P	  @���A���*

charts/episodic_length�{�D�/��-       <A��	  ����A���*

charts/learning_rate�e9�	*       ����	  ����A���*

losses/value_loss8c�=���+       ��K	  ����A���*

losses/policy_loss��:=cP'       ��F	  ����A���*

losses/entropy����#�9R-       <A��	  ����A���*

losses/old_approx_kle�u;�ά)       7�_ 	  ����A���*

losses/approx_kl�H�;氋(       �pJ	  ����A���*

losses/clipfracU�=�s5F2       $V�	  ����A���*"
 
losses/discounted_returnsZ��?N�җ#       ��wC	  ����A���*


charts/SPS��@Dx^{-'       ��F	  ����A���*

charts/restart    8�/       m]P	  @��A���*

charts/episodic_return- �A�3       � V	  @��A���*#
!
charts/windowed_avg_returnݐ�AT�VX/       m]P	  @��A���*

charts/episodic_length�h�D���
-       <A��	   ��A���*

charts/learning_rated�e9P�{*       ����	   ��A���*

losses/value_lossqX_=j���+       ��K	   ��A���*

losses/policy_loss|�ջS!EJ'       ��F	   ��A���*

losses/entropy-����<-       <A��	   ��A���*

losses/old_approx_kl2�<;BG��)       7�_ 	   ��A���*

losses/approx_kl,;���|(       �pJ	   ��A���*

losses/clipfrac��|=����2       $V�	   ��A���*"
 
losses/discounted_returns ��?t�5#       ��wC	   ��A���*


charts/SPS.�>DV�9�'       ��F	   ��A���*

charts/restart    ZG[�/       m]P	  @ ��A���*

charts/episodic_return���A�訮3       � V	  @ ��A���*#
!
charts/windowed_avg_return��Aj�L/       m]P	  @ ��A���*

charts/episodic_length%*�D!�-2-       <A��	   !��A���*

charts/learning_rate��e9���"*       ����	   !��A���*

losses/value_loss�!u=�ҥ+       ��K	   !��A���*

losses/policy_loss{��qR9�'       ��F	   !��A���*

losses/entropy ���6�~-       <A��	   !��A���*

losses/old_approx_kl�ߴ;�v�E)       7�_ 	   !��A���*

losses/approx_klL6;DO�(       �pJ	   !��A���*

losses/clipfrac �>�7+2       $V�	   !��A���*"
 
losses/discounted_returns�N�?�#E�#       ��wC	   !��A���*


charts/SPS�.<D��(x'       ��F	   !��A���*

charts/restart    k���/       m]P	  @=��A���*

charts/episodic_returnmB5Đ�3       � V	  @=��A���*#
!
charts/windowed_avg_returniG�A�H /       m]P	  @=��A���*

charts/episodic_length�E�eu-       <A��	  �=��A���*

charts/learning_rate�e9V��*       ����	  �=��A���*

losses/value_loss{!-=��{_+       ��K	  �=��A���*

losses/policy_loss��%�� �'       ��F	  �=��A���*

losses/entropy�����%�-       <A��	  �=��A���*

losses/old_approx_kl��;5��i)       7�_ 	  �=��A���*

losses/approx_kl�ܾ;�e�,(       �pJ	  �=��A���*

losses/clipfrac�ڪ=Y>�2       $V�	  �=��A���*"
 
losses/discounted_returns%��?��%H#       ��wC	  �=��A���*


charts/SPSUw9DO�ޒ'       ��F	  �=��A���*

charts/restart    b�&:/       m]P	  �Z��A���*

charts/episodic_returnZ�A"�߼3       � V	  �Z��A���*#
!
charts/windowed_avg_return�Z�A�'t�/       m]P	  �Z��A���*

charts/episodic_lengthrl�D8�G�-       <A��	  @[��A���*

charts/learning_rateg�e9��v&*       ����	  @[��A���*

losses/value_lossjg=����+       ��K	  @[��A���*

losses/policy_loss����;�s'       ��F	  @[��A���*

losses/entropy~���c��-       <A��	  @[��A���*

losses/old_approx_kl_W:]��)       7�_ 	  @[��A���*

losses/approx_klD�4;x{�(       �pJ	  @[��A���*

losses/clipfrac���=n@�2       $V�	  @[��A���*"
 
losses/discounted_returns>��?����#       ��wC	  @[��A���*


charts/SPS��6D�ڽS'       ��F	  @[��A���*

charts/restart    ���K/       m]P	   y��A���*

charts/episodic_return��B[#yj3       � V	   y��A���*#
!
charts/windowed_avg_return��A�B/       m]P	   y��A���*

charts/episodic_length�HEP'�E-       <A��	  �y��A���*

charts/learning_rate��e9�a$*       ����	  �y��A���*

losses/value_loss�GF= ��`+       ��K	  �y��A���*

losses/policy_loss$&����{�'       ��F	  �y��A���*

losses/entropyX�	����-       <A��	  �y��A���*

losses/old_approx_klN;6xZ)       7�_ 	  �y��A���*

losses/approx_kl�!4;�8��(       �pJ	  �y��A���*

losses/clipfracV��=���2       $V�	  �y��A���*"
 
losses/discounted_returns�.�?n玥#       ��wC	  �y��A���*


charts/SPSe4D�h��'       ��F	  �y��A���*

charts/restart    �6�/       m]P	  ����A���*

charts/episodic_return+� B�I��3       � V	  ����A���*#
!
charts/windowed_avg_return�c�A���/       m]P	  ����A���*

charts/episodic_length�B�D��F-       <A��	  ����A���*

charts/learning_rate�e9I+��*       ����	  ����A���*

losses/value_loss�XO=�`�o+       ��K	  ����A���*

losses/policy_loss����� '       ��F	  ����A���*

losses/entropy�b���b�-       <A��	  ����A���*

losses/old_approx_kl>�4;Щ�)       7�_ 	  ����A���*

losses/approx_klF�;���L(       �pJ	  ����A���*

losses/clipfrac 8�=����2       $V�	  ����A���*"
 
losses/discounted_returns��?[�]}#       ��wC	  ����A���*


charts/SPSNL1D��_�'       ��F	  ����A���*

charts/restart    ��w/       m]P	  ����A���*

charts/episodic_return;2�A�;�3       � V	  ����A���*#
!
charts/windowed_avg_return	�A)��'/       m]P	  ����A���*

charts/episodic_lengthf��D��y�-       <A��	  ����A���*

charts/learning_rateje9:5�*       ����	  ����A���*

losses/value_loss��c=u`�+       ��K	  ����A���*

losses/policy_loss��[��K~s'       ��F	  ����A���*

losses/entropy+��$��-       <A��	  ����A���*

losses/old_approx_klL�7;��¦)       7�_ 	  ����A���*

losses/approx_kl-�;��Bg(       �pJ	  ����A���*

losses/clipfrac  �=���2       $V�	  ����A���*"
 
losses/discounted_returnsX��?b.mO#       ��wC	  ����A���*


charts/SPSkm.D��9'       ��F	  ����A���*

charts/restart    ���/       m]P	  ����A���*

charts/episodic_returni��A�3       � V	  ����A���*#
!
charts/windowed_avg_return���A�K�A/       m]P	  ����A���*

charts/episodic_length��DGt�G-       <A��	  @���A���*

charts/learning_rate�pe9\g�*       ����	  @���A���*

losses/value_loss6la=��v6+       ��K	  @���A���*

losses/policy_loss+���V��'       ��F	  @���A���*

losses/entropy4�����-       <A��	  @���A���*

losses/old_approx_kl���:P��@)       7�_ 	  @���A���*

losses/approx_kl��;^���(       �pJ	  @���A���*

losses/clipfrac h�=v�q
2       $V�	  @���A���*"
 
losses/discounted_returns
`�?q���#       ��wC	  @���A���*


charts/SPS?�+D؃\!'       ��F	  @���A���*

charts/restart    �obB/       m]P	  ����A���*

charts/episodic_return���AmO�:3       � V	  ����A���*#
!
charts/windowed_avg_return`i�A(3E_/       m]P	  ����A���*

charts/episodic_lengthIj�D�b�C-       <A��	  ����A���*

charts/learning_ratebe9塝�*       ����	  ����A���*

losses/value_lossa�V=иT+       ��K	  ����A���*

losses/policy_loss��>��-W�'       ��F	  ����A���*

losses/entropy���� �G-       <A��	  ����A���*

losses/old_approx_kl���;|�h)       7�_ 	  ����A���*

losses/approx_kl|�^;Kcu(       �pJ	  ����A���*

losses/clipfrac ؠ=CU�`2       $V�	  ����A���*"
 
losses/discounted_returns���?���_#       ��wC	  ����A���*


charts/SPS.�(D�Ï'       ��F	  ����A���*

charts/restart    �Ax�/       m]P	  @ ��A���*

charts/episodic_return�&�APS�3       � V	  @ ��A���*#
!
charts/windowed_avg_return���A݃�c/       m]P	  @ ��A���*

charts/episodic_length���Dkq-|-       <A��	  � ��A���*

charts/learning_ratenSe9҄\�*       ����	  � ��A���*

losses/value_loss�`=3��+       ��K	  � ��A���*

losses/policy_loss�����'       ��F	  � ��A���*

losses/entropyzl�QU�-       <A��	  � ��A���*

losses/old_approx_kl��e;i���)       7�_ 	  � ��A���*

losses/approx_klm[;��C;(       �pJ	  � ��A���*

losses/clipfrac ��=�=�2       $V�	  � ��A���*"
 
losses/discounted_returnsʛ�?���(#       ��wC	  � ��A���*


charts/SPS��&D��B�'       ��F	  � ��A���*

charts/restart    ��/       m]P	  @;��A���*

charts/episodic_returnF�A�E|,3       � V	  @;��A���*#
!
charts/windowed_avg_returnQ��A�t/       m]P	  @;��A���*

charts/episodic_length�~�D��C-       <A��	   <��A���*

charts/learning_rate�De9.�?*       ����	   <��A���*

losses/value_loss��T=�p!�+       ��K	   <��A���*

losses/policy_loss`�ֺ|I��'       ��F	   <��A���*

losses/entropy	����W�F-       <A��	   <��A���*

losses/old_approx_kl�w^;px�)       7�_ 	   <��A���*

losses/approx_kl2�`;��4�(       �pJ	   <��A���*

losses/clipfracUU�=���P2       $V�	   <��A���*"
 
losses/discounted_returns`��?G��}#       ��wC	   <��A���*


charts/SPS�&D��'       ��F	   <��A���*

charts/restart    d�]/       m]P	   W��A���*

charts/episodic_return��Aҥ�z3       � V	   W��A���*#
!
charts/windowed_avg_return���A��/       m]P	   W��A���*

charts/episodic_length ��D���.-       <A��	  �W��A���*

charts/learning_rate6e9��_*       ����	  �W��A���*

losses/value_loss�,@=n�F+       ��K	  �W��A���*

losses/policy_loss ɺ[5'       ��F	  �W��A���*

losses/entropy ����h�u-       <A��	  �W��A���*

losses/old_approx_klSqB:(��)       7�_ 	  �W��A���*

losses/approx_kl�:8;D˅j(       �pJ	  �W��A���*

losses/clipfrac��=�ː(2       $V�	  �W��A���*"
 
losses/discounted_returnsp�?r���#       ��wC	  �W��A���*


charts/SPS�_%D	@�'       ��F	  �W��A���*

charts/restart    ���(/       m]P	  @q��A���*

charts/episodic_return/��A"�n3       � V	  @q��A���*#
!
charts/windowed_avg_returnS��AN��6/       m]P	  @q��A���*

charts/episodic_length���D�扊-       <A��	  �q��A���*

charts/learning_rateq'e9� �*       ����	  �q��A���*

losses/value_lossvDC=���+       ��K	  �q��A���*

losses/policy_loss�m��"Oϯ'       ��F	  �q��A���*

losses/entropy$���, K]-       <A��	  �q��A���*

losses/old_approx_kl� ;��)       7�_ 	  �q��A���*

losses/approx_kl�i/;ON_(       �pJ	  �q��A���*

losses/clipfracV-�=���2       $V�	  �q��A���*"
 
losses/discounted_returns��?�o?)#       ��wC	  �q��A���*


charts/SPSE%DxL9�'       ��F	  �q��A���*

charts/restart    Iv��/       m]P	  ����A���*

charts/episodic_return���A�bl�3       � V	  ����A���*#
!
charts/windowed_avg_returnP��A�"�/       m]P	  ����A���*

charts/episodic_length�u�D|g]-       <A��	  @���A���*

charts/learning_rate�e9��m*       ����	  @���A���*

losses/value_loss�\T=1Lm�+       ��K	  @���A���*

losses/policy_loss�<��g]�'       ��F	  @���A���*

losses/entropy(>��@-       <A��	  @���A���*

losses/old_approx_kl�L�:ґn�)       7�_ 	  @���A���*

losses/approx_kl���:���(       �pJ	  @���A���*

losses/clipfrac ��=��T�2       $V�	  @���A���*"
 
losses/discounted_returns���?RA��#       ��wC	  @���A���*


charts/SPS,�$D���'       ��F	  @���A���*

charts/restart    �6 �