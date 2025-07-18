�      <��	  �c��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000001354.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 88801280,
    "max_training_score" : 30.766647338867188,
    "best_iteration" : 1245,
    "latest_iteration" : 1354,
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
text�W��'       ��F	  @�c��A���**

charts/restart  �?��b�/       m]P	  @�c��A���**

charts/episodic_returnd��@��!�3       � V	  @�c��A���**#
!
charts/windowed_avg_returnd��@�۔/       m]P	  @�c��A���**

charts/episodic_length��FC�lZ-       <A��	  @�c��A���**

charts/learning_rate��89����*       ����	  @�c��A���**

losses/value_lossx��=�$�+       ��K	  @�c��A���**

losses/policy_loss�t�"��	'       ��F	  @�c��A���**

losses/entropy%�i�n�6-       <A��	  @�c��A���**

losses/old_approx_kl�k;A:��)       7�_ 	  @�c��A���**

losses/approx_kl�M;g*�V(       �pJ	  @�c��A���**

losses/clipfracVe>|�}2       $V�	  @�c��A���**"
 
losses/discounted_returnsL�@�:ͻ#       ��wC	  @�c��A���**


charts/SPS��[D�HtR'       ��F	  @�c��A���**

charts/restart    I��r/       m]P	  @�c��A���**

charts/episodic_return<XA<%��3       � V	  @�c��A���**#
!
charts/windowed_avg_return7&A�Y�/       m]P	  @�c��A���**

charts/episodic_lengthv�D��"-       <A��	   �c��A���**

charts/learning_rate�z89r�=V*       ����	   �c��A���**

losses/value_loss�ț=c�5Z+       ��K	   �c��A���**

losses/policy_loss������'       ��F	   �c��A���**

losses/entropy&b�n�w-       <A��	   �c��A���**

losses/old_approx_kl�
�;�+�)       7�_ 	   �c��A���**

losses/approx_klis;��_(       �pJ	   �c��A���**

losses/clipfrac  �=��A2       $V�	   �c��A���**"
 
losses/discounted_returns��@#Y5#       ��wC	   �c��A���**


charts/SPS��OD�pB'       ��F	   �c��A���**

charts/restart    �./       m]P	   �c��A���**

charts/episodic_return��B�u�=3       � V	   �c��A���**#
!
charts/windowed_avg_return{�A���;/       m]P	   �c��A���**

charts/episodic_length���DO5S-       <A��	  ��c��A���**

charts/learning_rateDl89��T�*       ����	  ��c��A���**

losses/value_loss�g�=�)��+       ��K	  ��c��A���**

losses/policy_loss��CF0�'       ��F	  ��c��A���**

losses/entropyӉR��-       <A��	  ��c��A���**

losses/old_approx_kl,��;��{)       7�_ 	  ��c��A���**

losses/approx_kllR;��-m(       �pJ	  ��c��A���**

losses/clipfrac�� >�9W2       $V�	  ��c��A���**"
 
losses/discounted_returns�@��#       ��wC	  ��c��A���**


charts/SPSt�ED����'       ��F	  ��c��A���**

charts/restart    ��y�/       m]P	  ��c��A���**

charts/episodic_return�"B]8:�3       � V	  ��c��A���**#
!
charts/windowed_avg_returnΗ�A��/       m]P	  ��c��A���**

charts/episodic_lengthӫ�D�JC]-       <A��	   �c��A���**

charts/learning_rate�]89��k�*       ����	   �c��A���**

losses/value_lossX�z=@�:+       ��K	   �c��A���**

losses/policy_loss�Ro�D��'       ��F	   �c��A���**

losses/entropy��K�����-       <A��	   �c��A���**

losses/old_approx_kl�R;�Da�)       7�_ 	   �c��A���**

losses/approx_kl��I;ZAk(       �pJ	   �c��A���**

losses/clipfracU��=����2       $V�	   �c��A���**"
 
losses/discounted_returns�D@�4H#       ��wC	   �c��A���**


charts/SPS��?D��'       ��F	   �c��A���**

charts/restart    ���^/       m]P	  ��c��A���**

charts/episodic_returnw�Bty�3       � V	  ��c��A���**#
!
charts/windowed_avg_return�\�A��/       m]P	  ��c��A���**

charts/episodic_length���DtRz?-       <A��	  @�c��A���**

charts/learning_rate�N89s�e*       ����	  @�c��A���**

losses/value_lossr8�=���O+       ��K	  @�c��A���**

losses/policy_lossӈd�(
�'       ��F	  @�c��A���**

losses/entropy�;M��no�-       <A��	  @�c��A���**

losses/old_approx_kls̓;��bp)       7�_ 	  @�c��A���**

losses/approx_klu
�;s�A|(       �pJ	  @�c��A���**

losses/clipfracVŬ=�I2       $V�	  @�c��A���**"
 
losses/discounted_returns�@��)5#       ��wC	  @�c��A���**


charts/SPS�ADۂ�'       ��F	  @�c��A���**

charts/restart    �ܢ/       m]P	  ��c��A���**

charts/episodic_return|hB`1u�3       � V	  ��c��A���**#
!
charts/windowed_avg_return0p�A��vE/       m]P	  ��c��A���**

charts/episodic_length�f�D�%-       <A��	  @ d��A���**

charts/learning_rateH@89 W.n*       ����	  @ d��A���**

losses/value_loss�i=��I6+       ��K	  @ d��A���**

losses/policy_lossB؟���'       ��F	  @ d��A���**

losses/entropy�>��:��-       <A��	  @ d��A���**

losses/old_approx_kl{�;9���)       7�_ 	  @ d��A���**

losses/approx_kl5FD;�5L(       �pJ	  @ d��A���**

losses/clipfracV�=��#/2       $V�	  @ d��A���**"
 
losses/discounted_returnsiG@�`�#       ��wC	  @ d��A���**


charts/SPSq�@D�o]�'       ��F	  @ d��A���**

charts/restart    0�Y�/       m]P	  �d��A���**

charts/episodic_returnP� B�~�3       � V	  �d��A���**#
!
charts/windowed_avg_return���A�a�/       m]P	  �d��A���**

charts/episodic_length��DC}�O-       <A��	  @d��A���**

charts/learning_rate�189�խ�*       ����	  @d��A���**

losses/value_loss���=�O��+       ��K	  @d��A���**

losses/policy_loss~���3�'i'       ��F	  @d��A���**

losses/entropy0�K��Cј-       <A��	  @d��A���**

losses/old_approx_kl��;ako�)       7�_ 	  @d��A���**

losses/approx_kl��-;�� (       �pJ	  @d��A���**

losses/clipfracUU�=ӈ8�2       $V�	  @d��A���**"
 
losses/discounted_returns	:@���#       ��wC	  @d��A���**


charts/SPSj	BD5w�'       ��F	  @d��A���**

charts/restart    S��/       m]P	  �'d��A���**

charts/episodic_return���A��3       � V	  �'d��A���**#
!
charts/windowed_avg_return���A���/       m]P	  �'d��A���**

charts/episodic_length1̨DK��-       <A��	  �(d��A���**

charts/learning_rate�"89��>�*       ����	  �(d��A���**

losses/value_loss3C�=�⸕+       ��K	  �(d��A���**

losses/policy_loss�h�|�'       ��F	  �(d��A���**

losses/entropyfPD��8-       <A��	  �(d��A���**

losses/old_approx_kl/�a;���)       7�_ 	  �(d��A���**

losses/approx_kl<�T;)��(       �pJ	  �(d��A���**

losses/clipfrac�Z�=K��2       $V�	  �(d��A���**"
 
losses/discounted_returns�@�2�~#       ��wC	  �(d��A���**


charts/SPS:CDD��'       ��F	  �(d��A���**

charts/restart    |�F�/       m]P	  �;d��A���**

charts/episodic_return��B �1�3       � V	  �;d��A���**#
!
charts/windowed_avg_return\c�AY�Yi/       m]P	  �;d��A���**

charts/episodic_length���D�? �-       <A��	  @<d��A���**

charts/learning_rateK89<�Px*       ����	  @<d��A���**

losses/value_loss!�]=mz+       ��K	  @<d��A���**

losses/policy_loss���M8��'       ��F	  @<d��A���**

losses/entropyPI�1��!-       <A��	  @<d��A���**

losses/old_approx_kl��l:�U�)       7�_ 	  @<d��A���**

losses/approx_klN�;y�:�(       �pJ	  @<d��A���**

losses/clipfrac ��=!�qt2       $V�	  @<d��A���**"
 
losses/discounted_returns�@�Uۺ#       ��wC	  @<d��A���**


charts/SPS�LDDs!�'       ��F	  @<d��A���**

charts/restart    s��/       m]P	   Pd��A���**

charts/episodic_return�m	Bk�=3       � V	   Pd��A���**#
!
charts/windowed_avg_return�"�A�:�M/       m]P	   Pd��A���**

charts/episodic_length��D�j��-       <A��	  �Pd��A���**

charts/learning_rate�896�L*       ����	  �Pd��A���**

losses/value_loss��^=�,�7+       ��K	  �Pd��A���**

losses/policy_loss�U�� m{'       ��F	  �Pd��A���**

losses/entropy
D�{5�-       <A��	  �Pd��A���**

losses/old_approx_kl�)w;��n�)       7�_ 	  �Pd��A���**

losses/approx_klcz:;j�J�(       �pJ	  �Pd��A���**

losses/clipfracU}�=��/2       $V�	  �Pd��A���**"
 
losses/discounted_returns�@�.�A#       ��wC	  �Pd��A���**


charts/SPS��DD�La�'       ��F	  �Pd��A���**

charts/restart    �%Fo/       m]P	   ed��A���**

charts/episodic_return��BiI�D3       � V	   ed��A���**#
!
charts/windowed_avg_returnM��A��	/       m]P	   ed��A���**

charts/episodic_lengthHc�D�9D1-       <A��	  �ed��A���**

charts/learning_rate��79;�OI*       ����	  �ed��A���**

losses/value_loss�}�=��C�+       ��K	  �ed��A���**

losses/policy_loss�N���bE2'       ��F	  �ed��A���**

losses/entropy}�Q�J)��-       <A��	  �ed��A���**

losses/old_approx_kl��;�)�2)       7�_ 	  �ed��A���**

losses/approx_kl:a;�0�(       �pJ	  �ed��A���**

losses/clipfrac��=���2       $V�	  �ed��A���**"
 
losses/discounted_returns��@�g�|#       ��wC	  �ed��A���**


charts/SPSJ�DD�(u'       ��F	  �ed��A���**

charts/restart    =�ٚ/       m]P	  �wd��A���**

charts/episodic_return��B���l3       � V	  �wd��A���**#
!
charts/windowed_avg_return�z�A k��/       m]P	  �wd��A���**

charts/episodic_length�z�DS�-       <A��	  �xd��A���**

charts/learning_rateN�79�Nk�*       ����	  �xd��A���**

losses/value_loss[�=����+       ��K	  �xd��A���**

losses/policy_loss�ѹ �'       ��F	  �xd��A���**

losses/entropyB/N�10��-       <A��	  �xd��A���**

losses/old_approx_kl�޷;���)       7�_ 	  �xd��A���**

losses/approx_kl̅J;5�I�(       �pJ	  �xd��A���**

losses/clipfrac ��=˻2       $V�	  �xd��A���**"
 
losses/discounted_returns��@Eۄ�#       ��wC	  �xd��A���**


charts/SPSC4FD&�m'       ��F	  �xd��A���**

charts/restart    �p0/       m]P	  ��d��A���**

charts/episodic_return��B�j3       � V	  ��d��A���**#
!
charts/windowed_avg_return7��AQ*D�/       m]P	  ��d��A���**

charts/episodic_length"�DW5��-       <A��	  @�d��A���**

charts/learning_rate��79�*Tv*       ����	  @�d��A���**

losses/value_loss��a=��,+       ��K	  @�d��A���**

losses/policy_loss=�����5'       ��F	  @�d��A���**

losses/entropy�xD�+��-       <A��	  @�d��A���**

losses/old_approx_kl��x;gn��)       7�_ 	  @�d��A���**

losses/approx_klCk0;����(       �pJ	  @�d��A���**

losses/clipfracU-�=�J2       $V�	  @�d��A���**"
 
losses/discounted_returns��@'��#       ��wC	  @�d��A���**


charts/SPS��FD�m\'       ��F	  @�d��A���**

charts/restart    ��l�/       m]P	  ��d��A���**

charts/episodic_returnqbB=k-�3       � V	  ��d��A���**#
!
charts/windowed_avg_return+( BU�}/       m]P	  ��d��A���**

charts/episodic_lengthDd�DfYW#-       <A��	  @�d��A���**

charts/learning_rate��79�ǧ*       ����	  @�d��A���**

losses/value_loss��B=3���+       ��K	  @�d��A���**

losses/policy_losspP���d��'       ��F	  @�d��A���**

losses/entropyr�6����-       <A��	  @�d��A���**

losses/old_approx_kl��9P�@)       7�_ 	  @�d��A���**

losses/approx_kl[$V;^	w3(       �pJ	  @�d��A���**

losses/clipfrac �>e>G�2       $V�	  @�d��A���**"
 
losses/discounted_returnsP=�?,͔�#       ��wC	  @�d��A���**


charts/SPS�HD ��'       ��F	  @�d��A���**

charts/restart    ��/       m]P	  ��d��A���**

charts/episodic_return��A=X7�3       � V	  ��d��A���**#
!
charts/windowed_avg_return. B�a��/       m]P	  ��d��A���**

charts/episodic_length�e�DѼ-q-       <A��	  @�d��A���**

charts/learning_rateR�79�k*       ����	  @�d��A���**

losses/value_loss��g=��+�+       ��K	  @�d��A���**

losses/policy_loss�_��'       ��F	  @�d��A���**

losses/entropy��@�Em�-       <A��	  @�d��A���**

losses/old_approx_kl�s�;X��M)       7�_ 	  @�d��A���**

losses/approx_kl�ā;�/S(       �pJ	  @�d��A���**

losses/clipfracU��=v>�*2       $V�	  @�d��A���**"
 
losses/discounted_returnsj8 @S��#       ��wC	  @�d��A���**


charts/SPSQ�ID���'       ��F	  @�d��A���**

charts/restart    �]�n/       m]P	  @�d��A���**

charts/episodic_return(�B��I�3       � V	  @�d��A���**#
!
charts/windowed_avg_return>� B��/       m]P	  @�d��A���**

charts/episodic_length{�D4b0-       <A��	   �d��A���**

charts/learning_rate��79&�8*       ����	   �d��A���**

losses/value_lossϏU=8K�+       ��K	   �d��A���**

losses/policy_loss7&��n��'       ��F	   �d��A���**

losses/entropy+9�T5-       <A��	   �d��A���**

losses/old_approx_kl��:&�Z�)       7�_ 	   �d��A���**

losses/approx_kl�0
;��}�(       �pJ	   �d��A���**

losses/clipfrac��z=Sl.�2       $V�	   �d��A���**"
 
losses/discounted_returnsl��?ʁ�a#       ��wC	   �d��A���**


charts/SPSp�ID�?�'       ��F	   �d��A���**

charts/restart    ����/       m]P	  ��d��A���**

charts/episodic_return��B�9g3       � V	  ��d��A���**#
!
charts/windowed_avg_return��B�1�/       m]P	  ��d��A���**

charts/episodic_length
�D��>-       <A��	   �d��A���**

charts/learning_rate �79u�m�*       ����	   �d��A���**

losses/value_loss�`u=�TBr+       ��K	   �d��A���**

losses/policy_loss:���\'       ��F	   �d��A���**

losses/entropyed?�FQ��-       <A��	   �d��A���**

losses/old_approx_kl+�/;�@dw)       7�_ 	   �d��A���**

losses/approx_kl�ޔ;����(       �pJ	   �d��A���**

losses/clipfrac��=]sU�2       $V�	   �d��A���**"
 
losses/discounted_returnsŖ�?��J�#       ��wC	   �d��A���**


charts/SPS�]HD��6!'       ��F	   �d��A���**

charts/restart    t5�/       m]P	  ��d��A���**

charts/episodic_return�TB�!� 3       � V	  ��d��A���**#
!
charts/windowed_avg_returnB[��;/       m]P	  ��d��A���**

charts/episodic_length��D�^-       <A��	  @�d��A���**

charts/learning_rateW�79���*       ����	  @�d��A���**

losses/value_loss�W2=m���+       ��K	  @�d��A���**

losses/policy_lossj��Kg<'       ��F	  @�d��A���**

losses/entropy>5G���-       <A��	  @�d��A���**

losses/old_approx_kl`�;�-�)       7�_ 	  @�d��A���**

losses/approx_kl��L;S���(       �pJ	  @�d��A���**

losses/clipfrac ��=X��2       $V�	  @�d��A���**"
 
losses/discounted_returnsys�?�#       ��wC	  @�d��A���**


charts/SPS�zGD`$ZM'       ��F	  @�d��A���**

charts/restart    ����/       m]P	  �e��A���**

charts/episodic_return�B��~�3       � V	  �e��A���**#
!
charts/windowed_avg_return�8B=\�/       m]P	  �e��A���**

charts/episodic_lengtht	�Dy��-       <A��	  @e��A���**

charts/learning_rate��79���*       ����	  @e��A���**

losses/value_lossy�e=aR/+       ��K	  @e��A���**

losses/policy_loss귦�1�4�'       ��F	  @e��A���**

losses/entropy�NG�c��U-       <A��	  @e��A���**

losses/old_approx_kl�t�:�%WE)       7�_ 	  @e��A���**

losses/approx_klK�.;�X�j(       �pJ	  @e��A���**

losses/clipfrac��|= �>2       $V�	  @e��A���**"
 
losses/discounted_returns&��?��g�#       ��wC	  @e��A���**


charts/SPS��FD�F�7'       ��F	  @e��A���**

charts/restart    �`N/       m]P	   e��A���**

charts/episodic_return��B��2�3       � V	   e��A���**#
!
charts/windowed_avg_return�rB�c�	/       m]P	   e��A���**

charts/episodic_length1,�D�ϣ�-       <A��	  �e��A���**

charts/learning_rates799Hp�*       ����	  �e��A���**

losses/value_loss�\=�_�+       ��K	  �e��A���**

losses/policy_lossI���W7�'       ��F	  �e��A���**

losses/entropy�">�i�W�-       <A��	  �e��A���**

losses/old_approx_kl�z:Ox��)       7�_ 	  �e��A���**

losses/approx_kl[�M;JDn(       �pJ	  �e��A���**

losses/clipfrac��=
,'�2       $V�	  �e��A���**"
 
losses/discounted_returnsJ/�?

d�#       ��wC	  �e��A���**


charts/SPS�FFD�z.?'       ��F	  �e��A���**

charts/restart    �z�S