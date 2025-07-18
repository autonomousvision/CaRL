�      ��`�	  �b��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000000724.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 47513600,
    "max_training_score" : 28.637149810791016,
    "best_iteration" : 652,
    "latest_iteration" : 724,
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
text"�'       ��F	   e��A���*

charts/restart  �?���F/       m]P	  @x��A���*

charts/episodic_return�b�@M�3       � V	  @x��A���*#
!
charts/windowed_avg_return�b�@gi	W/       m]P	  @x��A���*

charts/episodic_length�%uC���-       <A��	  @y��A���*

charts/learning_rateƞ\9u�O�*       ����	  @y��A���*

losses/value_loss���=�U�K+       ��K	  @y��A���*

losses/policy_loss�`�]k��'       ��F	  @y��A���*

losses/entropy�3G�n�ƥ-       <A��	  @y��A���*

losses/old_approx_kl
��;��)       7�_ 	  @y��A���*

losses/approx_kl�f;	Ƃ>(       �pJ	  @y��A���*

losses/clipfrac @�=����2       $V�	  @y��A���*"
 
losses/discounted_returnsp{@r�a)#       ��wC	  @y��A���*


charts/SPS;OD/[�'       ��F	  @y��A���*

charts/restart    �/�/       m]P	  ����A���*

charts/episodic_return	]AHe 3       � V	  ����A���*#
!
charts/windowed_avg_return'h$AR#�{/       m]P	  ����A���*

charts/episodic_length�
D�]~�-       <A��	  ����A���*

charts/learning_rate�\9�\7*       ����	  ����A���*

losses/value_loss�$�=�UX�+       ��K	  ����A���*

losses/policy_loss$��~��'       ��F	  ����A���*

losses/entropy��9�Ӳc&-       <A��	  ����A���*

losses/old_approx_kl%��;�4�+)       7�_ 	  ����A���*

losses/approx_klӃ�;Ey�V(       �pJ	  ����A���*

losses/clipfrac���=�\��2       $V�	  ����A���*"
 
losses/discounted_returns�6@Vr�#       ��wC	  ����A���*


charts/SPS�CMD|m'       ��F	  ����A���*

charts/restart    �;/       m]P	  ����A���*

charts/episodic_return�W�A���3       � V	  ����A���*#
!
charts/windowed_avg_return�*kA[
�[/       m]P	  ����A���*

charts/episodic_lengthp�QD(�]�-       <A��	  @���A���*

charts/learning_rates�\9�S�q*       ����	  @���A���*

losses/value_lossm�=�_��+       ��K	  @���A���*

losses/policy_loss�e���h��'       ��F	  @���A���*

losses/entropy��8�ڹ��-       <A��	  @���A���*

losses/old_approx_kla;>%�)       7�_ 	  @���A���*

losses/approx_kl���:U<��(       �pJ	  @���A���*

losses/clipfrac���=G�k�2       $V�	  @���A���*"
 
losses/discounted_returns�`@�G׉#       ��wC	  @���A���*


charts/SPS��ID*F�'       ��F	  @���A���*

charts/restart    �| /       m]P	   ���A���*

charts/episodic_return��AD��3       � V	   ���A���*#
!
charts/windowed_avg_return,�A���/       m]P	   ���A���*

charts/episodic_length)��Df�_-       <A��	  ����A���*

charts/learning_rate�r\9�s�X*       ����	  ����A���*

losses/value_loss��=(|�+       ��K	  ����A���*

losses/policy_loss��p���Xr'       ��F	  ����A���*

losses/entropy+#-�����-       <A��	  ����A���*

losses/old_approx_kl� �;<٣)       7�_ 	  ����A���*

losses/approx_klO�;�*z�(       �pJ	  ����A���*

losses/clipfrac�*�=�b;2       $V�	  ����A���*"
 
losses/discounted_returnsI��?x=��#       ��wC	  ����A���*


charts/SPS��ID��P'       ��F	  ����A���*

charts/restart    FF��/       m]P	  ����A���*

charts/episodic_returnr��A����3       � V	  ����A���*#
!
charts/windowed_avg_returnM��A8�$/       m]P	  ����A���*

charts/episodic_lengthD*�DÁ��-       <A��	   ���A���*

charts/learning_rate d\9\iأ*       ����	   ���A���*

losses/value_loss�&�=���+       ��K	   ���A���*

losses/policy_loss [���g�'       ��F	   ���A���*

losses/entropy�A)�28-       <A��	   ���A���*

losses/old_approx_klrO�:8��)       7�_ 	   ���A���*

losses/approx_kl��6;u(       �pJ	   ���A���*

losses/clipfrac�z�=�v2       $V�	   ���A���*"
 
losses/discounted_returns^Y�?ǿ�#       ��wC	   ���A���*


charts/SPSC�GD���'       ��F	   ���A���*

charts/restart    #��O/       m]P	   ���A���*

charts/episodic_return�C�A+)3       � V	   ���A���*#
!
charts/windowed_avg_return��A�j��/       m]P	   ���A���*

charts/episodic_lengthk_�Dw��-       <A��	  ����A���*

charts/learning_ratewU\9<�	�*       ����	  ����A���*

losses/value_loss喁=
ɣ�+       ��K	  ����A���*

losses/policy_loss]D@��N�P'       ��F	  ����A���*

losses/entropy֡�C��u-       <A��	  ����A���*

losses/old_approx_kl�0�;��)       7�_ 	  ����A���*

losses/approx_kl��K;f��(       �pJ	  ����A���*

losses/clipfracV�=�oL2       $V�	  ����A���*"
 
losses/discounted_returns���?e�u�#       ��wC	  ����A���*


charts/SPSdSGDwo'       ��F	  ����A���*

charts/restart    ϟ��/       m]P	  ����A���*

charts/episodic_return��A=�F�3       � V	  ����A���*#
!
charts/windowed_avg_return���A5���/       m]P	  ����A���*

charts/episodic_lengthݜ�DZ�4�-       <A��	  ����A���*

charts/learning_rate�F\9�X�P*       ����	  ����A���*

losses/value_lossµ�=i-O+       ��K	  ����A���*

losses/policy_lossj+���z�'       ��F	  ����A���*

losses/entropy(/�Q��-       <A��	  ����A���*

losses/old_approx_kl�;�]!=)       7�_ 	  ����A���*

losses/approx_kl�l9;���y(       �pJ	  ����A���*

losses/clipfrac ؗ=��02       $V�	  ����A���*"
 
losses/discounted_returns1�?۱�.#       ��wC	  ����A���*


charts/SPS��ED�a��'       ��F	  ����A���*

charts/restart    6Sx�/       m]P	   
��A���*

charts/episodic_return1��A���,3       � V	   
��A���*#
!
charts/windowed_avg_return�1�AI8�O/       m]P	   
��A���*

charts/episodic_length5�DG!�-       <A��	  �
��A���*

charts/learning_rate$8\9�k�*       ����	  �
��A���*

losses/value_loss��=�qb�+       ��K	  �
��A���*

losses/policy_lossZM��Igi�'       ��F	  �
��A���*

losses/entropy�3-���S9-       <A��	  �
��A���*

losses/old_approx_klV�;�K?x)       7�_ 	  �
��A���*

losses/approx_kl�9=;��I(       �pJ	  �
��A���*

losses/clipfrac���=��2       $V�	  �
��A���*"
 
losses/discounted_returns3��?�Wl#       ��wC	  �
��A���*


charts/SPS�_FD?M='       ��F	  �
��A���*

charts/restart    ܗ�A/       m]P	  @��A���*

charts/episodic_returnY�Bq>E3       � V	  @��A���*#
!
charts/windowed_avg_returnP�A@ͽf/       m]P	  @��A���*

charts/episodic_length���D4.�-       <A��	   ��A���*

charts/learning_rate{)\9��i*       ����	   ��A���*

losses/value_lossa�|=�2��+       ��K	   ��A���*

losses/policy_loss�>���A�'       ��F	   ��A���*

losses/entropyˉ,�.e�p-       <A��	   ��A���*

losses/old_approx_kl��	;�;;u)       7�_ 	   ��A���*

losses/approx_klm1;�q`�(       �pJ	   ��A���*

losses/clipfracU�=�v-2       $V�	   ��A���*"
 
losses/discounted_returns�h�?�*5�#       ��wC	   ��A���*


charts/SPS�FD���'       ��F	   ��A���*

charts/restart    Q]�/       m]P	   7��A���*

charts/episodic_return�F�A��3       � V	   7��A���*#
!
charts/windowed_avg_return&O�A��*/       m]P	   7��A���*

charts/episodic_length���Db�m�-       <A��	  �7��A���*

charts/learning_rate�\9 W�*       ����	  �7��A���*

losses/value_loss39�=����+       ��K	  �7��A���*

losses/policy_loss�`Y���{'       ��F	  �7��A���*

losses/entropy�:�+�i�-       <A��	  �7��A���*

losses/old_approx_kl���:O!��)       7�_ 	  �7��A���*

losses/approx_kl�5;��@�(       �pJ	  �7��A���*

losses/clipfracVE�=v��2       $V�	  �7��A���*"
 
losses/discounted_returnsm�@���#       ��wC	  �7��A���*


charts/SPS��BD�,s#'       ��F	  �7��A���*

charts/restart    ��E�/       m]P	  @P��A���*

charts/episodic_return��BB�O�3       � V	  @P��A���*#
!
charts/windowed_avg_return!b�A�鄢/       m]P	  @P��A���*

charts/episodic_length
��DT�ݪ-       <A��	   Q��A���*

charts/learning_rate(\90��*       ����	   Q��A���*

losses/value_loss�u=���+       ��K	   Q��A���*

losses/policy_lossH������c'       ��F	   Q��A���*

losses/entropy�%���-�-       <A��	   Q��A���*

losses/old_approx_klM�;b�-k)       7�_ 	   Q��A���*

losses/approx_kl�l;NTv�(       �pJ	   Q��A���*

losses/clipfracU��=+�8e2       $V�	   Q��A���*"
 
losses/discounted_returns��?���D#       ��wC	   Q��A���*


charts/SPS�F?Dt��'       ��F	   Q��A���*

charts/restart    �,9/       m]P	  �k��A���*

charts/episodic_returnm��At��W3       � V	  �k��A���*#
!
charts/windowed_avg_return�B�A�>�8/       m]P	  �k��A���*

charts/episodic_lengthLΐD�#��-       <A��	   l��A���*

charts/learning_rate~�[9 �J�*       ����	   l��A���*

losses/value_loss��t=h��+       ��K	   l��A���*

losses/policy_loss`�^�aJ'       ��F	   l��A���*

losses/entropyY'��xa-       <A��	   l��A���*

losses/old_approx_kl�i�:�_FO)       7�_ 	   l��A���*

losses/approx_kl�FY;g���(       �pJ	   l��A���*

losses/clipfracU��=� &2       $V�	   l��A���*"
 
losses/discounted_returnsF�?y/R#       ��wC	   l��A���*


charts/SPSf/;DG�X'       ��F	   l��A���*

charts/restart    {��/       m]P	  ����A���*

charts/episodic_return�v�AX���3       � V	  ����A���*#
!
charts/windowed_avg_return1��A��7�/       m]P	  ����A���*

charts/episodic_length�y�D���-       <A��	  ����A���*

charts/learning_rate��[9f�	*       ����	  ����A���*

losses/value_loss�J�=���\+       ��K	  ����A���*

losses/policy_loss-r��
�N'       ��F	  ����A���*

losses/entropy��CI�-       <A��	  ����A���*

losses/old_approx_kl}�;Q��)       7�_ 	  ����A���*

losses/approx_kl�kr;EN�(       �pJ	  ����A���*

losses/clipfracU՞=P�J�2       $V�	  ����A���*"
 
losses/discounted_returns���?�,0�#       ��wC	  ����A���*


charts/SPSٔ7D��:s'       ��F	  ����A���*

charts/restart    Bܣ�/       m]P	  @���A���*

charts/episodic_returnO$�AO��;3       � V	  @���A���*#
!
charts/windowed_avg_return3��A���/       m]P	  @���A���*

charts/episodic_length�`�D�|��-       <A��	   ���A���*

charts/learning_rate+�[9�2*       ����	   ���A���*

losses/value_loss�Pt=VJz�+       ��K	   ���A���*

losses/policy_loss���Q7��'       ��F	   ���A���*

losses/entropyN��
EN-       <A��	   ���A���*

losses/old_approx_kl�JC;Qˉt)       7�_ 	   ���A���*

losses/approx_kl4*g;˭��(       �pJ	   ���A���*

losses/clipfracVe�=Y��a2       $V�	   ���A���*"
 
losses/discounted_returnsN)�?���#       ��wC	   ���A���*


charts/SPS'�5D�~��'       ��F	   ���A���*

charts/restart    \�O�/       m]P	  ����A���*

charts/episodic_returnf]�AM���3       � V	  ����A���*#
!
charts/windowed_avg_return���A75��/       m]P	  ����A���*

charts/episodic_length9�D/
�-       <A��	   ���A���*

charts/learning_rate��[9E��q*       ����	   ���A���*

losses/value_loss�;�=�M+       ��K	   ���A���*

losses/policy_loss.���sV��'       ��F	   ���A���*

losses/entropy��"�K�3-       <A��	   ���A���*

losses/old_approx_kl08;��o�)       7�_ 	   ���A���*

losses/approx_klI�+;[] (       �pJ	   ���A���*

losses/clipfrac�z�=v�r2       $V�	   ���A���*"
 
losses/discounted_returns]�?T�`#       ��wC	   ���A���*


charts/SPS��6D��'       ��F	   ���A���*

charts/restart    1�z�/       m]P	   ���A���*

charts/episodic_return���A��~�3       � V	   ���A���*#
!
charts/windowed_avg_return��A�V�/       m]P	   ���A���*

charts/episodic_length{�D�?�-       <A��	  ����A���*

charts/learning_rate��[9b15�*       ����	  ����A���*

losses/value_lossa�=��ȭ+       ��K	  ����A���*

losses/policy_loss��A;l'       ��F	  ����A���*

losses/entropy<����"w-       <A��	  ����A���*

losses/old_approx_kln�T;��g)       7�_ 	  ����A���*

losses/approx_kl!Y�;/���(       �pJ	  ����A���*

losses/clipfracUm�=|�52       $V�	  ����A���*"
 
losses/discounted_returns&�?T��#       ��wC	  ����A���*


charts/SPS�8D�¬l'       ��F	  ����A���*

charts/restart    ���U/       m]P	   ���A���*

charts/episodic_return��A���,3       � V	   ���A���*#
!
charts/windowed_avg_return�;�A��8�/       m]P	   ���A���*

charts/episodic_length���D��;-       <A��	  ����A���*

charts/learning_rate.�[9�ry�*       ����	  ����A���*

losses/value_loss�'�=2?c|+       ��K	  ����A���*

losses/policy_loss�;!:�='       ��F	  ����A���*

losses/entropy���=�-       <A��	  ����A���*

losses/old_approx_klE��;�=e3)       7�_ 	  ����A���*

losses/approx_kl��;�(       �pJ	  ����A���*

losses/clipfrac X�=+�ב2       $V�	  ����A���*"
 
losses/discounted_returns+�?5��#       ��wC	  ����A���*


charts/SPS3�9D�&]g'       ��F	  ����A���*

charts/restart    ��7/       m]P	  ����A���*

charts/episodic_return��A ��(3       � V	  ����A���*#
!
charts/windowed_avg_return)%�Au��/       m]P	  ����A���*

charts/episodic_length/2�D�X��-       <A��	   ���A���*

charts/learning_rate��[9%���*       ����	   ���A���*

losses/value_lossي�=�Q�+       ��K	   ���A���*

losses/policy_loss�޺�l
:'       ��F	   ���A���*

losses/entropy�U����-       <A��	   ���A���*

losses/old_approx_kl���;�@�)       7�_ 	   ���A���*

losses/approx_kl��;�L�7(       �pJ	   ���A���*

losses/clipfracU��=��"�2       $V�	   ���A���*"
 
losses/discounted_returnsf�?�i��#       ��wC	   ���A���*


charts/SPSb:D���'       ��F	   ���A���*

charts/restart    �s�/       m]P	  ���A���*

charts/episodic_returno`�A�j1�3       � V	  ���A���*#
!
charts/windowed_avg_return���A8Ed*/       m]P	  ���A���*

charts/episodic_length�ϓD2;E-       <A��	   ��A���*

charts/learning_rateܖ[9���K*       ����	   ��A���*

losses/value_loss9�=�4��+       ��K	   ��A���*

losses/policy_loss(�D�	��'       ��F	   ��A���*

losses/entropyf���Կ-       <A��	   ��A���*

losses/old_approx_kl��z;��d)       7�_ 	   ��A���*

losses/approx_kl��c;���(       �pJ	   ��A���*

losses/clipfracVm�=�7�2       $V�	   ��A���*"
 
losses/discounted_returns��?���#       ��wC	   ��A���*


charts/SPSJ>;D�_�P'       ��F	   ��A���*

charts/restart    �ؑ�/       m]P	   ��A���*

charts/episodic_returnE�A�3       � V	   ��A���*#
!
charts/windowed_avg_returna%�A���/       m]P	   ��A���*

charts/episodic_length���D�B�-       <A��	  ���A���*

charts/learning_rate2�[9)�F4*       ����	  ���A���*

losses/value_lossAD�=����+       ��K	  ���A���*

losses/policy_loss�<��<~��'       ��F	  ���A���*

losses/entropy����� -       <A��	  ���A���*

losses/old_approx_kl��;��)       7�_ 	  ���A���*

losses/approx_kl��L;�G�(       �pJ	  ���A���*

losses/clipfrac�R�=�#d�2       $V�	  ���A���*"
 
losses/discounted_returns���?8NDP#       ��wC	  ���A���*


charts/SPSK�;DV�`'       ��F	  ���A���*

charts/restart    �}L�/       m]P	  �-��A���*

charts/episodic_returnF��A����3       � V	  �-��A���*#
!
charts/windowed_avg_return��A(�%w/       m]P	  �-��A���*

charts/episodic_length�D.ղ-       <A��	  �.��A���*

charts/learning_rate�y[9�pC*       ����	  �.��A���*

losses/value_loss&�o=�]m�+       ��K	  �.��A���*

losses/policy_loss �����'       ��F	  �.��A���*

losses/entropy~��k��-       <A��	  �.��A���*

losses/old_approx_kl$d�;+G��)       7�_ 	  �.��A���*

losses/approx_kl�T9;����(       �pJ	  �.��A���*

losses/clipfracV�=�+�!2       $V�	  �.��A���*"
 
losses/discounted_returns��?<��`#       ��wC	  �.��A���*


charts/SPS�2<D(K'       ��F	  �.��A���*

charts/restart    C��#/       m]P	   E��A���*

charts/episodic_return��AZ;�I3       � V	   E��A���*#
!
charts/windowed_avg_return�+�Aٯ��/       m]P	   E��A���*

charts/episodic_length�E�DF�[@-       <A��	  �E��A���*

charts/learning_rate�j[9료�*       ����	  �E��A���*

losses/value_loss�@i=hgU�+       ��K	  �E��A���*

losses/policy_loss��P�#qSF'       ��F	  �E��A���*

losses/entropyط��	|�-       <A��	  �E��A���*

losses/old_approx_kl��7;�W��)       7�_ 	  �E��A���*

losses/approx_klVm�;e�&�(       �pJ	  �E��A���*

losses/clipfracUE�=��O�2       $V�	  �E��A���*"
 
losses/discounted_returns�@�?F��#       ��wC	  �E��A���*


charts/SPS�;D�]D'       ��F	  �E��A���*

charts/restart    ��v�/       m]P	  �Z��A���*

charts/episodic_return��A�+�3       � V	  �Z��A���*#
!
charts/windowed_avg_return�4�A(./       m]P	  �Z��A���*

charts/episodic_length�\�D�1��-       <A��	  @[��A���*

charts/learning_rate5\[9���*       ����	  @[��A���*

losses/value_loss vt=vE{+       ��K	  @[��A���*

losses/policy_loss���;-�9'       ��F	  @[��A���*

losses/entropy����Q1-       <A��	  @[��A���*

losses/old_approx_kl <��k#)       7�_ 	  @[��A���*

losses/approx_kl&�<vo�(       �pJ	  @[��A���*

losses/clipfracU!!>j�D�2       $V�	  @[��A���*"
 
losses/discounted_returns�?#�G!#       ��wC	  @[��A���*


charts/SPS��;D���('       ��F	  @[��A���*

charts/restart    ��/       m]P	   n��A���*

charts/episodic_return@��A�w�3       � V	   n��A���*#
!
charts/windowed_avg_return���A��J/       m]P	   n��A���*

charts/episodic_length���Dt��A-       <A��	  �n��A���*

charts/learning_rate�M[9K�/t*       ����	  �n��A���*

losses/value_loss2�e=g�$+       ��K	  �n��A���*

losses/policy_loss�������'       ��F	  �n��A���*

losses/entropy��<� -       <A��	  �n��A���*

losses/old_approx_kl�w#;ʘW�)       7�_ 	  �n��A���*

losses/approx_kl4�.;;�׀(       �pJ	  �n��A���*

losses/clipfrac `�=��-2       $V�	  �n��A���*"
 
losses/discounted_returnsr,�?�s#       ��wC	  �n��A���*


charts/SPSԡ<D�F�u'       ��F	  �n��A���*

charts/restart    �a� /       m]P	  @���A���*

charts/episodic_return��Aۣ/3       � V	  @���A���*#
!
charts/windowed_avg_return��A*~<z/       m]P	  @���A���*

charts/episodic_lengthe-�D2Nd�-       <A��	   ���A���*

charts/learning_rate�>[9�w�*       ����	   ���A���*

losses/value_loss���=m��+       ��K	   ���A���*

losses/policy_lossI�
�v�'�'       ��F	   ���A���*

losses/entropy��&��7% -       <A��	   ���A���*

losses/old_approx_kl��;�f��)       7�_ 	   ���A���*

losses/approx_kl0ś;s�~�(       �pJ	   ���A���*

losses/clipfrac X�=��t2       $V�	   ���A���*"
 
losses/discounted_returns��?���#       ��wC	   ���A���*


charts/SPS{=D��'       ��F	   ���A���*

charts/restart    K6�5/       m]P	  @���A���*

charts/episodic_return(��Atl��3       � V	  @���A���*#
!
charts/windowed_avg_return��A���/       m]P	  @���A���*

charts/episodic_length ��DY�PY-       <A��	   ���A���*

charts/learning_rate80[9Az4	*       ����	   ���A���*

losses/value_losssOp=�.C�+       ��K	   ���A���*

losses/policy_loss����[�h'       ��F	   ���A���*

losses/entropyӔ��3<�-       <A��	   ���A���*

losses/old_approx_kl�F;����)       7�_ 	   ���A���*

losses/approx_kly;HU��(       �pJ	   ���A���*

losses/clipfrac�b�=��\2       $V�	   ���A���*"
 
losses/discounted_returns��?iB�#       ��wC	   ���A���*


charts/SPSY=Df���'       ��F	   ���A���*

charts/restart    o���/       m]P	  ����A���*

charts/episodic_return��A����3       � V	  ����A���*#
!
charts/windowed_avg_return�v�A�
HC/       m]P	  ����A���*

charts/episodic_length z�DFrH-       <A��	  @���A���*

charts/learning_rate�![9�_J�*       ����	  @���A���*

losses/value_loss?�J=g�y�+       ��K	  @���A���*

losses/policy_loss�x�^�'       ��F	  @���A���*

losses/entropy����5D-       <A��	  @���A���*

losses/old_approx_kl?C�;@�)       7�_ 	  @���A���*

losses/approx_kl���;9M_(       �pJ	  @���A���*

losses/clipfracU��=�S�u2       $V�	  @���A���*"
 
losses/discounted_returnsS��?A �#       ��wC	  @���A���*


charts/SPS=D�#½'       ��F	  @���A���*

charts/restart    ��7/       m]P	  @���A���*

charts/episodic_return&�A��Y3       � V	  @���A���*#
!
charts/windowed_avg_returne��A�O�M/       m]P	  @���A���*

charts/episodic_length۞�D��X-       <A��	   ���A���*

charts/learning_rate�[9�|��*       ����	   ���A���*

losses/value_loss��m=ŕ��+       ��K	   ���A���*

losses/policy_loss�����'       ��F	   ���A���*

losses/entropyj'��DC.-       <A��	   ���A���*

losses/old_approx_kl���9�6')       7�_ 	   ���A���*

losses/approx_kl�xh;��(       �pJ	   ���A���*

losses/clipfracUY>��82       $V�	   ���A���*"
 
losses/discounted_returns?��?+	�#       ��wC	   ���A���*


charts/SPSC�<Dn���'       ��F	   ���A���*

charts/restart    ^xh�/       m]P	   ���A���*

charts/episodic_return��A(��3       � V	   ���A���*#
!
charts/windowed_avg_return��A�%�/       m]P	   ���A���*

charts/episodic_length$n�Dj	['-       <A��	  ����A���*

charts/learning_rate<[9/�*       ����	  ����A���*

losses/value_loss��r=�]&�+       ��K	  ����A���*

losses/policy_loss1>�8��_'       ��F	  ����A���*

losses/entropy�F
�����-       <A��	  ����A���*

losses/old_approx_kl�v*;"<��)       7�_ 	  ����A���*

losses/approx_kls60;)�o{(       �pJ	  ����A���*

losses/clipfracU%�=��K2       $V�	  ����A���*"
 
losses/discounted_returnsP�?13ܵ#       ��wC	  ����A���*


charts/SPS��<DK�	'       ��F	  ����A���*

charts/restart    � u�/       m]P	   ���A���*

charts/episodic_return&��AC�c;3       � V	   ���A���*#
!
charts/windowed_avg_return���A���/       m]P	   ���A���*

charts/episodic_lengthf2�D��-       <A��	  ����A���*

charts/learning_rate��Z9]�~*       ����	  ����A���*

losses/value_lossG~=��+       ��K	  ����A���*

losses/policy_loss�ܘ��'       ��F	  ����A���*

losses/entropy����}�-       <A��	  ����A���*

losses/old_approx_klC�F;�E)       7�_ 	  ����A���*

losses/approx_klc�Q;}�[(       �pJ	  ����A���*

losses/clipfracU��=�zt2       $V�	  ����A���*"
 
losses/discounted_returns���?�e?P#       ��wC	  ����A���*


charts/SPS�e<D_��'       ��F	  ����A���*

charts/restart    t�"z/       m]P	  � ��A���*

charts/episodic_return���A�⭈3       � V	  � ��A���*#
!
charts/windowed_avg_return(_�Aǅ��/       m]P	  � ��A���*

charts/episodic_length%Y�DLR-       <A��	   	 ��A���*

charts/learning_rate��Z9ⅈ�*       ����	   	 ��A���*

losses/value_loss�x�=��?+       ��K	   	 ��A���*

losses/policy_loss*wz�_"3~'       ��F	   	 ��A���*

losses/entropyƖ�"�R�-       <A��	   	 ��A���*

losses/old_approx_kl:�;\4�)       7�_ 	   	 ��A���*

losses/approx_klPk ;R
��(       �pJ	   	 ��A���*

losses/clipfracU��=A� 2       $V�	   	 ��A���*"
 
losses/discounted_returns���?��=X#       ��wC	   	 ��A���*


charts/SPSd�;D�U�'       ��F	   	 ��A���*

charts/restart    �j`/       m]P	  � ��A���*

charts/episodic_return�JB����3       � V	  � ��A���*#
!
charts/windowed_avg_return��A�{�/       m]P	  � ��A���*

charts/episodic_lengthF��D=u\-       <A��	  @  ��A���*

charts/learning_rate?�Z9~cWE*       ����	  @  ��A���*

losses/value_loss6 :=��4�+       ��K	  @  ��A���*

losses/policy_loss�܉�(�\p'       ��F	  @  ��A���*

losses/entropy�����=�-       <A��	  @  ��A���*

losses/old_approx_klPE�:%�;�)       7�_ 	  @  ��A���*

losses/approx_kl�E;�$(       �pJ	  @  ��A���*

losses/clipfracV�v=&O�N2       $V�	  @  ��A���*"
 
losses/discounted_returns_��?���#       ��wC	  @  ��A���*


charts/SPS�;D��R'       ��F	  @  ��A���*

charts/restart    ��;/       m]P	  �6 ��A���*

charts/episodic_returnT<�A���3       � V	  �6 ��A���*#
!
charts/windowed_avg_return<P�A���O/       m]P	  �6 ��A���*

charts/episodic_lengthB��DB��T-       <A��	  �7 ��A���*

charts/learning_rate��Z9^(��*       ����	  �7 ��A���*

losses/value_loss�j=
n�X+       ��K	  �7 ��A���*

losses/policy_lossD�����'       ��F	  �7 ��A���*

losses/entropy���qb.-       <A��	  �7 ��A���*

losses/old_approx_kl�t;�3?�)       7�_ 	  �7 ��A���*

losses/approx_kl$ʅ;a�$�(       �pJ	  �7 ��A���*

losses/clipfrac �=Ę�2       $V�	  �7 ��A���*"
 
losses/discounted_returnsdx�?�U*l#       ��wC	  �7 ��A���*


charts/SPS6;D���'       ��F	  �7 ��A���*

charts/restart    `+��/       m]P	  @N ��A���*

charts/episodic_returnY��AB�m3       � V	  @N ��A���*#
!
charts/windowed_avg_return���A]N/       m]P	  @N ��A���*

charts/episodic_lengthUK�D�=-       <A��	   O ��A���*

charts/learning_rate�Z9�0�*       ����	   O ��A���*

losses/value_loss+K]=T� }+       ��K	   O ��A���*

losses/policy_loss#6���i'       ��F	   O ��A���*

losses/entropy�����To�-       <A��	   O ��A���*

losses/old_approx_kl�g;&Gi3)       7�_ 	   O ��A���*

losses/approx_kl^�;��>�(       �pJ	   O ��A���*

losses/clipfrac h�=��>2       $V�	   O ��A���*"
 
losses/discounted_returns�F�?�<�L#       ��wC	   O ��A���*


charts/SPS��:DyY�'       ��F	   O ��A���*

charts/restart    �/�