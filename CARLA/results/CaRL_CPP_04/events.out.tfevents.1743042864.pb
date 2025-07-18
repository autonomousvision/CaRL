�      �YW�	   L.��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000002903.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 190316544,
    "max_training_score" : 39.652854919433594,
    "best_iteration" : 2881,
    "latest_iteration" : 2903,
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
text^�0'       ��F	  �L.��A���Z*

charts/restart  �?�ۧ/       m]P	  �`.��A���Z*

charts/episodic_returnߦ�@����3       � V	  �`.��A���Z*#
!
charts/windowed_avg_returnߦ�@m(&�/       m]P	  �`.��A���Z*

charts/episodic_lengthC~C����-       <A��	  �a.��A���Z*

charts/learning_rate���8�N;*       ����	  �a.��A���Z*

losses/value_loss/��=���_+       ��K	  �a.��A���Z*

losses/policy_lossC�"�:�T�'       ��F	  �a.��A���Z*

losses/entropy`��h��-       <A��	  �a.��A���Z*

losses/old_approx_kl�7�:kw�|)       7�_ 	  �a.��A���Z*

losses/approx_kl0	k;��y(       �pJ	  �a.��A���Z*

losses/clipfracUm�=��>K2       $V�	  �a.��A���Z*"
 
losses/discounted_returns�)@m���#       ��wC	  �a.��A���Z*


charts/SPS��ED���'       ��F	  �a.��A���Z*

charts/restart    ��*�/       m]P	  @x.��A���Z*

charts/episodic_return1�A���3       � V	  @x.��A���Z*#
!
charts/windowed_avg_return�hAf�/       m]P	  @x.��A���Z*

charts/episodic_lengthY�3D�j�-       <A��	   y.��A���Z*

charts/learning_ratea��8Y���*       ����	   y.��A���Z*

losses/value_loss沋=��k�+       ��K	   y.��A���Z*

losses/policy_loss�ڷ���We'       ��F	   y.��A���Z*

losses/entropy�7��-       <A��	   y.��A���Z*

losses/old_approx_kl}�;´�,)       7�_ 	   y.��A���Z*

losses/approx_kl�g;�!��(       �pJ	   y.��A���Z*

losses/clipfrac 8�=�G�$2       $V�	   y.��A���Z*"
 
losses/discounted_returns!@����#       ��wC	   y.��A���Z*


charts/SPS��9D���'       ��F	   y.��A���Z*

charts/restart    �F?T/       m]P	  ��.��A���Z*

charts/episodic_return	B�R�3       � V	  ��.��A���Z*#
!
charts/windowed_avg_return���A
�u�/       m]P	  ��.��A���Z*

charts/episodic_length��Dɽen-       <A��	  @�.��A���Z*

charts/learning_ratei�8�$�=*       ����	  @�.��A���Z*

losses/value_lossp�=-�'�+       ��K	  @�.��A���Z*

losses/policy_loss�Հ�%%E2'       ��F	  @�.��A���Z*

losses/entropy����xm�-       <A��	  @�.��A���Z*

losses/old_approx_klH};��b)       7�_ 	  @�.��A���Z*

losses/approx_kl.G(;Sޮ�(       �pJ	  @�.��A���Z*

losses/clipfracV��=�D��2       $V�	  @�.��A���Z*"
 
losses/discounted_returnsN@�o�z#       ��wC	  @�.��A���Z*


charts/SPSz4DQ=��'       ��F	  @�.��A���Z*

charts/restart    Ĥo�/       m]P	   �.��A���Z*

charts/episodic_return��?B'4�3       � V	   �.��A���Z*#
!
charts/windowed_avg_return�Q�A��&�/       m]P	   �.��A���Z*

charts/episodic_length�{�D̒�-       <A��	  ��.��A���Z*

charts/learning_rate�K�8ַKX*       ����	  ��.��A���Z*

losses/value_loss�ǁ='�%+       ��K	  ��.��A���Z*

losses/policy_loss.T�����F'       ��F	  ��.��A���Z*

losses/entropy�ߗ��|��-       <A��	  ��.��A���Z*

losses/old_approx_kl���;I
�)       7�_ 	  ��.��A���Z*

losses/approx_kl� L;`��(       �pJ	  ��.��A���Z*

losses/clipfrac�z�=�% m2       $V�	  ��.��A���Z*"
 
losses/discounted_returns�.@H*�}#       ��wC	  ��.��A���Z*


charts/SPS�0D�� '       ��F	  ��.��A���Z*

charts/restart    ���/       m]P	  @�.��A���Z*

charts/episodic_returnE�5B�C�h3       � V	  @�.��A���Z*#
!
charts/windowed_avg_returnޕ�A h��/       m]P	  @�.��A���Z*

charts/episodic_lengtht1�D�`$&-       <A��	   �.��A���Z*

charts/learning_ratei.�8�Ȟ@*       ����	   �.��A���Z*

losses/value_loss�eg=�P�L+       ��K	   �.��A���Z*

losses/policy_loss*U���̧�'       ��F	   �.��A���Z*

losses/entropyTؐ��c�/-       <A��	   �.��A���Z*

losses/old_approx_kl%(;ǚ��)       7�_ 	   �.��A���Z*

losses/approx_kl{w5;s9(       �pJ	   �.��A���Z*

losses/clipfrac�2�=�J�c2       $V�	   �.��A���Z*"
 
losses/discounted_returnsJH@���F#       ��wC	   �.��A���Z*


charts/SPSO�-DKM'       ��F	   �.��A���Z*

charts/restart    ���/       m]P	  @�.��A���Z*

charts/episodic_return��&B����3       � V	  @�.��A���Z*#
!
charts/windowed_avg_returnE6Bo�v/       m]P	  @�.��A���Z*

charts/episodic_lengthD��DVϗB-       <A��	   �.��A���Z*

charts/learning_rate�8)��i*       ����	   �.��A���Z*

losses/value_loss@�=9��@+       ��K	   �.��A���Z*

losses/policy_loss����{L'       ��F	   �.��A���Z*

losses/entropy�Ǒ��z�;-       <A��	   �.��A���Z*

losses/old_approx_klw� :%�N()       7�_ 	   �.��A���Z*

losses/approx_kl(Q";�#�a(       �pJ	   �.��A���Z*

losses/clipfrac ��=���2       $V�	   �.��A���Z*"
 
losses/discounted_returns�.@����#       ��wC	   �.��A���Z*


charts/SPS��*D&�l�'       ��F	   �.��A���Z*

charts/restart    1~*o/       m]P	  @�.��A���Z*

charts/episodic_return %B�t��3       � V	  @�.��A���Z*#
!
charts/windowed_avg_return��B�E�/       m]P	  @�.��A���Z*

charts/episodic_length�J�D�f�q-       <A��	   �.��A���Z*

charts/learning_rate��8�|l*       ����	   �.��A���Z*

losses/value_loss8��=1+       ��K	   �.��A���Z*

losses/policy_loss�g��3��M'       ��F	   �.��A���Z*

losses/entropy�J�� �g�-       <A��	   �.��A���Z*

losses/old_approx_kl%��;&���)       7�_ 	   �.��A���Z*

losses/approx_kl,[];����(       �pJ	   �.��A���Z*

losses/clipfracU��=*a{D2       $V�	   �.��A���Z*"
 
losses/discounted_returns�
@ `�<#       ��wC	   �.��A���Z*


charts/SPS©(D�Y��'       ��F	   �.��A���Z*

charts/restart    ���/       m]P	   /��A���[*

charts/episodic_returnw�B@
o^3       � V	   /��A���[*#
!
charts/windowed_avg_return�=
B[�.�/       m]P	   /��A���[*

charts/episodic_length��DURL-       <A��	  �/��A���[*

charts/learning_rateo־8Ժ�*       ����	  �/��A���[*

losses/value_lossꇍ=|�Ks+       ��K	  �/��A���[*

losses/policy_lossF���3��+'       ��F	  �/��A���[*

losses/entropy�O����-       <A��	  �/��A���[*

losses/old_approx_klp��:6�)       7�_ 	  �/��A���[*

losses/approx_kl���:oN��(       �pJ	  �/��A���[*

losses/clipfrac p�=n6H�2       $V�	  �/��A���[*"
 
losses/discounted_returns��@�Ghx#       ��wC	  �/��A���[*


charts/SPS�)D��6'       ��F	  �/��A���[*

charts/restart    �J/       m]P	  �#/��A���[*

charts/episodic_returnif&B���3       � V	  �#/��A���[*#
!
charts/windowed_avg_return�^B�d�/       m]P	  �#/��A���[*

charts/episodic_lengthBh�D-��^-       <A��	  @$/��A���[*

charts/learning_rate��8\��*       ����	  @$/��A���[*

losses/value_loss)is=d})�+       ��K	  @$/��A���[*

losses/policy_lossh����>�.'       ��F	  @$/��A���[*

losses/entropy鎿M4�-       <A��	  @$/��A���[*

losses/old_approx_kl�_:;d(��)       7�_ 	  @$/��A���[*

losses/approx_kl8� ;��Ҹ(       �pJ	  @$/��A���[*

losses/clipfracVU�=�+��2       $V�	  @$/��A���[*"
 
losses/discounted_returns�@���#       ��wC	  @$/��A���[*


charts/SPS+F+D��'       ��F	  @$/��A���[*

charts/restart    ��BO/       m]P	   :/��A���[*

charts/episodic_returnY�#BYh�3       � V	   :/��A���[*#
!
charts/windowed_avg_return�B�h�{/       m]P	   :/��A���[*

charts/episodic_length���D�Ӊ)-       <A��	  �:/��A���[*

charts/learning_rateɛ�8�p *       ����	  �:/��A���[*

losses/value_loss��=� %+       ��K	  �:/��A���[*

losses/policy_lossF���\�}'       ��F	  �:/��A���[*

losses/entropy�f����5�-       <A��	  �:/��A���[*

losses/old_approx_kl�z;���()       7�_ 	  �:/��A���[*

losses/approx_kl�8C;´y�(       �pJ	  �:/��A���[*

losses/clipfrac�r�=����2       $V�	  �:/��A���[*"
 
losses/discounted_returns�Q@q�#       ��wC	  �:/��A���[*


charts/SPS�V,D:��'       ��F	  �:/��A���[*

charts/restart    9V+_/       m]P	  �O/��A���[*

charts/episodic_return��BQȋ3       � V	  �O/��A���[*#
!
charts/windowed_avg_return�/BI�V/       m]P	  �O/��A���[*

charts/episodic_length͌�D]ъ-       <A��	   P/��A���[*

charts/learning_rateu~�81�o*       ����	   P/��A���[*

losses/value_loss3�u=���+       ��K	   P/��A���[*

losses/policy_lossL������'       ��F	   P/��A���[*

losses/entropy/����--       <A��	   P/��A���[*

losses/old_approx_klF��:� �5)       7�_ 	   P/��A���[*

losses/approx_klT�&;(���(       �pJ	   P/��A���[*

losses/clipfrac 8�=����2       $V�	   P/��A���[*"
 
losses/discounted_returns.�@��I�#       ��wC	   P/��A���[*


charts/SPS9�-D����'       ��F	   P/��A���[*

charts/restart    W<	W/       m]P	  �e/��A���[*

charts/episodic_return��B�&%R3       � V	  �e/��A���[*#
!
charts/windowed_avg_return��Bl�4=/       m]P	  �e/��A���[*

charts/episodic_length 0�D-�\�-       <A��	  @f/��A���[*

charts/learning_rate#a�8����*       ����	  @f/��A���[*

losses/value_loss9�=�j�8+       ��K	  @f/��A���[*

losses/policy_loss�n�
/��'       ��F	  @f/��A���[*

losses/entropyr�������-       <A��	  @f/��A���[*

losses/old_approx_kl!�; ��@)       7�_ 	  @f/��A���[*

losses/approx_klE38;O��K(       �pJ	  @f/��A���[*

losses/clipfrac @�=�RH�2       $V�	  @f/��A���[*"
 
losses/discounted_returns�@ /�W#       ��wC	  @f/��A���[*


charts/SPSٹ.D@�'       ��F	  @f/��A���[*

charts/restart    轭�/       m]P	  �z/��A���[*

charts/episodic_return�:B���3       � V	  �z/��A���[*#
!
charts/windowed_avg_return:QB<��/       m]P	  �z/��A���[*

charts/episodic_length�5�D��,�-       <A��	  @{/��A���[*

charts/learning_rate�C�8��^*       ����	  @{/��A���[*

losses/value_loss�m~=���+       ��K	  @{/��A���[*

losses/policy_loss}h��{0X^'       ��F	  @{/��A���[*

losses/entropyԏ�R�Ny-       <A��	  @{/��A���[*

losses/old_approx_klIzm;��)       7�_ 	  @{/��A���[*

losses/approx_kl1�;��\(       �pJ	  @{/��A���[*

losses/clipfrac H�=�=�52       $V�	  @{/��A���[*"
 
losses/discounted_returnsk&@
gj#       ��wC	  @{/��A���[*


charts/SPS+0D���'       ��F	  @{/��A���[*

charts/restart    /Gd,/       m]P	   �/��A���[*

charts/episodic_returnl;BB���3       � V	   �/��A���[*#
!
charts/windowed_avg_return+B!���/       m]P	   �/��A���[*

charts/episodic_length `�D��;�-       <A��	  ��/��A���[*

charts/learning_rate~&�8��#�*       ����	  ��/��A���[*

losses/value_loss��=��i+       ��K	  ��/��A���[*

losses/policy_loss�YZ�d��'       ��F	  ��/��A���[*

losses/entropy�Ӓ���]�-       <A��	  ��/��A���[*

losses/old_approx_kl�;5;w��)       7�_ 	  ��/��A���[*

losses/approx_kl^�;b��(       �pJ	  ��/��A���[*

losses/clipfrac `�=��/2       $V�	  ��/��A���[*"
 
losses/discounted_returns�1@r��#       ��wC	  ��/��A���[*


charts/SPS�w.D�Ƽ"'       ��F	  ��/��A���[*

charts/restart    �[.�/       m]P	  @�/��A���[*

charts/episodic_return�*B��YE3       � V	  @�/��A���[*#
!
charts/windowed_avg_return��B]��/       m]P	  @�/��A���[*

charts/episodic_length���DG�]-       <A��	   �/��A���[*

charts/learning_rate*	�8��'�*       ����	   �/��A���[*

losses/value_loss��p=��+       ��K	   �/��A���[*

losses/policy_lossH���q�'       ��F	   �/��A���[*

losses/entropy!��m�W�-       <A��	   �/��A���[*

losses/old_approx_klE\";>=��)       7�_ 	   �/��A���[*

losses/approx_klߕ!;2 2q(       �pJ	   �/��A���[*

losses/clipfrac ��=�%�2       $V�	   �/��A���[*"
 
losses/discounted_returns�1
@L�]#       ��wC	   �/��A���[*


charts/SPSA�/Da�x'       ��F	   �/��A���[*

charts/restart    ��+�/       m]P	  ��/��A���[*

charts/episodic_return7�*B�d�3       � V	  ��/��A���[*#
!
charts/windowed_avg_return�1B$���/       m]P	  ��/��A���[*

charts/episodic_length���DJ�7�-       <A��	  ��/��A���[*

charts/learning_rate��8��:�*       ����	  ��/��A���[*

losses/value_loss�ͅ=�.+       ��K	  ��/��A���[*

losses/policy_loss$��[�0�'       ��F	  ��/��A���[*

losses/entropyf6���Y-       <A��	  ��/��A���[*

losses/old_approx_kl�U};@��)       7�_ 	  ��/��A���[*

losses/approx_kl��;�O��(       �pJ	  ��/��A���[*

losses/clipfracU��=���K2       $V�	  ��/��A���[*"
 
losses/discounted_returns��	@t��#       ��wC	  ��/��A���[*


charts/SPSK�/DD�:s'       ��F	  ��/��A���[*

charts/restart    ��	�/       m]P	  @�/��A���[*

charts/episodic_returnZuB�1�3       � V	  @�/��A���[*#
!
charts/windowed_avg_return�&B4�/       m]P	  @�/��A���[*

charts/episodic_length���D�r�8-       <A��	   �/��A���[*

charts/learning_rate�ν8�@�s*       ����	   �/��A���[*

losses/value_loss�<~=&���+       ��K	   �/��A���[*

losses/policy_loss�@��*�m'       ��F	   �/��A���[*

losses/entropy�a����x^-       <A��	   �/��A���[*

losses/old_approx_klM�!;��y))       7�_ 	   �/��A���[*

losses/approx_kl�l;Ӏ��(       �pJ	   �/��A���[*

losses/clipfracV5�=x^2       $V�	   �/��A���[*"
 
losses/discounted_returns�}	@�	0�#       ��wC	   �/��A���[*


charts/SPS�P0D�v��'       ��F	   �/��A���[*

charts/restart    \�	i/       m]P	  @�/��A���[*

charts/episodic_return7�*B�|3       � V	  @�/��A���[*#
!
charts/windowed_avg_return�hB�Y|/       m]P	  @�/��A���[*

charts/episodic_length#��D�cV!-       <A��	   �/��A���[*

charts/learning_rate2��8��*       ����	   �/��A���[*

losses/value_loss�R|=�:�Z+       ��K	   �/��A���[*

losses/policy_lossƌ-�E��'       ��F	   �/��A���[*

losses/entropy�����X�`-       <A��	   �/��A���[*

losses/old_approx_klހ;�LX)       7�_ 	   �/��A���[*

losses/approx_kl���:�(       �pJ	   �/��A���[*

losses/clipfracU�=u`#�2       $V�	   �/��A���[*"
 
losses/discounted_returnsD@�Kj#       ��wC	   �/��A���[*


charts/SPSR�/Dؚ�s'       ��F	   �/��A���[*

charts/restart    _.��/       m]P	  �0��A���[*

charts/episodic_return�vB�÷�3       � V	  �0��A���[*#
!
charts/windowed_avg_returnÑB�U�j/       m]P	  �0��A���[*

charts/episodic_length�r�Do��-       <A��	  �0��A���[*

charts/learning_rateޓ�8�߼�*       ����	  �0��A���[*

losses/value_lossڏm=;�s�+       ��K	  �0��A���[*

losses/policy_lossH���!��'       ��F	  �0��A���[*

losses/entropy�Č�c
l-       <A��	  �0��A���[*

losses/old_approx_kl#f;�+')       7�_ 	  �0��A���[*

losses/approx_klړ;A:ɱ(       �pJ	  �0��A���[*

losses/clipfrac�:y=���,2       $V�	  �0��A���[*"
 
losses/discounted_returns��?���#       ��wC	  �0��A���[*


charts/SPS2K0D��=�'       ��F	  �0��A���[*

charts/restart    ;2zy/       m]P	  @0��A���[*

charts/episodic_returnJrB[tx�3       � V	  @0��A���[*#
!
charts/windowed_avg_return0�BF��/       m]P	  @0��A���[*

charts/episodic_length(��D�s^0-       <A��	   0��A���[*

charts/learning_rate�v�8�.xA*       ����	   0��A���[*

losses/value_loss�=``��+       ��K	   0��A���[*

losses/policy_loss���L�A�'       ��F	   0��A���[*

losses/entropy*J���a'-       <A��	   0��A���[*

losses/old_approx_kl;w;���)       7�_ 	   0��A���[*

losses/approx_kl9]^;ݠ
(       �pJ	   0��A���[*

losses/clipfrac���=#�2       $V�	   0��A���[*"
 
losses/discounted_returnsAl@�ʀ�#       ��wC	   0��A���[*


charts/SPS�0D��J�'       ��F	   0��A���[*

charts/restart    �\W�/       m]P	  �20��A���[*

charts/episodic_return+�(B���3       � V	  �20��A���[*#
!
charts/windowed_avg_return�B�[R�/       m]P	  �20��A���[*

charts/episodic_length���DP�lm-       <A��	  @30��A���[*

charts/learning_rate8Y�8F/�O*       ����	  @30��A���[*

losses/value_loss~�{=4�f�+       ��K	  @30��A���[*

losses/policy_loss�(��R'       ��F	  @30��A���[*

losses/entropy�_��0���-       <A��	  @30��A���[*

losses/old_approx_kl(�;Χ_)       7�_ 	  @30��A���[*

losses/approx_kl�L;�No�(       �pJ	  @30��A���[*

losses/clipfrac�R�=�Ύ�2       $V�	  @30��A���[*"
 
losses/discounted_returns��@���B#       ��wC	  @30��A���[*


charts/SPS��0D2Q�'       ��F	  @30��A���[*

charts/restart    u�#/       m]P	   I0��A���[*

charts/episodic_return�-B%�3       � V	   I0��A���[*#
!
charts/windowed_avg_return@]B&	/       m]P	   I0��A���[*

charts/episodic_lengthy��D��W-       <A��	  �I0��A���[*

charts/learning_rate�;�8�:)t*       ����	  �I0��A���[*

losses/value_loss��=�0�8+       ��K	  �I0��A���[*

losses/policy_lossF��<�z'       ��F	  �I0��A���[*

losses/entropy������č-       <A��	  �I0��A���[*

losses/old_approx_kl��h;*{��)       7�_ 	  �I0��A���[*

losses/approx_kl�';��Z�(       �pJ	  �I0��A���[*

losses/clipfrac ��=Vn�2       $V�	  �I0��A���[*"
 
losses/discounted_returns�L@�'\^#       ��wC	  �I0��A���[*


charts/SPS01Da)�o'       ��F	  �I0��A���[*

charts/restart    ��4/       m]P	  �`0��A���[*

charts/episodic_return>�B�3       � V	  �`0��A���[*#
!
charts/windowed_avg_return� B�0�/       m]P	  �`0��A���[*

charts/episodic_length,W�D��U�-       <A��	   a0��A���[*

charts/learning_rate��8���*       ����	   a0��A���[*

losses/value_loss!�=���+       ��K	   a0��A���[*

losses/policy_loss�a�y��S'       ��F	   a0��A���[*

losses/entropy���$�M-       <A��	   a0��A���[*

losses/old_approx_kl�n�;�A�)       7�_ 	   a0��A���[*

losses/approx_kl��;�T2�(       �pJ	   a0��A���[*

losses/clipfrac���=�do�2       $V�	   a0��A���[*"
 
losses/discounted_returnsΌ@�_o[#       ��wC	   a0��A���[*


charts/SPS�1DY'       ��F	   a0��A���[*

charts/restart    ���^/       m]P	  �y0��A���[*

charts/episodic_returntu!B�!�w3       � V	  �y0��A���[*#
!
charts/windowed_avg_returnm�B
vx�/       m]P	  �y0��A���[*

charts/episodic_length�z�D&)�-       <A��	   z0��A���[*

charts/learning_rate>�8k.[�*       ����	   z0��A���[*

losses/value_loss�,�=sn+       ��K	   z0��A���[*

losses/policy_loss�j��=(�;'       ��F	   z0��A���[*

losses/entropyɤ�����-       <A��	   z0��A���[*

losses/old_approx_kl�b;L�Q�)       7�_ 	   z0��A���[*

losses/approx_kl�2;��"6(       �pJ	   z0��A���[*

losses/clipfracU�d=�8��2       $V�	   z0��A���[*"
 
losses/discounted_returns�@�ӽ	#       ��wC	   z0��A���[*


charts/SPS�t0D4b�P'       ��F	   z0��A���[*

charts/restart    .Ϗ�/       m]P	  @�0��A���[*

charts/episodic_return0uB�d��3       � V	  @�0��A���[*#
!
charts/windowed_avg_return��B���/       m]P	  @�0��A���[*

charts/episodic_length>��D8�K�-       <A��	   �0��A���[*

charts/learning_rate��8F��*       ����	   �0��A���[*

losses/value_loss��p=@YAO+       ��K	   �0��A���[*

losses/policy_loss��K����'       ��F	   �0��A���[*

losses/entropyV��o�L�-       <A��	   �0��A���[*

losses/old_approx_kl�<;<̓)       7�_ 	   �0��A���[*

losses/approx_klu�7;���(       �pJ	   �0��A���[*

losses/clipfracV��=�'R�2       $V�	   �0��A���[*"
 
losses/discounted_returnsԭ@Hg�#       ��wC	   �0��A���[*


charts/SPSp�0Dp։\'       ��F	   �0��A���[*

charts/restart    [��/       m]P	  ��0��A���[*

charts/episodic_return��%B��<�3       � V	  ��0��A���[*#
!
charts/windowed_avg_returntBg[G�/       m]P	  ��0��A���[*

charts/episodic_length[k�D���-       <A��	  ��0��A���[*

charts/learning_rate�Ƽ8��*W*       ����	  ��0��A���[*

losses/value_loss�6~=/�+       ��K	  ��0��A���[*

losses/policy_loss��k�'�h6'       ��F	  ��0��A���[*

losses/entropy�d���-       <A��	  ��0��A���[*

losses/old_approx_kl�)4;E9�c)       7�_ 	  ��0��A���[*

losses/approx_kl&;�,�U(       �pJ	  ��0��A���[*

losses/clipfracV��=
p�H2       $V�	  ��0��A���[*"
 
losses/discounted_returns�2@7�j�#       ��wC	  ��0��A���[*


charts/SPS,�0D����'       ��F	  ��0��A���[*

charts/restart    �ra/       m]P	  @�0��A���[*

charts/episodic_returnF$Bξ13       � V	  @�0��A���[*#
!
charts/windowed_avg_return3�BF�</       m]P	  @�0��A���[*

charts/episodic_lengthI��D�Ya�-       <A��	   �0��A���[*

charts/learning_rateF��8�4�*       ����	   �0��A���[*

losses/value_loss7�=f�-;+       ��K	   �0��A���[*

losses/policy_loss�\{�.�Z"'       ��F	   �0��A���[*

losses/entropy�j��c�h-       <A��	   �0��A���[*

losses/old_approx_kl��;��v�)       7�_ 	   �0��A���[*

losses/approx_klp�;����(       �pJ	   �0��A���[*

losses/clipfracU}�=~[j.2       $V�	   �0��A���[*"
 
losses/discounted_returns�@ꖌ�#       ��wC	   �0��A���[*


charts/SPS�0D�ZS�'       ��F	   �0��A���[*

charts/restart    �KHY/       m]P	   �0��A���[*

charts/episodic_return��	BE3       � V	   �0��A���[*#
!
charts/windowed_avg_return�|B��֟/       m]P	   �0��A���[*

charts/episodic_length$�Dg�:<-       <A��	  ��0��A���[*

charts/learning_rate�8��� *       ����	  ��0��A���[*

losses/value_loss�=��b+       ��K	  ��0��A���[*

losses/policy_loss0ͅ�Yŧ{'       ��F	  ��0��A���[*

losses/entropy�����b-       <A��	  ��0��A���[*

losses/old_approx_kl��D;?�8�)       7�_ 	  ��0��A���[*

losses/approx_klj;���(       �pJ	  ��0��A���[*

losses/clipfrac ȇ=� 2       $V�	  ��0��A���[*"
 
losses/discounted_returns�l@�t`-#       ��wC	  ��0��A���[*


charts/SPS�0Dt��'       ��F	  ��0��A���[*

charts/restart    �s�/       m]P	  ��0��A���[*

charts/episodic_returnIm,B�d3       � V	  ��0��A���[*#
!
charts/windowed_avg_return�>B��g/       m]P	  ��0��A���[*

charts/episodic_length;�D��I-       <A��	  ��0��A���[*

charts/learning_rate�n�8ӰXA*       ����	  ��0��A���[*

losses/value_lossG�h=e+       ��K	  ��0��A���[*

losses/policy_lossjs��zW/�'       ��F	  ��0��A���[*

losses/entropy7a���Y��-       <A��	  ��0��A���[*

losses/old_approx_kl΃M;� �")       7�_ 	  ��0��A���[*

losses/approx_kl#� ;jf�(       �pJ	  ��0��A���[*

losses/clipfrac�:|=|fgl2       $V�	  ��0��A���[*"
 
losses/discounted_returns|��?{��#       ��wC	  ��0��A���[*


charts/SPS�Z0DK��W'       ��F	  ��0��A���[*

charts/restart    NnM�/       m]P	  �1��A���[*

charts/episodic_returnN�,B�'�s3       � V	  �1��A���[*#
!
charts/windowed_avg_returns�B�zK�/       m]P	  �1��A���[*

charts/episodic_length�H�D��	x-       <A��	  @1��A���[*

charts/learning_rateLQ�8�Ω*       ����	  @1��A���[*

losses/value_loss�T�=�
c+       ��K	  @1��A���[*

losses/policy_loss:�s�F���'       ��F	  @1��A���[*

losses/entropy0w��GO�-       <A��	  @1��A���[*

losses/old_approx_kl焽:�`#)       7�_ 	  @1��A���[*

losses/approx_kl�;#���(       �pJ	  @1��A���[*

losses/clipfrac��c=G�/�2       $V�	  @1��A���[*"
 
losses/discounted_returns�k�?�.�0#       ��wC	  @1��A���[*


charts/SPSx�/D�G��'       ��F	  @1��A���[*

charts/restart    .��%/       m]P	  � 1��A���[*

charts/episodic_returnQzBX��3       � V	  � 1��A���[*#
!
charts/windowed_avg_returnZmB�(R/       m]P	  � 1��A���[*

charts/episodic_length�	E�&-       <A��	  @!1��A���[*

charts/learning_rate�3�8S�ϙ*       ����	  @!1��A���[*

losses/value_loss0�=��i<+       ��K	  @!1��A���[*

losses/policy_lossh�f�"I.T'       ��F	  @!1��A���[*

losses/entropy�����+�-       <A��	  @!1��A���[*

losses/old_approx_kl�W;�(�L)       7�_ 	  @!1��A���[*

losses/approx_klyK�:���(       �pJ	  @!1��A���[*

losses/clipfracU�e=��2       $V�	  @!1��A���[*"
 
losses/discounted_returnsБ�?k"rT#       ��wC	  @!1��A���[*


charts/SPS	L/D��db'       ��F	  @!1��A���[*

charts/restart    j��/       m]P	  �:1��A���[*

charts/episodic_return�%&B?��3       � V	  �:1��A���[*#
!
charts/windowed_avg_return�B���/       m]P	  �:1��A���[*

charts/episodic_length tE���-       <A��	  @;1��A���[*

charts/learning_rate��8�1 ^*       ����	  @;1��A���[*

losses/value_loss�Gk=|E��+       ��K	  @;1��A���[*

losses/policy_loss������'       ��F	  @;1��A���[*

losses/entropy��mS��-       <A��	  @;1��A���[*

losses/old_approx_kl7ǳ:��[=)       7�_ 	  @;1��A���[*

losses/approx_kl�%;"b��(       �pJ	  @;1��A���[*

losses/clipfracTEz=��2       $V�	  @;1��A���[*"
 
losses/discounted_returns_�	@��#       ��wC	  @;1��A���[*


charts/SPS�.DXb�'       ��F	  @;1��A���[*

charts/restart    %�Y6/       m]P	  �T1��A���[*

charts/episodic_return/#BL�c�3       � V	  �T1��A���[*#
!
charts/windowed_avg_return�:B�e�i/       m]P	  �T1��A���[*

charts/episodic_lengthF��D��+!-       <A��	   U1��A���[*

charts/learning_rateS��81�h*       ����	   U1��A���[*

losses/value_loss�i=Y�/+       ��K	   U1��A���[*

losses/policy_loss
�����d�'       ��F	   U1��A���[*

losses/entropy*�����t-       <A��	   U1��A���[*

losses/old_approx_kl��:� ��)       7�_ 	   U1��A���[*

losses/approx_klr ;�k�(       �pJ	   U1��A���[*

losses/clipfrac ��=���2       $V�	   U1��A���[*"
 
losses/discounted_returns�T�?�p�#       ��wC	   U1��A���[*


charts/SPS&.D4C��'       ��F	   U1��A���[*

charts/restart    %��l/       m]P	  �n1��A���[*

charts/episodic_return�B�e3       � V	  �n1��A���[*#
!
charts/windowed_avg_return^B�.=/       m]P	  �n1��A���[*

charts/episodic_length `�D�_��-       <A��	  @o1��A���[*

charts/learning_rateܻ8.�*       ����	  @o1��A���[*

losses/value_lossT&w=`W5+       ��K	  @o1��A���[*

losses/policy_loss�Ak�#y>�'       ��F	  @o1��A���[*

losses/entropy�\��3���-       <A��	  @o1��A���[*

losses/old_approx_klD�b;lRi)       7�_ 	  @o1��A���[*

losses/approx_kla�';l4��(       �pJ	  @o1��A���[*

losses/clipfracU�q=�2       $V�	  @o1��A���[*"
 
losses/discounted_returns�{�?�a~�#       ��wC	  @o1��A���[*


charts/SPS
�-D�	�['       ��F	  @o1��A���[*

charts/restart    ���T/       m]P	   �1��A���[*

charts/episodic_return�5!BLN"�3       � V	   �1��A���[*#
!
charts/windowed_avg_returnɞB0�Z/       m]P	   �1��A���[*

charts/episodic_lengthw�4E�˕-       <A��	  ��1��A���[*

charts/learning_rate���8d�Y*       ����	  ��1��A���[*

losses/value_loss�g=I	�X+       ��K	  ��1��A���[*

losses/policy_loss4x��[B��'       ��F	  ��1��A���[*

losses/entropyIh��^}��-       <A��	  ��1��A���[*

losses/old_approx_kl�":r0��)       7�_ 	  ��1��A���[*

losses/approx_kl�[%;Y�(       �pJ	  ��1��A���[*

losses/clipfrac�J�=W�
S2       $V�	  ��1��A���[*"
 
losses/discounted_returns'�?z�H�#       ��wC	  ��1��A���[*


charts/SPS!1-D�W'       ��F	  ��1��A���[*

charts/restart    D��/       m]P	   �1��A���[*

charts/episodic_return�B~R9�3       � V	   �1��A���[*#
!
charts/windowed_avg_returnШB���/       m]P	   �1��A���[*

charts/episodic_lengthCE� ,�-       <A��	  ��1��A���[*

charts/learning_rate[��8I���*       ����	  ��1��A���[*

losses/value_loss��=�w�+       ��K	  ��1��A���[*

losses/policy_loss��O��sr'       ��F	  ��1��A���[*

losses/entropy�P��q�I-       <A��	  ��1��A���[*

losses/old_approx_klGH ;��s)       7�_ 	  ��1��A���[*

losses/approx_kl܇;"�
�(       �pJ	  ��1��A���[*

losses/clipfrac�Z�=q[2       $V�	  ��1��A���[*"
 
losses/discounted_returns���?+�6�#       ��wC	  ��1��A���[*


charts/SPS��,D��'       ��F	  ��1��A���[*

charts/restart    ��i�