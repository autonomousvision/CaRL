�      �YW�	  ����A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000003996.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 261947392,
    "max_training_score" : 41.727436065673828,
    "best_iteration" : 3304,
    "latest_iteration" : 3996,
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
texti->'       ��F	  ����A���|*

charts/restart  �?gP#i/       m]P	  ����A���|*

charts/episodic_returnj��@��R�3       � V	  ����A���|*#
!
charts/windowed_avg_returnj��@k,�/       m]P	  ����A���|*

charts/episodic_length��pC6â-       <A��	  ����A���|*

charts/learning_rateJ�8����*       ����	  ����A���|*

losses/value_losssU�=GP��+       ��K	  ����A���|*

losses/policy_loss�J���(/p'       ��F	  ����A���|*

losses/entropy鮏�B��u-       <A��	  ����A���|*

losses/old_approx_kl�q�:N��)       7�_ 	  ����A���|*

losses/approx_kl�>�:M�O(       �pJ	  ����A���|*

losses/clipfrac  =Δ�2       $V�	  ����A���|*"
 
losses/discounted_returnsx�)@�	�#       ��wC	  ����A���|*


charts/SPSRGDW�?�'       ��F	  ����A���|*

charts/restart    ���/       m]P	  �,���A���|*

charts/episodic_return^�A��� 3       � V	  �,���A���|*#
!
charts/windowed_avg_return�FiA2 B�/       m]P	  �,���A���|*

charts/episodic_length-�-D�Y&-       <A��	  �-���A���|*

charts/learning_rate��8΢��*       ����	  �-���A���|*

losses/value_loss)�=���+       ��K	  �-���A���|*

losses/policy_loss�߁��1��'       ��F	  �-���A���|*

losses/entropy[J��4�E-       <A��	  �-���A���|*

losses/old_approx_klh^�:���y)       7�_ 	  �-���A���|*

losses/approx_klꫡ:k��(       �pJ	  �-���A���|*

losses/clipfrac��=�&�2       $V�	  �-���A���|*"
 
losses/discounted_returns��.@���#       ��wC	  �-���A���|*


charts/SPS��JD��u�'       ��F	  �-���A���|*

charts/restart    �|V/       m]P	  �@���A���}*

charts/episodic_returnSQBo�n3       � V	  �@���A���}*#
!
charts/windowed_avg_return�M�A/�e/       m]P	  �@���A���}*

charts/episodic_length6ԊD��-       <A��	  �A���A���}*

charts/learning_rate�j8n�|*       ����	  �A���A���}*

losses/value_loss���=W��+       ��K	  �A���A���}*

losses/policy_lossYZ��K�'       ��F	  �A���A���}*

losses/entropyI?��R��-       <A��	  �A���A���}*

losses/old_approx_kl�:'��)       7�_ 	  �A���A���}*

losses/approx_kl�g�:�KO(       �pJ	  �A���A���}*

losses/clipfrac��=ߝ�w2       $V�	  �A���A���}*"
 
losses/discounted_returnsJC%@xa�
#       ��wC	  �A���A���}*


charts/SPSהKD�P'       ��F	  �A���A���}*

charts/restart    �!�_/       m]P	  @T���A���}*

charts/episodic_returnc�B�%�c3       � V	  @T���A���}*#
!
charts/windowed_avg_return��A�Tuz/       m]P	  @T���A���}*

charts/episodic_lengthU�Dg���-       <A��	   U���A���}*

charts/learning_rate[08glj�*       ����	   U���A���}*

losses/value_loss	��=����+       ��K	   U���A���}*

losses/policy_loss��.��J�'       ��F	   U���A���}*

losses/entropy�N��ja�#-       <A��	   U���A���}*

losses/old_approx_kl�Ҿ:����)       7�_ 	   U���A���}*

losses/approx_kl�]�:n�E(       �pJ	   U���A���}*

losses/clipfracV%=�^,2       $V�	   U���A���}*"
 
losses/discounted_returns-� @��#       ��wC	   U���A���}*


charts/SPS3MD���'       ��F	   U���A���}*

charts/restart    ��./       m]P	  @i���A���}*

charts/episodic_return�BC˻C3       � V	  @i���A���}*#
!
charts/windowed_avg_returnM��A�V
�/       m]P	  @i���A���}*

charts/episodic_lengthj԰D�ɔ7-       <A��	   j���A���}*

charts/learning_rate��8��_*       ����	   j���A���}*

losses/value_loss�D�=U�� +       ��K	   j���A���}*

losses/policy_lossLQ�L4�'       ��F	   j���A���}*

losses/entropyb���1�Y�-       <A��	   j���A���}*

losses/old_approx_kl�C�:�Y�)       7�_ 	   j���A���}*

losses/approx_kl���:F�6>(       �pJ	   j���A���}*

losses/clipfrac  =v��2       $V�	   j���A���}*"
 
losses/discounted_returnsN�@�#P�#       ��wC	   j���A���}*


charts/SPSkKD���'       ��F	   j���A���}*

charts/restart    AV�
/       m]P	  �}���A���}*

charts/episodic_return.B�5��3       � V	  �}���A���}*#
!
charts/windowed_avg_returno0�A�G��/       m]P	  �}���A���}*

charts/episodic_length߹�D�n�-       <A��	  @~���A���}*

charts/learning_rate�8�f[	*       ����	  @~���A���}*

losses/value_lossÕ=���+       ��K	  @~���A���}*

losses/policy_loss\Z��IF�'       ��F	  @~���A���}*

losses/entropy���g=L-       <A��	  @~���A���}*

losses/old_approx_kl���:EW)       7�_ 	  @~���A���}*

losses/approx_kl��:�B�f(       �pJ	  @~���A���}*

losses/clipfrac��=�m��2       $V�	  @~���A���}*"
 
losses/discounted_returns^k@3oM�#       ��wC	  @~���A���}*


charts/SPS��JDzb�F'       ��F	  @~���A���}*

charts/restart    �[/       m]P	  @����A���}*

charts/episodic_returnR�B�y�3       � V	  @����A���}*#
!
charts/windowed_avg_return.X�Aa��./       m]P	  @����A���}*

charts/episodic_length��D_ ݡ-       <A��	   ����A���}*

charts/learning_rateg�8k�D�*       ����	   ����A���}*

losses/value_loss"�}=�y�s+       ��K	   ����A���}*

losses/policy_loss1�C��fr'       ��F	   ����A���}*

losses/entropy�F��X�=-       <A��	   ����A���}*

losses/old_approx_klx��:\��t)       7�_ 	   ����A���}*

losses/approx_klZ��:q��(       �pJ	   ����A���}*

losses/clipfracV� =����2       $V�	   ����A���}*"
 
losses/discounted_returns׍@Y(�#       ��wC	   ����A���}*


charts/SPS�KD��8g'       ��F	   ����A���}*

charts/restart    ���]/       m]P	  �����A���}*

charts/episodic_return\2B]��3       � V	  �����A���}*#
!
charts/windowed_avg_return )B��)�/       m]P	  �����A���}*

charts/episodic_length�'�D�P'�-       <A��	  �����A���}*

charts/learning_rate�E8�tЅ*       ����	  �����A���}*

losses/value_loss���=��+       ��K	  �����A���}*

losses/policy_loss������'       ��F	  �����A���}*

losses/entropyI������-       <A��	  �����A���}*

losses/old_approx_kl3{y:ę�I)       7�_ 	  �����A���}*

losses/approx_kl>f�:Nр#(       �pJ	  �����A���}*

losses/clipfrac��/=����2       $V�	  �����A���}*"
 
losses/discounted_returns�8@:�{#       ��wC	  �����A���}*


charts/SPS��ID�x�'       ��F	  �����A���}*

charts/restart    �)hA/       m]P	  @����A���}*

charts/episodic_return� Bc$��3       � V	  @����A���}*#
!
charts/windowed_avg_return�B14��/       m]P	  @����A���}*

charts/episodic_length��D�7�-       <A��	   ����A���}*

charts/learning_rate8�2�\*       ����	   ����A���}*

losses/value_lossh�=��+       ��K	   ����A���}*

losses/policy_loss�8���('       ��F	   ����A���}*

losses/entropy�����o-       <A��	   ����A���}*

losses/old_approx_kl��=:{�)       7�_ 	   ����A���}*

losses/approx_kl���:c��(       �pJ	   ����A���}*

losses/clipfrac�z=]��2       $V�	   ����A���}*"
 
losses/discounted_returns4~@9A7�#       ��wC	   ����A���}*


charts/SPS��GDT�'       ��F	   ����A���}*

charts/restart    �Z��/       m]P	  @ҹ��A���}*

charts/episodic_return��+B�Kf�3       � V	  @ҹ��A���}*#
!
charts/windowed_avg_return��B-�U/       m]P	  @ҹ��A���}*

charts/episodic_length�r�D�-       <A��	   ӹ��A���}*

charts/learning_rates�8�\�z*       ����	   ӹ��A���}*

losses/value_loss�Bd=��p�+       ��K	   ӹ��A���}*

losses/policy_loss@�B�g�.'       ��F	   ӹ��A���}*

losses/entropy�W��@-       <A��	   ӹ��A���}*

losses/old_approx_klظ';��FQ)       7�_ 	   ӹ��A���}*

losses/approx_kl"��:���(       �pJ	   ӹ��A���}*

losses/clipfrac @=��&2       $V�	   ӹ��A���}*"
 
losses/discounted_returns/�	@C$�>#       ��wC	   ӹ��A���}*


charts/SPSA�FD~��+'       ��F	   ӹ��A���}*

charts/restart    �/       m]P	  ����A���}*

charts/episodic_return��B���R3       � V	  ����A���}*#
!
charts/windowed_avg_return�
Ba�`�/       m]P	  ����A���}*

charts/episodic_length�s�DM��-       <A��	  ����A���}*

charts/learning_rateϕ8e0�{*       ����	  ����A���}*

losses/value_loss���=L.Tr+       ��K	  ����A���}*

losses/policy_loss��F��Z��'       ��F	  ����A���}*

losses/entropydJ����H�-       <A��	  ����A���}*

losses/old_approx_kl��;�dh)       7�_ 	  ����A���}*

losses/approx_klD��:�a2(       �pJ	  ����A���}*

losses/clipfrac�*0=JCk�2       $V�	  ����A���}*"
 
losses/discounted_returns�@�T��#       ��wC	  ����A���}*


charts/SPS��BD]Sg'       ��F	  ����A���}*

charts/restart    �nV?/       m]P	  ����A���}*

charts/episodic_return�1Bu��3       � V	  ����A���}*#
!
charts/windowed_avg_returnc�
B��J/       m]P	  ����A���}*

charts/episodic_length?t�D�Q K-       <A��	  @���A���}*

charts/learning_rate'[8����*       ����	  @���A���}*

losses/value_loss ɔ=�#�+       ��K	  @���A���}*

losses/policy_lossDJ��lp��'       ��F	  @���A���}*

losses/entropy���*�g�-       <A��	  @���A���}*

losses/old_approx_kl�.;Hc4�)       7�_ 	  @���A���}*

losses/approx_klse�:�j�;(       �pJ	  @���A���}*

losses/clipfrac��=����2       $V�	  @���A���}*"
 
losses/discounted_returnsh�@T�#�#       ��wC	  @���A���}*


charts/SPS�?D���<'       ��F	  @���A���}*

charts/restart    ��/       m]P	  ����A���}*

charts/episodic_returnb�B?Q��3       � V	  ����A���}*#
!
charts/windowed_avg_return�+BqV�i/       m]P	  ����A���}*

charts/episodic_length1��Dw�-       <A��	  @���A���}*

charts/learning_rate� 8� �-*       ����	  @���A���}*

losses/value_loss�X�=a,1+       ��K	  @���A���}*

losses/policy_loss������'       ��F	  @���A���}*

losses/entropyD��g�-       <A��	  @���A���}*

losses/old_approx_kl��9c��j)       7�_ 	  @���A���}*

losses/approx_kl
��:L�n(       �pJ	  @���A���}*

losses/clipfrac =gM?�2       $V�	  @���A���}*"
 
losses/discounted_returnsU
@�s8H#       ��wC	  @���A���}*


charts/SPS>D�R�'       ��F	  @���A���}*

charts/restart    #�/       m]P	  �1���A���}*

charts/episodic_return�fB,��3       � V	  �1���A���}*#
!
charts/windowed_avg_return��B^�/       m]P	  �1���A���}*

charts/episodic_lengthw��D��n-       <A��	  @2���A���}*

charts/learning_rate��87�*       ����	  @2���A���}*

losses/value_loss�=��0+       ��K	  @2���A���}*

losses/policy_loss��K��`�
'       ��F	  @2���A���}*

losses/entropyf�����e-       <A��	  @2���A���}*

losses/old_approx_klo�:	N�)       7�_ 	  @2���A���}*

losses/approx_kl��:��M(       �pJ	  @2���A���}*

losses/clipfrac�*='�Xl2       $V�	  @2���A���}*"
 
losses/discounted_returns��@�9#       ��wC	  @2���A���}*


charts/SPS�u>D����'       ��F	  @2���A���}*

charts/restart    �3��/       m]P	  �E���A���}*

charts/episodic_return��DBMXAB3       � V	  �E���A���}*#
!
charts/windowed_avg_return�_B8��/       m]P	  �E���A���}*

charts/episodic_length���D��O-       <A��	  �F���A���}*

charts/learning_rate8�8d��2*       ����	  �F���A���}*

losses/value_lossy֖=<�&�+       ��K	  �F���A���}*

losses/policy_lossʡ\�&M��'       ��F	  �F���A���}*

losses/entropyx��ba-       <A��	  �F���A���}*

losses/old_approx_kl��(:�|��)       7�_ 	  �F���A���}*

losses/approx_kl�{�:��.�(       �pJ	  �F���A���}*

losses/clipfrac �*=��BZ2       $V�	  �F���A���}*"
 
losses/discounted_returns*�@�m�x#       ��wC	  �F���A���}*


charts/SPSA?Dpm'       ��F	  �F���A���}*

charts/restart    ؾ�a/       m]P	  �\���A���}*

charts/episodic_returnY�	B�0�3       � V	  �\���A���}*#
!
charts/windowed_avg_return�BZ<�C/       m]P	  �\���A���}*

charts/episodic_lengthUy�DWQ��-       <A��	  �]���A���}*

charts/learning_rate�p8�t�*       ����	  �]���A���}*

losses/value_loss�a�=}�']+       ��K	  �]���A���}*

losses/policy_loss�X�t�'       ��F	  �]���A���}*

losses/entropy�;��7��-       <A��	  �]���A���}*

losses/old_approx_kl�8�:�Gq�)       7�_ 	  �]���A���}*

losses/approx_kl��:�)Y�(       �pJ	  �]���A���}*

losses/clipfracVU=c�Q�2       $V�	  �]���A���}*"
 
losses/discounted_returnsG�@� �#       ��wC	  �]���A���}*


charts/SPSc[>D��'       ��F	  �]���A���}*

charts/restart    A���/       m]P	  �t���A���}*

charts/episodic_return��B緡�3       � V	  �t���A���}*#
!
charts/windowed_avg_returnY�B���/       m]P	  �t���A���}*

charts/episodic_length�ӹDT���-       <A��	  �u���A���}*

charts/learning_rate�58;0�*       ����	  �u���A���}*

losses/value_loss�u=�%�+       ��K	  �u���A���}*

losses/policy_losspB��Mv�v'       ��F	  �u���A���}*

losses/entropy>̆���H�-       <A��	  �u���A���}*

losses/old_approx_klp�9q���)       7�_ 	  �u���A���}*

losses/approx_kl# �:6�s�(       �pJ	  �u���A���}*

losses/clipfracU%=x�n2       $V�	  �u���A���}*"
 
losses/discounted_returns��@?�Ƞ#       ��wC	  �u���A���}*


charts/SPS%=D�#۸'       ��F	  �u���A���}*

charts/restart    B��/       m]P	  @����A���}*

charts/episodic_return��B�S
�3       � V	  @����A���}*#
!
charts/windowed_avg_return�EBS��V/       m]P	  @����A���}*

charts/episodic_length `�D!0�-       <A��	   ����A���}*

charts/learning_rateD� 83U�!*       ����	   ����A���}*

losses/value_loss��u=:cr+       ��K	   ����A���}*

losses/policy_loss9�T����'       ��F	   ����A���}*

losses/entropyP�|�uI��-       <A��	   ����A���}*

losses/old_approx_kl��:�a�)       7�_ 	   ����A���}*

losses/approx_kl�h�:�^C(       �pJ	   ����A���}*

losses/clipfracVu=� (2       $V�	   ����A���}*"
 
losses/discounted_returnsg@��|�#       ��wC	   ����A���}*


charts/SPSb�:D���'       ��F	   ����A���}*

charts/restart    _��E/       m]P	  �����A���}*

charts/episodic_return+B�csV3       � V	  �����A���}*#
!
charts/windowed_avg_returnM�B���N/       m]P	  �����A���}*

charts/episodic_lengthŮ�D�$��-       <A��	  @����A���}*

charts/learning_rate�� 8�R8M*       ����	  @����A���}*

losses/value_loss7My=�6�b+       ��K	  @����A���}*

losses/policy_loss,k��
�'       ��F	  @����A���}*

losses/entropy�|���ه-       <A��	  @����A���}*

losses/old_approx_kl���:_W��)       7�_ 	  @����A���}*

losses/approx_kl(y�:��>�(       �pJ	  @����A���}*

losses/clipfrac �=�/�2       $V�	  @����A���}*"
 
losses/discounted_returnsN4@�~�#       ��wC	  @����A���}*


charts/SPS�K9D��'       ��F	  @����A���}*

charts/restart    ��