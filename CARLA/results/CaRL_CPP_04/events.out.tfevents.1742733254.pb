�      ��`�	  �����A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000000469.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 30801920,
    "max_training_score" : 20.704099655151367,
    "best_iteration" : 430,
    "latest_iteration" : 469,
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
text���M'       ��F	  �����A���*

charts/restart  �?�I/       m]P	  @ ��A���*

charts/episodic_return���@�P��3       � V	  @ ��A���*#
!
charts/windowed_avg_return���@|��/       m]P	  @ ��A���*

charts/episodic_length�IkC��7U-       <A��	  @ ��A���*

charts/learning_rate�9k9.*       ����	  @ ��A���*

losses/value_loss�ɥ=�i7+       ��K	  @ ��A���*

losses/policy_loss|ys��25'       ��F	  @ ��A���*

losses/entropy��J�5�-       <A��	  @ ��A���*

losses/old_approx_kl�Ƙ;�u�K)       7�_ 	  @ ��A���*

losses/approx_kl/��;����(       �pJ	  @ ��A���*

losses/clipfrac���=3��m2       $V�	  @ ��A���*"
 
losses/discounted_returnsb|�?Deb�#       ��wC	  @ ��A���*


charts/SPS�mTD��py'       ��F	  @ ��A���*

charts/restart    	ku/       m]P	  � ��A���*

charts/episodic_return��[A���3       � V	  � ��A���*#
!
charts/windowed_avg_return�["A��P/       m]P	  � ��A���*

charts/episodic_lengthD�ʁ�-       <A��	    ��A���*

charts/learning_rate�*k9��y�*       ����	    ��A���*

losses/value_loss?'�=lN��+       ��K	    ��A���*

losses/policy_loss`v���6n'       ��F	    ��A���*

losses/entropy��J��4!-       <A��	    ��A���*

losses/old_approx_klV�;��^p)       7�_ 	    ��A���*

losses/approx_kl��h;L��
(       �pJ	    ��A���*

losses/clipfrac �>�E2       $V�	    ��A���*"
 
losses/discounted_returnsV�?��O#       ��wC	    ��A���*


charts/SPS�3BD���'       ��F	    ��A���*

charts/restart    X7�]/       m]P	  �5 ��A���*

charts/episodic_return8:�A�F�!3       � V	  �5 ��A���*#
!
charts/windowed_avg_return�_Aqoی/       m]P	  �5 ��A���*

charts/episodic_length  KD���-       <A��	  @6 ��A���*

charts/learning_ratePk9�G�*       ����	  @6 ��A���*

losses/value_losse�v=�vD+       ��K	  @6 ��A���*

losses/policy_lossƖ���!��'       ��F	  @6 ��A���*

losses/entropyY�6��-�-       <A��	  @6 ��A���*

losses/old_approx_kl2�;%��)       7�_ 	  @6 ��A���*

losses/approx_kl��=;�H��(       �pJ	  @6 ��A���*

losses/clipfrac >�y�2       $V�	  @6 ��A���*"
 
losses/discounted_returnsr��?F�D�#       ��wC	  @6 ��A���*


charts/SPS��8D��'       ��F	  @6 ��A���*

charts/restart    �RD/       m]P	  �N ��A���*

charts/episodic_returnIw�A�D��3       � V	  �N ��A���*#
!
charts/windowed_avg_returnLC�AD��@/       m]P	  �N ��A���*

charts/episodic_length�B�D-�eD-       <A��	  @O ��A���*

charts/learning_rate�k9��h�*       ����	  @O ��A���*

losses/value_loss&��=��9%+       ��K	  @O ��A���*

losses/policy_lossU\-���@'       ��F	  @O ��A���*

losses/entropya5����-       <A��	  �O ��A���*

losses/old_approx_kl�Mg;�l%�)       7�_ 	  �O ��A���*

losses/approx_kl\52;����(       �pJ	  �O ��A���*

losses/clipfracU�6>XU��2       $V�	  �O ��A���*"
 
losses/discounted_returns���?��n�#       ��wC	  �O ��A���*


charts/SPS3D���'       ��F	  �O ��A���*

charts/restart    h8ǉ/       m]P	  @i ��A���*

charts/episodic_return!;�AD��3       � V	  @i ��A���*#
!
charts/windowed_avg_return��ACI��/       m]P	  @i ��A���*

charts/episodic_length�D��(s-       <A��	  �i ��A���*

charts/learning_rate��j9���*       ����	  �i ��A���*

losses/value_loss4��= G+       ��K	  �i ��A���*

losses/policy_loss aQ:τr'       ��F	  �i ��A���*

losses/entropy�3�`�j�-       <A��	  �i ��A���*

losses/old_approx_kl}c�;:<a�)       7�_ 	  �i ��A���*

losses/approx_kl���;�\^(       �pJ	  �i ��A���*

losses/clipfracU	>L�2       $V�	  �i ��A���*"
 
losses/discounted_returns��?�5��#       ��wC	  �i ��A���*


charts/SPS�-D�A[�'       ��F	  �i ��A���*

charts/restart    ��e�/       m]P	  @� ��A���*

charts/episodic_returnW�A= >p3       � V	  @� ��A���*#
!
charts/windowed_avg_return���A����/       m]P	  @� ��A���*

charts/episodic_length�sD1G��-       <A��	  �� ��A���*

charts/learning_rateS�j9P���*       ����	  �� ��A���*

losses/value_lossm�x=	U�+       ��K	  �� ��A���*

losses/policy_lossn�ʺr��'       ��F	  �� ��A���*

losses/entropy�K-�~K<-       <A��	  �� ��A���*

losses/old_approx_kl���;�/�)       7�_ 	  �� ��A���*

losses/approx_kl���;�+��(       �pJ	  �� ��A���*

losses/clipfracV59>ϊ�x2       $V�	  �� ��A���*"
 
losses/discounted_returns,)�?�'�#       ��wC	  �� ��A���*


charts/SPS1a.D�ڰ�'       ��F	  �� ��A���*

charts/restart    ��Fz/       m]P	  �� ��A���*

charts/episodic_returnp��AD3       � V	  �� ��A���*#
!
charts/windowed_avg_return���Ax���/       m]P	  �� ��A���*

charts/episodic_length��DK~�-       <A��	  @� ��A���*

charts/learning_rate��j9U]#**       ����	  @� ��A���*

losses/value_loss�ܬ=����+       ��K	  @� ��A���*

losses/policy_lossp�����'       ��F	  @� ��A���*

losses/entropy��'��&5w-       <A��	  @� ��A���*

losses/old_approx_kl΅�;�QS)       7�_ 	  @� ��A���*

losses/approx_kl�<W;��N/(       �pJ	  @� ��A���*

losses/clipfracU�>�;/�2       $V�	  @� ��A���*"
 
losses/discounted_returns���?���#       ��wC	  @� ��A���*


charts/SPS�q.D,�wf'       ��F	  @� ��A���*

charts/restart    ��/       m]P	  @� ��A���*

charts/episodic_return�S�AU�X�3       � V	  @� ��A���*#
!
charts/windowed_avg_return/4�A	if�/       m]P	  @� ��A���*

charts/episodic_length6T�D�@��-       <A��	   � ��A���*

charts/learning_rate �j9�*       ����	   � ��A���*

losses/value_loss~Ds=a^��+       ��K	   � ��A���*

losses/policy_loss3�J�L.�n'       ��F	   � ��A���*

losses/entropy0��SMC-       <A��	   � ��A���*

losses/old_approx_klU;;�X)       7�_ 	   � ��A���*

losses/approx_kl��D;��JV(       �pJ	   � ��A���*

losses/clipfracV��=b$��2       $V�	   � ��A���*"
 
losses/discounted_returnsB	�?����#       ��wC	   � ��A���*


charts/SPSRE.D���'       ��F	   � ��A���*

charts/restart    ��#�/       m]P	  �� ��A���*

charts/episodic_return�M�A�-�3       � V	  �� ��A���*#
!
charts/windowed_avg_return$��A((/       m]P	  �� ��A���*

charts/episodic_length���D����-       <A��	  @� ��A���*

charts/learning_rateW�j9 �\�*       ����	  @� ��A���*

losses/value_loss+�_=~�ַ+       ��K	  @� ��A���*

losses/policy_loss䂳��`'       ��F	  @� ��A���*

losses/entropy� �\��-       <A��	  @� ��A���*

losses/old_approx_kl��9�{�)       7�_ 	  @� ��A���*

losses/approx_kl�F8;�_Z�(       �pJ	  @� ��A���*

losses/clipfracV�=�&N62       $V�	  @� ��A���*"
 
losses/discounted_returns�R�?ޙ#       ��wC	  @� ��A���*


charts/SPSg�-DZ4$�'       ��F	  @� ��A���*

charts/restart    �A��/       m]P	  �� ��A���*

charts/episodic_return���A~յ!3       � V	  �� ��A���*#
!
charts/windowed_avg_returnH^�A(��/       m]P	  �� ��A���*

charts/episodic_length_ߠD��^p-       <A��	  @� ��A���*

charts/learning_rate��j9��6*       ����	  @� ��A���*

losses/value_lossP�m=�ҵa+       ��K	  @� ��A���*

losses/policy_loss4.����'       ��F	  @� ��A���*

losses/entropy�/�'§-       <A��	  @� ��A���*

losses/old_approx_kl��A;�n*�)       7�_ 	  @� ��A���*

losses/approx_klW�c;;a� (       �pJ	  @� ��A���*

losses/clipfrac ��=h�m2       $V�	  @� ��A���*"
 
losses/discounted_returns.N�?���q#       ��wC	  @� ��A���*


charts/SPS��,D�,��'       ��F	  @� ��A���*

charts/restart    ���D/       m]P	   � ��A���*

charts/episodic_returnlόA�(�3       � V	   � ��A���*#
!
charts/windowed_avg_return�"�Al���/       m]P	   � ��A���*

charts/episodic_length ��D�-�-       <A��	  �� ��A���*

charts/learning_rate�j9TE�*       ����	  �� ��A���*

losses/value_loss�d^=j�mX+       ��K	  �� ��A���*

losses/policy_lossh��X	0c'       ��F	  �� ��A���*

losses/entropy ��wm�f-       <A��	  �� ��A���*

losses/old_approx_kl�=;>K�l)       7�_ 	  �� ��A���*

losses/approx_kl�E;;�h�(       �pJ	  �� ��A���*

losses/clipfrac���=�bp�2       $V�	  �� ��A���*"
 
losses/discounted_returns�Y�?`��S#       ��wC	  �� ��A���*


charts/SPS�+D8\�a'       ��F	  �� ��A���*

charts/restart    R��z/       m]P	   ��A���*

charts/episodic_return�|�A�ɛ�3       � V	   ��A���*#
!
charts/windowed_avg_return�ԣAU'6</       m]P	   ��A���*

charts/episodic_length�ڗD��֢-       <A��	  ���A���*

charts/learning_rateZ�j9^��*       ����	  ���A���*

losses/value_lossE%K={l�&+       ��K	  ���A���*

losses/policy_losslЦ���m�'       ��F	  ���A���*

losses/entropy8S
�ۂ2�-       <A��	  ���A���*

losses/old_approx_kl]��;��
b)       7�_ 	  ���A���*

losses/approx_kl�F;0��(       �pJ	  ���A���*

losses/clipfrac���=:��2       $V�	  ���A���*"
 
losses/discounted_returns�͸?��|J#       ��wC	  ���A���*


charts/SPS�4*Dr1'       ��F	  ���A���*

charts/restart    �Lx=/       m]P	  �.��A���*

charts/episodic_return�T�Avv<?3       � V	  �.��A���*#
!
charts/windowed_avg_return�ޥAܰ�O/       m]P	  �.��A���*

charts/episodic_length0H�Dn��T-       <A��	  @/��A���*

charts/learning_rate��j9��w�*       ����	  @/��A���*

losses/value_loss�dJ=@��F+       ��K	  @/��A���*

losses/policy_loss 
���:�'       ��F	  @/��A���*

losses/entropy����?l-       <A��	  @/��A���*

losses/old_approx_kl6g�:ų��)       7�_ 	  @/��A���*

losses/approx_kl��E;���(       �pJ	  @/��A���*

losses/clipfrac ��=K��j2       $V�	  @/��A���*"
 
losses/discounted_returns ��?�1��#       ��wC	  @/��A���*


charts/SPST�(D�8 i'       ��F	  @/��A���*

charts/restart    .�{/       m]P	  �I��A���*

charts/episodic_return���A�KlI3       � V	  �I��A���*#
!
charts/windowed_avg_return��A�V1/       m]P	  �I��A���*

charts/episodic_length��D�|�-       <A��	  @J��A���*

charts/learning_rate{j9e��*       ����	  @J��A���*

losses/value_loss|�-=�1�++       ��K	  @J��A���*

losses/policy_loss@������'       ��F	  @J��A���*

losses/entropy��
�=�S
-       <A��	  @J��A���*

losses/old_approx_klܦ;wui�)       7�_ 	  @J��A���*

losses/approx_klBP2;��(       �pJ	  @J��A���*

losses/clipfracV��=�R�	2       $V�	  @J��A���*"
 
losses/discounted_returns.��?��[#       ��wC	  @J��A���*


charts/SPSk�'D��'       ��F	  @J��A���*

charts/restart    �I~%/       m]P	  @e��A���*

charts/episodic_returne4�Aw��[3       � V	  @e��A���*#
!
charts/windowed_avg_returnq]�A��Z/       m]P	  @e��A���*

charts/episodic_length���D'�5-       <A��	   f��A���*

charts/learning_rate]lj9 m*       ����	   f��A���*

losses/value_loss9 :=�E�+       ��K	   f��A���*

losses/policy_loss(�(�X��'       ��F	   f��A���*

losses/entropy���b�H-       <A��	   f��A���*

losses/old_approx_kl�<;;}6�@)       7�_ 	   f��A���*

losses/approx_kl�3Z;kP��(       �pJ	   f��A���*

losses/clipfracV}�=�P�2       $V�	   f��A���*"
 
losses/discounted_returns&�?l+:�#       ��wC	   f��A���*


charts/SPS�&D/B�'       ��F	   f��A���*

charts/restart    _��/       m]P	  ����A���*

charts/episodic_return�G�A���~3       � V	  ����A���*#
!
charts/windowed_avg_return̩A���/       m]P	  ����A���*

charts/episodic_length�j�D�?
-       <A��	   ���A���*

charts/learning_rate�]j9���*       ����	   ���A���*

losses/value_loss��K=��l�+       ��K	   ���A���*

losses/policy_loss�X����ep'       ��F	   ���A���*

losses/entropyG��K!<�-       <A��	   ���A���*

losses/old_approx_kl';��y>)       7�_ 	   ���A���*

losses/approx_kl�m|;B[{W(       �pJ	   ���A���*

losses/clipfrac�r�=�'�02       $V�	   ���A���*"
 
losses/discounted_returns��?�i�#       ��wC	   ���A���*


charts/SPS_�$D�o��'       ��F	   ���A���*

charts/restart    �G/       m]P	  ����A���*

charts/episodic_returnx��Aٞ�p3       � V	  ����A���*#
!
charts/windowed_avg_return V�A�J�}/       m]P	  ����A���*

charts/episodic_length��DS���-       <A��	  ����A���*

charts/learning_rate
Oj9��&U*       ����	  ����A���*

losses/value_loss��7=c�+       ��K	  ����A���*

losses/policy_loss#��M�z8'       ��F	  ����A���*

losses/entropy���lE7�-       <A��	  ����A���*

losses/old_approx_kl�=;7�)       7�_ 	  ����A���*

losses/approx_kl 8P;���(       �pJ	  ����A���*

losses/clipfrac�r�=/y*2       $V�	  ����A���*"
 
losses/discounted_returns�2�?<��+#       ��wC	  ����A���*


charts/SPS��"D)�5�'       ��F	  ����A���*

charts/restart    )2_�/       m]P	  ����A���*

charts/episodic_returnoE�An�)~3       � V	  ����A���*#
!
charts/windowed_avg_return1��A���/       m]P	  ����A���*

charts/episodic_length�A�D�	p}-       <A��	  @���A���*

charts/learning_ratea@j9ztm*       ����	  @���A���*

losses/value_loss��Y={��z+       ��K	  @���A���*

losses/policy_loss��s�4'       ��F	  @���A���*

losses/entropya4��r-       <A��	  @���A���*

losses/old_approx_kl�W;����)       7�_ 	  @���A���*

losses/approx_klt#Y;П�(       �pJ	  @���A���*

losses/clipfrac ��=�:\k2       $V�	  @���A���*"
 
losses/discounted_returns�z�?�f�%#       ��wC	  @���A���*


charts/SPSK=!D�Jp'       ��F	  @���A���*

charts/restart    !���/       m]P	   ���A���*

charts/episodic_return��AYA��3       � V	   ���A���*#
!
charts/windowed_avg_returnm�AP���/       m]P	   ���A���*

charts/episodic_length9��D����-       <A��	  ����A���*

charts/learning_rate�1j9��S*       ����	  ����A���*

losses/value_loss�J=�!�+       ��K	  ����A���*

losses/policy_loss �� >'       ��F	  ����A���*

losses/entropy� ��{�-       <A��	  ����A���*

losses/old_approx_klp�:~mW8)       7�_ 	  ����A���*

losses/approx_klQd�;O�8(       �pJ	  ����A���*

losses/clipfrac h�=�٠2       $V�	  ����A���*"
 
losses/discounted_returnsތ�?@��C#       ��wC	  ����A���*


charts/SPSK�D}o�O'       ��F	  ����A���*

charts/restart    �O�