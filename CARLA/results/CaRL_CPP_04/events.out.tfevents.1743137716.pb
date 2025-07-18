�      �YW�	   ���A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000003621.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 237371392,
    "max_training_score" : 41.727436065673828,
    "best_iteration" : 3304,
    "latest_iteration" : 3621,
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
text����'       ��F	  @���A���q*

charts/restart  �?��,	/       m]P	  ����A���q*

charts/episodic_returnŮ�@C��G3       � V	  ����A���q*#
!
charts/windowed_avg_returnŮ�@�5��/       m]P	  ����A���q*

charts/episodic_length�LaCx���-       <A��	  ����A���q*

charts/learning_rate��Z8M��*       ����	  ����A���q*

losses/value_loss�8�=b��+       ��K	  ����A���q*

losses/policy_loss�:��<�I'       ��F	  ����A���q*

losses/entropy�#��i?-       <A��	  ����A���q*

losses/old_approx_klt��ꅏ�)       7�_ 	  ����A���q*

losses/approx_klX�;�O(       �pJ	  ����A���q*

losses/clipfrac ؎=/���2       $V�	  ����A���q*"
 
losses/discounted_returns�1"@�^�#       ��wC	  ����A���q*


charts/SPSW�D
B�h'       ��F	  ����A���q*

charts/restart    X1��/       m]P	  � ���A���q*

charts/episodic_return��A'�Ə3       � V	  � ���A���q*#
!
charts/windowed_avg_return�jA�ڪw/       m]P	  � ���A���q*

charts/episodic_lengtho�6D{�-d-       <A��	   !���A���q*

charts/learning_rate�Z8%&��*       ����	   !���A���q*

losses/value_loss^%�=�'*+       ��K	   !���A���q*

losses/policy_losss*��Ǣ�'       ��F	   !���A���q*

losses/entropy����PC?-       <A��	   !���A���q*

losses/old_approx_kl)�E;/�)       7�_ 	   !���A���q*

losses/approx_kl��-;�l�R(       �pJ	   !���A���q*

losses/clipfrac �k=���2       $V�	   !���A���q*"
 
losses/discounted_returnsA*%@���#       ��wC	   !���A���q*


charts/SPS1�!D�'       ��F	   !���A���q*

charts/restart    K1�P/       m]P	  �6���A���q*

charts/episodic_return7"B�,�3       � V	  �6���A���q*#
!
charts/windowed_avg_returnc:�AX�x�/       m]P	  �6���A���q*

charts/episodic_length���D�n{�-       <A��	  @7���A���q*

charts/learning_rateJTZ8!���*       ����	  @7���A���q*

losses/value_loss���=QS��+       ��K	  @7���A���q*

losses/policy_lossf���|�'       ��F	  @7���A���q*

losses/entropygt���(Z-       <A��	  @7���A���q*

losses/old_approx_kl�;m��)       7�_ 	  @7���A���q*

losses/approx_kl�j�:��H3(       �pJ	  @7���A���q*

losses/clipfrac �N=���T2       $V�	  @7���A���q*"
 
losses/discounted_returns��@��#       ��wC	  @7���A���q*


charts/SPSض(D��5�'       ��F	  @7���A���q*

charts/restart    �x��/       m]P	  �N���A���q*

charts/episodic_return�B�ʴ�3       � V	  �N���A���q*#
!
charts/windowed_avg_return;8�AUG�/       m]P	  �N���A���q*

charts/episodic_length���D�!J-       <A��	  �O���A���q*

charts/learning_rate�Z8&�9*       ����	  �O���A���q*

losses/value_loss��={�M+       ��K	  �O���A���q*

losses/policy_lossLצ�[��'       ��F	  �O���A���q*

losses/entropyCInG�-       <A��	  �O���A���q*

losses/old_approx_kl�kJ;�7ԇ)       7�_ 	  �O���A���q*

losses/approx_kle�;I`I(       �pJ	  �O���A���q*

losses/clipfrac �D=�l�2       $V�	  �O���A���q*"
 
losses/discounted_returnslm@��P}#       ��wC	  �O���A���q*


charts/SPS	�(D���'       ��F	  �O���A���q*

charts/restart    ��n;/       m]P	  �k���A���q*

charts/episodic_return��*B��o�3       � V	  �k���A���q*#
!
charts/windowed_avg_return ��AR0��/       m]P	  �k���A���q*

charts/episodic_length3��D���S-       <A��	  @l���A���q*

charts/learning_rate��Y8S� �*       ����	  @l���A���q*

losses/value_loss��{=�NR(+       ��K	  @l���A���q*

losses/policy_loss|Ŋ����j'       ��F	  @l���A���q*

losses/entropyj��ȃ�-       <A��	  @l���A���q*

losses/old_approx_kl#��:|��)       7�_ 	  @l���A���q*

losses/approx_kl���:�Q�O(       �pJ	  @l���A���q*

losses/clipfracVuS=���2       $V�	  @l���A���q*"
 
losses/discounted_returnsVF@ʹ��#       ��wC	  @l���A���q*


charts/SPS:�"DȔ'       ��F	  @l���A���q*

charts/restart    R#D�/       m]P	  �����A���q*

charts/episodic_return��EB����3       � V	  �����A���q*#
!
charts/windowed_avg_returnW�B�I�/       m]P	  �����A���q*

charts/episodic_length�i�D$^��-       <A��	  @����A���q*

charts/learning_rateW�Y8yh*�*       ����	  @����A���q*

losses/value_loss�q�=;�Y�+       ��K	  @����A���q*

losses/policy_loss�Jv�o�yH'       ��F	  @����A���q*

losses/entropy���0�ى-       <A��	  @����A���q*

losses/old_approx_klqeh;����)       7�_ 	  @����A���q*

losses/approx_klT;;��>(       �pJ	  @����A���q*

losses/clipfrac py=�+��2       $V�	  @����A���q*"
 
losses/discounted_returnsuL@�#       ��wC	  @����A���q*


charts/SPS�W&D��'       ��F	  @����A���q*

charts/restart    ����/       m]P	   ����A���q*

charts/episodic_return��B��WS3       � V	   ����A���q*#
!
charts/windowed_avg_returnd!B,Kp/       m]P	   ����A���q*

charts/episodic_length�R�D�2-       <A��	  �����A���q*

charts/learning_rate�iY8w�*       ����	  �����A���q*

losses/value_lossS��=S�tG+       ��K	  �����A���q*

losses/policy_loss%��B#XN'       ��F	  �����A���q*

losses/entropyԬ��T�Z-       <A��	  �����A���q*

losses/old_approx_kl�^Y:R\�)       7�_ 	  �����A���q*

losses/approx_klH��:0�$'(       �pJ	  �����A���q*

losses/clipfracU5~=�6�%2       $V�	  �����A���q*"
 
losses/discounted_returns��@�X��#       ��wC	  �����A���q*


charts/SPSM[*D����'       ��F	  �����A���q*

charts/restart    �B�/       m]P	   ����A���q*

charts/episodic_return�p;BV\�3       � V	   ����A���q*#
!
charts/windowed_avg_returnV�B�g��/       m]P	   ����A���q*

charts/episodic_length��D{;c<-       <A��	  �����A���q*

charts/learning_rate/Y8�*       ����	  �����A���q*

losses/value_loss��=�~)�+       ��K	  �����A���q*

losses/policy_loss����͜�'       ��F	  �����A���q*

losses/entropy^��uG?�-       <A��	  �����A���q*

losses/old_approx_klX�;�)       7�_ 	  �����A���q*

losses/approx_kl���:��H7(       �pJ	  �����A���q*

losses/clipfrac Pv=���2       $V�	  �����A���q*"
 
losses/discounted_returns�@I�8#       ��wC	  �����A���q*


charts/SPS�G,D��*)'       ��F	  �����A���q*

charts/restart    Y���/       m]P	  @ċ��A���q*

charts/episodic_returnD�B�f+�3       � V	  @ċ��A���q*#
!
charts/windowed_avg_return7�B��*;/       m]P	  @ċ��A���q*

charts/episodic_length ��D�MU-       <A��	  �ċ��A���q*

charts/learning_rateg�X8 .*�*       ����	  �ċ��A���q*

losses/value_lossn�=��+       ��K	  �ċ��A���q*

losses/policy_lossaV����9'       ��F	  �ċ��A���q*

losses/entropyx/����t-       <A��	  �ċ��A���q*

losses/old_approx_kl1�;�T��)       7�_ 	  �ċ��A���q*

losses/approx_kl�t<;���;(       �pJ	  �ċ��A���q*

losses/clipfracVU�=��2       $V�	  �ċ��A���q*"
 
losses/discounted_returns �@��J#       ��wC	  �ċ��A���q*


charts/SPS�+DL&'       ��F	  �ċ��A���q*

charts/restart    .�E�/       m]P	  @ڋ��A���q*

charts/episodic_return��4B`�^3       � V	  @ڋ��A���q*#
!
charts/windowed_avg_return��B]g��/       m]P	  @ڋ��A���q*

charts/episodic_lengthn��D���-       <A��	   ۋ��A���q*

charts/learning_rate��X8$?�*       ����	   ۋ��A���q*

losses/value_loss��=�G��+       ��K	   ۋ��A���q*

losses/policy_loss�0��/��'       ��F	   ۋ��A���q*

losses/entropy���e�y-       <A��	   ۋ��A���q*

losses/old_approx_kl)��:8�y)       7�_ 	   ۋ��A���q*

losses/approx_kl���:���(       �pJ	   ۋ��A���q*

losses/clipfrac�ZC=�-W2       $V�	   ۋ��A���q*"
 
losses/discounted_returnsR�
@��c�#       ��wC	   ۋ��A���q*


charts/SPSGA-D���'       ��F	   ۋ��A���q*

charts/restart    (\�/       m]P	  ����A���q*

charts/episodic_return� B�Sc3       � V	  ����A���q*#
!
charts/windowed_avg_return;B��.�/       m]P	  ����A���q*

charts/episodic_length���DV�uZ-       <A��	  ����A���q*

charts/learning_rateX8���*       ����	  ����A���q*

losses/value_loss�<�={b��+       ��K	  ����A���q*

losses/policy_lossGk���|*'       ��F	  ����A���q*

losses/entropy맇��[��-       <A��	  ����A���q*

losses/old_approx_kl&q;y��n)       7�_ 	  ����A���q*

losses/approx_kl�p;���(       �pJ	  ����A���q*

losses/clipfracUz=sB��2       $V�	  ����A���q*"
 
losses/discounted_returns�@>���#       ��wC	  ����A���q*


charts/SPS�B/D�9'       ��F	  ����A���q*

charts/restart    ����/       m]P	   ���A���q*

charts/episodic_return��)B��cA3       � V	   ���A���q*#
!
charts/windowed_avg_return\�B]��n/       m]P	   ���A���q*

charts/episodic_length,��D�Ŀf-       <A��	  ����A���q*

charts/learning_ratesDX8ۤ�^*       ����	  ����A���q*

losses/value_loss�=����+       ��K	  ����A���q*

losses/policy_loss�낻��e'       ��F	  ����A���q*

losses/entropyÆ� w��-       <A��	  ����A���q*

losses/old_approx_kl��:�H)       7�_ 	  ����A���q*

losses/approx_kl��;NYk(       �pJ	  ����A���q*

losses/clipfrac�zK=A.A�2       $V�	  ����A���q*"
 
losses/discounted_returns#@SEO#       ��wC	  ����A���q*


charts/SPS��0D��f�'       ��F	  ����A���q*

charts/restart    ߞQ/       m]P	  ����A���q*

charts/episodic_return�.B���3       � V	  ����A���q*#
!
charts/windowed_avg_return�7B���/       m]P	  ����A���q*

charts/episodic_lengthܽD�S%-       <A��	  ����A���q*

charts/learning_rate�	X8���*       ����	  ����A���q*

losses/value_loss��=m��+       ��K	  ����A���q*

losses/policy_loss5[����'       ��F	  ����A���q*

losses/entropy]ԏ�-��s-       <A��	  ����A���q*

losses/old_approx_klƾ�:˯�r)       7�_ 	  ����A���q*

losses/approx_kll��:����(       �pJ	  ����A���q*

losses/clipfracU�P=��^a2       $V�	  ����A���q*"
 
losses/discounted_returns��@��"�#       ��wC	  ����A���q*


charts/SPS��2D�Z�'       ��F	  ����A���q*

charts/restart    �]�/       m]P	  �+���A���q*

charts/episodic_return6�1B��w3       � V	  �+���A���q*#
!
charts/windowed_avg_return�>B��[o/       m]P	  �+���A���q*

charts/episodic_lengthm�D��Z-       <A��	  @,���A���q*

charts/learning_rate(�W8\6�*       ����	  @,���A���q*

losses/value_loss���=��d�+       ��K	  @,���A���q*

losses/policy_loss�0���y�0'       ��F	  @,���A���q*

losses/entropy;���*B-       <A��	  @,���A���q*

losses/old_approx_kl��;���)       7�_ 	  @,���A���q*

losses/approx_kl���:q�x(       �pJ	  @,���A���q*

losses/clipfrac @s=�V�2       $V�	  @,���A���q*"
 
losses/discounted_returnsI�@F��Y#       ��wC	  @,���A���q*


charts/SPS�c4D9�C%'       ��F	  @,���A���q*

charts/restart    C��c/       m]P	  @@���A���q*

charts/episodic_return6BOp@3       � V	  @@���A���q*#
!
charts/windowed_avg_return+B�B&V/       m]P	  @@���A���q*

charts/episodic_length���DəF�-       <A��	   A���A���q*

charts/learning_rate��W8u�N�*       ����	   A���A���q*

losses/value_loss��=�֠X+       ��K	   A���A���q*

losses/policy_loss�����-'       ��F	   A���A���q*

losses/entropyh��� .V-       <A��	   A���A���q*

losses/old_approx_kld�:��|H)       7�_ 	   A���A���q*

losses/approx_kl߀;�A�U(       �pJ	   A���A���q*

losses/clipfrac pU=J�j�2       $V�	   A���A���q*"
 
losses/discounted_returnsF�@'rD�#       ��wC	   A���A���q*


charts/SPS_�5D�~�"'       ��F	   A���A���q*

charts/restart    ���/       m]P	   U���A���q*

charts/episodic_returnt;B���E3       � V	   U���A���q*#
!
charts/windowed_avg_returnB(��/       m]P	   U���A���q*

charts/episodic_length���D
xiQ-       <A��	  �U���A���q*

charts/learning_rate�YW8R���*       ����	  �U���A���q*

losses/value_lossvA�=L�<�+       ��K	  �U���A���q*

losses/policy_loss`���~ޣE'       ��F	  �U���A���q*

losses/entropyvU��Ѡ-       <A��	  �U���A���q*

losses/old_approx_kl�[<;3X�)       7�_ 	  �U���A���q*

losses/approx_kl��
;/�P(       �pJ	  �U���A���q*

losses/clipfrac�3=]ai2       $V�	  �U���A���q*"
 
losses/discounted_returns9F
@Z1Fb#       ��wC	  �U���A���q*


charts/SPS��6Dai�m'       ��F	  �U���A���q*

charts/restart    ����/       m]P	  @n���A���q*

charts/episodic_returnޫ�A���B3       � V	  @n���A���q*#
!
charts/windowed_avg_return�jB�3W/       m]P	  @n���A���q*

charts/episodic_length��D�k-       <A��	   o���A���q*

charts/learning_rate4W8����*       ����	   o���A���q*

losses/value_loss#m�=�<�+       ��K	   o���A���q*

losses/policy_loss���N?��'       ��F	   o���A���q*

losses/entropy���#Z-       <A��	   o���A���q*

losses/old_approx_kl^8;����)       7�_ 	   o���A���q*

losses/approx_kl�;���p(       �pJ	   o���A���q*

losses/clipfrac�jj=
���2       $V�	   o���A���q*"
 
losses/discounted_returns!@}��L#       ��wC	   o���A���q*


charts/SPS�5DZ��'       ��F	   o���A���q*

charts/restart    �P�/       m]P	   ����A���q*

charts/episodic_returnA$B��3       � V	   ����A���q*#
!
charts/windowed_avg_return�:B� ��/       m]P	   ����A���q*

charts/episodic_lengthU��D�B�i-       <A��	  �����A���q*

charts/learning_rate��V8���*       ����	  �����A���q*

losses/value_loss�=���+       ��K	  �����A���q*

losses/policy_loss������''       ��F	  �����A���q*

losses/entropy����K��-       <A��	  �����A���q*

losses/old_approx_kl� ;��)       7�_ 	  �����A���q*

losses/approx_kl$T�:���I(       �pJ	  �����A���q*

losses/clipfrac��b=�*Q�2       $V�	  �����A���q*"
 
losses/discounted_returns��@�o�f#       ��wC	  �����A���q*


charts/SPS��3DvX�-'       ��F	  �����A���q*

charts/restart    ��YP/       m]P	  @����A���q*

charts/episodic_return(B��3       � V	  @����A���q*#
!
charts/windowed_avg_return�*B�Ķ"/       m]P	  @����A���q*

charts/episodic_length6��D]*-       <A��	   ����A���q*

charts/learning_rate�V8S%aO*       ����	   ����A���q*

losses/value_loss�F�=G�'+       ��K	   ����A���q*

losses/policy_loss�^���̄L'       ��F	   ����A���q*

losses/entropy�q��zAC-       <A��	   ����A���q*

losses/old_approx_kl�S:���m)       7�_ 	   ����A���q*

losses/approx_kl�&�:IM�(       �pJ	   ����A���q*

losses/clipfrac��a=�:	2       $V�	   ����A���q*"
 
losses/discounted_returnsm@9T�:#       ��wC	   ����A���q*


charts/SPS��1D�	Q�'       ��F	   ����A���q*

charts/restart    L̀�/       m]P	  �����A���q*

charts/episodic_return,�BK��3       � V	  �����A���q*#
!
charts/windowed_avg_return5#B��0/       m]P	  �����A���q*

charts/episodic_lengthF��D?k�&-       <A��	  �����A���q*

charts/learning_rateDoV8��*       ����	  �����A���q*

losses/value_loss�=�`�+       ��K	  �����A���q*

losses/policy_lossU�����'       ��F	  �����A���q*

losses/entropy�l��P���-       <A��	  �����A���q*

losses/old_approx_klBd�:����)       7�_ 	  �����A���q*

losses/approx_kl��:(1�r(       �pJ	  �����A���q*

losses/clipfracV�d=Z�2       $V�	  �����A���q*"
 
losses/discounted_returnsۮ@���%#       ��wC	  �����A���q*


charts/SPSw2D
JǴ'       ��F	  �����A���q*

charts/restart    ���/       m]P	  �Ќ��A���q*

charts/episodic_returnǰ"Bi�;L3       � V	  �Ќ��A���q*#
!
charts/windowed_avg_return�BZ>�/       m]P	  �Ќ��A���q*

charts/episodic_length� �D�b-       <A��	  �ь��A���q*

charts/learning_rate�4V8��*       ����	  �ь��A���q*

losses/value_loss���=<3��+       ��K	  �ь��A���q*

losses/policy_loss��N���Ҳ'       ��F	  �ь��A���q*

losses/entropy�ω�n���-       <A��	  �ь��A���q*

losses/old_approx_klM�|:���)       7�_ 	  �ь��A���q*

losses/approx_kl/�;2r�(       �pJ	  �ь��A���q*

losses/clipfracU�n=�z�2       $V�	  �ь��A���q*"
 
losses/discounted_returnsbA@ΰs�#       ��wC	  �ь��A���q*


charts/SPS�2DG�F�'       ��F	  �ь��A���q*

charts/restart    n�K�/       m]P	   ���A���q*

charts/episodic_return��B.o�3       � V	   ���A���q*#
!
charts/windowed_avg_return��B�p��/       m]P	   ���A���q*

charts/episodic_length��DIޯ
-       <A��	  ����A���q*

charts/learning_rate��U8QjL�*       ����	  ����A���q*

losses/value_loss��=)�S�+       ��K	  ����A���q*

losses/policy_loss|F��ˌ��'       ��F	  ����A���q*

losses/entropy�Ն�k2e�-       <A��	  ����A���q*

losses/old_approx_kl�S;b��)       7�_ 	  ����A���q*

losses/approx_kl���:�$(       �pJ	  ����A���q*

losses/clipfracVP=��/2       $V�	  ����A���q*"
 
losses/discounted_returnsY@��|�#       ��wC	  ����A���q*


charts/SPS\2D�7Q?'       ��F	  ����A���q*

charts/restart    �b�/       m]P	  �����A���q*

charts/episodic_returnd�B�Ef�3       � V	  �����A���q*#
!
charts/windowed_avg_return� B/���/       m]P	  �����A���q*

charts/episodic_lengthU�D��-       <A��	  @����A���q*

charts/learning_rateQ�U8�K�U*       ����	  @����A���q*

losses/value_loss���=���]+       ��K	  @����A���q*

losses/policy_loss��t��'       ��F	  @����A���q*

losses/entropyp���6J6-       <A��	  @����A���q*

losses/old_approx_kl8�*;P���)       7�_ 	  @����A���q*

losses/approx_kl��;$��(       �pJ	  @����A���q*

losses/clipfrac�2�=�Ũ2       $V�	  @����A���q*"
 
losses/discounted_returnsƑ@t,`#       ��wC	  @����A���q*


charts/SPS�2Db�ip'       ��F	  @����A���q*

charts/restart    x��./       m]P	  ����A���q*

charts/episodic_return$�B�:.3       � V	  ����A���q*#
!
charts/windowed_avg_return.BBr�@/       m]P	  ����A���q*

charts/episodic_length�j�D�*�-       <A��	  ����A���q*

charts/learning_rate��U8�[*       ����	  ����A���q*

losses/value_lossĜ=�H1+       ��K	  ����A���q*

losses/policy_losso�<���M'       ��F	  ����A���q*

losses/entropy�y~���.�-       <A��	  ����A���q*

losses/old_approx_kl=�:����)       7�_ 	  ����A���q*

losses/approx_kl��:���(       �pJ	  ����A���q*

losses/clipfrac�Z�=J��f2       $V�	  ����A���q*"
 
losses/discounted_returns�@�6#       ��wC	  ����A���q*


charts/SPSm�2D׊	�'       ��F	  ����A���q*

charts/restart    �ɐA/       m]P	  �*���A���q*

charts/episodic_return�eB���O3       � V	  �*���A���q*#
!
charts/windowed_avg_return��B�/�/       m]P	  �*���A���q*

charts/episodic_length���D���-       <A��	  @+���A���q*

charts/learning_rateJU8OP�*       ����	  @+���A���q*

losses/value_loss�y�=��+       ��K	  @+���A���q*

losses/policy_lossV8���>'       ��F	  @+���A���q*

losses/entropyVZ��-��H-       <A��	  @+���A���q*

losses/old_approx_kl`�S;^��)       7�_ 	  @+���A���q*

losses/approx_klʙ";S���(       �pJ	  @+���A���q*

losses/clipfrac �M=H^�`2       $V�	  @+���A���q*"
 
losses/discounted_returns�$@�<W#       ��wC	  @+���A���q*


charts/SPS-�2D:Fq'       ��F	  @+���A���q*

charts/restart    7%�/       m]P	  �B���A���r*

charts/episodic_return��!B}��3       � V	  �B���A���r*#
!
charts/windowed_avg_returnlCB�OZ/       m]P	  �B���A���r*

charts/episodic_length.��D��%-       <A��	   C���A���r*

charts/learning_rate]U8�t�*       ����	   C���A���r*

losses/value_lossgl�=�f
�+       ��K	   C���A���r*

losses/policy_loss��\�M�R'       ��F	   C���A���r*

losses/entropy�����-       <A��	   C���A���r*

losses/old_approx_kl#�G;���)       7�_ 	   C���A���r*

losses/approx_klWl;jģ�(       �pJ	   C���A���r*

losses/clipfracUQ=���f2       $V�	   C���A���r*"
 
losses/discounted_returns��@I�c&#       ��wC	   C���A���r*


charts/SPS^�2D�e
c'       ��F	   C���A���r*

charts/restart    8��d/       m]P	  �X���A���r*

charts/episodic_return׏$B�-�D3       � V	  �X���A���r*#
!
charts/windowed_avg_return��B��j�/       m]P	  �X���A���r*

charts/episodic_length�5�D$��o-       <A��	  �Y���A���r*

charts/learning_rate��T8{@�*       ����	  �Y���A���r*

losses/value_loss��=��6�+       ��K	  �Y���A���r*

losses/policy_loss%n��Eh1'       ��F	  �Y���A���r*

losses/entropy����J�y-       <A��	  �Y���A���r*

losses/old_approx_kl��3;��F)       7�_ 	  �Y���A���r*

losses/approx_kl�/�:DB��(       �pJ	  �Y���A���r*

losses/clipfracVݜ=��2       $V�	  �Y���A���r*"
 
losses/discounted_returns�z�?��#       ��wC	  �Y���A���r*


charts/SPSɦ2D���'       ��F	  �Y���A���r*

charts/restart    �OI5/       m]P	   o���A���r*

charts/episodic_returnѹ,B�ߨQ3       � V	   o���A���r*#
!
charts/windowed_avg_return=�B/W/       m]P	   o���A���r*

charts/episodic_length:��D�VJ-       <A��	  �o���A���r*

charts/learning_rate�T8��*       ����	  �o���A���r*

losses/value_loss�@v=j �+       ��K	  �o���A���r*

losses/policy_loss�O��Kx�'       ��F	  �o���A���r*

losses/entropy�R���3�-       <A��	  �o���A���r*

losses/old_approx_klZk!;�"5�)       7�_ 	  �o���A���r*

losses/approx_kl@}�:]Y�(       �pJ	  �o���A���r*

losses/clipfrac @z=ڬ��2       $V�	  �o���A���r*"
 
losses/discounted_returns6�@-�#       ��wC	  �o���A���r*


charts/SPS��2D��'       ��F	  �o���A���r*

charts/restart    )}u/       m]P	  �����A���r*

charts/episodic_return�j%B LԾ3       � V	  �����A���r*#
!
charts/windowed_avg_return7�Bzw+/       m]P	  �����A���r*

charts/episodic_length�DA-       <A��	  @����A���r*

charts/learning_ratem_T8�t*       ����	  @����A���r*

losses/value_lossڅz=��:+       ��K	  @����A���r*

losses/policy_loss�M8�+#��'       ��F	  @����A���r*

losses/entropy��r�B#ϛ-       <A��	  @����A���r*

losses/old_approx_kl)��:��)       7�_ 	  @����A���r*

losses/approx_kl@O�:d��(       �pJ	  @����A���r*

losses/clipfrac�jK=���2       $V�	  @����A���r*"
 
losses/discounted_returns���?-L�f#       ��wC	  @����A���r*


charts/SPSD;3D�M��'       ��F	  @����A���r*

charts/restart    /��/       m]P	  �����A���r*

charts/episodic_return��)B{&_3       � V	  �����A���r*#
!
charts/windowed_avg_return��Be�k�/       m]P	  �����A���r*

charts/episodic_length�gEŠm�-       <A��	  @����A���r*

charts/learning_rate�$T8�ӣE*       ����	  @����A���r*

losses/value_loss'�=��L�+       ��K	  @����A���r*

losses/policy_loss��|�jfPF'       ��F	  @����A���r*

losses/entropy���o56-       <A��	  @����A���r*

losses/old_approx_kl��:Į��)       7�_ 	  @����A���r*

losses/approx_kld;��(       �pJ	  @����A���r*

losses/clipfrac�*M=�7��2       $V�	  @����A���r*"
 
losses/discounted_returns� @>�'�#       ��wC	  @����A���r*


charts/SPSڳ3D��'       ��F	  @����A���r*

charts/restart    ��/       m]P	   ����A���r*

charts/episodic_returnҪB��i3       � V	   ����A���r*#
!
charts/windowed_avg_return�ZBzxT�/       m]P	   ����A���r*

charts/episodic_length���DL��)-       <A��	  �����A���r*

charts/learning_rate�S8AEm*       ����	  �����A���r*

losses/value_lossј�=��h�+       ��K	  �����A���r*

losses/policy_lossR���w��h'       ��F	  �����A���r*

losses/entropy�����#�e-       <A��	  �����A���r*

losses/old_approx_kl���:�3c)       7�_ 	  �����A���r*

losses/approx_kl���:dq�(       �pJ	  �����A���r*

losses/clipfrac�z[=[t+�2       $V�	  �����A���r*"
 
losses/discounted_returnsy�@?��#       ��wC	  �����A���r*


charts/SPS��3DVт�'       ��F	  �����A���r*

charts/restart    (��'/       m]P	   ō��A���r*

charts/episodic_returnj�BJ��f3       � V	   ō��A���r*#
!
charts/windowed_avg_returnfB�a/�/       m]P	   ō��A���r*

charts/episodic_length̀�Dۂ��-       <A��	  �ō��A���r*

charts/learning_ratey�S8,���*       ����	  �ō��A���r*

losses/value_loss��=	��+       ��K	  �ō��A���r*

losses/policy_lossA��B
M'       ��F	  �ō��A���r*

losses/entropym��v
~;-       <A��	  �ō��A���r*

losses/old_approx_kl�	�9�)ʃ)       7�_ 	  �ō��A���r*

losses/approx_kl�s�:u��a(       �pJ	  �ō��A���r*

losses/clipfrac �a=ə,2       $V�	  �ō��A���r*"
 
losses/discounted_returns�@I�#       ��wC	  �ō��A���r*


charts/SPS#:4DĹ��'       ��F	  �ō��A���r*

charts/restart    �>�q/       m]P	   ٍ��A���r*

charts/episodic_return.F7Bj���3       � V	   ٍ��A���r*#
!
charts/windowed_avg_return�MBEτ�/       m]P	   ٍ��A���r*

charts/episodic_length�k�Dz.��-       <A��	  �ٍ��A���r*

charts/learning_rate�tS8��*       ����	  �ٍ��A���r*

losses/value_loss-Z=��
+       ��K	  �ٍ��A���r*

losses/policy_loss�_������'       ��F	  �ٍ��A���r*

losses/entropy�E��~�K-       <A��	  �ٍ��A���r*

losses/old_approx_kl~�8;w��])       7�_ 	  �ٍ��A���r*

losses/approx_kl�D�: �±(       �pJ	  �ٍ��A���r*

losses/clipfrac�3=��m�2       $V�	  �ٍ��A���r*"
 
losses/discounted_returnsN��?��0#       ��wC	  �ٍ��A���r*


charts/SPS��4D���'       ��F	  �ٍ��A���r*

charts/restart    V�S9/       m]P	   ����A���r*

charts/episodic_return�o0Bn�6�3       � V	   ����A���r*#
!
charts/windowed_avg_returnl�Bz.,R/       m]P	   ����A���r*

charts/episodic_length��D��͎-       <A��	  �����A���r*

charts/learning_rate.:S8���{*       ����	  �����A���r*

losses/value_loss%�v=e�l�+       ��K	  �����A���r*

losses/policy_loss;틻��s�'       ��F	  �����A���r*

losses/entropy��s�G/��-       <A��	  �����A���r*

losses/old_approx_kl�tq;��/�)       7�_ 	  �����A���r*

losses/approx_klD�;��g (       �pJ	  �����A���r*

losses/clipfrac�Ja=.S��2       $V�	  �����A���r*"
 
losses/discounted_returns/|�?Ơ�E#       ��wC	  �����A���r*


charts/SPS��4D�tn'       ��F	  �����A���r*

charts/restart    k��/       m]P	  @���A���r*

charts/episodic_return��B�0�3       � V	  @���A���r*#
!
charts/windowed_avg_returnC�Bm�)�/       m]P	  @���A���r*

charts/episodic_length���D�3�-       <A��	  ����A���r*

charts/learning_rate��R87=�*       ����	  ����A���r*

losses/value_loss5�=~x7�+       ��K	  ����A���r*

losses/policy_loss�����d'       ��F	  ����A���r*

losses/entropy�Vz��� �-       <A��	  ����A���r*

losses/old_approx_kl��:b=�
)       7�_ 	  ����A���r*

losses/approx_klB��:	݈�(       �pJ	  ����A���r*

losses/clipfracU:=�K|2       $V�	  ����A���r*"
 
losses/discounted_returnsfa@�B�#       ��wC	  ����A���r*


charts/SPS�'5D�B8K'       ��F	  ����A���r*

charts/restart    ��/       m]P	   ���A���r*

charts/episodic_returnُB��+�3       � V	   ���A���r*#
!
charts/windowed_avg_return�Br�ۚ/       m]P	   ���A���r*

charts/episodic_lengthL(�D���-       <A��	  ����A���r*

charts/learning_rate��R8��*       ����	  ����A���r*

losses/value_loss��=�Y��+       ��K	  ����A���r*

losses/policy_loss����
�T'       ��F	  ����A���r*

losses/entropy��m�]���-       <A��	  ����A���r*

losses/old_approx_kl	�:�K�)       7�_ 	  ����A���r*

losses/approx_kl�>;*d(       �pJ	  ����A���r*

losses/clipfrac @U=_�}�2       $V�	  ����A���r*"
 
losses/discounted_returns��@]�I#       ��wC	  ����A���r*


charts/SPS�T5D�?'       ��F	  ����A���r*

charts/restart    \�6�/       m]P	   3���A���r*

charts/episodic_return�Bz��|3       � V	   3���A���r*#
!
charts/windowed_avg_returnoB\
�/       m]P	   3���A���r*

charts/episodic_lengthZw�DT{��-       <A��	  �3���A���r*

charts/learning_rate:�R8|!�r*       ����	  �3���A���r*

losses/value_loss{�o=U���+       ��K	  �3���A���r*

losses/policy_lossTǱ��ln+'       ��F	  �3���A���r*

losses/entropy6_i����9-       <A��	  �3���A���r*

losses/old_approx_klJ;@;���h)       7�_ 	  �3���A���r*

losses/approx_klb;�K(       �pJ	  �3���A���r*

losses/clipfrac��c=Y��2       $V�	  �3���A���r*"
 
losses/discounted_returns���?�/x#       ��wC	  �3���A���r*


charts/SPSs�4D��F�'       ��F	  �3���A���r*

charts/restart    ��O�/       m]P	  @L���A���r*

charts/episodic_return1kBWPv3       � V	  @L���A���r*#
!
charts/windowed_avg_return� B$�Y/       m]P	  @L���A���r*

charts/episodic_length�f�D�z�P-       <A��	   M���A���r*

charts/learning_rate�OR80��S*       ����	   M���A���r*

losses/value_loss6��=��e�+       ��K	   M���A���r*

losses/policy_loss��\���x>'       ��F	   M���A���r*

losses/entropyy5h����{-       <A��	   M���A���r*

losses/old_approx_klY��:~6]w)       7�_ 	   M���A���r*

losses/approx_kl�j;3BW�(       �pJ	   M���A���r*

losses/clipfrac��I=*׬\2       $V�	   M���A���r*"
 
losses/discounted_returnsX��?��NW#       ��wC	   M���A���r*


charts/SPS�x4Dx�'       ��F	   M���A���r*

charts/restart    �t/       m]P	  �e���A���r*

charts/episodic_return��Br*�P3       � V	  �e���A���r*#
!
charts/windowed_avg_return��B>~/       m]P	  �e���A���r*

charts/episodic_length6��D���-       <A��	  �f���A���r*

charts/learning_rate�R8�]�*       ����	  �f���A���r*

losses/value_lossc��=��+       ��K	  �f���A���r*

losses/policy_loss驦�t=�'       ��F	  �f���A���r*

losses/entropy�Pi�NF�-       <A��	  �f���A���r*

losses/old_approx_kl�5;n�~")       7�_ 	  �f���A���r*

losses/approx_kl6;�|��(       �pJ	  �f���A���r*

losses/clipfrac  2=<�D�2       $V�	  �f���A���r*"
 
losses/discounted_returns1��?X*�4#       ��wC	  �f���A���r*


charts/SPS��3D;��'       ��F	  �f���A���r*

charts/restart    ��/       m]P	   ���A���r*

charts/episodic_returnZ"B�e�3       � V	   ���A���r*#
!
charts/windowed_avg_return�B�r,/       m]P	   ���A���r*

charts/episodic_length�k�D�:.-       <A��	  ����A���r*

charts/learning_rateJ�Q8]�T*       ����	  ����A���r*

losses/value_loss��=�D+       ��K	  ����A���r*

losses/policy_loss^��[�!V'       ��F	  ����A���r*

losses/entropy�#k�Q��*-       <A��	  ����A���r*

losses/old_approx_klz��:���)       7�_ 	  ����A���r*

losses/approx_kl,��:F5�(       �pJ	  ����A���r*

losses/clipfrac�jG=�S��2       $V�	  ����A���r*"
 
losses/discounted_returns?��?��O)#       ��wC	  ����A���r*


charts/SPS1g3DO��'       ��F	  ����A���r*

charts/restart    k���/       m]P	  �����A���r*

charts/episodic_return9BɆ�43       � V	  �����A���r*#
!
charts/windowed_avg_return�+BM#]�/       m]P	  �����A���r*

charts/episodic_lengthso�DJ�-       <A��	  �����A���r*

charts/learning_rate��Q8��u*       ����	  �����A���r*

losses/value_loss�3U=��8@+       ��K	  �����A���r*

losses/policy_lossl�m���2�'       ��F	  �����A���r*

losses/entropy��k�hҹ�-       <A��	  �����A���r*

losses/old_approx_kl�;����)       7�_ 	  �����A���r*

losses/approx_kl��
;�l%(       �pJ	  �����A���r*

losses/clipfrac�b�=� �2       $V�	  �����A���r*"
 
losses/discounted_returns�x�?���#       ��wC	  �����A���r*


charts/SPS��2D�|'       ��F	  �����A���r*

charts/restart    i��/       m]P	  �����A���r*

charts/episodic_return��B��L+3       � V	  �����A���r*#
!
charts/windowed_avg_return�ABZ�0/       m]P	  �����A���r*

charts/episodic_length��DÇT�-       <A��	  @����A���r*

charts/learning_rate�dQ8�n {*       ����	  @����A���r*

losses/value_lossnr=�Vz�+       ��K	  @����A���r*

losses/policy_loss�}��E׸�'       ��F	  @����A���r*

losses/entropy�o�`n�-       <A��	  @����A���r*

losses/old_approx_kl���:�vcC)       7�_ 	  @����A���r*

losses/approx_kl���:�CdN(       �pJ	  @����A���r*

losses/clipfracUE2=| ��2       $V�	  @����A���r*"
 
losses/discounted_returns��?�i�,#       ��wC	  @����A���r*


charts/SPS�%2D��,�'       ��F	  @����A���r*

charts/restart    �c�7