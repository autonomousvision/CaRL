�      �YW�	  ��Z��A*�
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
    "load_file" : "/mnt/bernhard/code/ad_planning/2_carla/results/AC_PPO_025_04/model_latest_000003258.pth",
    "debug" : false,
    "debug_type" : "render",
    "logdir" : "/mnt/bernhard/code/ad_planning/2_carla/results",
    "global_step" : 213581824,
    "max_training_score" : 39.693675994873047,
    "best_iteration" : 3081,
    "latest_iteration" : 3258,
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
textʩ�}'       ��F	   �Z��A���e*

charts/restart  �?\�ˈ/       m]P	   �Z��A���e*

charts/episodic_return�@�
�3       � V	   �Z��A���e*#
!
charts/windowed_avg_return�@��/       m]P	   �Z��A���e*

charts/episodic_length�IC-�6-       <A��	  ��Z��A���e*

charts/learning_rate���8�T�*       ����	  ��Z��A���e*

losses/value_loss���=���+       ��K	  ��Z��A���e*

losses/policy_loss�z��t�c'       ��F	  ��Z��A���e*

losses/entropy����&��-       <A��	  ��Z��A���e*

losses/old_approx_klKrz;ӷZ)       7�_ 	  ��Z��A���e*

losses/approx_kl,(%;��(       �pJ	  ��Z��A���e*

losses/clipfrac h�=�C�2       $V�	  ��Z��A���e*"
 
losses/discounted_returns�e@$�e#       ��wC	  ��Z��A���e*


charts/SPS�]D�<X'       ��F	  ��Z��A���e*

charts/restart    ,RB�/       m]P	  ��Z��A���e*

charts/episodic_return��AL�
3       � V	  ��Z��A���e*#
!
charts/windowed_avg_returnnpA�Q�/       m]P	  ��Z��A���e*

charts/episodic_length��-D|�-       <A��	  ��Z��A���e*

charts/learning_rate:ܖ8N_zl*       ����	  ��Z��A���e*

losses/value_loss�ˎ=��%c+       ��K	  ��Z��A���e*

losses/policy_loss�����?�'       ��F	  ��Z��A���e*

losses/entropy����<SYv-       <A��	  ��Z��A���e*

losses/old_approx_kl�;��)       7�_ 	  ��Z��A���e*

losses/approx_klv�;9���(       �pJ	  ��Z��A���e*

losses/clipfrac�:|=��N2       $V�	  ��Z��A���e*"
 
losses/discounted_returns��@��o�#       ��wC	  ��Z��A���e*


charts/SPSIbWDAH)�'       ��F	  ��Z��A���e*

charts/restart    &�Z�/       m]P	  @�Z��A���e*

charts/episodic_return��B�3       � V	  @�Z��A���e*#
!
charts/windowed_avg_returnaT�A7�/       m]P	  @�Z��A���e*

charts/episodic_length%�D8�y�-       <A��	   �Z��A���e*

charts/learning_rate澖8䀼�*       ����	   �Z��A���e*

losses/value_lossQ�="[�V+       ��K	   �Z��A���e*

losses/policy_loss>�������'       ��F	   �Z��A���e*

losses/entropy�1�����-       <A��	   �Z��A���e*

losses/old_approx_kly�};}�=�)       7�_ 	   �Z��A���e*

losses/approx_kl70;�ؿ�(       �pJ	   �Z��A���e*

losses/clipfracUm�=4ְ!2       $V�	   �Z��A���e*"
 
losses/discounted_returns��@%���#       ��wC	   �Z��A���e*


charts/SPSςND:��'       ��F	   �Z��A���e*

charts/restart    �%�/       m]P	  ��Z��A���e*

charts/episodic_return.*B E?R3       � V	  ��Z��A���e*#
!
charts/windowed_avg_return`�A�
i�/       m]P	  ��Z��A���e*

charts/episodic_lengthX�D��YN-       <A��	  @�Z��A���e*

charts/learning_rate���8�LDX*       ����	  @�Z��A���e*

losses/value_loss)G�=s�R�+       ��K	  @�Z��A���e*

losses/policy_loss�������'       ��F	  @�Z��A���e*

losses/entropy0���$F�-       <A��	  @�Z��A���e*

losses/old_approx_kl�7�;*���)       7�_ 	  @�Z��A���e*

losses/approx_klg�W;0J(       �pJ	  @�Z��A���e*

losses/clipfrac �=8O�O2       $V�	  @�Z��A���e*"
 
losses/discounted_returns�\@PD�S#       ��wC	  @�Z��A���e*


charts/SPS߀ODq��z'       ��F	  @�Z��A���e*

charts/restart    ��_/       m]P	  �[��A���f*

charts/episodic_return��6Bd�Z�3       � V	  �[��A���f*#
!
charts/windowed_avg_return.��A��/       m]P	  �[��A���f*

charts/episodic_lengthP��D�k�L-       <A��	  �[��A���f*

charts/learning_rate@��84���*       ����	  �[��A���f*

losses/value_loss7��=��`�+       ��K	  �[��A���f*

losses/policy_loss�_����\'       ��F	  �[��A���f*

losses/entropyRd������-       <A��	  �[��A���f*

losses/old_approx_klM��;kb�)       7�_ 	  �[��A���f*

losses/approx_kl�&;��,�(       �pJ	  �[��A���f*

losses/clipfracU�=��<I2       $V�	  �[��A���f*"
 
losses/discounted_returnsb�@�F��#       ��wC	  �[��A���f*


charts/SPSŖND[iܘ'       ��F	  �[��A���f*

charts/restart    ���Q/       m]P	  �#[��A���f*

charts/episodic_return��9BK��3       � V	  �#[��A���f*#
!
charts/windowed_avg_returnB�?|{/       m]P	  �#[��A���f*

charts/episodic_lengthO��D��-       <A��	  �$[��A���f*

charts/learning_rate�f�8���*       ����	  �$[��A���f*

losses/value_loss_��=���+       ��K	  �$[��A���f*

losses/policy_loss�}���/��'       ��F	  �$[��A���f*

losses/entropy4����Ȓn-       <A��	  �$[��A���f*

losses/old_approx_kl7;�:�.�)       7�_ 	  �$[��A���f*

losses/approx_kl4�;m|�,(       �pJ	  �$[��A���f*

losses/clipfracV��=�I�2       $V�	  �$[��A���f*"
 
losses/discounted_returns��@chg�#       ��wC	  �$[��A���f*


charts/SPS6UND�[��'       ��F	  �$[��A���f*

charts/restart    T��/       m]P	  �9[��A���f*

charts/episodic_return� Bqw3       � V	  �9[��A���f*#
!
charts/windowed_avg_return�
BCo�/       m]P	  �9[��A���f*

charts/episodic_lengthU��D�-       <A��	  @:[��A���f*

charts/learning_rate�I�86c`�*       ����	  @:[��A���f*

losses/value_lossް~=٧{�+       ��K	  @:[��A���f*

losses/policy_loss�^>�{�}Z'       ��F	  @:[��A���f*

losses/entropy"ė��l-       <A��	  @:[��A���f*

losses/old_approx_kl��p;��b�)       7�_ 	  @:[��A���f*

losses/approx_kl{4�;��<(       �pJ	  @:[��A���f*

losses/clipfrac�b�=#6��2       $V�	  @:[��A���f*"
 
losses/discounted_returns�@2�z�#       ��wC	  @:[��A���f*


charts/SPSްKD�ט'       ��F	  @:[��A���f*

charts/restart    ��/       m]P	   M[��A���f*

charts/episodic_return.�"BA��F3       � V	   M[��A���f*#
!
charts/windowed_avg_return��B�~N/       m]P	   M[��A���f*

charts/episodic_length���D^�Lc-       <A��	  �M[��A���f*

charts/learning_rateF,�8�Na *       ����	  �M[��A���f*

losses/value_lossax�=���+       ��K	  �M[��A���f*

losses/policy_lossEYr�C�'       ��F	  �M[��A���f*

losses/entropy����,�0-       <A��	  �M[��A���f*

losses/old_approx_kl��`;}z&�)       7�_ 	  �M[��A���f*

losses/approx_kl�*&;B��(       �pJ	  �M[��A���f*

losses/clipfrac�R�=h�]@2       $V�	  �M[��A���f*"
 
losses/discounted_returns��@h�Ҫ#       ��wC	  �M[��A���f*


charts/SPS�|LD�'w'       ��F	  �M[��A���f*

charts/restart    *L�/       m]P	  �c[��A���f*

charts/episodic_return�!B͂��3       � V	  �c[��A���f*#
!
charts/windowed_avg_returnT�BD;��/       m]P	  �c[��A���f*

charts/episodic_lengthӒ�D�,�-       <A��	  �d[��A���f*

charts/learning_rate��8��_A*       ����	  �d[��A���f*

losses/value_lossv}�=����+       ��K	  �d[��A���f*

losses/policy_loss�ݝ�;�o'       ��F	  �d[��A���f*

losses/entropy�m��m<�I-       <A��	  �d[��A���f*

losses/old_approx_klU.k;?��)       7�_ 	  �d[��A���f*

losses/approx_kl�;�x�j(       �pJ	  �d[��A���f*

losses/clipfrac ��=/�2       $V�	  �d[��A���f*"
 
losses/discounted_returnsMw	@�bz#       ��wC	  �d[��A���f*


charts/SPS�\ID{ ��'       ��F	  �d[��A���f*

charts/restart    �q��/       m]P	  �x[��A���f*

charts/episodic_return�BQ�س3       � V	  �x[��A���f*#
!
charts/windowed_avg_return��B�ѥ]/       m]P	  �x[��A���f*

charts/episodic_length֔�DL�=�-       <A��	  @y[��A���f*

charts/learning_rate��8g�
�*       ����	  @y[��A���f*

losses/value_loss�}�=�Wv+       ��K	  @y[��A���f*

losses/policy_loss��z�� '       ��F	  @y[��A���f*

losses/entropyZ󠿥WjD-       <A��	  @y[��A���f*

losses/old_approx_kl���;���)       7�_ 	  @y[��A���f*

losses/approx_klQO;XIU�(       �pJ	  @y[��A���f*

losses/clipfrac ر=�\σ2       $V�	  @y[��A���f*"
 
losses/discounted_returns��@��
�#       ��wC	  @y[��A���f*


charts/SPS�IDx�7�'       ��F	  @y[��A���f*

charts/restart    ���/       m]P	  ��[��A���f*

charts/episodic_return|=BR���3       � V	  ��[��A���f*#
!
charts/windowed_avg_returnO�B��t�/       m]P	  ��[��A���f*

charts/episodic_length���D�YR�-       <A��	  @�[��A���f*

charts/learning_rateNԕ8|�A *       ����	  @�[��A���f*

losses/value_loss��=�~+       ��K	  @�[��A���f*

losses/policy_lossޘ��q.� '       ��F	  @�[��A���f*

losses/entropyt���4�o]-       <A��	  @�[��A���f*

losses/old_approx_kl";��%�)       7�_ 	  @�[��A���f*

losses/approx_klM�:��
e(       �pJ	  @�[��A���f*

losses/clipfrac �m=,bE�2       $V�	  @�[��A���f*"
 
losses/discounted_returnsmn@�Kl#       ��wC	  @�[��A���f*


charts/SPS�@HDC��'       ��F	  @�[��A���f*

charts/restart    ����/       m]P	  @�[��A���f*

charts/episodic_return(Bw�3       � V	  @�[��A���f*#
!
charts/windowed_avg_returnʿB���/       m]P	  @�[��A���f*

charts/episodic_length���D��<H-       <A��	   �[��A���f*

charts/learning_rate���8`*       ����	   �[��A���f*

losses/value_loss���=��=+       ��K	   �[��A���f*

losses/policy_loss$����ך'       ��F	   �[��A���f*

losses/entropyS,��ڳ*�-       <A��	   �[��A���f*

losses/old_approx_kl9+%;v-��)       7�_ 	   �[��A���f*

losses/approx_klO#,;�g�(       �pJ	   �[��A���f*

losses/clipfracV��=�<��2       $V�	   �[��A���f*"
 
losses/discounted_returnst@����#       ��wC	   �[��A���f*


charts/SPS��FD>a��'       ��F	   �[��A���f*

charts/restart    ��*/       m]P	  @�[��A���f*

charts/episodic_return��B!�Wf3       � V	  @�[��A���f*#
!
charts/windowed_avg_return:�B9)Z�/       m]P	  @�[��A���f*

charts/episodic_length���D�a�-       <A��	   �[��A���f*

charts/learning_rate���8�l�*       ����	   �[��A���f*

losses/value_loss�۶=�e��+       ��K	   �[��A���f*

losses/policy_loss��l�v�	'       ��F	   �[��A���f*

losses/entropy?�������-       <A��	   �[��A���f*

losses/old_approx_klːd;��Y)       7�_ 	   �[��A���f*

losses/approx_klj�!;.N<�(       �pJ	   �[��A���f*

losses/clipfracV�=�s�!2       $V�	   �[��A���f*"
 
losses/discounted_returnsfr@�+L#       ��wC	   �[��A���f*


charts/SPS�zED4.g'       ��F	   �[��A���f*

charts/restart    �*/       m]P	  ��[��A���f*

charts/episodic_return��	B�g��3       � V	  ��[��A���f*#
!
charts/windowed_avg_return�	B-�\/       m]P	  ��[��A���f*

charts/episodic_length��DagP--       <A��	   �[��A���f*

charts/learning_rateT|�8�'|w*       ����	   �[��A���f*

losses/value_loss�\z=f��+       ��K	   �[��A���f*

losses/policy_lossFv��S�=�'       ��F	   �[��A���f*

losses/entropyL���R�9-       <A��	   �[��A���f*

losses/old_approx_klHy;C]'�)       7�_ 	   �[��A���f*

losses/approx_kl!;;�3 �(       �pJ	   �[��A���f*

losses/clipfracVEz=��,2       $V�	   �[��A���f*"
 
losses/discounted_returnsf/@�J6�#       ��wC	   �[��A���f*


charts/SPSj�ED��%'       ��F	   �[��A���f*

charts/restart     k��/       m]P	  @�[��A���f*

charts/episodic_return��B~�)(3       � V	  @�[��A���f*#
!
charts/windowed_avg_return[Bb�A�/       m]P	  @�[��A���f*

charts/episodic_length ��D-��-       <A��	  ��[��A���f*

charts/learning_rate _�8m�f�*       ����	  ��[��A���f*

losses/value_loss��a=B +       ��K	  ��[��A���f*

losses/policy_loss�R���P��'       ��F	  ��[��A���f*

losses/entropyQ��l�-�-       <A��	  ��[��A���f*

losses/old_approx_kl�i�:���)       7�_ 	  ��[��A���f*

losses/approx_klt�;�R!(       �pJ	  ��[��A���f*

losses/clipfracV�~=�\�2       $V�	  ��[��A���f*"
 
losses/discounted_returns�@ݺ�s#       ��wC	  ��[��A���f*


charts/SPS�FED�z a'       ��F	  ��[��A���f*

charts/restart    G��]/       m]P	  @�[��A���f*

charts/episodic_returnL�B�uq 3       � V	  @�[��A���f*#
!
charts/windowed_avg_return��BiJͨ/       m]P	  @�[��A���f*

charts/episodic_length�0�D�n��-       <A��	   �[��A���f*

charts/learning_rate�A�8ܟ��*       ����	   �[��A���f*

losses/value_lossͩ\=�"X�+       ��K	   �[��A���f*

losses/policy_loss��\��n�'       ��F	   �[��A���f*

losses/entropy���BP�-       <A��	   �[��A���f*

losses/old_approx_klQŴ���0`)       7�_ 	   �[��A���f*

losses/approx_kl��%;^>|(       �pJ	   �[��A���f*

losses/clipfrac  �=&,�2       $V�	   �[��A���f*"
 
losses/discounted_returns�
�?��{#       ��wC	   �[��A���f*


charts/SPS��ED��`'       ��F	   �[��A���f*

charts/restart    �w6�/       m]P	  �\��A���f*

charts/episodic_return0!B��\�3       � V	  �\��A���f*#
!
charts/windowed_avg_return��B��c/       m]P	  �\��A���f*

charts/episodic_length"��Dz+
-       <A��	   \��A���f*

charts/learning_rate[$�8ʭi�*       ����	   \��A���f*

losses/value_lossK
h=i(/�+       ��K	   \��A���f*

losses/policy_loss~t��E$��'       ��F	   \��A���f*

losses/entropyP��N�=i-       <A��	   \��A���f*

losses/old_approx_klrkn:��g)       7�_ 	   \��A���f*

losses/approx_kl�g3;=�-(       �pJ	   \��A���f*

losses/clipfrac �=��;R2       $V�	   \��A���f*"
 
losses/discounted_returns0( @�&�#       ��wC	   \��A���f*


charts/SPS�MDD�Qd�'       ��F	   \��A���f*

charts/restart    �B��/       m]P	  �'\��A���f*

charts/episodic_return�*6B���3       � V	  �'\��A���f*#
!
charts/windowed_avg_returnQ�BF��/       m]P	  �'\��A���f*

charts/episodic_length��E5�mv-       <A��	  @(\��A���f*

charts/learning_rate	�8���,*       ����	  @(\��A���f*

losses/value_loss�f�=��a+       ��K	  @(\��A���f*

losses/policy_loss4u��F�R�'       ��F	  @(\��A���f*

losses/entropy�K��N���-       <A��	  @(\��A���f*

losses/old_approx_kl���:k+)�)       7�_ 	  @(\��A���f*

losses/approx_klJ��:�B8(       �pJ	  @(\��A���f*

losses/clipfrac �k=wS_2       $V�	  @(\��A���f*"
 
losses/discounted_returns�@%��#       ��wC	  @(\��A���f*


charts/SPS�BD��B�'       ��F	  @(\��A���f*

charts/restart    �Fތ