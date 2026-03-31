'''
Config class that contains all the hyperparameters needed to build any model.
'''
import numpy as np


class GlobalConfig:
  '''
  Config class that contains all the hyperparameters needed to build any model.
  '''

  def __init__(self):
    self.frame_rate = 10.0  # Frames per second of the CARLA simulator
    self.original_frame_rate = 20.0  # Frames per second of the CARLA evaluation server. 20 comes from leaderboard
    self.time_interval = 1.0 / self.frame_rate  # ms per step in CARLA time.

    self.pixels_per_meter = 2  # 1 / pixels_per_meter = size of pixel in meters
    self.bev_semantics_width = 256  # Numer of pixels the bev_semantics is wide
    self.pixels_ev_to_bottom = 100  # Numer of pixels from the vehicle to the bottom.
    self.bev_semantics_height = 256  # Numer of pixels the bev_semantics is high
    # Distance of traffic lights considered relevant (in meters)
    self.light_radius = 15.0
    self.debug = False  # Whether to turn on debugging functions, like visualizations.
    self.logging_freq = 10  # Log every 10 th frame
    self.logger_region_of_interest = 30.0  # Meters around the car that will be logged.
    self.route_points = 10  # Number of route points to render in logger

    half_second = int(self.frame_rate * 0.5)
    self.history_idx = [-3 * half_second - 1, -2 * half_second - 1, -1 * half_second - 1, -0 * half_second - 1]

    self.num_route_points_rendered = 150  # Number of route points rendered into the BEV seg observation.
    self.use_history = False  # Whether to use the history in bev_observation
    # History index for bev_observation
    # Current observation is not included as it is rendered anyway
    self.history_idx_2 = [-3 * half_second - 1, -2 * half_second - 1, -1 * half_second - 1]

    # Color format BGR
    self.bev_classes_list = (
        (0, 0, 0),  # unlabeled
        (150, 150, 150),  # road
        (255, 255, 255),  # route
        (255, 255, 0),  # lane marking
        (0, 0, 255),  # vehicle
        (0, 255, 255),  # pedestrian
        (255, 255, 0),  # traffic light
        (160, 160, 0),  # stop sign
        (0, 255, 0),  # speed sign
    )

    # Extent of the ego vehicles bounding box
    self.ego_extent_x = 2.44619083404541
    self.ego_extent_y = 0.9183566570281982
    self.ego_extent_z = 0.7451388239860535

    # New bev observation parameters
    self.use_new_bev_obs = True  # Whether to use bev_observation.py instead of chauffeurnet.py
    self.route_width = 6  # Width of the rendered route in pixel.
    self.red_light_thickness = 3  # Width of the red light line
    self.use_extra_control_inputs = False  # Whether to use extra control inputs such as integral of past steering.
    # Rough avg steering angle in degree that the wheel can be set to
    # The steering angle for individual wheels is  +- 70° and +-48° for the other wheel respectively
    self.max_avg_steer_angle = 60.0
    self.condition_outside_junction = False  # Whether to render the route outside junctions.
    self.use_target_point = False  # Whether to input a target point in the measurements.
    self.use_value_measurements = True  # Whether to use value measurements (otherwise all are set to 0)
    self.num_value_measurements = 11  # Number of measurements exclusive to the value head.
    self.render_speed_lines = True  # Whether to render the speed lines for moving objects.
    self.use_positional_encoding = False  # Whether to add positional encoding to the image
    self.render_yellow_time = True  # Whether to indicate the remaining time to red in yellow light rendering
    self.render_shoulder = False  # Whether to render shoulder lanes as roads.
    self.use_shoulder_channel = True  # Whether to use an extra channel for shoulder lanes
    self.use_lidar = False  # Whether to use LiDAR instead of ground truth perception
    # Attributes of the LiDAR sensor. Values copied from CARLA leaderboard LiDAR for now.
    self.lidar_x = 0.0
    self.lidar_y = 0.0
    self.lidar_z = 2.5
    self.lidar_pitch = 0.0
    self.lidar_roll = 0.0
    self.lidar_yaw = 0.0
    self.lidar_points_per_second = 600000  # How many LiDAR hits one sweep generates
    self.lidar_channels = 64
    self.lidar_upper_fov = 10
    self.lidar_lower_fov = -30
    self.lidar_atmosphere_attenuation_rate = 0.004
    self.lidar_dropoff_general_rate = 0.45
    self.lidar_dropoff_intensity_limit = 0.8
    self.lidar_dropoff_zero_intensity = 0.4
    # Unit meters. Points from the LiDAR higher than this threshold are discarded.
    self.lidar_max_height = self.ego_extent_z * 2.0 + 1.0
    self.lidar_split_height = 0.2  # Unit meters. Points lower this value are dropped from the LiDAR image
    # Max and minimum LiDAR ranges used for voxelization
    self.lidar_min_x = -64
    self.lidar_max_x = 64
    self.lidar_min_y = -64
    self.lidar_max_y = 64
    self.lidar_min_z = -4
    self.lidar_max_z = 4
    # Max number of LiDAR points per pixel in voxelized LiDAR
    self.hist_max_per_pixel = 5
    self.lidar_last_frame_delay = 0.5  # Unit is second. The second LiDAR the agent sees will be from this long ago.

    self.use_occlusion_check = False  # Whether to only render bounding boxes that are unoccluded
    self.occlusion_origin_pos = (0.0, 0.0, 2.5)  # Point from which the occlusion check happens

    self.scale_bbox = True  # Whether to scale up the bounding boxes extends 1.0 for vehicles, 2.0 for ped. 0.8 max
    self.scale_factor_vehicle = 1.0
    self.scale_factor_walker = 2.0
    self.min_ext_bounding_box = 0.8
    self.scale_mask_col = 1.0  # Scaling factor for ego vehicle bounding box.
    self.map_folder = 'maps_2ppm_cv'  # Map folder for the preprocessed map data
    self.max_speed_actor = 33.33  # In m/s maximum speed we expect from other actors. = 120 km/h
    self.min_speed_actor = -2.67  # In m/s minimum speed we expect from other actors. = -10 km/h

    self.use_speed_limit = True  # Whether to input the current speed limit as measurement to the policy.

    # Roach reward hyperparameters. rr stands for roach reward
    self.reward_type = 'simple_reward'  # Reward function to be used during training. Options: roach, simple_reward
    self.use_exploration_suggest = False  # Whether to use the exploration loss from roach.
    self.rr_maximum_speed = 6.0  # Maximum speed in m/s encouraged by the roach reward function.
    self.vehicle_distance_threshold = 15  # Distance in meters within which vehicles are considered for the reward.
    self.max_vehicle_detection_number = 10  # Maximum number of vehicles considered for the roach reward.
    self.rr_vehicle_proximity_threshold = 9.5  # Threshold within which vehicles are considered hazard in the reward.
    # Distance in meters within which pedestrians are considered for the reward.
    self.pedestrian_distance_threshold = 15
    self.max_pedestrian_detection_number = 10  # Maximum number of pedestrians considered for the roach reward.
    # Threshold within which pedestrians are considered hazard in the reward.
    self.rr_pedestrian_proximity_threshold = 9.5
    self.rr_tl_offset = -0.8 * self.ego_extent_x  # Probably offset to be kept to the entrance of the intersection.
    self.rr_tl_dist_threshold = 18.0  # Distance at which traffic lights are considered for the speed reward.
    # Meters. If the agent is father away from the centerline (laterally) it counts as route deviation in the reward
    self.min_thresh_lat_dist = 2.0
    self.eval_time = 1200.0  # Seconds. After this time a timeout is triggered in the reward which counts as truncation.
    # Number of frames before the end of the episode where the exploration loss is applied.
    self.n_step_exploration = 100
    # If true rr_maximum_speed will be overwritten to the current speed limit affecting the ego vehicle.
    self.use_speed_limit_as_max_speed = False

    # Simple reward hyperparameters
    self.consider_tl = True  # If set to false traffic light infractions are turned off. Used in simple reward
    self.terminal_reward = 0.0  # Reward at the end of the episode
    self.terminal_hint = 1.0  # Reward at the end of the episode when colliding, the number will be subtracted.
    self.normalize_rewards = False  # Whether to use gymnasiums reward normalization.
    self.speeding_infraction = True  # Whether to terminate the route if the agent drives too fast.
    self.use_comfort_infraction = True  # Whether to apply a soft penalty if comfort limits are exceeded
    # These values are tuned for the nuPlan dataset
    self.max_abs_lon_jerk = 4.13  # m/s^3 Comfort limit for longitudinal jerk, nuPlan 4.13
    self.max_abs_mag_jerk = 8.37  # m/s^3 Comfort limit for jerk magnitude, nuPlan 8.37
    self.min_lon_accel = -4.05  # m/s^2 Comfort limit for longitudinal acceleration, nuPlan -4.05
    self.max_lon_accel = 2.40  # m/s^2 Comfort limit for longitudinal acceleration nuPlan 2.40
    self.max_abs_lat_accel = 4.89  # m/s^2 Comfort limit for lateral acceleration, nuPlan 4.89
    self.max_abs_yaw_rate = 0.95  # rad/s Comfort limit for angular velocity, nuPlan 0.95
    self.max_abs_yaw_accel = 1.93  # rad/s^2 Comfort limit for angular yaw acceleration, nuPlan 1.93
    self.comfort_penalty_ticks = 500  # Number of simulator steps that comfort penalties are applied for
    self.comfort_penalty_factor = 0.5  # Max comfort penalty if all comfort metrics are violated.
    self.use_vehicle_close_penalty = False  # Whether to use a penalty for being too close to the front vehicle.
    # Whether to give a penalty depending on vehicle speed when crashing or running red light
    self.use_termination_hint = True
    self.ego_forecast_time = 1.0  # Number of seconds that the ego agent is forecasted.
    self.ego_forecast_min_speed = 2.5  # In m/s. Minimum speed in the ego forecast.
    self.use_perc_progress = True  # Whether to multiply RC reward by percentage away from lane center.
    self.lane_distance_violation_threshold = 0.5  # Grace distance in m at which no lane perc penalty is applied
    # For how many frames the agent can deviate from the center. This is for making lane changes. We give 3s currently.
    self.lane_dist_penalty_time = int(3.0 * self.frame_rate)
    self.use_min_speed_infraction = False  # Whether to penalize the agent for driving slower than other agents on avg.
    self.use_leave_route_done = False  # Whether to terminate the route when leaving the precomputed path.
    self.use_outside_route_lanes = True  # Whether to terminate the route when invading opposing lanes or sidewalks.
    self.use_max_change_penalty = False  # Whether to apply a soft penalty when the action changes too fast.
    self.max_change = 0.25  # Maximum change in action allowed compared to last frame before a penalty is applied
    self.penalize_yellow_light = False  # Whether to penalize running a yellow light.
    self.use_off_road_term = True  # Whether to terminate when he agent drives off the drivable area
    self.off_road_term_perc = 0.95  # Percentage of agent overlap with off-road, that triggers the termination
    # Whether to use a different stop sign detector that prevents the policy from cheating by changing lanes.
    self.use_new_stop_sign_detector = True
    self.use_ttc = True  # Whether to use TTC in the reward.
    self.ttc_resolution = 2  # Interval of frame_rate time steps at which TTC is evaluated
    self.ttc_penalty_ticks = 500  # Number of simulator steps that a TTC penalty is applied for
    # m/s. Number adopted from nuPlan.
    # If the agent exceeds speed limit within this threshold he will get a penalty between [0,1] linearly
    self.max_overspeed_value_threshold = 2.23
    # Whether to only use RC als reward source in simple reward, else adds TTC, comfort and speed like in nuPlan
    self.use_single_reward = True
    self.use_rl_termination_hint = True  # Whether to include red light infraction for termination hints
    self.use_survival_reward = False  # Whether to add a constant reward every frame
    self.survival_reward_magnitude = 0.0001  # How large the survival reward is.

    # How often an action is repeated.
    self.action_repeat = 1

    # Action and observation space
    self.obs_num_measurements = 8  # Number of scalar measurements in observation.
    self.obs_num_channels = 10  # Number of channels in the bev observation.

    ####### Distribution parameters ############
    self.distribution = 'beta'  # Distribution used for the action space. Options beta, normal, beta_uni_mix
    # Minimum value for a, b of the beta distribution that the model can predict. Gets added to the softplus output.
    self.beta_min_a_b_value = 1.0

    self.normal_dist_init = ((0, -2), (0, -2))  # Initial bias parameters of the normal distribution
    self.normal_dist_action_dep_std = True  # Whether the std of the normal distribution is dependent of the input

    self.uniform_percentage_z = 0.03  # Mixing percentage of uniform distribution in beta_uni_mix

    # We have 2 actions, corresponding to left right steering and negative to positive acceleration.
    self.action_space_dim = 2
    self.action_space_min = -1.0  # Minimum value of the action space
    self.action_space_max = 1.0  # Maximum value of the action space
    # Number of frames at the beginning before learning starts, return brake
    self.start_delay_frames = int(2.0 / self.time_interval + 0.5)

    # PPO training hyperparameters
    self.exp_name = 'PPO_000'  # the name of this experiment
    self.gym_id = 'CARLAEnv-v0'  # the id of the gym environment
    self.learning_rate = 0.00025  # the learning rate of the optimizer
    self.seed = 1  # seed of the experiment
    self.total_timesteps = 10000000  # total time steps of the experiments
    self.torch_deterministic = True  # if toggled, `torch.backends.cudnn.deterministic=False`
    self.cuda = True  # if toggled, cuda will be enabled by default
    self.track = False  # if toggled, this experiment will be tracked with Weights and Biases
    self.wandb_project_name = 'ppo-roach'  # the wandb project name
    self.wandb_entity = None  # the entity (team) of wandb project
    self.capture_video = False  # whether to capture videos of the agent performances (check out `videos` folder)
    self.num_envs = 1  # the number of parallel game environments
    self.lr_schedule = 'linear'  # Which lr schedule to use. Options: (linear, kl, none, step, cosine, cosine_restart)
    self.gae = True  # Use GAE for advantage computation
    self.gamma = 0.99  # the discount factor gamma
    self.gae_lambda = 0.95  # the lambda for the general advantage estimation
    self.update_epochs = 3  # the K epochs to update the policy
    self.norm_adv = True  # Toggles advantages normalization
    self.clip_coef = 0.1  # the surrogate clipping coefficient
    self.clip_vloss = True  # Toggles whether to use a clipped loss for the value function, as per the paper.
    self.ent_coef = 0.01  # coefficient of the entropy
    self.vf_coef = 0.5  # coefficient of the value function
    self.max_grad_norm = 0.5  # the maximum norm for the gradient clipping
    self.target_kl = 0.015  # the target KL divergence threshold
    self.visualize = False  # if toggled, Game will render on screen
    self.logdir = ''  # The directory to log the data into.
    self.load_file = None  # model weights for initialization
    # Ports of the carla_gym wrapper to connect to. It requires to submit a port for every envs ports == --num_envs
    self.ports = (5555,)
    self.gpu_ids = (0,)  # Which GPUs to train on. Index 0 indicates GPU for rank 0 etc.
    self.compile_model = False  # Whether to use torch compile on the model.
    self.total_batch_size = 512  # The total amount of data collected at every step across all environments
    self.total_minibatch_size = 256  # The total minibatch sized used for training (across all environments)
    self.expl_coef = 0.05  # Weight / coefficient of the exploration loss
    self.lr_schedule_step = 8  # Number of time the KL divergence can be triggered before the lr reduces.
    self.current_learning_rate = self.learning_rate  # Learning rate at the latest iteration.
    self.kl_early_stop = 0  # Counter that reduces lr once it reaches lr_schedule_step
    self.adam_eps = 1e-5  # Adam optimizer parameter parameter. Standard PPO value is 1e-5
    # Did not observe a significant speedup with these so we turn them off for better numerical precision.
    self.allow_tf32 = False  # Whether to use tf32 format, which has better speed but lower numeric precision.
    self.benchmark = False  # Whether to use cudnn benchmarking
    self.matmul_precision = 'highest'  # Options highest float32, high tf32, medium bfloat16
    # Whether to collect data on cpu. This can be a bit faster, since it avoid CPU GPU ping pong,
    # at the cost of running the model on the CPU during data collection.
    self.cpu_collect = False
    # Robust policy optimization https://arxiv.org/abs/2212.07536
    self.use_rpo = False
    self.rpo_alpha = 0.5  # Size of the uniform random value that gets added to a, b
    self.use_green_wave = False  # If true in some routes all TL that the agent encounters are set to green.
    self.green_wave_prob = 0.05  # Probability of a route using green wave (if use_green_wave=True)
    # You should pick tiny networks for efficiency e.g. convnext_atto.d2_in1k,
    #  tinynet_e.in1k, vit_small_patch32_224
    self.image_encoder = 'roach_ln2'  # Which image cnn encoder to use. Either roach, roach_ln, or timm model name
    self.use_layer_norm = True  # Whether to use LayerNorm before ReLU in MLPs.
    # Applicable if use_layer_norm=True, whether to also apply layernorm to the policy head.
    # Can be useful to remove to allow the policy to predict large values (for a, b of Beta).
    self.use_layer_norm_policy_head = True
    self.features_dim = 256  # Dimension of features produced by the state encoder
    self.use_lstm = False  # Whether to use an LSTM after the feature encoder.
    self.num_lstm_layers = 1  # How many LSTM layers to use.
    # How many steps to accumulate the gradient for. Real minibatch size is total_minibatch_size * grad_acc_steps
    # Should be multiple and smaller than the number of steps per epoch.
    self.num_grad_acc_steps = 1
    self.use_bfloat16 = False  # Whether to do forward and backward passes in bfloat16 mixed precision.
    self.teacher_path = None  # Path to teacher model.pth Used for distillation loss
    self.forward_kl = False  # Whether to use forward KL or reverse KL for distillation
    self.start_kl_percentage = 1.0  # How much percent to do distillation vs RL at the start
    self.end_kl_percentage = 1.0  # How much percent to do distillation vs RL at the end. Interpolation is linear.

    self.render_green_tl = True  # Whether to render green traffic lights into the observation.
    self.lr_schedule_step_factor = 0.1  # Multiplier when doing a step decrease in learning rate
    self.lr_schedule_step_perc = (0.5, 0.75)  # Percentage of training run after which the lr is decayed
    self.weight_decay = 0.0  # Weight decay applied to optimizer. AdamW is used when > 0.0
    self.beta_1 = 0.9  # Beta 1 parameter of adam
    self.beta_2 = 0.999  # Beta 2 parameter of adam
    self.lr_schedule_cosine_restarts = (0.0, 0.25, 0.50, 0.75, 1.0)  # Percentage of training to do a restart
    # https://arxiv.org/abs/1911.00357
    self.use_dd_ppo_preempt = False  # Whether to use the dd-ppo preemption technique to early stop stragglers
    self.dd_ppo_preempt_threshold = 0.6  # Percentage of nodes that need to be finished before the rest is stopped.
    self.dd_ppo_min_perc = 0.25  # Minimum percentage of data points that need to be collected before preemption.
    self.num_envs_per_proc = 1  # Number of environments to use per process.
    # Percentage of training at which the model is evaluated
    self.eval_intervals = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    self.current_eval_interval_idx = 0  # Helper variable to remember which model to save next.
    self.use_temperature = False  # Whether the output distribution parameters are divided by a learned temperature
    self.min_temperature = 0.1  # Whether the output distribution parameters are divided by a learned temperature

    # Whether to use the histogram loss gauss to train the value head via classification (instead of regression + L2)
    self.use_hl_gauss_value_loss = False
    self.hl_gauss_std = 0.75  # Standard deviation use for the gaussian histogram loss
    self.hl_gauss_vmin = -10.0  # Min value of the histogram in HL_Gauss. Tune to be in return range
    self.hl_gauss_vmax = 30.0  # Max value of the histogram in HL_Gauss. Tune to be in return range
    self.hl_gauss_bucket_size = 1.0  # Size of each bucket in the HL_Gauss histogram.
    self.hl_gauss_num_classes = int((self.hl_gauss_vmax - self.hl_gauss_vmin) / self.hl_gauss_bucket_size) + 1

    self.global_step = 0  # Current iteration of the training
    self.max_training_score = -np.inf  # Highest training score achieved so far
    self.best_iteration = 0  # Iteration of the best model
    self.latest_iteration = 0  # Iteration of the latest model

    # Sensorimotor parameters
    self.use_sensorimotor = False  # Whether to train a sensorimotor agent.
    self.cam_configs = [{'type': 'sensor.camera.rgb',
                         'x': 0.1,
                         'y': -0.35,
                         'z': 2.25,
                         'roll': 0.0,
                         'pitch': 0.0,
                         'yaw': -54.5,
                         'width': 1152 // 3,
                         'height': 384,
                         'fov': 60,
                         'id': 'rgb_left'},
                        {'type': 'sensor.camera.rgb',
                         'x': 0.35,
                         'y': 0.0,
                         'z': 2.25,
                         'roll': 0.0,
                         'pitch': 0.0,
                         'yaw': 0.0,
                         'width': 1152 // 3,
                         'height': 384,
                         'fov': 60,
                         'id': 'rgb_front'},
                        {'type': 'sensor.camera.rgb',
                         'x': 0.1,
                         'y': 0.35,
                         'z': 2.25,
                         'roll': 0.0,
                         'pitch': 0.0,
                         'yaw': 54.5,
                         'width': 1152 // 3,
                         'height': 384,
                         'fov': 60,
                         'id': 'rgb_right'}]

    self.imu_sensor_config = {'type': 'sensor.other.imu',
                              'x': 0.0,
                              'y': 0.0,
                              'z': 0.0,
                              'roll': 0.0,
                              'pitch': 0.0,
                              'yaw': 0.0,
                              'sensor_tick': self.frame_rate,
                              'id': 'imu'}

    self.speedometer_config = {
      'type': 'sensor.speedometer',
      'reading_frequency': self.frame_rate,
      'id': 'speed'
    }

    self.gnss_config = {
      'type': 'sensor.other.gnss',
      'x': 0.0,
      'y': 0.0,
      'z': 0.0,
      'roll': 0.0,
      'pitch': 0.0,
      'yaw': 0.0,
      'sensor_tick': 0.01,
      'id': 'gps'
    }

    self.lidar_configs = [{'type': 'sensor.lidar.ray_cast',
                           'x': 0.0,
                           'y': 0.0,
                           'z': 2.5,
                           'roll': 0.0,
                           'pitch': 0.0,
                           'yaw': -90.0,
                           'id': 'lidar_0'},
                          {'type': 'sensor.lidar.ray_cast',
                           'x': 0.0,
                           'y': 0.0,
                           'z': 2.5,
                           'roll': 0.0,
                           'pitch': 0.0,
                           'yaw': -270.0,
                           'id': 'lidar_1'}]
    # Minimum distance for route planner
    self.route_planner_min_distance = 7.5  # TODO TFv6 uses 5.0
    # Maximum distance for route planner
    self.route_planner_max_distance = 50

    # TransFuser Hyperparameters
    # convnext_pico.d1_in1k
    # convnext_femto.d1_in1k
    self.image_architecture = 'resnet34' #'regnety_032'  # Image architecture used in the backbone resnet34, regnety_032
    self.lidar_architecture = 'resnet18' #'regnety_032'  # LiDAR architecture used in the backbone resnet34, regnety_032

    # Width and height of the LiDAR grid that the point cloud is voxelized into.
    self.lidar_resolution_width = 256
    self.lidar_resolution_height = 256

    self.use_ground_plane = False
    self.lidar_seq_len = 1
    self.camera_width = self.cam_configs[0]['width'] + self.cam_configs[1]['width'] + self.cam_configs[2]['width']
    self.img_vert_anchors = self.cam_configs[0]['height'] // 32
    self.img_horz_anchors = self.camera_width // 32
    self.lidar_vert_anchors = self.lidar_resolution_height // 32
    self.lidar_horz_anchors = self.lidar_resolution_width // 32

    # Whether to normalize the camera image by the imagenet distribution
    self.normalize_imagenet = True

    # Resolution at which the perspective auxiliary tasks are predicted
    self.perspective_downsample_factor = 1

    self.detect_boxes = False  # Whether to use the bounding box auxiliary task
    self.use_bev_semantic = False  # Whether to use bev semantic segmentation as auxiliary loss for training.
    self.use_depth = False  # Whether to use depth prediction as auxiliary loss for training.
    self.use_semantic = False  # Whether to use semantic segmentation as auxiliary loss

    self.bev_features_chanels = 64  # Number of channels for the BEV feature pyramid
    # Resolution at which the BEV auxiliary tasks are predicted
    self.bev_down_sample_factor = 4
    self.bev_upsample_factor = 2

    # Semantic Segmentation
    self.num_semantic_classes = 7
    self.classes = {
        0: [0, 0, 0],  # unlabeled
        1: [30, 170, 250],  # vehicle
        2: [200, 200, 200],  # road
        3: [255, 255, 0],  # light
        4: [0, 255, 0],  # pedestrian
        5: [0, 255, 255],  # road line
        6: [255, 255, 255],  # sidewalk
    }
    # Color format BGR
    self.classes_list = [
        [0, 0, 0],  # unlabeled
        [250, 170, 30],  # vehicle
        [200, 200, 200],  # road
        [0, 255, 255],  # light
        [0, 255, 0],  # pedestrian
        [255, 255, 0],  # road line
        [255, 255, 255],  # sidewalk
    ]

    # GPT Encoder
    self.block_exp = 4
    self.n_layer = 2  # Number of transformer layers used in the vision backbone
    self.n_head = 4
    self.embd_pdrop = 0.0 # TODO set to 0 since it causes off-policy ness tune? 0.1
    self.resid_pdrop = 0.0 # TODO set to 0 since it causes off-policy ness tune? 0.1
    self.attn_pdrop = 0.0 # TODO set to 0 since it causes off-policy ness tune? 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    self.gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    self.gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    self.gpt_layer_norm_init_weight = 1.0

    self.num_transformer_decoder_layers = 6  # Number of layers in the TransFormer decoder
    self.num_decoder_heads = 8
    self.tf_decoder_channels = 256

  def initialize(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
