'''
TransFuser based policy
'''
import math
from typing import Dict
from copy import deepcopy

import gymnasium as gym
import torch
from torch import nn
import numpy as np
import cv2

from distributions import BetaDistribution
from model.transfuser import TransfuserBackbone
import model.custom_resnet as backbones
import model.tf_utils as t_u
# import matplotlib.pyplot as plt


class TransFuserPolicy(nn.Module):
  '''
    Neural network policy designed for driving and training with the PPO algorithm.
  '''

  def __init__(self,
               observation_space: gym.spaces.Space,
               action_space: gym.spaces.Space,
               config=None):

    super().__init__()
    self.action_space = action_space
    self.config = config

    # self.vision_backbone = TransfuserBackbone(config)

    self.action_dist = BetaDistribution(int(np.prod(action_space.shape)))

    self.action_space_low = nn.Parameter(torch.from_numpy(self.action_space.low), requires_grad=False)
    self.action_space_high = nn.Parameter(torch.from_numpy(self.action_space.high), requires_grad=False)

    #
    # decoder_layer = nn.TransformerDecoderLayer(self.config.tf_decoder_channels,
    #                                            self.config.num_decoder_heads,
    #                                            activation=nn.GELU(),
    #                                            dropout=0.0,  # TODO might harm on-policy if turned on
    #                                            batch_first=True)
    #
    # self.tf_decoder_join = torch.nn.TransformerDecoder(decoder_layer,
    #                                                    num_layers=self.config.num_transformer_decoder_layers,
    #                                                    norm=nn.LayerNorm(self.config.tf_decoder_channels))
    #
    # self.encoder_pos_encoding = PositionEmbeddingSine(self.config.tf_decoder_channels // 2, normalize=True)
    # extra_size = 1 + 2 + 2  # speed, target point, target point next
    # self.extra_sensor_encoder = nn.Sequential(nn.Linear(extra_size, 128), nn.LayerNorm(128), nn.ReLU(inplace=True),
    #                                           nn.Linear(128, self.config.tf_decoder_channels),
    #                                           nn.LayerNorm(self.config.tf_decoder_channels), nn.ReLU(inplace=True))
    # self.extra_sensor_pos_embed = nn.Parameter(torch.zeros(1, self.config.tf_decoder_channels))
    # nn.init.uniform_(self.extra_sensor_pos_embed)
    # self.learned_query = nn.Parameter(torch.zeros(1, 3, self.config.tf_decoder_channels))
    # nn.init.uniform_(self.learned_query)

    self.linear_alpha = nn.Linear(self.config.features_dim, self.action_space.shape[0])
    self.linear_beta = nn.Linear(self.config.features_dim, self.action_space.shape[0])


    self.value_measurements_encoder = nn.Sequential(nn.Linear(self.config.num_value_measurements, 256),
                                               nn.LayerNorm(256),
                                               nn.ReLU(),
                                               nn.Linear(256, 256),
                                               nn.LayerNorm(256),
                                               nn.ReLU())

    self._value_head = nn.Sequential(nn.Linear(config.features_dim + 256, 256),
                                      nn.LayerNorm(256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256),
                                      nn.LayerNorm(256),
                                      nn.ReLU(),
                                      nn.Linear(256, 1))


    input_channels = 3

    # Roach test:
    if self.config.image_architecture == 'rl_resnet34':
      self.cnn = backbones.ResNet34_PPO(in_channel=input_channels)
    elif self.config.image_architecture == 'rl_resnet18':
      self.cnn = backbones.ResNet18_PPO(in_channel=input_channels)
    elif self.config.image_architecture == 'None':
      pass
    else:
      raise ValueError("Unsupported network")

    if self.config.lidar_architecture == 'rl_resnet34':
      self.lidar_cnn = backbones.ResNet34_PPO(in_channel=1)
    elif self.config.lidar_architecture == 'rl_resnet18':
      self.lidar_cnn = backbones.ResNet18_PPO(in_channel=1)
    elif self.config.lidar_architecture == 'None':
      pass
    else:
      raise ValueError("Unsupported network")

    if self.config.image_architecture != 'None':
      self.down_img = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.GroupNorm(32, 512),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(32, 128),
                                    nn.ReLU())
      self.weights_init(self.down_img)

    if self.config.lidar_architecture != 'None':
      self.down_lidar = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, bias=False),
                                      nn.GroupNorm(32, 128),
                                      nn.ReLU())
      self.weights_init(self.down_lidar)

    self.state_linear = []

    # +4 comes from 2 dim each for target point and target point next.
    if self.config.use_speed_limit:
      num_measure = self.config.obs_num_measurements
    else:
      num_measure = self.config.obs_num_measurements - 1

    self.state_linear.append(nn.Linear(num_measure + 4, 256))
    self.state_linear.append(nn.LayerNorm(256))
    self.state_linear.append(nn.ReLU())
    self.state_linear.append(nn.Linear(256, 256))
    self.state_linear.append(nn.LayerNorm(256))
    self.state_linear.append(nn.ReLU())
    self.state_linear = nn.Sequential(*self.state_linear)

    # Compute shape by doing one forward pass
    with torch.no_grad():
      self.n_flatten = 0
      if self.config.image_architecture != 'None':
        width = self.config.cam_configs[0]['width'] + self.config.cam_configs[1]['width'] + self.config.cam_configs[2]['width']
        sample_cam = torch.zeros([1, input_channels, self.config.cam_configs[0]['height'], width])
        self.cnn_out_shape = self.down_img(self.cnn(sample_cam)).shape
        self.n_flatten += math.prod(self.cnn_out_shape[1:])

      if self.config.lidar_architecture != 'None':
        lidar_shape = (int((self.config.lidar_max_x - self.config.lidar_min_x) * self.config.pixels_per_meter),
                       int((self.config.lidar_max_y - self.config.lidar_min_y) * self.config.pixels_per_meter))
        sample_lidar = torch.zeros([1, 1, lidar_shape[0], lidar_shape[1]])
        self.lidar_out_shape = self.down_lidar(self.lidar_cnn(sample_lidar)).shape
        self.n_flatten += math.prod(self.lidar_out_shape[1:])

    assert (self.n_flatten != 0)

    self.linear = nn.Sequential(nn.Linear(self.n_flatten + 256, 512), nn.LayerNorm(512), nn.ReLU(),
                                nn.Linear(512, config.features_dim), nn.LayerNorm(config.features_dim), nn.ReLU())

    self.precision = torch.float32
    if self.config.use_bfloat16:
      self.precision = torch.bfloat16

    if self.config.use_lstm:
      self.lstm = nn.LSTM(config.features_dim, config.features_dim, num_layers=config.num_lstm_layers)
      for name, param in self.lstm.named_parameters():
        if 'bias' in name:
          nn.init.constant_(param, 0)
        elif 'weight' in name:
          nn.init.orthogonal_(param, 1.0)
      self.lstm_mlp = nn.Sequential(nn.Linear(config.features_dim, config.features_dim),
                                    nn.LayerNorm(config.features_dim),
                                    nn.ReLU(),
                                    nn.Linear(config.features_dim, config.features_dim),
                                    nn.LayerNorm(config.features_dim),
                                    nn.ReLU())

    # plt.ion()  # interactive mode
    # self.fig, self.ax = plt.subplots()
    #
    # self.img = None

  def weights_init(self, module):
    for m in module.modules():
      if isinstance(m, nn.Conv2d):
        # Orthogonal init with the recommended gain for ReLU
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.GroupNorm):
        # Standard initialization for normalization layers
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


  def get_features(self, obs_dict) -> torch.Tensor:
    rgb = obs_dict['rgb'].to(dtype=self.precision)
    measurements = obs_dict['measurements']

    if not self.config.use_speed_limit:
      with torch.no_grad():
        if self.config.obs_num_measurements <= 8:
          measurements = measurements[:, :7]
        else:
          measurements = torch.cat((measurements[:, :7], measurements[:, 8:]), dim=1)


    target_point = obs_dict['target_point']
    target_point_next = obs_dict['target_point_next']
    # 256 because it's lossless in float
    lidar_bev = obs_dict['lidar'].unsqueeze(dim=1).to(dtype=self.precision) / 256.0
    measurements = torch.cat((measurements, target_point, target_point_next), dim=1)

    # with torch.no_grad():
    #   t = rgb.detach().cpu()
    #
    #   # [1,3,H,W] -> [H,W,3]
    #   t = t[0].permute(1, 2, 0) / 255.0
    #
    # if self.img is None:
    #   self.img = self.ax.imshow(t)
    #   self.ax.axis("off")
    # else:
    #   self.img.set_data(t)
    #
    # self.fig.canvas.draw()
    # self.fig.canvas.flush_events()
    # plt.pause(0.01)  # controls refresh rate
    latent_state = self.state_linear(measurements)
    feature_list = [latent_state]
    if self.config.image_architecture != 'None':
      if self.config.normalize_imagenet:
        with torch.no_grad():
          if self.config.use_bfloat16:
            rgb = t_u.normalize_image_bf16(rgb)
          else:
            rgb = t_u.normalize_imagenet_func(rgb)

      rgb_enc = self.cnn(rgb)
      rgb_enc = self.down_img(rgb_enc)
      rgb_enc = torch.flatten(rgb_enc, start_dim=1)
      feature_list.append(rgb_enc)

    if self.config.lidar_architecture != 'None':
      lidar_enc = self.lidar_cnn(lidar_bev)
      lidar_enc = self.down_lidar(lidar_enc)
      lidar_enc = torch.flatten(lidar_enc, start_dim=1)
      feature_list.append(lidar_enc)

    x = torch.cat(feature_list, dim=1)
    x = self.linear(x)

    return x

  def lstm_forward(self, features, lstm_state, done):
    # LSTM logic
    batch_size = lstm_state[0].shape[1]
    hidden = features.reshape((-1, batch_size, self.lstm.input_size))
    done = done.reshape((-1, batch_size))
    new_hidden = []
    for h, d in zip(hidden, done):
      h, lstm_state = self.lstm(
          h.unsqueeze(0),
          (
              (1.0 - d).view(1, -1, 1) * lstm_state[0],
              (1.0 - d).view(1, -1, 1) * lstm_state[1],
          ),
      )
      new_hidden += [h]
    new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
    return new_hidden, lstm_state

  def get_value(self, obs_dict: Dict[str, torch.Tensor], lstm_state=None, done=None):
    features = self.get_features(obs_dict)

    if self.config.use_lstm:
      features, _ = self.lstm_forward(features, lstm_state, done)
      features = self.lstm_mlp(features)

    value_states = self.value_measurements_encoder(obs_dict['value_measurements'])
    values = self._value_head(torch.cat((features, value_states), dim=1))

    return values

  def forward(self,
              obs_dict,
              actions=None,
              sample_type='sample',
              exploration_suggests=None,
              lstm_state=None,
              done=None):
    '''
        actions are expected to be unscaled actions!
        '''
    features = self.get_features(obs_dict)

    if self.config.use_lstm:
      features, lstm_state = self.lstm_forward(features, lstm_state, done)
      features = self.lstm_mlp(features)

    alpha = self.linear_alpha(features)
    beta = self.linear_beta(features)

    value_states = self.value_measurements_encoder(obs_dict['value_measurements'])
    values = self._value_head(torch.cat((features, value_states), dim=1))

    alpha = nn.functional.softplus(alpha)
    beta = nn.functional.softplus(beta)
    alpha = alpha + self.config.beta_min_a_b_value
    beta = beta + self.config.beta_min_a_b_value
    distribution = self.action_dist.proba_distribution(alpha, beta)

    pred_sem = pred_measure = None

    if actions is None:
      actions = distribution.get_actions(sample_type)
    else:
      actions = self.scale_action(actions)

    log_prob = distribution.log_prob(actions)

    actions = self.unscale_action(actions)

    entropy = distribution.entropy().sum(1)
    exp_loss = None

    if exploration_suggests is not None:
      exp_loss = distribution.exploration_loss(exploration_suggests)

    return (actions, log_prob, entropy, values, exp_loss, alpha.detach(), beta.detach(), distribution.distribution, pred_sem, pred_measure,
            lstm_state)

  def scale_action(self, action: torch.Tensor, eps=1e-7) -> torch.Tensor:
    # input action \in [a_low, a_high]
    # output action \in [d_low+eps, d_high-eps]
    d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

    if d_low is not None and d_high is not None:
      a_low, a_high = self.action_space_low, self.action_space_high
      action = (action - a_low) / (a_high - a_low) * (d_high - d_low) + d_low
      action = torch.clamp(action, d_low + eps, d_high - eps)
    return action

  def unscale_action(self, action: torch.Tensor) -> torch.Tensor:
    # input action \in [d_low, d_high]
    # output action \in [a_low+eps, a_high-eps]
    d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

    if d_low is not None and d_high is not None:
      a_low, a_high = self.action_space_low, self.action_space_high
      action = (action - d_low) / (d_high - d_low) * (a_high - a_low) + a_low
    return action

  @torch.no_grad()
  def visualize_model(self,
                      distribution,
                      obs,
                      control,
                      value,
                      pred_sem,
                      pred_measure,
                      upscale_factor=1):
    cv2.setNumThreads(0) # Set to something reasonable for CPUs with 100s of cores.

    font = cv2.FONT_HERSHEY_SIMPLEX
    obs_rendered = obs['rendered']
    measurements = obs['measurements']
    value_measurements = obs['value_measurements']

    obs_rendered_upscaled = obs_rendered.repeat(upscale_factor, axis=0).repeat(upscale_factor, axis=1)
    width, height, _ = obs_rendered_upscaled.shape

    if distribution is not None:
      if self.config.distribution in ('beta', 'beta_uni_mix'):
        device = distribution.concentration1.device
        granularity = torch.arange(start=0.0, end=1.0, step=0.001 / upscale_factor).unsqueeze(1)
        granularity = torch.ones((granularity.shape[0], self.action_space.shape[0])) * granularity
        granularity = granularity.to(device)
        granularity_cpu = deepcopy(granularity).cpu()
      elif self.config.distribution == 'normal':
        device = distribution.mean.device
        granularity_cpu = torch.arange(start=0.0, end=1.0, step=0.001 / upscale_factor).unsqueeze(1)
        granularity = torch.arange(start=-1.0, end=1.0, step=0.002 / upscale_factor).unsqueeze(1)
        granularity = torch.ones((granularity.shape[0], self.action_space.shape[0])) * granularity
        granularity = granularity.to(device)

      if self.config.distribution == 'beta_uni_mix':
        uniform_pdf = torch.ones_like(granularity, device=device, requires_grad=False)
        distribution = (self.action_dist.beta_perc * distribution.log_prob(granularity).exp() +
                        self.action_dist.uniform_perc * uniform_pdf)
        distribution = distribution.cpu().numpy()
      else:
        distribution = distribution.log_prob(granularity)
        distribution = torch.exp(distribution).cpu().numpy()

      action_type = ['Steering', 'Brake | Throttle']
      action_plots = []
      plot_height = int(round(height / (self.action_space.shape[0] + 1), 0))
      actions = [control.steer, control.throttle - control.brake]
      y_max = 12.0  # Continuous PDFs can be arbitrary high. We clipp after 25.

      for i in range(self.action_space.shape[0]):
        action_plot = np.zeros((plot_height, width, 3), dtype=np.uint8)
        cv2.line(action_plot, (width // 2, 0), (width // 2, (plot_height - 1)), (0, 255, 0), thickness=2 * upscale_factor)
        cv2.line(action_plot, (0, 0), (0, (plot_height - 1)), (0, 255, 0), thickness=2 * upscale_factor)
        cv2.line(action_plot, (width - 1, 0), (width - 1, (plot_height - 1)), (0, 255, 0), thickness=2 * upscale_factor)

        # Plot actions:
        control_pixel = int(((actions[i] + 1.0) / 2.0) * (width - 1))
        cv2.line(action_plot, (control_pixel, 0), (control_pixel, (plot_height - 1)), (255, 255, 0),
                 thickness=2 * upscale_factor)

        granularity_numpy = granularity_cpu.numpy()
        xs = (granularity_numpy[:, 0] * width).astype(np.int32)
        y_pixels = distribution[:, i] / y_max * (plot_height - 1)
        clipped_pixels = np.clip(y_pixels, a_min=None, a_max=int(plot_height - 1)).astype(np.int32)
        ys = (plot_height - 1) - clipped_pixels

        points = np.stack([xs, ys], axis=1).astype(np.int32)
        action_plot = cv2.polylines(action_plot, [points],
                                    isClosed=False,
                                    color=(255, 255, 0),
                                    lineType=cv2.LINE_AA,
                                    thickness=2 * upscale_factor)

        cv2.putText(action_plot, action_type[i], (5 * upscale_factor, 10 * upscale_factor), font, 0.33 * upscale_factor,
                    (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
        action_plots.append(action_plot)

      action_plots = np.concatenate(action_plots, axis=0)
      measure_plot_height = height - action_plots.shape[0]
      measurement_plot = np.zeros((measure_plot_height, width, 3), dtype=np.uint8)
    else:
      measurement_plot = np.zeros((height, width, 3), dtype=np.uint8)

    y_point = 10 * upscale_factor
    cv2.putText(obs_rendered_upscaled, f'Last steer: {measurements[0]:.2f}', (0, y_point), font, 0.33 * upscale_factor,
                (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(obs_rendered_upscaled, f'Last throt: {measurements[1]:.2f}', (0, y_point), font, 0.33 * upscale_factor,
                (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(obs_rendered_upscaled, f'Last brake: {measurements[2]:.2f}', (0, y_point), font, 0.33 * upscale_factor,
                (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)

    if self.config.use_target_point:
      y_point += 15 * upscale_factor
      cv2.putText(obs_rendered_upscaled, f'TP: {measurements[8]:.1f} {measurements[9]:.1f}', (0, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1, cv2.LINE_AA)

    y_point = 10 * upscale_factor
    cv2.putText(obs_rendered_upscaled, f'Gear: {measurements[3]:.2f}', (width // 2, y_point), font,
                0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(obs_rendered_upscaled, f'Velocity: {measurements[4]:.1f} {measurements[5]:.1f}', (width // 2, y_point),
                font, 0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    #y_point += 15 * upscale_factor
    #cv2.putText(obs_rendered_upscaled, f'X Velocity: {measurements[6]:.2f}', (width // 2, y_point), font,
    #            0.33 * upscale_factor, (255, 255, 255), 1*upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    if self.config.use_speed_limit:
      cv2.putText(obs_rendered_upscaled, f'Speed lim.: {measurements[7]:.2f}', (width // 2, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)

    y_point = 10 * upscale_factor
    cv2.putText(measurement_plot, 'Model predictions:', (130 * upscale_factor, y_point), font, 0.33 * upscale_factor,
                (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(measurement_plot, f'Steer:{control.steer:.2f}', (130 * upscale_factor, y_point), font,
                0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(measurement_plot, f'Throt:{control.throttle:.2f}', (130 * upscale_factor, y_point), font,
                0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(measurement_plot, f'Brake:{control.brake:.2f}', (130 * upscale_factor, y_point), font,
                0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor

    if value is not None:
      cv2.putText(measurement_plot, f'Value:{value.item():.2f}', (130 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)

    if self.config.use_value_measurements:
      y_point = 10 * upscale_factor
      cv2.putText(measurement_plot, 'Critic inputs:', (5 * upscale_factor, y_point), font, 0.33 * upscale_factor,
                  (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'Timeout:{value_measurements[0]:.2f}', (5 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'Blocked:{value_measurements[1]:.2f}', (5 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'Route:{value_measurements[2]:.2f}', (5 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)

    if self.config.use_extra_control_inputs:
      y_point = 140 * upscale_factor
      cv2.putText(measurement_plot, f'wheel: {measurements[8]:.2f}', (130 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'error: {measurements[9]:.2f}', (130 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'deriv: {measurements[10]:.2f}', (130 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'integ: {measurements[11]:.2f}', (130 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)

    if distribution is not None:
      action_plots = np.concatenate((measurement_plot, action_plots), axis=0)
      bev_visu = np.concatenate((obs_rendered_upscaled, action_plots), axis=1)
    else:
      bev_visu = np.concatenate((obs_rendered_upscaled, measurement_plot), axis=1)

    if self.config.use_sensorimotor:
      sensor_visu = self.visualize_sensors(obs)
      scale_factor = sensor_visu.shape[1] / bev_visu.shape[1]
      bev_visu = cv2.resize(bev_visu,
                                dsize=(int(bev_visu.shape[1] * scale_factor), int(bev_visu.shape[0] * scale_factor)),
                                interpolation=cv2.INTER_NEAREST)

      bev_visu = np.concatenate((sensor_visu, bev_visu), axis=0)

    # # Life visu
    # if self.img is None:
    #   self.img = self.ax.imshow(bev_visu)
    #   self.ax.axis("off")
    # else:
    #   self.img.set_data(bev_visu)
    #
    # self.fig.canvas.draw()
    # self.fig.canvas.flush_events()
    # plt.pause(0.01)  # controls refresh rate

    return bev_visu


  def visualize_sensors(self, obs):
    # 0 Car, 1 Pedestrian, 2 Red light, 3 Stop sign, 4 emergency vehicle
    color_classes = [
      np.array([255, 165, 0]),
      np.array([0, 255, 0]),
      np.array([255, 0, 0]),
      np.array([250, 160, 160]),
      np.array([16, 133, 133])
    ]

    lidar_bev = obs['lidar']
    target_point = obs['target_point']
    target_point_next = obs['target_point_next']
    rgb = obs['rgb']

    size_width = int((self.config.lidar_max_y - self.config.lidar_min_y) * self.config.pixels_per_meter)
    size_height = int((self.config.lidar_max_x - self.config.lidar_min_x) * self.config.pixels_per_meter)

    scale_factor = 4.5
    origin = ((size_width * scale_factor) // 2, (size_height * scale_factor) // 2)
    loc_pixels_per_meter = self.config.pixels_per_meter * scale_factor

    images_lidar = np.stack([lidar_bev, lidar_bev, lidar_bev], axis=-1)

    images_lidar = cv2.resize(images_lidar,
                              dsize=(int(images_lidar.shape[1] * scale_factor), int(images_lidar.shape[0] * scale_factor)),
                              interpolation=cv2.INTER_NEAREST)

    # Rotate so that image coordinates align with CARLA coordinates before rendering target point.
    images_lidar = np.rot90(images_lidar, k=-1)
    images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)

    # Draw target points
    x_tp = target_point[0] * loc_pixels_per_meter + origin[0]
    y_tp = target_point[1] * loc_pixels_per_meter + origin[1]
    cv2.circle(images_lidar, (int(x_tp), int(y_tp)), radius=12, lineType=cv2.LINE_AA, color=(255, 0, 0),
               thickness=-1)

    # draw next tp too

    x_tpn = target_point_next[0] * loc_pixels_per_meter + origin[0]
    y_tpn = target_point_next[1] * loc_pixels_per_meter + origin[1]
    cv2.circle(images_lidar, (int(x_tpn), int(y_tpn)),
               radius=12,
               lineType=cv2.LINE_AA,
               color=(255, 0, 0),
               thickness=-1)
    # Rotate back so that top is up in image. Looks nicer and aligns with map.
    images_lidar = np.rot90(images_lidar, k=1)
    images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)

    # draw ego
    sample_box = np.array([
      int(images_lidar.shape[0] / 2),
      int(images_lidar.shape[1] / 2), self.config.ego_extent_x * loc_pixels_per_meter,
                                                       self.config.ego_extent_y * loc_pixels_per_meter,
      np.deg2rad(0.0), 0.0
    ])
    images_lidar = t_u.draw_box(images_lidar, sample_box, color=(0, 200, 0), pixel_per_meter=16, thickness=4)

    rgb_image = rgb.transpose(1, 2, 0)

    all_images = np.concatenate((rgb_image, images_lidar), axis=0)

    return all_images
