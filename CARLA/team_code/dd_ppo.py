'''
Self-contained PPO training algorithm. Adapted from CleanRL https://github.com/vwxyzjn/cleanrl
'''

import argparse
import os
import random
import time
import pathlib
import re
import gc
import datetime
import math
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import wandb
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from pytictoc import TicToc
import zmq

from model import PPOPolicy
from rl_config import GlobalConfig
from birds_eye_view.bev_observation import image_to_class_labels
import rl_utils as rl_u

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
torch.set_num_threads(1)


def strtobool(v):
  return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


def none_or_str(value):
  if value == 'None':
    return None
  return value


def save(model, optimizer, config, folder, model_file, optimizer_file):
  model_file = os.path.join(folder, model_file)
  torch.save(model.module.state_dict(), model_file)

  if optimizer is not None:
    optimizer_file = os.path.join(folder, optimizer_file)
    torch.save(optimizer.state_dict(), optimizer_file)

  json_config = jsonpickle.encode(config)
  with open(os.path.join(folder, 'config.json'), 'wt', encoding='utf-8') as f2:
    f2.write(json_config)


def parse_args(config):
  # fmt: off
  parser = argparse.ArgumentParser(allow_abbrev=False)
  parser.add_argument('--rdzv_addr', default='localhost', type=str, help='Master address for the TCP store.')
  parser.add_argument('--exp_name', type=str, default=config.exp_name, help='the name of this experiment')
  parser.add_argument('--gym_id', type=str, default=config.gym_id, help='the id of the gym environment')
  parser.add_argument('--tcp_store_port', type=int, required=True, help='port for the key value store')
  parser.add_argument('--learning_rate',
                      type=float,
                      default=config.learning_rate,
                      help='the learning rate of the optimizer')
  parser.add_argument('--seed', type=int, default=config.seed, help='seed of the experiment')
  parser.add_argument('--total_timesteps',
                      type=int,
                      default=config.total_timesteps,
                      help='total timesteps of the experiments')
  parser.add_argument('--torch_deterministic',
                      type=lambda x: bool(strtobool(x)),
                      default=config.torch_deterministic,
                      nargs='?',
                      const=True,
                      help='if toggled, `torch.backends.cudnn.deterministic=False`')
  parser.add_argument('--allow_tf32',
                      type=lambda x: bool(strtobool(x)),
                      default=config.allow_tf32,
                      nargs='?',
                      const=True,
                      help='whether to use tensor cores, lower numeric precision but faster speed')
  parser.add_argument('--benchmark',
                      type=lambda x: bool(strtobool(x)),
                      default=config.benchmark,
                      nargs='?',
                      const=True,
                      help='Whether to benchmark different algorithms on hardware to find the fastest')
  parser.add_argument('--matmul_precision',
                      type=str,
                      default=config.matmul_precision,
                      help=' Options highest=float32, high=tf32, medium=bfloat16')

  parser.add_argument('--cuda',
                      type=lambda x: bool(strtobool(x)),
                      default=config.cuda,
                      nargs='?',
                      const=True,
                      help='if toggled, cuda will be enabled by default')
  parser.add_argument('--track',
                      type=lambda x: bool(strtobool(x)),
                      default=config.track,
                      nargs='?',
                      const=True,
                      help='if toggled, this experiment will be tracked with Weights and Biases')
  parser.add_argument('--wandb_project_name',
                      type=str,
                      default=config.wandb_project_name,
                      help='the wandb project name')
  parser.add_argument('--wandb_entity',
                      type=str,
                      default=config.wandb_entity,
                      help='the entity (team) of wandb project')
  parser.add_argument('--capture_video',
                      type=lambda x: bool(strtobool(x)),
                      default=config.capture_video,
                      nargs='?',
                      const=True,
                      help='whether to capture videos of the agent performances (check out `videos` folder)')

  # Algorithm specific arguments
  parser.add_argument('--total_batch_size',
                      type=int,
                      default=config.total_batch_size,
                      help='the total amount of data collected at every step across all environments')
  parser.add_argument('--total_minibatch_size',
                      type=int,
                      default=config.total_minibatch_size,
                      help='the total minibatch sized used for training (across all environments)')
  parser.add_argument('--lr_schedule',
                      default=config.lr_schedule,
                      type=str,
                      help='Which lr schedule to use. Options: (linear, kl, none, step, cosine, cosine_restart)')
  parser.add_argument('--gae',
                      type=lambda x: bool(strtobool(x)),
                      default=config.gae,
                      nargs='?',
                      const=True,
                      help='Use GAE for advantage computation')
  parser.add_argument('--gamma', type=float, default=config.gamma, help='the discount factor gamma')
  parser.add_argument('--gae_lambda',
                      type=float,
                      default=config.gae_lambda,
                      help='the lambda for the general advantage estimation')
  parser.add_argument('--update_epochs',
                      type=int,
                      default=config.update_epochs,
                      help='the K epochs to update the policy')
  parser.add_argument('--norm_adv',
                      type=lambda x: bool(strtobool(x)),
                      default=config.norm_adv,
                      nargs='?',
                      const=True,
                      help='Toggles advantages normalization')
  parser.add_argument('--clip_coef', type=float, default=config.clip_coef, help='the surrogate clipping coefficient')
  parser.add_argument('--clip_vloss',
                      type=lambda x: bool(strtobool(x)),
                      default=config.clip_vloss,
                      nargs='?',
                      const=True,
                      help='Toggles whether or not to use a clipped loss for the value function, as per the paper.')
  parser.add_argument('--ent_coef', type=float, default=config.ent_coef, help='coefficient of the entropy')
  parser.add_argument('--vf_coef', type=float, default=config.vf_coef, help='coefficient of the value function')
  parser.add_argument('--max_grad_norm',
                      type=float,
                      default=config.max_grad_norm,
                      help='the maximum norm for the gradient clipping')
  parser.add_argument('--target_kl', type=float, default=config.target_kl, help='the target KL divergence threshold')
  parser.add_argument('--visualize',
                      type=lambda x: bool(strtobool(x)),
                      default=config.visualize,
                      nargs='?',
                      const=True,
                      help='if toggled, Game will render on screen')
  parser.add_argument('--logdir', type=str, default=config.logdir, help='The directory to log the data into.')
  parser.add_argument('--load_file',
                      type=none_or_str,
                      nargs='?',
                      default=config.load_file,
                      help='model weights for initialization')
  parser.add_argument('--ports',
                      nargs='+',
                      default=config.ports,
                      type=int,
                      help='Ports of the carla_gym wrapper to connect to.'
                      'It requires to submit a port for every envs'
                      '#ports == --nproc_per_node')
  parser.add_argument('--gpu_ids',
                      nargs='+',
                      default=config.gpu_ids,
                      type=int,
                      help='Which GPUs to train on. Index 0 indicates GPU for rank 0 etc.')
  parser.add_argument('--num_envs_per_proc',
                      type=int,
                      default=config.num_envs_per_proc,
                      help='Number of environments per process. Only considered for dd_ppo.py')
  parser.add_argument('--compile_model',
                      type=lambda x: bool(strtobool(x)),
                      default=config.compile_model,
                      nargs='?',
                      const=True,
                      help='whether to compile the model with torch.compile.')
  parser.add_argument('--expl_coef', type=float, default=config.expl_coef, help='coefficient of the exploration')
  parser.add_argument('--adam_eps', type=float, default=config.adam_eps, help='eps parameter of adam optimizer')
  parser.add_argument('--cpu_collect',
                      type=lambda x: bool(strtobool(x)),
                      default=config.cpu_collect,
                      nargs='?',
                      const=True,
                      help='If true than the agent will be run on the cpu during data collection. Can be faster.')
  parser.add_argument('--use_exploration_suggest',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_exploration_suggest,
                      nargs='?',
                      const=True,
                      help='Whether to use the exploration loss from roach.')
  parser.add_argument('--use_speed_limit_as_max_speed',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_speed_limit_as_max_speed,
                      nargs='?',
                      const=True,
                      help='If true rr_maximum_speed will be overwritten to the current speed limit affecting the ego '
                      'vehicle.')
  parser.add_argument('--beta_min_a_b_value',
                      type=float,
                      default=config.beta_min_a_b_value,
                      help='Nugget that gets added to the softplus output of the network.'
                      'Aims to prevent degenerate distributions with the Beta.')
  parser.add_argument('--use_rpo',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_rpo,
                      nargs='?',
                      const=True,
                      help='Whether to use robust policy optimization (add noise to mean during training)')
  parser.add_argument('--rpo_alpha',
                      type=float,
                      default=config.rpo_alpha,
                      help='Noise added during training is of uniform shape [-rpo_alpha, rpo_alpha]')
  parser.add_argument('--use_new_bev_obs',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_new_bev_obs,
                      nargs='?',
                      const=True,
                      help='Whether to use the new bev observation or the old roach one.')
  parser.add_argument('--obs_num_channels',
                      type=int,
                      default=config.obs_num_channels,
                      help='Number of channels is the BEV image.')
  parser.add_argument('--map_folder',
                      type=str,
                      default=config.map_folder,
                      help='The map folder to use with the new representation. Needs to align with pixels_per_meter')
  parser.add_argument('--pixels_per_meter',
                      type=float,
                      default=config.pixels_per_meter,
                      help='Pixels per meter in the new BEV image.')
  parser.add_argument('--route_width',
                      type=int,
                      default=config.route_width,
                      help='Width of the rendered route in pixel. Only affects new obs.')
  parser.add_argument('--reward_type',
                      type=str,
                      default=config.reward_type,
                      help='Reward function to be used during training. Options: roach, simple_reward')
  parser.add_argument('--consider_tl',
                      type=lambda x: bool(strtobool(x)),
                      default=config.consider_tl,
                      nargs='?',
                      const=True,
                      help='If set to false traffic light infractions are turned off. Used in simple reward')
  parser.add_argument('--eval_time', type=float, default=config.eval_time, help='Time until the model times out.')
  parser.add_argument('--terminal_reward',
                      type=float,
                      default=config.terminal_reward,
                      help='Time until the model times out.')
  parser.add_argument('--normalize_rewards',
                      type=lambda x: bool(strtobool(x)),
                      default=config.normalize_rewards,
                      nargs='?',
                      const=True,
                      help=' Whether to use gymnasiums reward normalization.')
  parser.add_argument('--speeding_infraction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.speeding_infraction,
                      nargs='?',
                      const=True,
                      help='Whether to terminate the route if the agent drives too fast. Only simple reward.')
  parser.add_argument('--min_thresh_lat_dist',
                      type=float,
                      default=config.min_thresh_lat_dist,
                      help='Time until the model times out.')
  parser.add_argument('--num_route_points_rendered',
                      type=int,
                      default=config.num_route_points_rendered,
                      help='Number of route points rendered into the BEV seg observation.')
  parser.add_argument('--use_green_wave',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_green_wave,
                      nargs='?',
                      const=True,
                      help='If True in some routes all TL that the agent encounters are set to green.')
  parser.add_argument('--image_encoder',
                      type=str,
                      default=config.image_encoder,
                      help='Which image cnn encoder to use. Either roach, roach_ln, or timm model name')
  parser.add_argument('--use_comfort_infraction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_comfort_infraction,
                      nargs='?',
                      const=True,
                      help='Whether to apply a soft penalty if comfort limits are exceeded.')
  parser.add_argument('--use_layer_norm',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_layer_norm,
                      nargs='?',
                      const=True,
                      help='Whether to use LayerNorm before ReLU in MLPs.')
  parser.add_argument('--use_vehicle_close_penalty',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_vehicle_close_penalty,
                      nargs='?',
                      const=True,
                      help='Whether to use a penalty for being too close to the front vehicle.')
  parser.add_argument('--render_green_tl',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_green_tl,
                      nargs='?',
                      const=True,
                      help='Whether to render green traffic lights into the observation.')
  parser.add_argument('--distribution',
                      type=str,
                      default=config.distribution,
                      help='Distribution used for the action space. Options beta, normal, beta_uni_mix')
  parser.add_argument('--weight_decay',
                      type=float,
                      default=config.weight_decay,
                      help='Weight decay applied to optimizer. AdamW is used when > 0.0')
  parser.add_argument('--use_dd_ppo_preempt',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_dd_ppo_preempt,
                      nargs='?',
                      const=True,
                      help='Whether to use the dd-ppo preemption technique to early stop stragglers.')
  parser.add_argument('--use_termination_hint',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_termination_hint,
                      nargs='?',
                      const=True,
                      help='Whether to give a penalty depending on vehicle speed when crashing or running red light.')
  parser.add_argument('--use_perc_progress',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_perc_progress,
                      nargs='?',
                      const=True,
                      help='Whether to multiply RC reward by percentage away from lane center.')
  parser.add_argument('--use_min_speed_infraction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_min_speed_infraction,
                      nargs='?',
                      const=True,
                      help='Whether to penalize the agent for driving slower than other agents on avg.')
  parser.add_argument('--use_leave_route_done',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_leave_route_done,
                      nargs='?',
                      const=True,
                      help='Whether to terminate the route when leaving the precomputed path.')
  parser.add_argument('--use_temperature',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_temperature,
                      nargs='?',
                      const=True,
                      help='Whether to scale the output distribution values with a predicted temperature.')
  parser.add_argument('--obs_num_measurements',
                      type=int,
                      default=config.obs_num_measurements,
                      help='Number of scalar measurements in observation.')
  parser.add_argument('--use_extra_control_inputs',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_extra_control_inputs,
                      nargs='?',
                      const=True,
                      help='Whether to use extra control inputs such as integral of past steering.')
  parser.add_argument('--condition_outside_junction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.condition_outside_junction,
                      nargs='?',
                      const=True,
                      help=' Whether to render the route outside junctions.')
  parser.add_argument('--use_layer_norm_policy_head',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_layer_norm_policy_head,
                      nargs='?',
                      const=True,
                      help='Applicable if use_layer_norm=True, whether to also apply layernorm to the policy head.'
                      'Can be useful to remove to allow the policy to predict large values (for a, b of Beta).')
  parser.add_argument('--use_hl_gauss_value_loss',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_hl_gauss_value_loss,
                      nargs='?',
                      const=True,
                      help='Whether to use the histogram loss gauss to train the value head via classification '
                      '(instead of regression + L2)')
  parser.add_argument('--use_outside_route_lanes',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_outside_route_lanes,
                      nargs='?',
                      const=True,
                      help='Whether to terminate the route when invading opposing lanes or sidewalks.')
  parser.add_argument('--use_max_change_penalty',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_max_change_penalty,
                      nargs='?',
                      const=True,
                      help='Whether to apply a soft penalty when the action changes too fast.')
  parser.add_argument('--use_lstm',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_lstm,
                      nargs='?',
                      const=True,
                      help=' Whether to use an LSTM after the feature encoder.')
  parser.add_argument('--terminal_hint',
                      type=float,
                      default=config.terminal_hint,
                      help='Reward at the end of the episode when colliding, the number will be subtracted.')
  parser.add_argument('--penalize_yellow_light',
                      type=lambda x: bool(strtobool(x)),
                      default=config.penalize_yellow_light,
                      nargs='?',
                      const=True,
                      help='Whether to penalize running a yellow light.')
  parser.add_argument('--use_target_point',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_target_point,
                      nargs='?',
                      const=True,
                      help='Whether to input a target point in the measurements.')
  parser.add_argument('--use_value_measurements',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_value_measurements,
                      nargs='?',
                      const=True,
                      help='Whether to use value measurements (otherwise all are set to 0)')
  parser.add_argument('--bev_semantics_width',
                      type=int,
                      default=config.bev_semantics_width,
                      help='Numer of pixels the bev_semantics is wide')
  parser.add_argument('--bev_semantics_height',
                      type=int,
                      default=config.bev_semantics_height,
                      help='Numer of pixels the bev_semantics is high')
  parser.add_argument('--pixels_ev_to_bottom',
                      type=int,
                      default=config.pixels_ev_to_bottom,
                      help='Numer of pixels from the vehicle to the bottom.')
  parser.add_argument('--use_history',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_history,
                      nargs='?',
                      const=True,
                      help='Whether to use historic vehicle and pedestrian observations in BEV obs')
  parser.add_argument('--use_off_road_term',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_off_road_term,
                      nargs='?',
                      const=True,
                      help='Whether to terminate when the agent drives off the drivable area')
  parser.add_argument('--off_road_term_perc',
                      type=float,
                      default=config.off_road_term_perc,
                      help='Percentage of agent overlap with off-road, that triggers the termination')
  parser.add_argument('--beta_1', type=float, default=config.beta_1, help='Beta 1 parameter of adam')
  parser.add_argument('--beta_2', type=float, default=config.beta_2, help='Beta 2 parameter of adam')
  parser.add_argument('--render_speed_lines',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_speed_lines,
                      nargs='?',
                      const=True,
                      help='Whether to render the speed lines for moving objects')
  parser.add_argument('--use_new_stop_sign_detector',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_new_stop_sign_detector,
                      nargs='?',
                      const=True,
                      help='Whether to use a different stop sign detector that prevents the policy from cheating by '
                      'changing lanes.')
  parser.add_argument('--use_positional_encoding',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_positional_encoding,
                      nargs='?',
                      const=True,
                      help='Whether to add positional encoding to the image')
  parser.add_argument('--use_ttc',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_ttc,
                      nargs='?',
                      const=True,
                      help='Whether to use TTC in the reward.')
  parser.add_argument('--num_value_measurements',
                      type=int,
                      default=config.num_value_measurements,
                      help='Number of measurements exclusive to the value head.')
  parser.add_argument('--render_yellow_time',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_yellow_time,
                      nargs='?',
                      const=True,
                      help='Whether to indicate the remaining time to red in yellow light rendering.')
  parser.add_argument('--use_single_reward',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_single_reward,
                      nargs='?',
                      const=True,
                      help='Whether to only use RC als reward source in simple reward')
  parser.add_argument('--use_rl_termination_hint',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_rl_termination_hint,
                      nargs='?',
                      const=True,
                      help='Whether to include rl infraction for termination hints')
  parser.add_argument('--render_shoulder',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_shoulder,
                      nargs='?',
                      const=True,
                      help='Whether to render shoulder lanes as roads.')
  parser.add_argument('--use_shoulder_channel',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_shoulder_channel,
                      nargs='?',
                      const=True,
                      help='Whether to use an extra channel for shoulder lanes.')
  parser.add_argument('--lane_distance_violation_threshold',
                      type=float,
                      default=config.lane_distance_violation_threshold,
                      help='Grace distance in m at which no lane perc penalty is applied')
  parser.add_argument('--lane_dist_penalty_softener',
                      type=float,
                      default=config.lane_dist_penalty_softener,
                      help='If smaller than 1 reduces lane distance penalty')
  parser.add_argument('--comfort_penalty_factor',
                      type=float,
                      default=config.comfort_penalty_factor,
                      help='Max comfort penalty if all comfort metrics are violated.')
  parser.add_argument('--use_survival_reward',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_survival_reward,
                      nargs='?',
                      const=True,
                      help='Whether to add a constant reward every frame')

  args, unknown = parser.parse_known_args()
  print('Unkown Arguments', unknown)
  # fmt: on
  return args


def make_env(gym_id, args, run_name, port, config):

  def thunk():
    if args.visualize:
      render_mode = 'human'  # Only works well with num envs 1 right now
    else:
      render_mode = 'rgb_array'

    env = gym.make(gym_id, port=port, config=config, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if args.capture_video:
      env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
    env = gym.wrappers.ClipAction(env)
    if config.normalize_rewards:
      env = gym.wrappers.NormalizeReward(env, gamma=config.gamma)
      env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

  return thunk


def main():
  register(
      id='CARLAEnv-v0',
      entry_point='env_gym:CARLAEnv',
      max_episode_steps=None,
  )
  config = GlobalConfig()
  args = parse_args(config)

  # Torchrun initialization
  # Use torchrun for starting because it has proper error handling. Local rank will be set automatically
  rank = int(os.environ['RANK'])  # Rank across all processes
  local_rank = int(os.environ['LOCAL_RANK'])  # Rank on Node
  world_size = int(os.environ['WORLD_SIZE'])  # Number of processes
  print(f'RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}')

  local_batch_size = args.total_batch_size // world_size
  local_bs_per_env = local_batch_size // args.num_envs_per_proc
  local_minibatch_size = args.total_minibatch_size // world_size
  num_minibatches = local_batch_size // local_minibatch_size

  run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}'
  if rank == 0:
    exp_folder = os.path.join(args.logdir, f'{args.exp_name}')
    wandb_folder = os.path.join(exp_folder, 'wandb')
    pathlib.Path(exp_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(wandb_folder).mkdir(parents=True, exist_ok=True)
    if args.track:

      wandb.init(
          project=args.wandb_project_name,
          entity=args.wandb_entity,
          sync_tensorboard=True,
          config=vars(args),
          name=run_name,
          monitor_gym=False,
          allow_val_change=True,
          save_code=False,
          mode='online',
          resume='auto',
          dir=wandb_folder,
          settings=wandb.Settings(_disable_stats=True, _disable_meta=True,
                                  start_method='fork')  # Can get large if we log all the cpu cores.
      )

    writer = SummaryWriter(exp_folder)
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )

  # TRY NOT TO MODIFY: seeding
  random.seed(args.seed * world_size + rank)
  np.random.seed(args.seed * world_size + rank)
  torch.manual_seed(args.seed * world_size + rank)

  print('Is cuda available?:', torch.cuda.is_available())

  # Load the config before overwriting values with current arguments
  if args.load_file is not None:
    load_folder = pathlib.Path(args.load_file).parent.resolve()
    with open(os.path.join(load_folder, 'config.json'), 'rt', encoding='utf-8') as f:
      json_config = f.read()
    # 4 ms, might need to move outside the agent.
    loaded_config = jsonpickle.decode(json_config)
    # Overwrite all properties that were set in the saved config.
    config.__dict__.update(loaded_config.__dict__)

  # Configure config. Converts all arguments into config attributes
  config.initialize(**vars(args))

  if config.use_dd_ppo_preempt:
    # Gloo is used for the parallelization strategy where multiple processes with 1 env run on 1 GPU because NCCL
    # does not support this use case currently.
    num_rollouts_done_store = torch.distributed.TCPStore(args.rdzv_addr, args.tcp_store_port, world_size, rank == 0)
    torch.distributed.init_process_group(backend='gloo' if
                                         (args.cpu_collect or args.num_envs_per_proc == 1) else 'nccl',
                                         store=num_rollouts_done_store,
                                         world_size=world_size,
                                         rank=rank,
                                         timeout=datetime.timedelta(minutes=45))
    num_rollouts_done_store.set('num_done', '0')
    print(f'Rank:{rank}, TCP_Store_Port: {args.tcp_store_port}')
  else:
    torch.distributed.init_process_group(backend='gloo' if
                                         (args.cpu_collect or args.num_envs_per_proc == 1) else 'nccl',
                                         init_method='env://',
                                         world_size=world_size,
                                         rank=rank,
                                         timeout=datetime.timedelta(minutes=45))

  device_id = args.gpu_ids[rank]
  print(device_id)
  if device_id < 0:
    print('ERROR! Device id must be positive.')

  device = torch.device(f'cuda:{args.gpu_ids[rank]}') if torch.cuda.is_available() and args.cuda else torch.device(
      'cpu')

  if torch.cuda.is_available() and args.cuda:
    torch.cuda.device(device)

  torch.backends.cudnn.deterministic = args.torch_deterministic
  torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
  torch.backends.cudnn.benchmark = args.benchmark
  torch.backends.cudnn.allow_tf32 = args.allow_tf32
  torch.set_float32_matmul_precision(args.matmul_precision)

  if rank == 0:
    json_config = jsonpickle.encode(config)

    with open(os.path.join(exp_folder, 'config.json'), 'w', encoding='utf-8') as f2:
      f2.write(json_config)

  # NOTE: need to update the config with the argparse arguments before creating the gym environment because the gym env
  # will make a copy of the config and send it to the carla leaderboard process.
  # Sends the updated config to the different environments.
  context = zmq.Context()
  socket_list = []  # So that the garbage collector doesn't clean up sockets before we are done.
  for i in range(args.num_envs_per_proc):

    socket_list.append(context.socket(zmq.PAIR))

    current_folder = pathlib.Path(__file__).parent.resolve()
    comm_folder = os.path.join(current_folder, 'comm_files')
    pathlib.Path(comm_folder).mkdir(parents=True, exist_ok=True)
    communication_file = os.path.join(comm_folder, str(args.ports[args.num_envs_per_proc * local_rank + i]))
    socket_list[i].bind(f'ipc://{communication_file}.conf_lock')
    json_config = jsonpickle.encode(config)
    socket_list[i].send_string(json_config)
    _ = socket_list[i].recv_string()
    socket_list[i].close()

  context.term()
  del socket_list

  print(f'Rank {rank}, sent CONFIG files')

  if args.num_envs_per_proc > 1:
    env = gym.vector.AsyncVectorEnv([
        make_env(args.gym_id, args, run_name, args.ports[args.num_envs_per_proc * local_rank + i], config)
        for i in range(args.num_envs_per_proc)
    ])
  else:
    # We don't want to spawn subprocesses for a single environment, so we use Sync==Sequential Vector env.
    env = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args, run_name, args.ports[args.num_envs_per_proc * local_rank + i], config)])
  assert isinstance(env.single_action_space, gym.spaces.Box), 'only continuous action space is supported'

  agent = PPOPolicy(env.single_observation_space, env.single_action_space, config=config).to(device)

  if config.compile_model:
    agent = torch.compile(agent)

  start_step = 0
  if args.load_file is not None:
    load_file_name = os.path.basename(args.load_file)
    algo_step = re.findall(r'\d+', load_file_name)
    if len(algo_step) > 0:
      start_step = int(algo_step[0]) + 1  # That step was already finished.
      print('Start training from step:', start_step)
    agent.load_state_dict(torch.load(args.load_file, map_location=device), strict=True)

  print(f'Rank {rank}, START DistributedDataParallel')
  agent = torch.nn.parallel.DistributedDataParallel(agent,
                                                    device_ids=None,
                                                    output_device=None,
                                                    broadcast_buffers=False,
                                                    find_unused_parameters=False)
  print(f'Rank:{rank}, Created DistributedDataParallel')
  # If we are resuming training use last learning rate from config.
  # If we start a fresh training set the current learning rate according to arguments.
  if args.load_file is None:
    config.current_learning_rate = args.learning_rate

  if rank == 0:
    model_parameters = filter(lambda p: p.requires_grad, agent.parameters())
    num_params = sum(np.prod(p.size()) for p in model_parameters)

    print('Total trainable parameters: ', num_params)

  if config.weight_decay > 0.0:
    optimizer = optim.AdamW(agent.parameters(),
                            lr=config.current_learning_rate,
                            eps=config.adam_eps,
                            weight_decay=config.weight_decay,
                            betas=(config.beta_1, config.beta_2))
  else:
    optimizer = optim.Adam(agent.parameters(),
                           lr=config.current_learning_rate,
                           eps=config.adam_eps,
                           betas=(config.beta_1, config.beta_2))

  print(f'Rank:{rank}, Created Optimizer')

  # Load optimizer
  if args.load_file is not None:
    optimizer.load_state_dict(torch.load(args.load_file.replace('model_', 'optimizer_'), map_location=device))
    print(f'Rank:{rank}, Load model')
    if rank == 0:
      writer.add_scalar('charts/restart', 1, config.global_step)  # Log that a restart happened

  if config.cpu_collect:
    device = 'cpu'

  # ALGO Logic: Storage setup
  obs = {
      'bev_semantics':
          torch.zeros(
              (local_bs_per_env, args.num_envs_per_proc) + env.single_observation_space.spaces['bev_semantics'].shape,
              device=device,
              dtype=torch.uint8),
      'measurements':
          torch.zeros(
              (local_bs_per_env, args.num_envs_per_proc) + env.single_observation_space.spaces['measurements'].shape,
              device=device),
      'value_measurements':
          torch.zeros((local_bs_per_env, args.num_envs_per_proc) +
                      env.single_observation_space.spaces['value_measurements'].shape,
                      device=device),
  }
  actions = torch.zeros((local_bs_per_env, args.num_envs_per_proc) + env.single_action_space.shape, device=device)
  old_mus = torch.zeros((local_bs_per_env, args.num_envs_per_proc) + env.single_action_space.shape, device=device)
  old_sigmas = torch.zeros((local_bs_per_env, args.num_envs_per_proc) + env.single_action_space.shape, device=device)
  logprobs = torch.zeros((local_bs_per_env, args.num_envs_per_proc), device=device)
  rewards = torch.zeros((local_bs_per_env, args.num_envs_per_proc), device=device)
  dones = torch.zeros((local_bs_per_env, args.num_envs_per_proc), device=device)
  values = torch.zeros((local_bs_per_env, args.num_envs_per_proc), device=device)
  exp_n_steps = np.zeros((local_bs_per_env, args.num_envs_per_proc), dtype=np.int32)
  exp_suggest = np.zeros((local_bs_per_env, args.num_envs_per_proc), dtype=np.int32)

  # TRY NOT TO MODIFY: start the game
  reset_obs = env.reset(seed=[args.seed + rank * args.num_envs_per_proc + i for i in range(args.num_envs_per_proc)])
  next_obs = {
      'bev_semantics': torch.tensor(reset_obs[0]['bev_semantics'], device=device, dtype=torch.uint8),
      'measurements': torch.tensor(reset_obs[0]['measurements'], device=device, dtype=torch.float32),
      'value_measurements': torch.tensor(reset_obs[0]['value_measurements'], device=device, dtype=torch.float32)
  }
  next_done = torch.zeros(args.num_envs_per_proc, device=device)
  next_lstm_state = (
      torch.zeros(config.num_lstm_layers, args.num_envs_per_proc, config.features_dim, device=device),
      torch.zeros(config.num_lstm_layers, args.num_envs_per_proc, config.features_dim, device=device),
  )
  num_updates = args.total_timesteps // args.total_batch_size
  local_processed_samples = 0
  start_time = time.time()
  # TODO This doesn't affect our agent, but it should probably be eval for data collection and train for training
  agent.train()

  if rank == 0:
    avg_returns = deque(maxlen=100)
  # from matplotlib import pyplot as plt
  # pyplots = []
  # fig = plt.figure(figsize=(config.obs_num_channels + 1, config.obs_num_channels + 1))
  # for i in range(1, config.obs_num_channels + 1):
  #   fig.add_subplot(4, 4, i)
  #   pyplots.append(plt.imshow(np.zeros((192, 192)), cmap='gray', vmin=0, vmax=255))
  # plt.show(block=False)

  if config.use_hl_gauss_value_loss:
    hl_gauss_bins = rl_u.hl_gaus_bins(config.hl_gauss_vmin, config.hl_gauss_vmax, config.hl_gauss_bucket_size, device)

  print(f'Rank:{rank}, Created obs', flush=True)

  for update in tqdm(range(start_step, num_updates), disable=rank != 0):
    if config.cpu_collect:
      device = 'cpu'
      agent.to(device)
    # Free all data from last interation.
    gc.collect()
    with torch.no_grad():
      torch.cuda.empty_cache()

    if config.use_dd_ppo_preempt:
      num_rollouts_done_store.set('num_done', '0')

    # Buffers we use to store returns and aggregate them later to rank 0 for logging.
    total_returns = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
    total_lengths = torch.zeros(world_size, device=device, dtype=torch.float32, requires_grad=False)
    num_total_returns = torch.zeros(world_size, device=device, dtype=torch.int32, requires_grad=False)

    if config.use_lstm:
      initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

    # Annealing the rate if instructed to do so.
    if config.lr_schedule == 'linear':
      frac = 1.0 - (update - 1.0) / num_updates
      config.current_learning_rate = frac * config.learning_rate
    elif config.lr_schedule == 'step':
      frac = update / num_updates
      lr_multiplier = 1.0
      for change_percentage in config.lr_schedule_step_perc:
        if frac > change_percentage:
          lr_multiplier *= config.lr_schedule_step_factor
      config.current_learning_rate = lr_multiplier * config.learning_rate
    elif config.lr_schedule == 'cosine':
      frac = update / num_updates
      config.current_learning_rate = 0.5 * config.learning_rate * (1 + math.cos(frac * math.pi))
    elif config.lr_schedule == 'cosine_restart':
      frac = update / (num_updates + 1)  # + 1 so it doesn't become 100 %
      for idx, frac_restart in enumerate(config.lr_schedule_cosine_restarts):
        if frac >= frac_restart:
          current_idx = idx
      base_frac = config.lr_schedule_cosine_restarts[current_idx]
      length_current_interval = (config.lr_schedule_cosine_restarts[current_idx + 1] -
                                 config.lr_schedule_cosine_restarts[current_idx])
      frac_current_iter = (frac - base_frac) / length_current_interval
      config.current_learning_rate = 0.5 * config.learning_rate * (1 + math.cos(frac_current_iter * math.pi))

    for param_group in optimizer.param_groups:
      param_group['lr'] = config.current_learning_rate

    t0 = TicToc()  # Data collect
    t1 = TicToc()  # Forward pass
    t2 = TicToc()  # Env step
    t3 = TicToc()  # Pre-processing
    t4 = TicToc()  # Train inter
    t5 = TicToc()  # Logging
    t0.tic()
    inference_times = []
    env_times = []
    for step in range(0, local_bs_per_env):
      config.global_step += 1 * world_size * args.num_envs_per_proc
      local_processed_samples += 1 * world_size * args.num_envs_per_proc

      obs['bev_semantics'][step] = next_obs['bev_semantics']
      obs['measurements'][step] = next_obs['measurements']
      obs['value_measurements'][step] = next_obs['value_measurements']
      dones[step] = next_done

      # ALGO LOGIC: action logic
      with torch.no_grad():
        t1.tic()
        action, logprob, _, value, _, mu, sigma, _, _, _, next_lstm_state = agent.forward(next_obs,
                                                                                          lstm_state=next_lstm_state,
                                                                                          done=next_done)
        if config.use_hl_gauss_value_loss:
          value_pdf = F.softmax(value, dim=1)
          value = torch.sum(value_pdf * hl_gauss_bins.unsqueeze(0), dim=1)
        inference_times.append(t1.tocvalue())
        values[step] = value.flatten()
      actions[step] = action
      logprobs[step] = logprob
      old_mus[step] = mu
      old_sigmas[step] = sigma

      # TRY NOT TO MODIFY: execute the game and log data.
      t2.tic()
      next_obs, reward, termination, truncation, info = env.step(action.cpu().numpy())
      env_times.append(t2.tocvalue())

      # for i in range(config.obs_num_channels):
      #   pyplots[i].set_data(next_obs['bev_semantics'][0, i])
      # plt.pause(0.0001)

      done = np.logical_or(termination, truncation)  # Not treated separately in original PPO
      rewards[step] = torch.tensor(reward, device=device, dtype=torch.float32)
      next_done = torch.tensor(done, device=device, dtype=torch.float32)
      next_obs = {
          'bev_semantics': torch.tensor(next_obs['bev_semantics'], device=device, dtype=torch.uint8),
          'measurements': torch.tensor(next_obs['measurements'], device=device, dtype=torch.float32),
          'value_measurements': torch.tensor(next_obs['value_measurements'], device=device, dtype=torch.float32)
      }

      if 'final_info' in info.keys():
        for idx, single_info in enumerate(info['final_info']):
          if single_info is not None:
            if config.use_exploration_suggest:
              # Exploration loss
              exp_n_steps[step, idx] = single_info['n_steps']
              exp_suggest[step, idx] = single_info['suggest']

            # Sum up total returns and how often the env was reset during this iteration.
            if 'episode' in single_info.keys():
              print(f'rank: {rank}, config.global_step={config.global_step}, '
                    f'episodic_return={single_info["episode"]["r"]}')
              total_returns[rank] += single_info['episode']['r'].item()
              total_lengths[rank] += single_info['episode']['l'].item()
              num_total_returns[rank] += 1

      if config.use_dd_ppo_preempt:
        num_done = int(num_rollouts_done_store.get('num_done'))
        min_steps = int(config.dd_ppo_min_perc * local_bs_per_env)
        if (num_done / world_size) > config.dd_ppo_preempt_threshold and step > min_steps:
          print(f'Rank:{rank}, Preempt at step: {step}, Num done: {num_done}')
          break  # End data collection early the other workers are finished.

    t0.toc(msg=f'Rank:{rank}, Data collection.')
    print(f'Rank:{rank}, Avg forward time {sum(inference_times)}')
    print(f'Rank:{rank}, Avg env time {sum(env_times)}')
    t3.tic()

    if config.use_dd_ppo_preempt:
      num_rollouts_done_store.add('num_done', 1)

    # In case of a dd-ppo preempt this can be smaller than local batch size
    num_collected_steps = step + 1

    # bootstrap value if not done
    with torch.no_grad():
      if config.use_hl_gauss_value_loss:
        next_value = agent.module.get_value(next_obs, next_lstm_state, next_done)
        value_pdf = F.softmax(next_value, dim=1)
        next_value = torch.sum(value_pdf * hl_gauss_bins.unsqueeze(0), dim=1)
      else:
        next_value = agent.module.get_value(next_obs, next_lstm_state, next_done).squeeze(1)
      if args.gae:
        advantages = torch.zeros_like(rewards, device=device)
        lastgaelam = 0.0
        for t in reversed(range(num_collected_steps)):
          if t == local_bs_per_env - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
          else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
          delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
          advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
      else:
        returns = torch.zeros_like(rewards, device=device)
        for t in reversed(range(num_collected_steps)):
          if t == local_bs_per_env - 1:
            nextnonterminal = 1.0 - next_done
            next_return = next_value
          else:
            nextnonterminal = 1.0 - dones[t + 1]
            next_return = returns[t + 1]
          returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
        advantages = returns - values

    if config.cpu_collect:
      device = torch.device(f'cuda:{args.gpu_ids[rank]}') if torch.cuda.is_available() and args.cuda else torch.device(
          'cpu')
      agent.to(device)

    exploration_suggests = np.zeros((num_collected_steps, args.num_envs_per_proc), dtype=np.int32)
    if config.use_exploration_suggest:
      for step in range(num_collected_steps):
        for idx in range(args.num_envs_per_proc):
          n_steps = exp_n_steps[step, idx]
          if n_steps > 0:
            n_start = max(0, step - n_steps)
            exploration_suggests[n_start:step, idx] = exp_suggest[step, idx]

    b_obs = {
        'bev_semantics':
            obs['bev_semantics']
            [:num_collected_steps].reshape((-1,) + env.single_observation_space.spaces['bev_semantics'].shape),
        'measurements':
            obs['measurements']
            [:num_collected_steps].reshape((-1,) + env.single_observation_space.spaces['measurements'].shape),
        'value_measurements':
            obs['value_measurements']
            [:num_collected_steps].reshape((-1,) + env.single_observation_space.spaces['value_measurements'].shape)
    }
    b_logprobs = logprobs[:num_collected_steps].reshape(-1)
    b_actions = actions[:num_collected_steps].reshape((-1,) + env.single_action_space.shape)
    b_dones = dones[:num_collected_steps].reshape(-1)  # TODO check if pre-emption trick causes problems with LSTM.
    b_advantages = advantages[:num_collected_steps].reshape(-1)
    b_returns = returns[:num_collected_steps].reshape(-1)
    b_values = values[:num_collected_steps].reshape(-1)
    b_old_mus = old_mus[:num_collected_steps].reshape((-1,) + env.single_action_space.shape)
    b_old_sigmas = old_sigmas[:num_collected_steps].reshape((-1,) + env.single_action_space.shape)
    b_exploration_suggests = exploration_suggests[:num_collected_steps].reshape(-1)

    # When the data was collected on the CPU, move it to GPU before training
    if config.cpu_collect:
      b_obs['bev_semantics'] = b_obs['bev_semantics'].to(device)
      b_obs['measurements'] = b_obs['measurements'].to(device)
      b_obs['value_measurements'] = b_obs['value_measurements'].to(device)
      b_logprobs = b_logprobs.to(device)
      b_actions = b_actions.to(device)
      b_dones = b_dones.to(device)
      b_advantages = b_advantages.to(device)
      b_returns = b_returns.to(device)
      b_values = b_values.to(device)
      b_old_mus = b_old_mus.to(device)
      b_old_sigmas = b_old_sigmas.to(device)

    # Synchronize all processes
    torch.distributed.barrier()
    # Aggregate returns to GPU 0 for logging and storing the best model.
    # Gloo doesn't support AVG, so we implement it via sum / num returns
    torch.distributed.all_reduce(total_returns, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_lengths, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(num_total_returns, op=torch.distributed.ReduceOp.SUM)

    if rank == 0:
      num_total_returns_all_processes = torch.sum(num_total_returns)
      # Only can log return if there was any episode that finished
      if num_total_returns_all_processes > 0:
        total_returns_all_processes = torch.sum(total_returns)
        total_lengths_all_processes = torch.sum(total_lengths)
        avg_return = total_returns_all_processes / num_total_returns_all_processes
        avg_return = avg_return.item()
        avg_length = total_lengths_all_processes / num_total_returns_all_processes
        avg_length = avg_length.item()

        avg_returns.append(avg_return)
        windowed_avg_return = sum(avg_returns) / len(avg_returns)

        writer.add_scalar('charts/episodic_return', avg_return, config.global_step)
        writer.add_scalar('charts/windowed_avg_return', windowed_avg_return, config.global_step)
        writer.add_scalar('charts/episodic_length', avg_length, config.global_step)
        if windowed_avg_return >= config.max_training_score:
          config.max_training_score = windowed_avg_return
          # Same model could reach multiple high scores
          if config.best_iteration != update:
            config.best_iteration = update
            save(agent, None, config, exp_folder, 'model_best.pth', None)

    # Optimizing the policy and value network
    if config.use_lstm:
      assert args.num_envs_per_proc % num_minibatches == 0
      assert not config.use_dd_ppo_preempt

      envsperbatch = args.num_envs_per_proc // num_minibatches
      envinds = np.arange(args.num_envs_per_proc)
      flatinds = np.arange(local_batch_size).reshape(local_bs_per_env, args.num_envs_per_proc)

    b_inds_original = np.arange(num_collected_steps * args.num_envs_per_proc)

    if config.use_dd_ppo_preempt:
      b_inds_original = np.resize(b_inds_original, (local_batch_size,))

    clipfracs = []

    # Free all data accumulated from data collection.
    gc.collect()
    with torch.no_grad():
      torch.cuda.empty_cache()

    t3.toc(msg=f'Rank:{rank}, Data pre-processing.')
    t4.tic()
    for latest_epoch in range(args.update_epochs):
      approx_kl_divs = []
      if config.use_lstm:
        np.random.shuffle(envinds)
      else:
        p = np.random.permutation(len(b_inds_original))
        b_inds = b_inds_original[p]

      total_steps = local_batch_size
      step_size = local_minibatch_size
      if config.use_lstm:
        total_steps = args.num_envs_per_proc
        step_size = envsperbatch

      for start in range(0, total_steps, step_size):
        if config.use_lstm:
          end = start + envsperbatch
          mbenvinds = envinds[start:end]
          lstm_state = (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds])
          mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
        else:
          end = start + local_minibatch_size
          mb_inds = b_inds[start:end]
          lstm_state = None

        if config.use_exploration_suggest:
          b_exploration_suggests_sampled = b_exploration_suggests[mb_inds]
        else:
          b_exploration_suggests_sampled = None
        b_obs_sampled = {
            'bev_semantics': b_obs['bev_semantics'][mb_inds],
            'measurements': b_obs['measurements'][mb_inds],
            'value_measurements': b_obs['value_measurements'][mb_inds]
        }
        # Don't need action, so we don't unscale
        _, newlogprob, entropy, newvalue, exploration_loss, _, _, distribution, pred_sem, pred_measure, _ = \
          agent.forward(
            b_obs_sampled,
            actions=b_actions[mb_inds],
            exploration_suggests=b_exploration_suggests_sampled,
            lstm_state=lstm_state,
            done=b_dones[mb_inds])
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        mb_advantages = b_advantages[mb_inds]
        if args.norm_adv:
          with torch.no_grad():
            # Distributed mean
            advantage_mean = mb_advantages.mean()
            torch.distributed.all_reduce(advantage_mean, op=torch.distributed.ReduceOp.SUM)
            advantage_mean = advantage_mean / world_size

            # Distributed std
            advantage_std = torch.sum(torch.square(mb_advantages - advantage_mean))
            torch.distributed.all_reduce(advantage_std, op=torch.distributed.ReduceOp.SUM)
            advantage_std = advantage_std / (world_size * torch.numel(mb_advantages) - 1)  # -1 is bessel's correction
            advantage_std = torch.sqrt(advantage_std)

            mb_advantages = (mb_advantages - advantage_mean) / (advantage_std + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        if args.clip_vloss:
          # Value clipping is not implemented with HL_Gauss loss
          assert config.use_hl_gauss_value_loss is False
          newvalue = newvalue.view(-1)
          v_clipped = b_values[mb_inds] + torch.clamp(
              newvalue - b_values[mb_inds],
              -args.clip_coef,
              args.clip_coef,
          )
          v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
          v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
          v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
          v_loss = 0.5 * v_loss_max.mean()
        else:
          if config.use_hl_gauss_value_loss:
            target_pdf = rl_u.hl_gaus_pdf(b_returns[mb_inds], config.hl_gauss_std, config.hl_gauss_vmin,
                                          config.hl_gauss_vmax, config.hl_gauss_bucket_size)
            v_loss = F.cross_entropy(newvalue, target_pdf)
          else:
            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
        if config.use_exploration_suggest:
          loss = loss + args.expl_coef * exploration_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        old_mu_sampled = b_old_mus[mb_inds]
        old_sigmas_sampled = b_old_sigmas[mb_inds]
        with torch.no_grad():
          # calculate approx_kl http://joschu.net/blog/kl-approx.html
          old_approx_kl = (-logratio).mean()
          #approx_kl = ((ratio - 1) - logratio).mean()

          # We compute approx KL according to roach
          old_distribution = agent.module.action_dist.proba_distribution(old_mu_sampled, old_sigmas_sampled)
          kl_div = torch.distributions.kl_divergence(old_distribution.distribution, distribution)
          approx_kl_divs.append(kl_div.mean())

          clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean()]

      approx_kl = torch.mean(torch.stack(approx_kl_divs))
      # Gloo doesn't support AVG, so we implement it via sum / world size
      torch.distributed.barrier()
      torch.distributed.all_reduce(approx_kl, op=torch.distributed.ReduceOp.SUM)
      approx_kl = approx_kl / world_size
      if args.target_kl is not None and config.lr_schedule == 'kl':
        if approx_kl > args.target_kl:
          if config.lr_schedule_step is not None:
            config.kl_early_stop += 1
            if config.kl_early_stop >= config.lr_schedule_step:
              config.current_learning_rate *= 0.5
              config.kl_early_stop = 0

          break

    del b_obs  # Remove large array
    t4.toc(msg=f'Rank:{rank}, Training.')
    t5.tic()

    config.latest_iteration = update
    # Avg value to log over all Environments
    # Sync with 3 envs takes 4 ms.
    torch.distributed.barrier()
    # Gloo doesn't support AVG, so we implement it via sum / world size
    torch.distributed.all_reduce(v_loss, op=torch.distributed.ReduceOp.SUM)
    v_loss = v_loss / world_size

    torch.distributed.all_reduce(pg_loss, op=torch.distributed.ReduceOp.SUM)
    pg_loss = pg_loss / world_size

    torch.distributed.all_reduce(entropy_loss, op=torch.distributed.ReduceOp.SUM)
    entropy_loss = entropy_loss / world_size

    if config.use_exploration_suggest:
      torch.distributed.all_reduce(exploration_loss, op=torch.distributed.ReduceOp.SUM)
      exploration_loss = exploration_loss / world_size

    torch.distributed.all_reduce(old_approx_kl, op=torch.distributed.ReduceOp.SUM)
    old_approx_kl = old_approx_kl / world_size

    torch.distributed.all_reduce(approx_kl, op=torch.distributed.ReduceOp.SUM)
    approx_kl = approx_kl / world_size

    b_values = b_values[b_inds_original]
    torch.distributed.all_reduce(b_values, op=torch.distributed.ReduceOp.SUM)
    b_values = b_values / world_size

    b_returns = b_returns[b_inds_original]
    torch.distributed.all_reduce(b_returns, op=torch.distributed.ReduceOp.SUM)
    b_returns = b_returns / world_size

    b_advantages = b_advantages[b_inds_original]
    torch.distributed.all_reduce(b_advantages, op=torch.distributed.ReduceOp.SUM)
    b_advantages = b_advantages / world_size

    clipfracs = torch.mean(torch.stack(clipfracs))
    torch.distributed.all_reduce(clipfracs, op=torch.distributed.ReduceOp.SUM)
    clipfracs = clipfracs / world_size

    if rank == 0:
      save(agent, optimizer, config, exp_folder, f'model_latest_{update:09d}.pth', f'optimizer_latest_{update:09d}.pth')
      frac = update / num_updates
      if config.current_eval_interval_idx < len(config.eval_intervals):
        if frac >= config.eval_intervals[config.current_eval_interval_idx]:
          save(agent, None, config, exp_folder, f'model_eval_{update:09d}.pth', None)
          config.current_eval_interval_idx += 1

      # Cleanup file from last epoch
      for file in os.listdir(exp_folder):
        if file.startswith('model_latest_') and file.endswith('.pth'):
          if file != f'model_latest_{update:09d}.pth':
            old_model_file = os.path.join(exp_folder, file)
            if os.path.isfile(old_model_file):
              os.remove(old_model_file)
        if file.startswith('optimizer_latest_') and file.endswith('.pth'):
          if file != f'optimizer_latest_{update:09d}.pth':
            old_model_file = os.path.join(exp_folder, file)
            if os.path.isfile(old_model_file):
              os.remove(old_model_file)

      y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
      var_y = np.var(y_true)
      explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

      # TRY NOT TO MODIFY: record rewards for plotting purposes
      writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], config.global_step)
      writer.add_scalar('losses/value_loss', v_loss.item(), config.global_step)
      writer.add_scalar('losses/policy_loss', pg_loss.item(), config.global_step)
      writer.add_scalar('losses/entropy', entropy_loss.item(), config.global_step)
      if config.use_exploration_suggest:
        writer.add_scalar('losses/exploration', exploration_loss.item(), config.global_step)
      writer.add_scalar('losses/old_approx_kl', old_approx_kl.item(), config.global_step)
      writer.add_scalar('losses/approx_kl', approx_kl.item(), config.global_step)
      writer.add_scalar('losses/clipfrac', clipfracs.item(), config.global_step)
      writer.add_scalar('losses/explained_variance', explained_var, config.global_step)
      writer.add_scalar('losses/latest_epoch', latest_epoch, config.global_step)
      writer.add_scalar('charts/discounted_returns', b_returns.mean().item(), config.global_step)
      writer.add_scalar('charts/advantages', b_advantages.mean().item(), config.global_step)
      # Adjusted so it doesn't count the first epoch which is slower than the rest (converges faster)
      print('SPS:', int(local_processed_samples / (time.time() - start_time)))
      writer.add_scalar('charts/SPS', int(local_processed_samples / (time.time() - start_time)), config.global_step)
      writer.add_scalar('charts/restart', 0, config.global_step)

    t5.toc(msg=f'Rank:{rank}, Logging')

  env.close()
  if rank == 0:
    writer.close()

    save(agent, optimizer, config, exp_folder, 'model_final.pth', 'optimizer_final.pth')
    wandb.finish(exit_code=0, quiet=True)
    print('Done training.')


if __name__ == '__main__':
  main()
