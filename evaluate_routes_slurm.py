"""
Evaluates a driving model on a set of CARLA routes wherein each route is evaluated on a separate machine in parallel.
This script generates the necessary shell files to run this on a SLURM cluster.
It also monitors the evaluation and resubmits crashed routes.
At the end all results files are aggregated and parsed.
Best run inside a tmux terminal.
"""

import subprocess
import time
from pathlib import Path
import os
import fnmatch
import ujson
import argparse
import sys

# Our centOS is missing some c libraries.
# Usually miniconda has them, so we tell the linker to look there as well.
newlib = '/mnt/lustre/work/geiger/bjaeger25/home/miniconda3/lib'
if not newlib in os.environ.get('LD_LIBRARY_PATH', None):
  os.environ['LD_LIBRARY_PATH'] += ':' + newlib


def create_run_eval_bash(bash_save_dir, results_save_dir, route_path, route, checkpoint, logs_save_dir,
                         carla_tm_port_start, carla_root, seed, team_code, use_cpp, cpp_port_start, code_root,
                         sample_type, high_freq_inference, ppo_cpp_install_path, path_to_singularity,
                         pytorch_kernel_cache_path, record):
  Path(f'{results_save_dir}').mkdir(parents=True, exist_ok=True)
  with open(f'{bash_save_dir}/eval_{route}.sh', 'w', encoding='utf-8') as rsh:
    rsh.write(f'''\
export CARLA_ROOT={carla_root}
export CARLA_SERVER=${{CARLA_ROOT}}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=original_leaderboard/scenario_runner
export LEADERBOARD_ROOT=original_leaderboard/leaderboard
export PYTHONPATH="${{SCENARIO_RUNNER_ROOT}}":"${{LEADERBOARD_ROOT}}":${{PYTHONPATH}}
''')
    rsh.write(f"""
export PORT=$1
echo 'World Port:' $PORT
export TM_PORT=`comm -23 <(seq {carla_tm_port_start} {carla_tm_port_start+49} | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'TM Port:' $TM_PORT
export ROUTES={route_path}{route}.xml
export TEAM_AGENT={team_code}/eval_agent.py
export TEAM_CONFIG={code_root}/{team_code}/checkpoints/{checkpoint}/
export CHALLENGE_TRACK_CODENAME=MAP
export REPETITIONS=1
export RESUME=1
export SEED={seed}
export CHECKPOINT_ENDPOINT={results_save_dir}/{route}.json
export DEBUG_ENV_AGENT=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CPP={use_cpp}
export CPP_PORT=`comm -23 <(seq {cpp_port_start} {cpp_port_start+49} | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'C++ Port:' $CPP_PORT
export PPO_CPP_INSTALL_PATH={ppo_cpp_install_path}
export PATH_TO_SINGULARITY={path_to_singularity}
export PYTORCH_KERNEL_CACHE_PATH={pytorch_kernel_cache_path}
export RECORD={record}
export HIGH_FREQ_INFERENCE={high_freq_inference}
export SAMPLE_TYPE={sample_type}
export SAVE_PATH={logs_save_dir}
""")
    rsh.write('''
python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=0 \
--traffic-manager-seed=${SEED} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--timeout=900 \
--traffic-manager-port=${TM_PORT}
''')


def make_jobsub_file(commands, job_number, exp_name, exp_root_name, partition):
  os.makedirs(f'evaluation/{exp_root_name}/{exp_name}/run_files/logs', exist_ok=True)
  os.makedirs(f'evaluation/{exp_root_name}/{exp_name}/run_files/job_files', exist_ok=True)
  job_file = f'evaluation/{exp_root_name}/{exp_name}/run_files/job_files/{job_number}.sh'
  qsub_template = f"""#!/bin/bash
#SBATCH --job-name={exp_name}{job_number}
#SBATCH --partition={partition}
#SBATCH -o evaluation/{exp_root_name}/{exp_name}/run_files/logs/qsub_out{job_number}.log
#SBATCH -e evaluation/{exp_root_name}/{exp_name}/run_files/logs/qsub_err{job_number}.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00-06:00
#SBATCH --gres=gpu:1
"""
  for cmd in commands:
    qsub_template = qsub_template + f"""
{cmd}

"""

  with open(job_file, 'w', encoding='utf-8') as f:
    f.write(qsub_template)
  return job_file


# All jobs of the user.
def get_num_jobs(job_name, username):
  len_usrn = len(username)
  num_running_jobs = int(
      subprocess.check_output(
          f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | wc -l",
          shell=True,
      ).decode('utf-8').replace('\n', ''))
  with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
    max_num_parallel_jobs = int(f.read())

  return num_running_jobs, max_num_parallel_jobs

# All jobs of the evaluation.
def get_num_jobs_eval(job_name, username):
  len_usrn = len(username)
  num_running_jobs = int(
      subprocess.check_output(
          f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
          shell=True,
      ).decode('utf-8').replace('\n', ''))
  with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
    max_num_parallel_jobs = int(f.read())

  return num_running_jobs, max_num_parallel_jobs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--benchmark', type=str, default='roach_leaderboard',
                      help='Route files need to be stored in {benchmark}_split folder'
                           'Options: roach_town_01_02, roach_leaderboard')
  parser.add_argument('--experiment',
                      type=str,
                      required=True,
                      help='Name of folder where the model files are stored in e.g. PPO_020_00')
  parser.add_argument('--model_dir',
                      type=str,
                      default='/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results',
                      help='Folder containing all the experiment folders.')
  parser.add_argument('--code_root',
                      type=str,
                      default='/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla',
                      help='Root folder containing all the code folders.')
  parser.add_argument('--carla_root',
                      type=str,
                      default='/mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15',
                      help='Directory of the CARLA installation')
  parser.add_argument('--partition',
                      type=str,
                      default='2080-galvani',
                      help='Slurm partition to run the job on.')
  parser.add_argument('--username',
                      type=str,
                      default='bjaeger25',
                      help='Slurm username')
  parser.add_argument('--epochs',
                      nargs='+',
                      default=('model_best', 'model_final'),
                      type=str,
                      help='Model names to be evaluated')
  parser.add_argument('--team_code',
                      type=str,
                      default='team_code_roach',
                      help='Which team code folder to use')
  parser.add_argument('--num_repetitions',
                      type=int,
                      default=3,
                      help='How often to repeat the same routes.')
  parser.add_argument('--use_cpp',
                      type=int,
                      default=0,
                      help='Whether the inference is a c++ model.')
  parser.add_argument('--sample_type',
                      type=str,
                      default='mean',
                      help='How to sample from the distribution. options: roach, mean, sample')
  parser.add_argument('--high_freq_inference',
                      type=int,
                      default=0,
                      help='Whether to evaluate at 20 Hz (1) or use action repeat 2 (0).')
  parser.add_argument('--ppo_cpp_install_path',
                      type=str,
                      default='/mnt/lustre/work/geiger/bjaeger25/ppo.cpp/build',
                      help='Install path of PPO c++ executable')
  parser.add_argument('--path_to_singularity',
                      type=str,
                      default='/mnt/lustre/work/geiger/bjaeger25/ppo.cpp/tools/ppo_cpp.sif',
                      help='Path to singularity container that can run PPO c++ executable')
  parser.add_argument('--pytorch_kernel_cache_path',
                      type=str,
                      default='/mnt/lustre/work/geiger/bjaeger25/home/.cache',
                      help='Path to pytorch cache folder')
  parser.add_argument('--record',
                      type=int,
                      default=0,
                      help='Whether to create short video clips about infractions')
  parser.add_argument('--use_sensor',
                      type=int,
                      default=0,
                      help='Whether to evaluate a sensor model')

  parser.add_argument('--num_envs', default=1, type=int, help='How many environments to train with 6 max right now.')

  args, unknown = parser.parse_known_args()

  print(f'Unkown arguments: {unknown}')
  num_repetitions = args.num_repetitions
  benchmark = args.benchmark
  experiment = args.experiment
  model_dir = args.model_dir
  code_root = args.code_root
  carla_root = args.carla_root
  partition = args.partition
  username = args.username
  experiment_name_stem = f'{experiment}_{benchmark}'
  exp_names_tmp = []
  seeds = []
  for i in range(num_repetitions):
    exp_names_tmp.append(experiment_name_stem + f'_e{i}')
    seeds.append(i)
  route_path = f'custom_leaderboard/leaderboard/data/{benchmark}_split/'
  route_pattern = '*.xml'

  carla_world_port_start = 10000
  carla_streaming_port_start = 20000
  carla_tm_port_start = 30000
  cpp_port_start = 500

  epochs = args.epochs
  job_nr = 0
  experiment_result_folders = []
  for epoch in epochs:
    # Root folder in which each of the evaluation seeds will be stored
    experiment_name_root = experiment_name_stem + '_' + epoch
    experiment_result_folders.append(experiment_name_root)
    exp_names = []
    for name in exp_names_tmp:
      exp_names.append(name + '_' + epoch)

    checkpoint = experiment
    checkpoint_new_name = checkpoint + '_' + epoch

    # Links the model file into team_code
    copy_model = True

    if copy_model:
      # copy checkpoint to my folder
      cmd = f'mkdir -p {args.team_code}/checkpoints/{checkpoint_new_name}'
      print(cmd)
      os.system(cmd)
      cmd = f'cp {model_dir}/{checkpoint}/config.json {args.team_code}/checkpoints/{checkpoint_new_name}/'
      print(cmd)
      os.system(cmd)
      cmd = f'cp {model_dir}/{checkpoint}/{epoch}.pth {args.team_code}/checkpoints/{checkpoint_new_name}/model.pth'
      print(cmd)
      os.system(cmd)

    route_files = []
    for root, _, files in os.walk(route_path):
      for name in files:
        if fnmatch.fnmatch(name, route_pattern):
          route_files.append(os.path.join(root, name))

    print('Number of routes:', len(route_files))

    for exp_name in exp_names:
      bash_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/run_bashs')
      results_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/results')
      logs_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/logs')
      bash_save_dir.mkdir(parents=True, exist_ok=True)
      results_save_dir.mkdir(parents=True, exist_ok=True)
      logs_save_dir.mkdir(parents=True, exist_ok=True)

    meta_jobs = {}

    for idx, exp_name in enumerate(exp_names):
      for route in route_files:
        route = Path(route).stem

        bash_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/run_bashs')
        results_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/results')
        logs_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/logs')

        commands = []

        # Finds a free port
        commands.append(
            f'FREE_WORLD_PORT=`comm -23 <(seq {carla_world_port_start} {carla_world_port_start + 49} | sort) '
            f'<(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`')
        commands.append("echo 'World Port:' $FREE_WORLD_PORT")
        commands.append(
            f'FREE_STREAMING_PORT=`comm -23 <(seq {carla_streaming_port_start} {carla_streaming_port_start + 49} '
            f'| sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`')
        commands.append("echo 'Streaming Port:' $FREE_STREAMING_PORT")

        nullrhi = '-nullrhi'
        if args.use_sensor != 0:
          nullrhi = ''
          print(f"Use sensor, nullrhi: {nullrhi}")
        commands.append(
            f'{carla_root}/CarlaUE4.sh -carla-rpc-port=${{FREE_WORLD_PORT}} -nosound -RenderOffScreen {nullrhi} '
            f'-carla-primary-port=0 -graphicsadapter=0 -carla-streaming-port=${{FREE_STREAMING_PORT}} &')
        commands.append('sleep 180')  # Waits for CARLA to finish starting
        create_run_eval_bash(bash_save_dir,
                             results_save_dir,
                             route_path,
                             route,
                             checkpoint_new_name,
                             logs_save_dir,
                             carla_tm_port_start,
                             carla_root=carla_root,
                             seed=seeds[idx],
                             team_code=args.team_code,
                             use_cpp=args.use_cpp,
                             cpp_port_start=cpp_port_start,
                             code_root=code_root,
                             sample_type=args.sample_type,
                             high_freq_inference=args.high_freq_inference,
                             ppo_cpp_install_path=args.ppo_cpp_install_path,
                             path_to_singularity=args.path_to_singularity,
                             pytorch_kernel_cache_path=args.pytorch_kernel_cache_path,
                             record=args.record)
        commands.append(f'chmod u+x {bash_save_dir}/eval_{route}.sh')
        commands.append(f'./{bash_save_dir}/eval_{route}.sh $FREE_WORLD_PORT')
        commands.append('sleep 2')

        carla_world_port_start += 50
        carla_streaming_port_start += 50
        carla_tm_port_start += 50
        cpp_port_start += 50

        job_file = make_jobsub_file(commands=commands,
                                    job_number=job_nr,
                                    exp_name=experiment_name_stem,
                                    exp_root_name=experiment_name_root,
                                    partition=partition)
        result_file = f'{results_save_dir}/{route}.json'

        # Wait until submitting new jobs that the #jobs are at below max
        num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=experiment_name_stem, username=username)
        print(f'{num_running_jobs}/{max_num_parallel_jobs} jobs are running...')
        while num_running_jobs >= max_num_parallel_jobs:
          num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=experiment_name_stem, username=username)
        time.sleep(0.05)
        print(f'Submitting job {job_nr}/{len(route_files) * num_repetitions}: {job_file}')
        jobid = subprocess.check_output(f'sbatch --exclude=galvani-cn115 {job_file}', shell=True).decode('utf-8').strip().rsplit(' ',
                                                                                                         maxsplit=1)[-1]
        meta_jobs[jobid] = (False, job_file, result_file, 0)

        job_nr += 1

  training_finished = False
  while not training_finished:
    num_running_jobs, max_num_parallel_jobs = get_num_jobs_eval(job_name=experiment_name_stem, username=username)
    print(f'{num_running_jobs} eval jobs are running...')
    time.sleep(10)

    # resubmit unfinished jobs
    for k in list(meta_jobs.keys()):
      job_finished, job_file, result_file, resubmitted = meta_jobs[k]
      need_to_resubmit = False
      if not job_finished and resubmitted < 5:
        # check whether job is running
        if int(subprocess.check_output(f'squeue | grep {k} | wc -l', shell=True).decode('utf-8').strip()) == 0:
          # check whether result file is finished?
          if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f_result:
              evaluation_data = ujson.load(f_result)
            progress = evaluation_data['_checkpoint']['progress']

            if len(progress) < 2 or progress[0] < progress[1]:
              need_to_resubmit = True
            else:
              for record in evaluation_data['_checkpoint']['records']:
                if record['status'] == 'Failed - Agent couldn\'t be set up':
                  need_to_resubmit = True
                  print('Resubmit - Agent not setup')
                elif record['status'] == 'Failed':
                  need_to_resubmit = True
                elif record['status'] == 'Failed - Simulation crashed':
                  need_to_resubmit = True
                elif record['status'] == 'Failed - Agent crashed':
                  need_to_resubmit = True

            if not need_to_resubmit:
              # delete old job
              print(f'Finished job {job_file}')
              meta_jobs[k] = (True, None, None, 0)
          else:
            need_to_resubmit = True

      if need_to_resubmit:
        # Remove crashed results file
        if os.path.exists(result_file):
          print('Remove file: ', result_file)
          Path(result_file).unlink()

        num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=experiment_name_stem, username=username)
        print(f'{num_running_jobs}/{max_num_parallel_jobs} jobs are running...')
        while num_running_jobs >= max_num_parallel_jobs:
          num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=experiment_name_stem, username=username)
          time.sleep(0.3)
        print(f'resubmit sbatch {job_file}')
        jobid = subprocess.check_output(f'sbatch --exclude=galvani-cn115 {job_file}', shell=True).decode('utf-8').strip().rsplit(' ',
                                                                                                         maxsplit=1)[-1]
        meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
        meta_jobs[k] = (True, None, None, 0)

    time.sleep(10)

    num_running_jobs, max_num_parallel_jobs = get_num_jobs_eval(job_name=experiment_name_stem, username=username)
    if num_running_jobs == 0:
      training_finished = True

  for exp_result_root in experiment_result_folders:
    print('Evaluation finished. Start parsing results.')
    eval_root = f'{code_root}/evaluation/{exp_result_root}'
    subprocess.check_call(
        f'python {code_root}/tools/result_parser.py '
        f'--xml {code_root}/custom_leaderboard/leaderboard/data/{benchmark}.xml '
        f'--results {eval_root} --strict',
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=True)


if __name__ == '__main__':
  main()
