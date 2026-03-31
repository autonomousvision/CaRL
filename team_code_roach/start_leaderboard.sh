#!/bin/bash

export git_root=$1
export route_file=$2
export logdir=$3
export index=$4
export client_port=$5
export tm_port=$6
export rl_port=$7
export random_seed=$8
export skip_next_route=$9
export repetitions=${10}
export no_rendering_mode=${11}
export NO_CARS=${12}

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1


echo python -u ${git_root}/custom_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py --routes ${route_file} --agent ${git_root}/team_code_roach/env_agent.py --checkpoint ${logdir}/route_${index}.json --track MAP --port ${client_port} --traffic-manager-port ${tm_port} --agent-config ${logdir} --gym_port ${rl_port} --traffic-manager-seed ${random_seed} --skip_next_route ${skip_next_route} --frame_rate 10 --resume 1 --no_rendering_mode ${no_rendering_mode} --runtime_timeout 500.0 --timeout 2000.0  --repetitions ${repetitions}

python -u ${git_root}/custom_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py --routes ${route_file} --agent ${git_root}/team_code_roach/env_agent.py --checkpoint ${logdir}/route_${index}.json --track MAP --port ${client_port} --traffic-manager-port ${tm_port} --agent-config ${logdir} --gym_port ${rl_port} --traffic-manager-seed ${random_seed} --skip_next_route ${skip_next_route} --frame_rate 10 --resume 1 --no_rendering_mode ${no_rendering_mode} --runtime_timeout 500.0 --timeout 2000.0  --repetitions ${repetitions}