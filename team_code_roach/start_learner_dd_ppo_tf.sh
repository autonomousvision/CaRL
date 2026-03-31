#!/bin/bash

export git_root=$1
export num_envs=$2
export num_nodes=$3
export rdzv_addr=$4
export rdzv_port=$5
export node_rank=$6

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8 # TODO tune
export MASTER_ADDR=${rdzv_addr}
#export NCCL_BLOCKING_WAIT=1 # Experimental for debugging.
#export CUDA_LAUNCH_BLOCKING=1

NCCL_P2P_LEVEL=SYS
NCCL_NET_GDR_LEVEL=PHB
NCCL_ALGO=Tree

#--rdzv-backend=c10d
echo torchrun --start-method spawn --nproc_per_node=${num_envs} --node_rank=${node_rank} --nnodes=${num_nodes} --max_restarts=0 --rdzv-endpoint=${rdzv_addr}:${rdzv_port} start_learner_dd_ppo_tf_numa.py ${git_root}/team_code_roach/dd_ppo_tf.py "${@:7}"

torchrun --start-method spawn --nproc_per_node=${num_envs} --node_rank=${node_rank} --nnodes=${num_nodes} --max_restarts=0 --rdzv-endpoint=${rdzv_addr}:${rdzv_port} start_learner_dd_ppo_tf_numa.py ${git_root}/team_code_roach/dd_ppo_tf.py "${@:7}"