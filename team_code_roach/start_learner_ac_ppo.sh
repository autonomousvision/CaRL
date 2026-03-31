#!/bin/bash

export git_root=$1
export num_processes=$2
export num_nodes=$3
export rdzv_addr=$4
export rdzv_port=$5
export PYTORCH_KERNEL_CACHE_PATH=$6
export ppo_cpp_install_path=$7
export logdir=$8
export cpp_singularity_file_path=$9
export system_lib_path_1=${10}
export system_lib_path_2=${11}

# We occupy our threads with multi-processing/threading so we turn of multi-threading inside libraries
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MASTER_ADDR=${rdzv_addr}
#export NCCL_BLOCKING_WAIT=1 # Experimental for debugging.
#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=INFO
#export TORCH_CPP_LOG_LEVEL=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL

SCRATCH="${SCRATCH:=/tmp}"

#  --map-by ppr:1:numa
 #  --bind-to numa
 #  --rank-by numa

# Construct the command for clarity
mpi_cmd=(
  mpirun
  -n "${num_processes}"
  --map-by ppr:4:numa
  --bind-to numa
  --report-bindings
  "${ppo_cpp_install_path}/ac_ppo_carla"
  --rdzv_addr "${rdzv_addr}"
  --logdir "${logdir}"
  "${@:12}"
)

# Print the full command for debugging
echo "Executing inside Singularity:"
printf ' %q' "${mpi_cmd[@]}"
echo -e "\n"

# Execute inside Singularity
singularity exec --nv \
  --env LD_LIBRARY_PATH="${ppo_cpp_install_path}:${system_lib_path_1}:${system_lib_path_2}:$LD_LIBRARY_PATH" \
  --env PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH}" \
  --env HWLOC_COMPONENTS=-gl \
  --bind "${system_lib_path_1}:${system_lib_path_1},${system_lib_path_2}:${system_lib_path_2},${ppo_cpp_install_path}:${ppo_cpp_install_path},${git_root}:${git_root},${PYTORCH_KERNEL_CACHE_PATH}:${PYTORCH_KERNEL_CACHE_PATH},${SCRATCH}:${SCRATCH},${logdir}:${logdir}" \
  "${cpp_singularity_file_path}" \
  "${mpi_cmd[@]}"
