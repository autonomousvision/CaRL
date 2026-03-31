#!/bin/bash


torchrun --start-method spawn --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 test_cuda.py