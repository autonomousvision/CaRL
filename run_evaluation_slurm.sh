#!/bin/bash
#SBATCH --job-name=eval_server
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/eval_server_%a_%A.out
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/results/logs/eval_server_%a_%A.err
#SBATCH --partition=2080-galvani

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

for i in $(seq 1 1); do
  ex_name=$(printf "TF_PPO_065_%02d" ${i})
  python -u evaluate_routes_slurm.py --experiment "${ex_name}" --benchmark longest6 --team_code team_code_roach --partition 2080-galvani --epochs model_final --num_repetitions 3 --use_cpp 0 --sample_type mode --high_freq_inference 0 --record 1 --use_sensor 1 &
done
wait

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
