#!/bin/bash
#SBATCH --job-name=eval_server
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --output=/weka/geiger/bjaeger25/ad_planning/2_carla/results/logs/eval_server_%a_%A.out
#SBATCH --error=/weka/geiger/bjaeger25/ad_planning/2_carla/results/logs/eval_server_%a_%A.err
#SBATCH --partition=h100-ferranti

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

export LD_LIBRARY_PATH=/weka/geiger/bjaeger25/miniconda3/lib:$LD_LIBRARY_PATH

for i in $(seq 2 2); do
  ex_name=$(printf "Fixed_DD_PPO_237_%02d" ${i})
  python -u evaluate_routes_slurm.py --experiment "${ex_name}" --benchmark longest6 --team_code team_code_roach --epochs model_final model_best --num_repetitions 3 --use_cpp 0 --sample_type mean --high_freq_inference 0 --model_dir /weka/geiger/bjaeger25/ad_planning/2_carla/results --code_root /weka/geiger/bjaeger25/ad_planning/2_carla --carla_root /weka/geiger/bjaeger25/carla_0_9_15 --partition h100-ferranti &
done
wait

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
