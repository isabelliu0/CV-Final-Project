#!/bin/bash
#SBATCH --job-name=firsttry         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yl3030@princeton.edu

module purge

python -m torch.distributed.launch --nproc_per_node 1 main.py --cfg '/home/yl3030/COS429_A3/InternImage/classification/configs/without_lr_decay/internimage_t_1k_224.yaml' --local-rank 0