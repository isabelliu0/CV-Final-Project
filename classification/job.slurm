#!/bin/sh
#SBATCH -o "slurm/outfile_%A_%a"
#SBATCH -e "slurm/errfile_%A_%a"
#SBATCH --time 00:30:00
#SBATCH --gres gpu:2
#SBATCH -c 80
#SBATCH --mem=32G
#SBATCH --array=0
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yl3030@princeton.edu

module purge

python -m torch.distributed.launch --nproc_per_node 2 main.py --cfg '/home/yl3030/COS429_A3/InternImage/classification/configs/without_lr_decay/internimage_t_1k_224.yaml' --local-rank 0