#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=14000
#SBATCH --job-name=train_sentence_cnn
#SBATCH --mail-type=END
#SBATCH --mail-user=ab8700@nyu.edu
#SBATCH --output=slurm_%j.out

python cnn_cls.py
