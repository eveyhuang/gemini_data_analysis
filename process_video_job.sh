#!/bin/bash
#SBATCH -A p32824
#SBATCH --partition=gengpu
#SBATCH --time=02:00:00
#SBATCH --mail-user=evey.huang@kellogg.northwestern.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=process_video

# unload modules that may have been loaded when job was submitted
module purge all
module load mamba
source activate gemini

source .env
echo "GOOGLE_API_KEY: $GOOGLE_API_KEY"

# By default all file paths are relative to the directory where you submitted the job.
# To change to another path, use `cd <path>`, for example:
# cd /projects/<allocationID>

python analyze_video.py --dir "/home/jhs0727/p32824/scialog/2021MND" --process-video yes --annotate-video no

