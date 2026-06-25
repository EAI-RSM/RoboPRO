#!/bin/bash
#SBATCH --job-name=pi05_smoke
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --nodelist=aaron-bb8
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/shared_work/robotwin_bench/customized_robotwin/policy/pi05/logs/smoke_%j.log

set -euo pipefail
mkdir -p /shared_work/robotwin_bench/customized_robotwin/policy/pi05/logs

bash /shared_work/robotwin_bench/customized_robotwin/policy/pi05/smoke_test.sh
