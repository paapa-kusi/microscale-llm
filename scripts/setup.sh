#!/usr/bin/env bash
set -e
conda env create -f env/environment.yml || conda env update -f env/environment.yml
conda activate microscale-llm
python -c "import torch, transformers, datasets; print('OK:', torch.__version__)"
