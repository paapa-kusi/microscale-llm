#!/usr/bin/env bash
set -e

# Choose environment file: prefer env/environment.yml if it exists, otherwise fallback to env/environment.linux-gpu.yml
ENV_FILE="env/environment.yml"
if [ ! -f "$ENV_FILE" ]; then
	if [ -f "env/environment.linux-gpu.yml" ]; then
		ENV_FILE="env/environment.linux-gpu.yml"
	fi
fi

echo "Using environment file: $ENV_FILE"

# Create or update the conda env
if command -v conda &> /dev/null; then
	# shellcheck source=/dev/null
	source "$(conda info --base)/etc/profile.d/conda.sh" || true
	conda env create -f "$ENV_FILE" || conda env update -f "$ENV_FILE"
	conda activate microscale-llm || true
else
	echo "Warning: conda not found on PATH. Please create the environment manually using:"
	echo "  conda env create -f $ENV_FILE"
fi

python - <<'PY'
try:
		import torch, transformers, datasets
		print('Environment OK')
		print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
		print('transformers:', transformers.__version__)
		print('datasets:', datasets.__version__)
except Exception as e:
		print('Environment check failed:', e)
		raise
PY
