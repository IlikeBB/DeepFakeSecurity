#!/usr/bin/env bash
set -eo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 非互動式 shell 需先載入 Conda，才能啟用環境。
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
conda activate pt230

exec python -u main.py "$@"
