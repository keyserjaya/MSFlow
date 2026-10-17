# WSL (windows)
## First time setup
sudo apt update
sudo apt install -y build-essential pkg-config ffmpeg nodejs npm   # compilers, mp4 writer, node

### Node 18 (apt) is too old for gradio's component build → user-space Node 22 via nvm (no sudo)
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
. ~/.nvm/nvm.sh && nvm install 22

### Two env vars in ~/.bashrc — both REQUIRED for this project:
cat >> ~/.bashrc <<'EOF'
export UV_PROJECT_ENVIRONMENT="$HOME/venvs/msflow"   # venv on Linux fs (drvfs venvs break: os error 1)
export MPLBACKEND=Agg                                 # headless matplotlib (old matplotlib + WSL = TkAgg crash)
EOF
source ~/.bashrc

### so git/pysen accept the repo on the Windows mount
git config --global --add safe.directory /mnt/g/AIProjects/MSFlow

## Project setup
cd /mnt/g/AIProjects/MSFlow
uv sync                 # core research env (downloads CPython 3.10.9 itself; conda not involved)
uv sync --group ui      # optional: adds the webui stack (gradio + the custom canvas component)

## Rebuilding UI component after editing it
set -euo pipefail
export PATH="$HOME/venvs/msflow/bin:$PATH"
. "$HOME/.nvm/nvm.sh" >/dev/null 2>&1 || true
nvm use 22 >/dev/null 2>&1 || true

SRC=/mnt/g/AIProjects/MSFlow/webui/components/constraint_canvas
BUILD=~/ccbuild
rm -rf "$BUILD" && mkdir -p "$BUILD"
(cd "$SRC" && tar --exclude=node_modules -cf - .) | (cd "$BUILD" && tar -xf -) || true

cd "$BUILD/frontend" && npm install           # pulls/updates three.js etc.
cd "$BUILD"
gradio cc install 2>&1 | tail -1              # pip-installs the package into the venv
gradio cc build --no-generate-docs 2>&1 | tail -6   # compiles Svelte+three → backend/.../templates

rm -rf "$SRC/backend" "$SRC/dist"             # copy artifacts back into the repo
cp -r "$BUILD/backend" "$SRC/backend"
cp -r "$BUILD/dist" "$SRC/dist"
cp "$BUILD/frontend/package-lock.json" "$SRC/frontend/package-lock.json"
rm -rf "$SRC/backend/gradio_constraint_canvas/__pycache__"

cd /mnt/g/AIProjects/MSFlow
uv sync --group ui --reinstall-package gradio-constraint-canvas   # the critical step

## Validation command
# Sanity-check that the bundle actually swapped (hash names should match):
ls /home/jayav2/venvs/msflow/lib/python3.10/site-packages/gradio_constraint_canvas/templates/component/
ls webui/components/constraint_canvas/backend/gradio_constraint_canvas/templates/component/

# another check
uv run --group ui python webui/selftest.py        # editor logic, spec round-trips, guard rails
uv run --group ui python webui/serve_test.py      # app builds + serves headlessly
uv run --group ui python webui/parity_check.py --spec kick --gpu cpu --num_steps 20   # GUI vs CLI bit-parity