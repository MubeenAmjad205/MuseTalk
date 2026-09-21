"""
One-shot Colab launcher for MuseTalk.

Builds a Python 3.10 environment (MuseTalk's pinned torch/mmcv versions need it),
downloads the model weights, and launches the Gradio app with a public link.
Safe to re-run: finished steps are skipped.

Environment variables:
    HF_TOKEN            Hugging Face token (faster, rate-limit-free downloads)
    MUSETALK_DRIVE_DIR  If set, model weights are cached in this folder (e.g. on Google Drive)
"""

import os
import shutil
import subprocess
import sys

VENV = "/content/musetalk-venv"
PY = f"{VENV}/bin/python"
SETUP_DONE = f"{VENV}/.setup_done"
LOG = "/content/musetalk_setup.log"

MMCV_INDEX = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"


def run(cmd, env=None):
    """Run a shell command quietly; on failure show the tail of the log and exit."""
    with open(LOG, "a") as log:
        log.write(f"\n$ {cmd}\n")
        log.flush()
        result = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}\n--- last lines of {LOG} ---")
        with open(LOG) as log:
            print("".join(log.readlines()[-30:]))
        sys.exit(result.returncode)


def check_gpu():
    if shutil.which("nvidia-smi") is None or subprocess.run(["nvidia-smi"], capture_output=True).returncode != 0:
        print("❌ No GPU found. Go to Runtime > Change runtime type and select T4 GPU, then run again.")
        sys.exit(1)
    name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    print(f"✅ GPU: {name}")


def install_dependencies():
    if os.path.exists(SETUP_DONE):
        print("✅ Dependencies already installed.")
        return
    print("⏳ Installing dependencies (about 5 minutes on first run)...")
    run(f"{sys.executable} -m pip install -q uv")
    run(f"uv venv --python 3.10 {VENV}")
    pip = f"uv pip install --python {PY}"
    run(f"{pip} torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url {TORCH_INDEX}")
    run(f"{pip} -r requirements.txt")
    run(f'{pip} mmengine "mmcv==2.0.1" -f {MMCV_INDEX}')
    run(f'{pip} "mmdet==3.1.0" "mmpose==1.1.0"')
    run(f'{PY} -c "import torch, mmcv, mmpose; assert torch.cuda.is_available()"')
    open(SETUP_DONE, "w").close()
    print("✅ Dependencies installed.")


def link_drive_cache():
    drive_dir = os.environ.get("MUSETALK_DRIVE_DIR")
    if not drive_dir:
        return
    os.makedirs(drive_dir, exist_ok=True)
    if os.path.islink("models"):
        os.remove("models")
    elif os.path.isdir("models"):
        shutil.rmtree("models")
    os.symlink(drive_dir, "models")
    print(f"✅ Model weights cached in {drive_dir}")


def download_weights():
    if os.path.exists("models/musetalkV15/unet.pth") and os.path.exists("models/face-parse-bisent/79999_iter.pth"):
        print("✅ Model weights already downloaded.")
        return
    print("⏳ Downloading model weights (about 5 GB)...")
    # The script points at a Chinese HF mirror; use the official Hub instead.
    with open("download_weights.sh") as f:
        script = "".join(line for line in f if "HF_ENDPOINT" not in line)
    with open("/content/download_weights_colab.sh", "w") as f:
        f.write(script)
    env = dict(os.environ, PATH=f"{VENV}/bin:{os.environ['PATH']}")
    run("bash /content/download_weights_colab.sh", env=env)
    print("✅ Model weights downloaded.")


def launch():
    print("🚀 Launching MuseTalk. Wait for the public *.gradio.live link below...")
    try:
        subprocess.run([PY, "app.py", "--use_float16", "--share", "--ffmpeg_path", "/usr/bin"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 MuseTalk stopped.")


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    check_gpu()
    install_dependencies()
    link_drive_cache()
    download_weights()
    launch()


if __name__ == "__main__":
    main()
