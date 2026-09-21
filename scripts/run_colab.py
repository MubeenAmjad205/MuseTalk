"""
One-shot Colab launcher for MuseTalk.

Builds a Python 3.10 environment (MuseTalk's pinned torch/mmcv versions need it),
downloads the model weights, and launches the Gradio app with a public link,
showing a live progress bar for every step. Safe to re-run: finished steps are skipped.

Environment variables:
    HF_TOKEN            Hugging Face token (faster, rate-limit-free downloads)
    MUSETALK_DRIVE_DIR  If set, model weights are cached in this folder (e.g. on Google Drive)
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque

# Colab points MPLBACKEND at its notebook backend, which doesn't exist inside our venv;
# use the headless backend for everything this script starts.
os.environ["MPLBACKEND"] = "Agg"

VENV = "/content/musetalk-venv"
PY = f"{VENV}/bin/python"
SETUP_DONE = f"{VENV}/.setup_done"
LOG = "/content/musetalk_setup.log"

MMCV_INDEX = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
# Keep numpy at MuseTalk's pinned version while adding the OpenMMLab packages.
NUMPY_PIN = '"numpy==1.23.5"'

# Weights needed by the app. MuseTalk 1.0 (models/musetalk) is downloaded by the app on first use.
HF_WEIGHTS = [
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", "models/musetalkV15/musetalk.json"),
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", "models/musetalkV15/unet.pth"),
    ("stabilityai/sd-vae-ft-mse", "config.json", "models/sd-vae/config.json"),
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", "models/sd-vae/diffusion_pytorch_model.bin"),
    ("openai/whisper-tiny", "config.json", "models/whisper/config.json"),
    ("openai/whisper-tiny", "pytorch_model.bin", "models/whisper/pytorch_model.bin"),
    ("openai/whisper-tiny", "preprocessor_config.json", "models/whisper/preprocessor_config.json"),
    ("yzd-v/DWPose", "dw-ll_ucoco_384.pth", "models/dwpose/dw-ll_ucoco_384.pth"),
    ("ByteDance/LatentSync", "latentsync_syncnet.pt", "models/syncnet/latentsync_syncnet.pt"),
]
RESNET_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
RESNET_DEST = "models/face-parse-bisent/resnet18-5c106cde.pth"
FACE_PARSE_GDRIVE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"
FACE_PARSE_DEST = "models/face-parse-bisent/79999_iter.pth"
FACE_PARSE_EST_BYTES = 53_000_000


# ---------------------------------------------------------------- progress display

def network_bytes():
    """Total bytes received on all non-loopback interfaces (Linux only)."""
    try:
        with open("/proc/net/dev") as f:
            lines = f.readlines()[2:]
        return sum(int(line.split(":")[1].split()[0]) for line in lines if not line.strip().startswith("lo:"))
    except (OSError, IndexError, ValueError):
        return None


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


class Progress:
    """One live line: overall bar, percentage, current step, spinner, elapsed time and download speed."""

    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    WIDTH = 24

    def __init__(self):
        self.percent = 0.0
        self.label = ""
        self.detail = ""
        self.step_start = time.time()
        self.net_start = network_bytes()
        self.net_samples = deque(maxlen=8)
        self._lock = threading.Lock()
        self._running = False
        self._tick = 0

    def start_step(self, label, percent):
        with self._lock:
            self.label, self.detail, self.percent = label, "", percent
            self.step_start = time.time()
            self.net_start = network_bytes()
            self.net_samples.clear()
        self._start_spinner()

    def set(self, percent=None, detail=None):
        with self._lock:
            if percent is not None:
                self.percent = max(self.percent, min(percent, 100.0))
            if detail is not None:
                self.detail = detail

    def finish_step(self, percent):
        self._stop_spinner()
        with self._lock:
            self.percent = percent
            elapsed = time.time() - self.step_start
        self._clear_line()
        print(f"  ✅ {self.label}  ({fmt_time(elapsed)})", flush=True)

    def render_final(self):
        self._render(final=True)

    def fail(self):
        self._stop_spinner()
        self._clear_line()
        print(f"  ❌ {self.label} failed", flush=True)

    def _net_text(self):
        now_bytes = network_bytes()
        if now_bytes is None or self.net_start is None:
            return ""
        self.net_samples.append((time.time(), now_bytes))
        received = now_bytes - self.net_start
        if received < 1_000_000:
            return ""
        speed = ""
        if len(self.net_samples) >= 2:
            (t0, b0), (t1, b1) = self.net_samples[0], self.net_samples[-1]
            if t1 > t0:
                speed = f" @ {fmt_bytes((b1 - b0) / (t1 - t0))}/s"
        return f"  ↓ {fmt_bytes(received)}{speed}"

    def _render(self, final=False):
        with self._lock:
            filled = int(self.WIDTH * self.percent / 100)
            bar = "█" * filled + "░" * (self.WIDTH - filled)
            if final:
                line = f"  [{bar}] {self.percent:5.1f}%"
            else:
                spin = self.SPINNER[self._tick % len(self.SPINNER)]
                detail = f" · {self.detail}" if self.detail else ""
                elapsed = fmt_time(time.time() - self.step_start)
                line = f"  [{bar}] {self.percent:5.1f}%  {spin} {self.label}{detail}  {elapsed}{self._net_text()}"
        sys.stdout.write("\r" + line.ljust(110) + ("\n" if final else ""))
        sys.stdout.flush()

    def _clear_line(self):
        sys.stdout.write("\r" + " " * 110 + "\r")
        sys.stdout.flush()

    def _spin(self):
        while self._running:
            self._tick += 1
            self._render()
            time.sleep(0.25)

    def _start_spinner(self):
        self._stop_spinner()
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _stop_spinner(self):
        if self._running:
            self._running = False
            self._thread.join()


progress = Progress()


def step(label, start, end, fn):
    """Run `fn(start, end)` as one step spanning `start`..`end` percent of the overall bar."""
    progress.start_step(label, start)
    try:
        fn(start, end)
    except BaseException:
        progress.fail()
        raise
    progress.finish_step(end)


def run(cmd, env=None):
    """Run a shell command quietly (output goes to LOG); on failure show the tail of the log and exit."""
    with open(LOG, "a") as log:
        log.write(f"\n$ {cmd}\n")
        log.flush()
        result = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        progress.fail()
        print(f"\nCommand failed: {cmd}\n--- last lines of {LOG} ---")
        with open(LOG) as log:
            print("".join(log.readlines()[-30:]))
        os._exit(result.returncode)


def run_steps(commands):
    """Return a step function that runs `commands` in order, advancing the bar after each one."""
    def fn(start, end):
        for i, cmd in enumerate(commands):
            run(cmd)
            progress.set(start + (end - start) * (i + 1) / len(commands))
    return fn


# ---------------------------------------------------------------- setup steps

def check_gpu():
    if shutil.which("nvidia-smi") is None or subprocess.run(["nvidia-smi"], capture_output=True).returncode != 0:
        print("❌ No GPU found. Go to Runtime > Change runtime type and select T4 GPU, then run again.")
        sys.exit(1)
    name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    print(f"  ✅ GPU: {name}", flush=True)


def install_dependencies(start, end):
    """Install everything, splitting start..end between the sub-steps by their typical duration."""
    if os.path.exists(SETUP_DONE):
        return
    if os.path.exists(VENV):
        shutil.rmtree(VENV)  # left over from an interrupted run
    pip = f"uv pip install --python {PY}"
    sub_steps = [
        ("Creating Python 3.10 environment", 5,
         [f"{sys.executable} -m pip install -q uv", f"uv venv --clear --python 3.10 {VENV}"]),
        ("Installing PyTorch 2.0.1 (CUDA 11.8)", 35,
         [f"{pip} torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url {TORCH_INDEX}"]),
        ("Installing MuseTalk requirements", 35, [f"{pip} -r requirements.txt"]),
        ("Installing mmcv / mmengine", 10, [f'{pip} mmengine "mmcv==2.0.1" {NUMPY_PIN} -f {MMCV_INDEX}']),
        # mmpose depends on chumpy, which can't be built with modern pip and isn't used by MuseTalk's DWPose,
        # so install mmpose without dependencies and add the ones it actually needs.
        ("Installing mmdet / mmpose", 10, [
            f'{pip} "mmdet==3.1.0" {NUMPY_PIN}',
            f'{pip} --no-deps "mmpose==1.1.0"',
            f"{pip} json_tricks munkres xtcocotools {NUMPY_PIN}",
        ]),
        ("Verifying installation", 5,
         [f'{PY} -c "import torch, mmcv, mmdet, mmpose; assert torch.cuda.is_available(), \'CUDA not available\'"']),
    ]
    total = sum(weight for _, weight, _ in sub_steps)
    pos = start
    for label, weight, commands in sub_steps:
        span = (end - start) * weight / total
        step(label, pos, pos + span, run_steps(commands))
        pos += span
    open(SETUP_DONE, "w").close()


def link_drive_cache():
    drive_dir = os.environ.get("MUSETALK_DRIVE_DIR")
    if not drive_dir:
        return
    os.makedirs(drive_dir, exist_ok=True)
    if os.path.islink("models"):
        os.remove("models")
    elif os.path.isdir("models"):
        # Move anything already downloaded into the Drive cache instead of discarding it.
        subprocess.run(f'cp -rn models/. "{drive_dir}/" && rm -rf models', shell=True, check=True)
    os.symlink(drive_dir, "models")
    print(f"  ✅ Model weights cached in {drive_dir}", flush=True)


def hf_headers():
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def remote_size(requests, url, headers):
    """File size in bytes, or 0 if the server doesn't say (the bar then just fills a little early)."""
    try:
        r = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
        r.raise_for_status()
        return int(r.headers.get("content-length", 0))
    except (requests.RequestException, ValueError):
        return 0


def stream_download(requests, url, dest, headers, on_bytes):
    """Download `url` to `dest` (resuming a partial .part file), calling on_bytes(n) for each chunk."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    counted = 0

    def add(n):
        nonlocal counted
        counted += n
        on_bytes(n)

    for attempt in range(4):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if have > counted:  # count data already on disk from an interrupted download
            add(have - counted)
        req_headers = dict(headers, Range=f"bytes={have}-") if have else headers
        try:
            with requests.get(url, headers=req_headers, stream=True, timeout=60) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                if mode == "wb" and have:  # server ignored the Range header; start over
                    add(-have)
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        add(len(chunk))
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    os.replace(tmp, dest)


def download_weights(start, end):
    import requests

    headers = hf_headers()
    jobs = []  # (url, dest, headers)
    for repo, filename, dest in HF_WEIGHTS:
        if not os.path.exists(dest):
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            jobs.append((url, dest, headers))
    if not os.path.exists(RESNET_DEST):
        jobs.append((RESNET_URL, RESNET_DEST, {}))
    need_face_parse = not os.path.exists(FACE_PARSE_DEST)
    if not jobs and not need_face_parse:
        return

    progress.set(detail="checking file sizes")
    sizes = [remote_size(requests, url, h) for url, _, h in jobs]
    total = sum(sizes) + (FACE_PARSE_EST_BYTES if need_face_parse else 0)
    done = [0]

    def report(name):
        def on_bytes(n):
            done[0] += n
            frac = min(done[0] / total, 1.0) if total else 0
            progress.set(start + (end - start) * frac,
                         f"{name} · {fmt_bytes(done[0])} / {fmt_bytes(total)}")
        return on_bytes

    for (url, dest, h), size in zip(jobs, sizes):
        stream_download(requests, url, dest, h, report(os.path.basename(dest)))

    if need_face_parse:
        # Google Drive downloads need gdown; track its progress by the partial file it writes.
        run(f"{sys.executable} -m pip install -q gdown")
        before = done[0]
        progress.set(detail="79999_iter.pth (Google Drive)")
        proc = subprocess.Popen([sys.executable, "-m", "gdown", "--id", FACE_PARSE_GDRIVE_ID, "-O", FACE_PARSE_DEST],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        folder = os.path.dirname(FACE_PARSE_DEST)
        while proc.poll() is None:
            partial = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)
                          if f.startswith("79999_iter") or f.endswith(".part"))
            done[0] = before + min(partial, FACE_PARSE_EST_BYTES)
            report("79999_iter.pth")(0)
            time.sleep(0.5)
        if proc.returncode != 0 or not os.path.exists(FACE_PARSE_DEST):
            progress.fail()
            print("\nCould not download the face-parsing model from Google Drive (it may be rate-limited).")
            print(f"Try again in a few minutes. Details:\n{proc.stderr.read()[-1500:]}")
            os._exit(1)


def launch():
    """Start the app, show a loader while models load, then print the public link."""
    step_start = 90.0
    progress.start_step("Loading models onto the GPU", step_start)
    proc = subprocess.Popen([PY, "-u", "app.py", "--use_float16", "--share", "--ffmpeg_path", "/usr/bin"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    recent = deque(maxlen=40)
    milestones = [("load unet model", 93, "loading MuseTalk 1.5"),
                  ("Running on local URL", 97, "starting web server"),
                  ("Running on public URL", 100, "")]
    public_url = None
    for line in proc.stdout:
        recent.append(line)
        for marker, percent, detail in milestones:
            if marker in line:
                progress.set(percent, detail)
        if "Running on public URL" in line:
            public_url = line.split("Running on public URL:")[-1].strip()
            break
    if public_url is None:
        proc.wait()
        progress.fail()
        print("\nMuseTalk exited before it was ready. Last output:\n" + "".join(recent))
        sys.exit(proc.returncode or 1)

    progress.finish_step(100.0)
    progress.render_final()
    print("\n" + "=" * 70)
    print(f"  🎉 MuseTalk Studio is ready:  {public_url}")
    print("=" * 70)
    print("  Keep this cell running while you use the app. Stop the cell to shut it down.\n", flush=True)
    try:
        for line in proc.stdout:  # keep showing the app's own log (errors, generation messages)
            print(line, end="", flush=True)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n🛑 MuseTalk stopped.")


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    open(LOG, "w").close()  # start each run with a fresh log so errors aren't mixed with old ones
    print("\n🎬 MuseTalk Studio setup\n", flush=True)
    check_gpu()
    if os.path.exists(SETUP_DONE):
        print("  ✅ Dependencies already installed", flush=True)
    else:
        install_dependencies(2, 50)
    link_drive_cache()
    step("Downloading model weights", 50, 90, download_weights)
    launch()


if __name__ == "__main__":
    main()
