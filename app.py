import os
import time
import pdb
import re

import gradio as gr
import numpy as np
import sys
import subprocess

from huggingface_hub import snapshot_download
import requests

import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from argparse import Namespace
import shutil
import gdown
import imageio
import ffmpeg
from moviepy.editor import *
from transformers import WhisperModel

ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")

@torch.no_grad()
def debug_inpainting(video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", 
                    left_cheek_width=90, right_cheek_width=90, version="v15", progress=gr.Progress()):
    """Debug inpainting parameters, only process the first frame"""
    # Set default parameters
    args_dict = {
        "result_dir": './results/debug', 
        "fps": 25, 
        "batch_size": 1, 
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": version,
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)

    # Create debug directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    stage_list = []
    if version not in loaded_unets:
        stage_list.append(("model", f"Loading {MODEL_VERSIONS[version][0]} (first use only)", 20))
    stage_list += [("read", "Reading first frame", 1), ("face", "Detecting face", 4), ("preview", "Generating preview", 3)]
    stages = StageProgress(progress, stage_list)
    model_unet = get_unet(version, stages, "model")
    # Read first frame
    stages.update("read")
    if get_file_type(video_path) == "video":
        reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        reader.close()
    else:
        first_frame = cv2.imread(video_path)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Save first frame
    debug_frame_path = os.path.join(args.result_dir, "debug_frame.png")
    cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    
    # Get face coordinates
    stages.update("face")
    coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], bbox_shift)
    bbox = coord_list[0]
    frame = frame_list[0]
    
    if bbox == coord_placeholder:
        return None, "No face detected, please adjust bbox_shift parameter"
    
    # Initialize face parser
    stages.update("preview")
    if version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:
        fp = FaceParsing()
    
    # Process first frame
    x1, y1, x2, y2 = bbox
    if version == "v15":
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
    crop_frame = frame[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    
    # Generate random audio features
    random_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
    audio_feature = pe(random_audio)
    
    # Get latents
    latents = vae.get_latents_for_unet(crop_frame)
    latents = latents.to(dtype=weight_dtype)
    
    # Generate prediction results
    pred_latents = model_unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
    recon = vae.decode_latents(pred_latents)
    
    # Inpaint back to original image
    res_frame = recon[0]
    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
    if version == "v15":
        combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
    else:
        combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], fp=fp)
    
    # Save results (no need to convert color space again since get_image already returns RGB format)
    debug_result_path = os.path.join(args.result_dir, "debug_result.png")
    cv2.imwrite(debug_result_path, combine_frame)
    
    # Create information text
    info_text = f"Model: {MODEL_VERSIONS[version][0]}\n" + \
                f"Parameter information:\n" + \
                f"bbox_shift: {bbox_shift}\n" + \
                f"extra_margin: {extra_margin}\n" + \
                f"parsing_mode: {parsing_mode}\n" + \
                f"left_cheek_width: {left_cheek_width}\n" + \
                f"right_cheek_width: {right_cheek_width}\n" + \
                f"Detected face coordinates: [{x1}, {y1}, {x2}, {y2}]"
    
    return cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR), info_text

def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print(child_path)

def download_model():
    # 检查必需的模型文件是否存在
    required_models = {
        "MuseTalk": f"{CheckpointsDir}/musetalkV15/unet.pth",
        "MuseTalk": f"{CheckpointsDir}/musetalkV15/musetalk.json",
        "SD VAE": f"{CheckpointsDir}/sd-vae/config.json",
        "Whisper": f"{CheckpointsDir}/whisper/config.json",
        "DWPose": f"{CheckpointsDir}/dwpose/dw-ll_ucoco_384.pth",
        "SyncNet": f"{CheckpointsDir}/syncnet/latentsync_syncnet.pt",
        "Face Parse": f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth",
        "ResNet": f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
    }
    
    missing_models = []
    for model_name, model_path in required_models.items():
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        # 全用英文
        print("The following required model files are missing:")
        for model in missing_models:
            print(f"- {model}")
        print("\nPlease run the download script to download the missing models:")
        if sys.platform == "win32":
            print("Windows: Run download_weights.bat")
        else:
            print("Linux/Mac: Run ./download_weights.sh")
        sys.exit(1)
    else:
        print("All required model files exist.")




download_model()  # for huggingface deployment.

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.models.unet import UNet
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range


import musetalk.utils.preprocessing as preprocessing_module
from contextlib import contextmanager
import proglog


class StageProgress:
    """Maps the progress of several named stages onto one overall 0-100% Gradio progress bar."""

    def __init__(self, progress, stages):
        """`stages` is a list of (key, label, weight); weights set each stage's share of the bar."""
        self.progress = progress
        self.index = {key: i for i, (key, _, _) in enumerate(stages)}
        self.names = [label for _, label, _ in stages]
        total_weight = sum(weight for _, _, weight in stages)
        self.starts, self.spans, acc = [], [], 0
        for _, _, weight in stages:
            self.starts.append(acc / total_weight)
            self.spans.append(weight / total_weight)
            acc += weight
        self._last_update = 0.0

    def update(self, key, frac=0.0, detail="", force=True):
        idx = self.index[key]
        now = time.time()
        if not force and now - self._last_update < 0.25:
            return
        self._last_update = now
        overall = self.starts[idx] + self.spans[idx] * min(max(frac, 0.0), 1.0)
        desc = f"Step {idx + 1}/{len(self.names)} · {self.names[idx]}" + (f" · {detail}" if detail else "")
        self.progress(overall, desc=desc)

    def track(self, iterable, key, total=None, lo=0.0, hi=1.0):
        """Yield from `iterable`, moving stage `key` from `lo` to `hi` of its span."""
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        self.update(key, lo)
        for i, item in enumerate(iterable, 1):
            yield item
            if total:
                self.update(key, lo + (hi - lo) * min(i / total, 1.0), f"{min(i, total)}/{total}", force=False)
        self.update(key, hi)

    @contextmanager
    def patch_tqdm(self, module, key, weights=(1,)):
        """Route a library module's successive tqdm loops into stage `key`, giving loop k `weights[k]` of the stage."""
        original = module.tqdm
        bounds = [sum(weights[:k]) / sum(weights) for k in range(len(weights) + 1)]
        calls = [0]

        def staged_tqdm(iterable, *args, **kwargs):
            k = min(calls[0], len(weights) - 1)
            calls[0] += 1
            return self.track(iterable, key, total=kwargs.get("total"), lo=bounds[k], hi=bounds[k + 1])

        module.tqdm = staged_tqdm
        try:
            yield
        finally:
            module.tqdm = original


class StageVideoLogger(proglog.ProgressBarLogger):
    """Reports moviepy's video-encoding progress to a StageProgress stage."""

    def __init__(self, stages, key, lo=0.0, hi=1.0):
        super().__init__()
        self.stages, self.key, self.lo, self.hi = stages, key, lo, hi

    def bars_callback(self, bar, attr, value, old_value=None):
        total = self.bars[bar].get("total")
        if bar == "t" and attr == "index" and total:
            frac = min(value / total, 1.0)
            self.stages.update(self.key, self.lo + (self.hi - self.lo) * frac, f"encoding {int(100 * frac)}%", force=False)


MODEL_VERSIONS = {
    # version: (label, folder under models/, weights file)
    "v15": ("MuseTalk 1.5", "musetalkV15", "unet.pth"),
    "v1": ("MuseTalk 1.0", "musetalk", "pytorch_model.bin"),
}
loaded_unets = {}


def download_hf_file(repo_id, filename, dest, stages=None, key=None, span=1.0):
    """Stream a file from the Hugging Face Hub, reporting real byte progress to a stage."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if stages and total:
                    stages.update(key, span * done / total,
                                  f"downloading {done / 1e9:.2f} / {total / 1e9:.2f} GB", force=False)
    os.replace(tmp, dest)


def get_unet(version, stages=None, key=None):
    """Return the UNet for `version`, downloading and loading it on first use."""
    if version in loaded_unets:
        return loaded_unets[version]
    _, folder, weights_file = MODEL_VERSIONS[version]
    config_path = os.path.join(CheckpointsDir, folder, "musetalk.json")
    weights_path = os.path.join(CheckpointsDir, folder, weights_file)
    if not os.path.exists(config_path):
        download_hf_file("TMElyralab/MuseTalk", f"{folder}/musetalk.json", config_path)
    if not os.path.exists(weights_path):
        download_hf_file("TMElyralab/MuseTalk", f"{folder}/{weights_file}", weights_path, stages, key, span=0.85)
    if stages:
        stages.update(key, 0.9, "loading onto GPU")
    model = UNet(unet_config=config_path, model_path=weights_path, device=device)
    if weight_dtype == torch.float16:
        model.model = model.model.half()
    model.model = model.model.to(device)
    loaded_unets[version] = model
    if stages:
        stages.update(key, 1.0)
    return model


def estimate_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count or None


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


@torch.no_grad()
def inference(audio_path, video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", 
              left_cheek_width=90, right_cheek_width=90, version="v15", progress=gr.Progress()):
    stage_list = []
    if version not in loaded_unets:
        stage_list.append(("model", f"Loading {MODEL_VERSIONS[version][0]} (first use only)", 40))
    stage_list += [
        ("frames", "Extracting video frames", 6),
        ("audio", "Analyzing audio", 4),
        ("landmarks", "Detecting face landmarks", 25),
        ("range", "Measuring face-box range", 15),
        ("encode", "Encoding face frames", 8),
        ("generate", "Generating lip movements", 22),
        ("blend", "Blending onto original video", 12),
        ("render", "Rendering final video", 8),
    ]
    stages = StageProgress(progress, stage_list)
    model_unet = get_unet(version, stages, "model")
    # Set default parameters, aligned with inference.py
    args_dict = {
        "result_dir": './results/output', 
        "fps": 25, 
        "batch_size": 8, 
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": version,
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)

    # Check ffmpeg
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    
    # Create temporary directory
    temp_dir = os.path.join(args.result_dir, f"{args.version}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set result save path
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)

    if args.output_vid_name == "":
        output_vid_name = os.path.join(temp_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(temp_dir, args.output_vid_name)
        
    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(temp_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        # Read video
        reader = imageio.get_reader(video_path)

        # Save images
        for i, im in enumerate(stages.track(reader, "frames", total=estimate_frame_count(video_path))):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    else: # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
        
    ############################################## extract audio feature ##############################################
    # Extract audio features
    stages.update("audio", 0.1, "reading audio")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, 
        device, 
        weight_dtype, 
        whisper, 
        librosa_length,
        fps=fps,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )
    stages.update("audio", 1.0)
        
    ############################################## preprocess input image  ##############################################
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates")
        with open(crop_coord_save_path,'rb') as f:
            coord_list = pickle.load(f)
        with stages.patch_tqdm(preprocessing_module, "landmarks"):
            frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming")
        with stages.patch_tqdm(preprocessing_module, "landmarks", weights=(1, 9)):
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    with stages.patch_tqdm(preprocessing_module, "range", weights=(1, 9)):
        bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)
    
    # Initialize face parser (v1.0 uses the default cheek widths)
    if version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:
        fp = FaceParsing()
    
    i = 0
    input_latent_list = []
    for bbox, frame in stages.track(list(zip(coord_list, frame_list)), "encode"):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        if version == "v15":
            y2 = y2 + args.extra_margin
            y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    ############################################## inference batch by batch ##############################################
    print("start inference")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )
    res_frame_list = []
    for i, (whisper_batch,latent_batch) in enumerate(stages.track(gen, "generate", total=int(np.ceil(float(video_num)/batch_size)))):
        audio_feature_batch = pe(whisper_batch)
        # Ensure latent_batch is consistent with model weight type
        latent_batch = latent_batch.to(dtype=weight_dtype)
        
        pred_latents = model_unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
            
    ############################################## pad to full image ##############################################
    print("pad talking image to original video")
    for i, res_frame in enumerate(stages.track(res_frame_list, "blend")):
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        if version == "v15":
            y2 = y2 + args.extra_margin
            y2 = min(y2, frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except:
            continue
        
        # Merge results with version-specific blending
        if version == "v15":
            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
        else:
            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
            
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
        
    # Frame rate
    fps = 25
    # Output video path
    output_video = 'temp.mp4'

    # Read images
    def is_valid_image(file):
        pattern = re.compile(r'\d{8}\.png')
        return pattern.match(file)

    images = []
    files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
    files.sort(key=lambda x: int(x.split('.')[0]))

    for file in stages.track(files, "render", hi=0.3):
        filename = os.path.join(result_img_save_path, file)
        images.append(imageio.imread(filename))
        

    # Save video
    stages.update("render", 0.35, "writing frames")
    imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')

    input_video = './temp.mp4'
    # Check if the input_video and audio_path exist
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Read video
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']  # Get original video frame rate
    reader.close() # Otherwise, error on win11: PermissionError: [WinError 32] Another program is using this file, process cannot access. : 'temp.mp4'
    # Store frames in list
    frames = images
    
    print(len(frames))

    # Load the video
    video_clip = VideoFileClip(input_video)

    # Load the audio
    audio_clip = AudioFileClip(audio_path)

    # Set the audio to the video
    video_clip = video_clip.set_audio(audio_clip)

    # Write the output video
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=25,
                               logger=StageVideoLogger(stages, "render", lo=0.4, hi=1.0))

    os.remove("temp.mp4")
    #shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")
    return output_vid_name,bbox_shift_text



# load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth", 
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device
)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_path", type=str, default=r"ffmpeg-master-latest-win64-gpl-shared\bin", help="Path to ffmpeg executable")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
parser.add_argument("--share", action="store_true", help="Create a public link")
parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
args = parser.parse_args()

# Set data type
if args.use_float16:
    # Convert models to half precision for better performance
    pe = pe.half()
    vae.vae = vae.vae.half()
    unet.model = unet.model.half()
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

# Move models to specified device
pe = pe.to(device)
vae.vae = vae.vae.to(device)
unet.model = unet.model.to(device)

loaded_unets["v15"] = unet

timesteps = torch.tensor([0], device=device)

# Initialize audio processor and Whisper model
audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
whisper = WhisperModel.from_pretrained("./models/whisper")
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)


def check_video(video, progress=gr.Progress()):
    if not isinstance(video, str):
        return video # in case of none type
    # Define the output video file name
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    # Add the output prefix to the file name
    output_file_name = "outputxxx_" + file_name

    os.makedirs('./results',exist_ok=True)
    os.makedirs('./results/output',exist_ok=True)
    os.makedirs('./results/input',exist_ok=True)

    # Combine the directory path and the new file name
    output_video = os.path.join('./results/input', output_file_name)


    # read video
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']  # get fps from original video

    # conver fps to 25
    stages = StageProgress(progress, [("prepare", "Preparing video (converting to 25 fps)", 1)])
    frames = [im for im in stages.track(reader, "prepare", total=estimate_frame_count(video), hi=0.6)]
    target_fps = 25
    
    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L+1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target+1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1      # find the first t_idx so that target_t / target_fps <= original_t[t_idx]
            if t_idx >= L:
                break
        target_frames.append(frames[t_idx])

    # save video
    stages.update("prepare", 0.7, "saving")
    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video




def generate(video, audio, version, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width,
             progress=gr.Progress()):
    """Validate inputs, run inference, and report status for the UI."""
    if not video:
        raise gr.Error("Please upload a face video first.")
    if not audio:
        raise gr.Error("Please add the audio you want the face to speak.")
    start = time.time()
    output_path, bbox_range = inference(audio, video, bbox_shift, extra_margin, parsing_mode,
                                        left_cheek_width, right_cheek_width, version=version, progress=progress)
    status = (f"✅ **Done in {time.time() - start:.0f}s** with {MODEL_VERSIONS[version][0]}. "
              "Use the ⬇ button on the video to download it.")
    return output_path, status, bbox_range


def preview(video, version, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width,
            progress=gr.Progress()):
    if not video:
        raise gr.Error("Please upload a face video first.")
    return debug_inpainting(video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width,
                            version=version, progress=progress)


def on_version_change(version):
    """Show the v1.5-only blending controls only when v1.5 is selected."""
    is_v15 = version == "v15"
    return [gr.update(visible=is_v15)] * 3


theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="indigo",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="linear-gradient(90deg, #06b6d4, #6366f1)",
    button_primary_background_fill_hover="linear-gradient(90deg, #0891b2, #4f46e5)",
    button_primary_text_color="white",
    block_title_text_weight="600",
)

css = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#hero {text-align: center; padding: 28px 16px 8px;}
#hero h1 {font-size: 2.3rem; font-weight: 800; margin: 0;
          background: linear-gradient(90deg, #06b6d4, #6366f1); -webkit-background-clip: text; background-clip: text; color: transparent;}
#hero p {margin: 6px 0 16px; opacity: .75; font-size: 1.05rem;}
.steps {display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;}
.steps span {padding: 6px 14px; border-radius: 999px; font-size: .9rem;
             border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary);}
.card {border: 1px solid var(--border-color-primary) !important; border-radius: 16px !important; padding: 16px !important;}
#generate {font-size: 1.1rem; min-height: 52px;}
#output_vid {max-height: 560px;}
#footer {text-align: center; opacity: .6; font-size: .85rem; padding: 12px;}
"""

with gr.Blocks(theme=theme, css=css, title="MuseTalk Studio") as demo:
    gr.HTML("""
    <div id="hero">
      <h1>🎬 MuseTalk Studio</h1>
      <p>Make any face video speak any audio: real-time, high-fidelity lip-sync.</p>
      <div class="steps"><span>① Upload a face video</span><span>② Add the audio</span><span>③ Generate</span></div>
    </div>""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="card"):
            gr.Markdown("### 📥 Inputs")
            video = gr.Video(label="Face video", sources=["upload"], height=300)
            audio = gr.Audio(label="Audio to speak", type="filepath", sources=["upload", "microphone"])
            version = gr.Radio(
                label="Model",
                choices=[("MuseTalk 1.5 (recommended)", "v15"), ("MuseTalk 1.0", "v1")],
                value="v15",
                info="1.5 gives sharper, better-synced lips. 1.0 downloads ~3.4 GB the first time you use it.",
            )
            btn = gr.Button("✨ Generate lip-sync", variant="primary", elem_id="generate")

            with gr.Accordion("🎛️ Fine-tune (optional)", open=False):
                gr.Markdown("Defaults work for most videos. If the mouth looks off, tweak these and click **Preview** to check the first frame.")
                extra_margin = gr.Slider(label="Jaw movement range (extra margin)", minimum=0, maximum=40, value=10, step=1)
                parsing_mode = gr.Radio(label="Blend mode", choices=["jaw", "raw"], value="jaw",
                                        info="'jaw' blends the lower face naturally; 'raw' pastes the whole generated crop.")
                with gr.Row() as cheek_row:
                    left_cheek_width = gr.Slider(label="Left cheek width", minimum=20, maximum=160, value=90, step=5)
                    right_cheek_width = gr.Slider(label="Right cheek width", minimum=20, maximum=160, value=90, step=5)
                bbox_shift = gr.Number(label="Face box shift (px)", value=0,
                                       info="Positive values open the mouth wider, negative values less.")
                debug_btn = gr.Button("🔍 Preview first frame", variant="secondary")
                debug_image = gr.Image(label="Preview", height=260)
                debug_info = gr.Textbox(label="Preview details", lines=4)
                bbox_shift_scale = gr.Textbox(label="Detected face-box range", interactive=False)

        with gr.Column(scale=6, elem_classes="card"):
            gr.Markdown("### 🎞️ Result")
            out1 = gr.Video(label="Lip-synced video", elem_id="output_vid", interactive=False, show_download_button=True)
            status = gr.Markdown("Your video will appear here. Generation usually takes 1–3 minutes on a T4 GPU.")
            gr.Markdown("""
**Tips for best results**
- Use a clear, front-facing video with one visible face and good lighting.
- Clean speech works best; background music can reduce accuracy.
- The output is as long as the audio; the video loops if the audio is longer.
""")

    gr.Examples(
        label="⚡ Try an example",
        examples=[["data/video/yongen.mp4", "data/audio/yongen.wav"],
                  ["data/video/yongen.mp4", "data/audio/eng.wav"],
                  ["data/video/sun.mp4", "data/audio/sun.wav"]],
        inputs=[video, audio],
    )

    gr.HTML("""<div id="footer">Powered by <a href="https://github.com/TMElyralab/MuseTalk" target="_blank">MuseTalk</a>
    (Lyra Lab, Tencent Music Entertainment) · Only use videos and voices you have permission to use.</div>""")

    video.change(fn=check_video, inputs=[video], outputs=[video])
    version.change(fn=on_version_change, inputs=[version], outputs=[extra_margin, parsing_mode, cheek_row])
    btn.click(
        fn=lambda: "⏳ **Generating…** extracting frames, detecting the face, then syncing the lips.",
        outputs=status,
    ).then(
        fn=generate,
        inputs=[video, audio, version, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
        outputs=[out1, status, bbox_shift_scale],
    )
    debug_btn.click(
        fn=preview,
        inputs=[video, version, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
        outputs=[debug_image, debug_info],
    )


# Check ffmpeg and add to PATH
if not fast_check_ffmpeg():
    print(f"Adding ffmpeg to PATH: {args.ffmpeg_path}")
    # According to operating system, choose path separator
    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

# Solve asynchronous IO issues on Windows
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Start Gradio application
demo.queue().launch(
    share=args.share, 
    debug=True, 
    server_name=args.ip, 
    server_port=args.port
)
