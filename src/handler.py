"""
RunPod serverless handler for RobustVideoMatting (video background removal).
Download video from URL -> RVM inference -> upload to Cloudinary. Return {"output_url"} or {"error"}.
RunPod wraps the return in {"output": ...}; do not double-wrap.
"""
try:
    import brotlicffi as _brotli_mod
except ImportError:
    import brotli as _brotli_mod

import logging
import os
import sys
import tempfile
import urllib.parse

_LOG_LEVEL = os.environ.get("RUNPOD_LOG_LEVEL", os.environ.get("LOG_LEVEL", "INFO")).upper()
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    stream=sys.stderr,
)
logger = logging.getLogger("rvm-worker")

sys.path.insert(0, os.environ.get("PYTHONPATH", "/app/RobustVideoMatting"))

import runpod
import torch
import requests
from torch.utils.data import DataLoader
from torchvision import transforms

MODEL = None
MODEL_PATH = "/app/RobustVideoMatting/checkpoints/rvm_mobilenetv3.pth"
DOWNLOAD_TIMEOUT_SEC = 600
DOWNLOAD_MAX_BYTES = 500 * 1024 * 1024
PROGRESS_INTERVAL_FRAMES = 10


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    from model import MattingNetwork
    MODEL = MattingNetwork("mobilenetv3")
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    MODEL = MODEL.eval().cuda()
    return MODEL


def auto_downsample_ratio(h, w):
    return min(512 / max(h, w), 1)


load_model()


def download_video_to_path(url: str, path: str, job, progress_update_fn) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("video_url must be http or https")
    logger.debug("download start url_host=%s path=%s", parsed.netloc or "(none)", path)
    progress_update_fn("Downloading (0%)")
    resp = requests.get(url, stream=True, timeout=(30, DOWNLOAD_TIMEOUT_SEC))
    resp.raise_for_status()
    total = int(resp.headers.get("content-length") or 0)
    logger.info("download response status=%s content_length=%s", resp.status_code, total or "unknown")
    downloaded = 0
    chunk_size = 1024 * 1024
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded > DOWNLOAD_MAX_BYTES:
                    raise ValueError(f"Video exceeds max size ({DOWNLOAD_MAX_BYTES // (1024*1024)}MB)")
                if total and total > 0 and (downloaded % (5 * chunk_size) == 0 or downloaded == total):
                    pct = min(100, int(100 * downloaded / total)) if total else 0
                    progress_update_fn(f"Downloading ({pct}%)")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info("download done path=%s size_mb=%.2f", path, size_mb)
    progress_update_fn("Downloading (100%)")


def run_rvm(input_path: str, output_path: str, downsample_ratio: float, job, progress_update_fn) -> None:
    from inference_utils import VideoReader, VideoWriter

    logger.debug("rvm start input=%s output=%s downsample_ratio=%s", input_path, output_path, downsample_ratio)
    model = load_model()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    transform = transforms.ToTensor()
    source = VideoReader(input_path, transform)
    total_frames = len(source)
    if total_frames == 0:
        raise ValueError("Video has no frames")
    logger.info("rvm input_frames=%s frame_rate=%s", total_frames, getattr(source, "frame_rate", "?"))
    reader = DataLoader(source, batch_size=1, pin_memory=True, num_workers=0)
    frame_rate = source.frame_rate
    writer = VideoWriter(output_path, frame_rate=frame_rate, bit_rate=4 * 1000000)
    bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    rec = [None] * 4
    processed = 0
    try:
        with torch.no_grad():
            for src in reader:
                if downsample_ratio is None or downsample_ratio <= 0:
                    downsample_ratio = auto_downsample_ratio(*src.shape[2:])
                src = src.to(device, dtype, non_blocking=True).unsqueeze(0)
                fgr, pha, *rec = model(src, *rec, downsample_ratio)
                com = fgr * pha + bgr * (1 - pha)
                writer.write(com[0])
                processed += src.size(1)
                if processed % PROGRESS_INTERVAL_FRAMES == 0 or processed == total_frames:
                    pct = int(100 * processed / total_frames)
                    progress_update_fn(f"Processing ({pct}%)")
    finally:
        writer.close()
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
    logger.info("rvm done output=%s size_mb=%.2f frames=%s", output_path, out_size_mb, processed)
    progress_update_fn("Processing (100%)")


def upload_to_cloudinary(path: str, job, progress_update_fn, folder: str = "rvm-output") -> str:
    import cloudinary
    import cloudinary.uploader

    url = os.environ.get("CLOUDINARY_URL")
    if url:
        cloudinary.config()
    else:
        cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
        api_key = os.environ.get("CLOUDINARY_API_KEY")
        api_secret = os.environ.get("CLOUDINARY_API_SECRET")
        if not all([cloud_name, api_key, api_secret]):
            raise RuntimeError(
                "Set CLOUDINARY_URL or CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET"
            )
        cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret)
    logger.debug("cloudinary upload start path=%s folder=%s", path, folder)
    progress_update_fn("Uploading to Cloudinary (0%)")
    result = cloudinary.uploader.upload_large(
        path, resource_type="video", folder=folder, chunk_size=6 * 1024 * 1024,
    )
    url_out = result.get("secure_url", "")
    logger.info("cloudinary upload done url_len=%s", len(url_out))
    progress_update_fn("Uploading to Cloudinary (100%)")
    return url_out


def handler(job):
    job_id = job.get("id", "unknown")
    job_input = job.get("input") or {}
    video_url = (job_input.get("video_url") or "").strip()
    downsample_ratio = job_input.get("downsample_ratio")
    cloudinary_folder = (job_input.get("cloudinary_folder") or "rvm-output").strip() or "rvm-output"
    url_host = urllib.parse.urlparse(video_url).netloc if video_url else ""
    logger.info("job_start job_id=%s video_host=%s downsample_ratio=%s folder=%s", job_id, url_host, downsample_ratio, cloudinary_folder)

    if not video_url:
        logger.warning("job validation failed job_id=%s reason=missing_video_url", job_id)
        return {"error": "Missing required input: video_url"}

    try:
        downsample_ratio = float(downsample_ratio) if downsample_ratio is not None else 0.25
        if not (0 < downsample_ratio <= 1):
            downsample_ratio = 0.25
    except (TypeError, ValueError):
        downsample_ratio = 0.25

    input_path = os.path.join(tempfile.gettempdir(), f"rvm_input_{job_id}.mp4")
    output_path = os.path.join(tempfile.gettempdir(), f"rvm_output_{job_id}.mp4")

    def progress_update(msg):
        runpod.serverless.progress_update(job, msg)

    try:
        progress_update("Downloading (0%)")
        download_video_to_path(video_url, input_path, job, progress_update)
        progress_update("Processing (0%)")
        run_rvm(input_path, output_path, downsample_ratio, job, progress_update)
        progress_update("Uploading to Cloudinary (0%)")
        output_url = upload_to_cloudinary(output_path, job, progress_update, folder=cloudinary_folder)
        progress_update("Done (100%)")
        logger.info("job_done job_id=%s output_url_len=%s", job_id, len(output_url or ""))
        # Return plain URL; SDK wraps in {"output": ...}. Nested {"output": {"output_url": ...}} can still cause 400.
        return output_url
    except Exception as e:
        logger.exception("job_failed job_id=%s error_type=%s error=%s", job_id, type(e).__name__, e)
        # error must be a string (dict causes 400)
        return {"error": str(e)}
    finally:
        for p in (input_path, output_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _verify_brotli():
    try:
        raw = b'{"input":{}}'
        compressed = _brotli_mod.compress(raw)
        decoded = _brotli_mod.decompress(compressed)
        if decoded != raw:
            raise RuntimeError("Brotli round-trip mismatch")
    except Exception as e:
        print(
            f"Brotli check failed: {e}. RunPod sends content-encoding: br; install brotlicffi or brotli.",
            file=sys.stderr,
        )
        sys.exit(1)


_verify_brotli()
runpod.serverless.start({"handler": handler})
