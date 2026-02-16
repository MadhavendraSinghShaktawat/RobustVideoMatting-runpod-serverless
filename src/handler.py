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
import shutil
import subprocess
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

# RVM package path: Docker uses PYTHONPATH=/app/RobustVideoMatting; for local runs use sibling RobustVideoMatting if present
_rvm_path = os.environ.get("PYTHONPATH", "").strip() or "/app/RobustVideoMatting"
if not os.path.isdir(_rvm_path):
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _fallback = os.path.abspath(os.path.join(_script_dir, "..", "..", "RobustVideoMatting"))
    if os.path.isdir(_fallback):
        _rvm_path = _fallback
        logger.debug("using local RVM path: %s", _rvm_path)
sys.path.insert(0, _rvm_path)

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
# Cloudinary upload limit (100 MB); ProRes often exceeds this for long videos
CLOUDINARY_MAX_FILE_BYTES = int(os.environ.get("CLOUDINARY_MAX_FILE_BYTES", 100 * 1024 * 1024))

# Required encoders: (display_name, aliases_in_ffmpeg_encoders_output).
# FFmpeg prints encoder names with hyphen (libvpx-vp9); some builds use underscore (libvpx_vp9).
_FFMPEG_REQUIRED_ENCODERS = (
    ("libvpx-vp9", ("libvpx-vp9", "libvpx_vp9")),
    ("prores_ks", ("prores_ks",)),
)


def _check_ffmpeg_capabilities() -> None:
    """Verify FFmpeg version and required encoders at startup. Fail fast if missing (best practice)."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-version", "-hide_banner", "-loglevel", "error"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_line = (r.stdout or r.stderr or "").split("\n")[0] if (r.stdout or r.stderr) else "unknown"
        logger.info("ffmpeg version: %s", version_line.strip())
    except FileNotFoundError:
        logger.error("ffmpeg not found in PATH. Install FFmpeg with libvpx-vp9 and prores_ks (see README).")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg -version timed out")
        sys.exit(1)

    try:
        r = subprocess.run(
            ["ffmpeg", "-encoders", "-hide_banner", "-loglevel", "error"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            logger.error("ffmpeg -encoders failed (returncode=%s). Ensure FFmpeg is built with libvpx and prores_ks.", r.returncode)
            sys.exit(1)
        missing = [
            name for name, aliases in _FFMPEG_REQUIRED_ENCODERS
            if not any(alias in out for alias in aliases)
        ]
        if missing:
            logger.error(
                "ffmpeg missing required encoders: %s. Distro/conda builds often omit these. "
                "Use a static build (e.g. BtbN/FFmpeg-Builds) or see README.",
                missing,
            )
            sys.exit(1)
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg -encoders timed out")
        sys.exit(1)


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    from model import MattingNetwork
    MODEL = MattingNetwork("mobilenetv3")
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    MODEL = MODEL.eval().cuda().half()
    return MODEL


def auto_downsample_ratio(h, w):
    return min(512 / max(h, w), 1)


_check_ffmpeg_capabilities()
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


def run_rvm_greenscreen(
    input_path: str, output_path: str, downsample_ratio: float, job, progress_update_fn
) -> None:
    from inference_utils import VideoReader, VideoWriter

    logger.debug("rvm greenscreen input=%s output=%s downsample_ratio=%s", input_path, output_path, downsample_ratio)
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
        with torch.inference_mode():
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
    logger.info("rvm greenscreen done output=%s size_mb=%.2f frames=%s", output_path, out_size_mb, processed)
    progress_update_fn("Processing (100%)")


def run_rvm_alpha(
    input_path: str,
    output_path: str,
    png_dir: str,
    downsample_ratio: float,
    job,
    progress_update_fn,
    encode_format: str = "webm",
) -> None:
    """Run RVM, then encode to WebM (via RGBA pipe, no PNGs) or ProRes 4444 (via PNG sequence)."""
    from PIL import Image
    from inference_utils import VideoReader, StreamingVideoReader

    logger.debug("rvm alpha input=%s output=%s png_dir=%s encode=%s", input_path, output_path, png_dir, encode_format)
    model = load_model()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    transform = transforms.ToTensor()
    source = StreamingVideoReader(input_path, transform)
    total_frames = len(source)
    if total_frames == 0:
        source = VideoReader(input_path, transform)
        total_frames = len(source)
    if total_frames == 0:
        raise ValueError("Video has no frames")
    frame_rate = getattr(source, "frame_rate", 30.0)
    logger.info("rvm alpha input_frames=%s frame_rate=%s", total_frames, frame_rate)
    use_webm_pipe = encode_format == "webm"
    if not use_webm_pipe:
        os.makedirs(png_dir, exist_ok=True)
    use_streaming = isinstance(source, StreamingVideoReader)
    reader = DataLoader(
        source, batch_size=1, pin_memory=True, num_workers=0 if use_streaming else 1
    )
    rec = [None] * 4
    processed = 0
    ffmpeg_proc = None
    PIPE_BUF_FRAMES = 8
    pipe_buf = []
    with torch.inference_mode():
        for src in reader:
            if downsample_ratio is None or downsample_ratio <= 0:
                downsample_ratio = auto_downsample_ratio(*src.shape[2:])
            src = src.to(device, dtype, non_blocking=True).unsqueeze(0)
            fgr, pha, *rec = model(src, *rec, downsample_ratio)
            fgr_clean = fgr * (pha > 0).to(dtype)
            rgba = torch.cat([fgr_clean[0], pha[0]], dim=1)[0].clamp(0, 1).mul(255).byte().cpu().permute(1, 2, 0).numpy()
            if use_webm_pipe:
                if processed == 0:
                    h, w = rgba.shape[0], rgba.shape[1]
                    progress_update_fn("Encoding WebM (alpha)")
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(frame_rate),
                        "-i", "pipe:0",
                        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                        "-crf", "31", "-b:v", "0",
                        "-cpu-used", "6", "-deadline", "good", "-row-mt", "1",
                        "-an", output_path,
                    ]
                    ffmpeg_proc = subprocess.Popen(
                        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                    )
                pipe_buf.append(rgba.tobytes())
                if len(pipe_buf) >= PIPE_BUF_FRAMES:
                    ffmpeg_proc.stdin.write(b"".join(pipe_buf))
                    pipe_buf.clear()
            else:
                frame_path = os.path.join(png_dir, f"frame_{processed:04d}.png")
                Image.fromarray(rgba, mode="RGBA").save(frame_path)
            processed += src.size(1)
            if processed % PROGRESS_INTERVAL_FRAMES == 0 or processed == total_frames:
                pct = min(100, int(100 * processed / total_frames)) if total_frames else 0
                progress_update_fn(f"Processing ({pct}%)")

    if use_webm_pipe and ffmpeg_proc is not None:
        if pipe_buf:
            ffmpeg_proc.stdin.write(b"".join(pipe_buf))
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait(timeout=3600)
        stderr = ffmpeg_proc.stderr.read() if ffmpeg_proc.stderr else b""
        if ffmpeg_proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode(errors='replace')}")
    else:
        progress_update_fn("Encoding ProRes 4444 (alpha)")
        cmd = [
            "ffmpeg", "-y", "-threads", "0",
            "-framerate", str(frame_rate),
            "-i", os.path.join(png_dir, "frame_%04d.png"),
            "-c:v", "prores_ks", "-profile:v", "4",
            "-pix_fmt", "yuva444p10le", "-an", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr or result.stdout}")
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
    logger.info("rvm alpha done output=%s size_mb=%.2f frames=%s", output_path, out_size_mb, processed)
    progress_update_fn("Processing (100%)")


def _convert_prores_to_webm(mov_path: str, job_id: str, progress_update_fn) -> str:
    """Convert ProRes .mov to WebM with alpha so it fits Cloudinary size limit. Returns path to .webm."""
    webm_path = os.path.join(tempfile.gettempdir(), f"rvm_webm_fallback_{job_id}.webm")
    progress_update_fn("Encoding WebM (alpha, for upload)")
    cmd = [
        "ffmpeg", "-y", "-i", mov_path,
        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
        "-crf", "31", "-b:v", "0",
        "-cpu-used", "6", "-deadline", "good", "-row-mt", "1",
        "-an", webm_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg ProRes→WebM failed: {result.stderr or result.stdout}")
    logger.info("prores_to_webm fallback done path=%s size_mb=%.2f", webm_path, os.path.getsize(webm_path) / (1024 * 1024))
    return webm_path


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


def _normalize_output_type(value) -> str:
    # Default to "alpha" (WebM) for fast single-encode pipeline; avoids ProRes then WebM double-encode.
    if value is None:
        return "alpha"
    s = (value or "").strip().lower()
    if s in ("alpha", "webm", "transparent"):
        return "alpha"
    if s in ("alpha_prores", "prores", "prores4444"):
        return "alpha_prores"
    if s == "greenscreen":
        return "greenscreen"
    return "alpha"


def _resolve_downsample_ratio(job_input) -> float:
    """Resolve downsample_ratio from input; apply speed preset if set."""
    raw = job_input.get("downsample_ratio")
    speed = (job_input.get("speed") or job_input.get("preset") or "").strip().lower()
    try:
        ratio = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        ratio = None
    if speed == "fast":
        return 0.125
    if speed in ("quality", "best", "high"):
        return 0.5 if ratio is None else ratio
    if ratio is not None and 0 < ratio <= 1:
        return ratio
    return 0.25


def handler(job):
    job_id = job.get("id", "unknown")
    job_input = job.get("input") or {}
    video_url = (job_input.get("video_url") or "").strip()
    downsample_ratio = job_input.get("downsample_ratio")
    cloudinary_folder = (job_input.get("cloudinary_folder") or "rvm-output").strip() or "rvm-output"
    output_type = _normalize_output_type(job_input.get("output_type"))
    url_host = urllib.parse.urlparse(video_url).netloc if video_url else ""
    logger.info("job_start job_id=%s video_host=%s downsample_ratio=%s folder=%s output_type=%s", job_id, url_host, downsample_ratio, cloudinary_folder, output_type)

    if not video_url:
        logger.warning("job validation failed job_id=%s reason=missing_video_url", job_id)
        return {"error": "Missing required input: video_url"}

    downsample_ratio = _resolve_downsample_ratio(job_input)

    input_path = os.path.join(tempfile.gettempdir(), f"rvm_input_{job_id}.mp4")
    if output_type == "alpha":
        ext, use_png_dir, encode_fmt = ".webm", True, "webm"
    elif output_type == "alpha_prores":
        ext, use_png_dir, encode_fmt = ".mov", True, "prores"
    else:
        ext, use_png_dir, encode_fmt = ".mp4", False, None
    output_path = os.path.join(tempfile.gettempdir(), f"rvm_output_{job_id}{ext}")
    png_dir = os.path.join(tempfile.gettempdir(), f"rvm_alpha_{job_id}") if use_png_dir else None

    def progress_update(msg):
        runpod.serverless.progress_update(job, msg)

    upload_path = output_path
    effective_format = encode_fmt  # "webm", "prores", or None
    fallback_webm_path = None

    try:
        progress_update("Downloading (0%)")
        download_video_to_path(video_url, input_path, job, progress_update)
        progress_update("Processing (0%)")
        if output_type in ("alpha", "alpha_prores"):
            run_rvm_alpha(input_path, output_path, png_dir, downsample_ratio, job, progress_update, encode_format=encode_fmt)
        else:
            run_rvm_greenscreen(input_path, output_path, downsample_ratio, job, progress_update)

        if output_type == "alpha_prores" and os.path.getsize(output_path) > CLOUDINARY_MAX_FILE_BYTES:
            logger.info("prores file over %s MB, converting to WebM for upload", CLOUDINARY_MAX_FILE_BYTES // (1024 * 1024))
            fallback_webm_path = _convert_prores_to_webm(output_path, job_id, progress_update)
            upload_path = fallback_webm_path
            effective_format = "webm"

        progress_update("Uploading to Cloudinary (0%)")
        output_url = upload_to_cloudinary(upload_path, job, progress_update, folder=cloudinary_folder)
        progress_update("Done (100%)")
        logger.info("job_done job_id=%s output_url_len=%s output_type=%s", job_id, len(output_url or ""), output_type)
        result = {"output_url": output_url}
        if effective_format:
            result["output_format"] = effective_format
        return result
    except Exception as e:
        logger.exception("job_failed job_id=%s error_type=%s error=%s", job_id, type(e).__name__, e)
        return {"error": str(e)}
    finally:
        for p in (input_path, output_path, fallback_webm_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        if png_dir and os.path.isdir(png_dir):
            try:
                shutil.rmtree(png_dir, ignore_errors=True)
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
