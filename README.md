# RunPod RVM Worker

[![Runpod](https://api.runpod.io/badge/MadhavendraSinghShaktawat/RobustVideoMatting-runpod-serverless)](https://console.runpod.io/hub/MadhavendraSinghShaktawat/RobustVideoMatting-runpod-serverless)

**Serverless video background removal** on [RunPod](https://runpod.io) using [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) (RVM). Submit a video URL; the worker downloads it, runs matting inference on GPU, uploads the result to **Cloudinary**, and returns the output URL. Scale-to-zero, pay-per-use.

---

## Features

- **Queue-based serverless** — RunPod manages scaling and GPU allocation
- **Input:** public HTTP(S) video URL (streaming download, max 500 MB)
- **Output:** Cloudinary URL of the processed video (transparent background)
- **Progress updates** — Poll the status API for messages: Downloading → Processing → Uploading → Done
- **RVM MobilenetV3** — Fast, good quality; runs on a single GPU (e.g. RTX 3090 / 4090)

---

## Prerequisites

- **Docker** (for building the image)
- **RunPod account** — [Sign up](https://runpod.io)
- **Cloudinary account** — [Sign up](https://cloudinary.com) (for result storage)
- **Docker Hub** (or another registry RunPod supports) — to push the image

---

## Quick Start

### 1. Clone and build

```bash
git clone https://github.com/YOUR_USERNAME/runpod-rvm-worker.git
cd runpod-rvm-worker
docker build -t runpod-rvm-worker:latest .
```

On **Apple Silicon (M1/M2)** or other ARM, build for Linux:

```bash
docker build --platform linux/amd64 -t runpod-rvm-worker:latest .
```

### 2. Tag and push to Docker Hub

Replace `YOUR_DOCKERHUB_USERNAME` with your Docker Hub username.

```bash
docker tag runpod-rvm-worker:latest YOUR_DOCKERHUB_USERNAME/runpod-rvm-worker:latest
docker login
docker push YOUR_DOCKERHUB_USERNAME/runpod-rvm-worker:latest
```

Image URL: **`docker.io/YOUR_DOCKERHUB_USERNAME/runpod-rvm-worker:latest`**

### 3. Create a RunPod Serverless endpoint

1. Go to [RunPod Console](https://www.runpod.io/console) → **Serverless** → **New Endpoint**.
2. **Container image:** `docker.io/YOUR_DOCKERHUB_USERNAME/runpod-rvm-worker:latest`
3. **GPU:** e.g. RTX 3090 or RTX 4090 (1 GPU per worker).
4. **Endpoint type:** Queue.
5. **Environment variables** (required):

   | Name | Value |
   |------|--------|
   | `RUNPOD_INIT_TIMEOUT` | `600` (model load can take several minutes) |
   | `CLOUDINARY_URL` | `cloudinary://API_KEY:API_SECRET@CLOUD_NAME` |

   Get your Cloudinary URL from [Cloudinary Console](https://cloudinary.com/console) → Account → API Keys.

   **Or** use separate vars: `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`.

6. **Deploy** and note the **Endpoint ID**.

### 4. Call the API

**Submit a job** (use your RunPod API key and endpoint ID):

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{"input":{"video_url":"https://example.com/your-video.mp4","output_type":"alpha"}}'
```

Response: `{"id":"JOB_ID","status":"IN_QUEUE",...}`

**Poll status** until `status` is `COMPLETED`:

```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

When completed, `output` contains `output_url` (Cloudinary URL) and, for alpha jobs, `output_format: "webm"`. On failure, `output` contains `error` (string).

---

## Configuration

### Environment variables (RunPod endpoint)

| Variable | Required | Description |
|----------|----------|-------------|
| `RUNPOD_INIT_TIMEOUT` | Recommended | Seconds for worker init (model load). Default may be ~420; set to `600` or higher if workers fail to start. |
| `CLOUDINARY_URL` | Yes* | `cloudinary://API_KEY:API_SECRET@CLOUD_NAME`. Prefer this if you have it. |
| `CLOUDINARY_CLOUD_NAME` | Yes* | Cloud name (if not using `CLOUDINARY_URL`). |
| `CLOUDINARY_API_KEY` | Yes* | API key. |
| `CLOUDINARY_API_SECRET` | Yes* | API secret. |
| `RUNPOD_LOG_LEVEL` | No | `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default `INFO`. |

\* Either `CLOUDINARY_URL` or all three of `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`.

### Job input

| Field | Required | Description |
|-------|----------|-------------|
| `video_url` | **Yes** | HTTP(S) URL of the source video. Max size 500 MB. |
| `output_type` | No | `"greenscreen"` (default) = H.264 MP4 with green background; `"alpha"` = WebM VP9 with alpha; `"alpha_prores"` = ProRes 4444 with alpha (faster encode, larger file; good for Safari). |
| `downsample_ratio` | No | RVM downsample ratio (e.g. `0.25` for 1080p). Default `0.25`. Lower = faster, higher = better quality. |
| `speed` or `preset` | No | `"fast"` = use downsample_ratio 0.125 (faster, lower quality); `"quality"` / `"best"` = prefer quality when downsample_ratio not set. |
| `cloudinary_folder` | No | Cloudinary folder for the output video. Default `rvm-output`. |

### Job output

- **Success:** `output` has `output_url` (Cloudinary URL) and, for alpha jobs, `output_format` (`"webm"` or `"prores"`) so the client can treat the asset as transparent video.
- **Failure:** `output` has `error` (string message).

### Progress

While the job runs, the status response includes a progress message, e.g.:

- `Downloading (0%)` … `Downloading (100%)`
- `Processing (0%)` … `Processing (100%)`
- `Uploading to Cloudinary (0%)` … `Uploading to Cloudinary (100%)`
- `Done (100%)`

---

## Local testing (optional)

With a local GPU and Python environment (or by running the Docker image with `--gpus all`):

```bash
# Direct test (single job, then exit)
python src/handler.py --test_input '{"input":{"video_url":"https://example.com/sample.mp4"}}'

# Local API server (simulate RunPod endpoint)
python src/handler.py --rp_serve_api
# Then: curl -X POST http://localhost:8000/runsync -H "Content-Type: application/json" -d '{"input":{"video_url":"..."}}'
```

Requires GPU and dependencies; see [RunPod local testing](https://docs.runpod.io/serverless/development/local-testing).

---

## Project structure

```
runpod-rvm-worker/
├── Dockerfile          # PyTorch CUDA base, RVM clone, patched inference_utils
├── requirements.txt    # runpod, brotli, torch, av, cloudinary, etc.
├── src/
│   └── handler.py      # RunPod handler: download → RVM → Cloudinary → return URL
├── patches/
│   └── inference_utils.py  # RVM VideoWriter fix for PyAV (Fraction for frame rate)
└── README.md
```

The image is self-contained: RVM and the MobilenetV3 checkpoint are cloned and downloaded during the Docker build. **FFmpeg** is installed from a pinned static build ([BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds)) so `output_type` **alpha** (WebM VP9) and **alpha_prores** (ProRes 4444) work reliably; distro/conda FFmpeg often lacks the required encoders or uses different CLI options.

---

## FFmpeg requirements (root cause of encoder errors)

The worker needs FFmpeg with:

- **libvpx_vp9** — for WebM VP9 with alpha (`output_type=alpha` and ProRes→WebM fallback).
- **prores_ks** — for ProRes 4444 with alpha (`output_type=alpha_prores`).

**Why this matters:** Unpinned FFmpeg (e.g. from `apt install ffmpeg` or the base image's conda) can be an older or minimal build that (1) misses these encoders, or (2) uses different options (e.g. no `-quality realtime` in 4.3). The Dockerfile pins a static build so version and options are consistent. At startup the worker runs `ffmpeg -encoders` and exits with a clear error if any required encoder is missing.

**If you use a custom image:** Install a full FFmpeg build (e.g. [BtbN](https://github.com/BtbN/FFmpeg-Builds) or [jrottenberg/ffmpeg](https://github.com/jrottenberg/ffmpeg) Docker image) and ensure `ffmpeg` in `PATH` reports both encoders.

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| **"ffmpeg missing required encoders"** / **"Unrecognized option 'quality'"** | Use the project's Dockerfile (it pins a static FFmpeg). If using your own image, install FFmpeg with libvpx and prores_ks (see [FFmpeg requirements](#ffmpeg-requirements-root-cause-of-encoder-errors)). |
| **"Can not decode content-encoding: br"** | Image includes `brotlicffi`/`brotli`. Rebuild and redeploy so new workers use the latest image. |
| **Worker fails to start / init timeout** | Increase `RUNPOD_INIT_TIMEOUT` (e.g. `600` or `800`). |
| **"Failed to return job results. 400 Bad Request"** | Handler returns a plain URL string on success and `{"error": "string"}` on failure. Ensure you’re on the latest handler (no nested `output` or `null` error). |
| **Job stuck IN_PROGRESS** | Usually means the job-done API rejected the result (400). Check worker logs; fix return format and redeploy. |
| **"File size too large" (Cloudinary)** | ProRes files >100 MB are auto-converted to WebM for upload. If upload still fails, check Cloudinary plan limits or set `CLOUDINARY_MAX_FILE_BYTES`. |

---

## License

This project uses:

- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) (license in that repo).
- RunPod SDK, PyTorch, Cloudinary, and other dependencies under their respective licenses.

Use and deploy at your own responsibility; ensure your use of RVM and Cloudinary complies with their terms.
