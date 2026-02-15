# Release notes (copy for GitHub Release)

## Release title (use as-is or shorten)

**v1.0.0 – RobustVideoMatting serverless on RunPod**

---

## Release description (paste into "Describe this release")

### What's in this release

- **Video background removal** on RunPod Serverless using [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) (RVM).
- **Input:** Public video URL (HTTP/HTTPS, max 500 MB).
- **Output:** Cloudinary URL of the processed video (transparent background).
- **Progress:** Poll the status API for Downloading → Processing → Uploading → Done.

### Requirements

- **GPU:** 1x GPU (e.g. RTX 3090 / RTX 4090).
- **Env:** Set `CLOUDINARY_URL` (from [Cloudinary Console](https://cloudinary.com/console)) so the worker can upload results. Optionally set `RUNPOD_INIT_TIMEOUT=600` if model load is slow.

### Job input

| Field | Required | Description |
|-------|----------|-------------|
| `video_url` | Yes | HTTP(S) URL of the source video. |
| `downsample_ratio` | No | RVM downsample ratio (default `0.25`). |
| `cloudinary_folder` | No | Cloudinary folder for output (default `rvm-output`). |

### Quick test

After deploying from the Hub, submit a job with your RunPod API key and endpoint ID:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{"input":{"video_url":"https://example.com/sample.mp4"}}'
```

Then poll `/status/JOB_ID` until `status` is `COMPLETED`; `output` is the result URL.

---

**Full docs:** See the [README](https://github.com/MadhavendraSinghShaktawat/RobustVideoMatting-runpod-serverless#readme) in this repo.
