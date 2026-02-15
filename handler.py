"""
Root entrypoint for RunPod Hub. Delegates to the real handler in src/handler.py.
When RunPod builds the image, the Dockerfile copies src/handler.py to /app/handler.py and runs it.
This file exists so the repo has a handler at root for Hub detection; local runs can use it too.
"""
import sys
import os

# Ensure src is on path when running from repo root (e.g. python handler.py)
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Import runs the real handler (brotli check + runpod.serverless.start)
import src.handler
