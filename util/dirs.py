# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# util/dirs.py
import os
from pathlib import Path

# code-location based canonical repo root
REINFLOW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ensure env var exists; if exists, normalize; if not, fallback to REINFLOW_DIR
_env_reinflow_dir = os.environ.get("REINFLOW_DIR")
if _env_reinflow_dir:
    # normalize both
    try:
        if os.path.exists(_env_reinflow_dir) and os.path.exists(REINFLOW_DIR):
            # use samefile if filesystem supports it
            try:
                if not os.path.samefile(REINFLOW_DIR, os.path.abspath(_env_reinflow_dir)):
                    # Not the same file — warn but continue using REINFLOW_DIR
                    print(
                        f"Warning: REINFLOW_DIR environment variable {_env_reinflow_dir} "
                        f"does not match detected REINFLOW_DIR {REINFLOW_DIR}. "
                        "Using code-detected REINFLOW_DIR."
                    )
            except Exception:
                # fallback to comparing normalized strings
                if os.path.abspath(_env_reinflow_dir) != os.path.abspath(REINFLOW_DIR):
                    print(
                        f"Warning: REINFLOW_DIR env ({_env_reinflow_dir}) != detected ({REINFLOW_DIR})."
                    )
        else:
            # If env path doesn't exist, warn
            print(f"Warning: REINFLOW_DIR env set to {_env_reinflow_dir} which does not exist.")
    except Exception:
        pass
else:
    # set env var for downstream code that expects it (optional)
    os.environ.setdefault("REINFLOW_DIR", REINFLOW_DIR)

REINFLOW_CFG_DIR = os.path.join(REINFLOW_DIR, "cfg")

# Use env vars if present, otherwise sensible defaults inside repository
REINFLOW_DATA_DIR = os.environ.get("REINFLOW_DATA_DIR", os.path.join(REINFLOW_DIR, "data"))
REINFLOW_LOG_DIR = os.environ.get("REINFLOW_LOG_DIR", os.path.join(REINFLOW_DIR, "logs"))

# Optionally create dirs if they don't exist
for d in (REINFLOW_DATA_DIR, REINFLOW_LOG_DIR, REINFLOW_CFG_DIR):
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
    except Exception:
        # don't crash on permission errors; leave it to the caller
        pass
