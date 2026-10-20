"""Record the installed publication stack and verify packaged Python sources."""

import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch
import triton


def main():
    root = Path(__file__).resolve().parent
    records = []
    packages = {}
    for name in ("torch", "triton", "triton-rocm", "pytorch-triton-rocm", "pytorch-triton"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        packages[name] = dist.version
        for item in dist.files or []:
            if item.suffix not in {".py", ".jinja"} or not item.hash:
                continue
            data = dist.locate_file(item).read_bytes()
            digest = hashlib.new(item.hash.mode, data).digest()
            encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
            if encoded != item.hash.value:
                raise RuntimeError(f"Installed package source differs from wheel RECORD: {item}")
            records.append({"distribution": name, "path": str(item),
                            "sha256": hashlib.sha256(data).hexdigest()})
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (root / "environment/requirements.txt").write_text(freeze)
    (root / "environment/packaged-source-sha256.json").write_text(
        json.dumps(records, indent=2) + "\n")
    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "torch_path": str(Path(torch.__file__).resolve()), "hip": torch.version.hip,
        "triton": triton.__version__, "distributions": packages,
        "wheel_record_verified_python_files": len(records),
        "torch_build_config": torch.__config__.show(),
    }
    (root / "environment/software.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
