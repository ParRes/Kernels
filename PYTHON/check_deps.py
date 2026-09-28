#!/usr/bin/env python3
"""
Check for (and optionally install) the third-party Python packages
used by the scripts in this directory: numpy, scipy, numba, cupy,
mpi4py, shmem4py.

Usage:
    python3 check_deps.py            # check only, report missing
    python3 check_deps.py --install  # also pip-install anything missing
"""

import argparse
import importlib
import subprocess
import sys

# module name -> pip package name
DEPENDENCIES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "numba": "numba",
    "cupy": "cupy",
    "mpi4py": "mpi4py",
    "shmem4py": "shmem4py",
}


def check(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# packages whose pyproject.toml build needs to see the already-installed
# cffi rather than the fresh one pip would fetch into an isolated build env
# (avoids a cffi/_cffi_backend version-mismatch build failure)
NO_BUILD_ISOLATION = {"shmem4py"}


def install(pip_name):
    cmd = [sys.executable, "-m", "pip", "install"]
    if pip_name in NO_BUILD_ISOLATION:
        cmd.append("--no-build-isolation")
    cmd.append(pip_name)
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true",
                         help="install any missing dependencies with pip")
    args = parser.parse_args()

    missing = []
    for module_name, pip_name in DEPENDENCIES.items():
        if check(module_name):
            print(f"[ok]      {module_name}")
        else:
            print(f"[missing] {module_name}")
            missing.append((module_name, pip_name))

    if not missing:
        print("\nAll dependencies are installed.")
        return 0

    print(f"\nMissing {len(missing)} package(s): "
          f"{', '.join(m for m, _ in missing)}")

    if not args.install:
        print("Re-run with --install to install them.")
        return 1

    failed = []
    for module_name, pip_name in missing:
        print(f"\nInstalling {pip_name} ...")
        try:
            install(pip_name)
        except subprocess.CalledProcessError:
            failed.append(pip_name)

    if failed:
        print(f"\nFailed to install: {', '.join(failed)}")
        print("(cupy and shmem4py typically require a matching CUDA "
              "toolkit / OpenSHMEM installation; see their docs.)")
        return 1

    print("\nAll missing dependencies installed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
