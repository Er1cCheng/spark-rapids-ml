import argparse
import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple, Union

from pylint import epylint

# This script is copied from dmlc/xgboost

CURDIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
PROJECT_ROOT = os.path.normpath(os.path.join(CURDIR, os.path.pardir))
SRC_PATHS = [
  "src/spark_rapids_ml",
  "tests",
  "benchmark",
]


class DirectoryExcursion:
    def __init__(self, path: Union[os.PathLike, str]):
        self.path = path
        self.curdir = os.path.normpath(os.path.abspath(os.path.curdir))

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *args):
        os.chdir(self.curdir)


def run_formatter(rel_path: str) -> bool:
    path = os.path.join(PROJECT_ROOT, rel_path)
    isort_ret = subprocess.run(["isort", "--check", "--profile=black", path]).returncode
    black_ret = subprocess.run(["black", "--check", rel_path]).returncode
    if isort_ret != 0 or black_ret != 0:
        msg = (
            "Please run the following command on your machine to address the format"
            f" errors:\n isort --profile=black {rel_path}\n black {rel_path}\n"
        )
        print(msg, file=sys.stdout)
        return False
    return True


def run_mypy(rel_path: str) -> bool:
    with DirectoryExcursion(PROJECT_ROOT):
        path = os.path.join(PROJECT_ROOT, rel_path)
        ret = subprocess.run(["mypy", path])
        return ret.returncode == 0


class PyLint:
    """A helper for running pylint, mostly copied from dmlc-core/scripts."""

    def __init__(self) -> None:
        self.pypackage_root = PROJECT_ROOT
        self.pylint_cats = set(["error", "warning", "convention", "refactor"])
        self.pylint_opts = [
            "--extension-pkg-whitelist=numpy",
            "--rcfile=" + os.path.join(self.pypackage_root, ".pylintrc"),
        ]

    def run(self, path: str) -> Tuple[Dict, str, str]:
        (pylint_stdout, pylint_stderr) = epylint.py_run(
            " ".join([str(path)] + self.pylint_opts), return_std=True
        )
        emap = {}
        err = pylint_stderr.read()

        out = []
        for line in pylint_stdout:
            out.append(line)
            key = line.split(":")[-1].split("(")[0].strip()
            if key not in self.pylint_cats:
                continue
            if key not in emap:
                emap[key] = 1
            else:
                emap[key] += 1

        return {path: emap}, err, "\n".join(out)

    def __call__(self) -> bool:
        all_errors: Dict[str, Dict[str, int]] = {}

        def print_summary_map(result_map: Dict[str, Dict[str, int]]) -> int:
            """Print summary of certain result map."""
            if len(result_map) == 0:
                return 0
            ftype = "Python"
            npass = sum(1 for x in result_map.values() if len(x) == 0)
            print(f"====={npass}/{len(result_map)} {ftype} files passed check=====")
            for fname, emap in result_map.items():
                if len(emap) == 0:
                    continue
                print(
                    f"{fname}: {sum(emap.values())} Errors of {len(emap)} Categories map={str(emap)}"
                )
            return len(result_map) - npass

        all_scripts = []
        for root, dirs, files in os.walk(self.pypackage_root):
            for f in files:
                if f.endswith(".py"):
                    all_scripts.append(os.path.join(root, f))

        with Pool(cpu_count()) as pool:
            error_maps = pool.map(self.run, all_scripts)
            for emap, err, out in error_maps:
                print(out)
                if len(err) != 0:
                    print(err)
                all_errors.update(emap)

        nerr = print_summary_map(all_errors)
        return nerr == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", action='store_true', default=False)
    parser.add_argument("--type-check", action='store_true', default=False)
    parser.add_argument("--pylint", action='store_true', default=False)
    args = parser.parse_args()
    if args.format:
        print("Formatting...")
        if not all(run_formatter(path) for path in SRC_PATHS):
            sys.exit(-1)

    if args.type_check:
        print("Type checking...")
        if not all(run_mypy(path) for path in SRC_PATHS):
            sys.exit(-1)

    if args.pylint:
        print("Running PyLint...")
        if not PyLint()():
            sys.exit(-1)
