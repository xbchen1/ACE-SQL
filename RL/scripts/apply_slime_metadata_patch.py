#!/usr/bin/env python3
"""Forward ACE metadata and exclude it from numeric logs in pinned slime v0.2.3."""

import argparse
import subprocess
from pathlib import Path

PINNED_COMMIT = "6195417d45f272b72d04f619ea334613135d1c1f"
FILES = {
    "slime/backends/megatron_utils/model.py": (
        '                "advantages",\n                "returns",',
        '                "advantages",\n                "metadata",\n                "returns",',
    ),
    "slime/ray/rollout.py": (
        '                "sample_indices",\n                "rollout_log_probs",',
        '                "sample_indices",\n                "metadata",\n                "rollout_log_probs",',
    ),
    "slime/backends/megatron_utils/data.py": (
        '                "dynamic_global_batch_size",\n            ]:',
        '                "dynamic_global_batch_size",\n                "metadata",\n            ]:',
    ),
}


def run_git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def apply(root: Path, check_only=False):
    root = root.resolve()
    if run_git(root, "rev-parse", "HEAD").strip() != PINNED_COMMIT:
        raise ValueError(f"Expected slime v0.2.3 commit {PINNED_COMMIT}; checkout does not match")
    states = []
    for relative, (before, after) in FILES.items():
        original = run_git(root, "show", f"{PINNED_COMMIT}:{relative}")
        if original.count(before) != 1:
            raise ValueError(f"Pinned source context is ambiguous in {relative}")
        patched = original.replace(before, after)
        current = (root / relative).read_text()
        if current == patched:
            states.append("applied")
        elif current == original:
            states.append("clean")
        else:
            raise ValueError(f"Refusing to modify local changes in {relative}")
    if states == ["applied"] * len(FILES):
        print("ACE-SQL metadata patch already applied and verified")
        return
    if check_only:
        raise ValueError("ACE-SQL metadata patch is not applied")
    if states != ["clean"] * len(FILES):
        raise ValueError("ACE-SQL metadata patch is partially applied; restore its files before retrying")
    patch = Path(__file__).resolve().parents[1] / "patches" / "slime-v0.2.3-ace-metadata.patch"
    run_git(root, "apply", "--check", str(patch))
    run_git(root, "apply", str(patch))
    apply(root, check_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slime_dir", type=Path, help="Pinned slime source checkout")
    parser.add_argument("--check", action="store_true", help="Verify without writing")
    options = parser.parse_args()
    try:
        apply(options.slime_dir, check_only=options.check)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ACE-SQL metadata patch: {error}\n")
