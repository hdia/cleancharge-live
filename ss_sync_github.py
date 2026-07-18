"""
Synchronise the local CleanCharge Live repository with GitHub.

Workflow:
1. Display repository status.
2. Stage all changes.
3. Commit only when staged changes exist.
4. Push the main branch to origin.

Usage:
    python sync_github.py
    python sync_github.py "Custom commit message"

This script does not resolve merge conflicts automatically.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_DIR = Path(r"C:\CleanCharge\cleancharge-live")
REMOTE = "origin"
BRANCH = "main"


def run_git(
    arguments: Sequence[str],
    *,
    capture_output: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a Git command in the configured repository."""
    command = ["git", *arguments]
    print(f">> {' '.join(command)}")

    result = subprocess.run(
        command,
        cwd=REPO_DIR,
        text=True,
        capture_output=capture_output,
        check=False,
    )

    if capture_output:
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Git command failed with exit code {result.returncode}: "
            f"{' '.join(command)}"
        )

    return result


def ensure_repository() -> None:
    """Confirm that the configured folder is a Git repository."""
    if not REPO_DIR.exists():
        raise FileNotFoundError(
            f"Repository directory does not exist: {REPO_DIR}"
        )

    result = run_git(
        ["rev-parse", "--is-inside-work-tree"],
        capture_output=True,
    )

    if result.returncode != 0 or result.stdout.strip() != "true":
        raise RuntimeError(
            f"The configured folder is not a Git repository: {REPO_DIR}"
        )


def remote_exists() -> bool:
    """Return True when the configured remote exists."""
    result = run_git(
        ["remote", "get-url", REMOTE],
        capture_output=True,
    )
    return result.returncode == 0


def staged_changes_exist() -> bool:
    """Return True when staged changes are ready to commit."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=REPO_DIR,
        text=True,
        check=False,
    )
    return result.returncode == 1


def main() -> int:
    """Stage, commit and push repository changes."""
    try:
        ensure_repository()

        print()
        print("==============================================")
        print(" CleanCharge Live Git synchronisation")
        print("==============================================")
        print()
        print(f"Repository: {REPO_DIR}")
        print(f"Branch: {BRANCH}")
        print()

        run_git(["status", "--short"], capture_output=True)
        run_git(["add", "--all"], check=True)

        if staged_changes_exist():
            if len(sys.argv) > 1:
                commit_message = " ".join(sys.argv[1:]).strip()
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                commit_message = f"Update CleanCharge Live {timestamp}"

            result = run_git(
                ["commit", "-m", commit_message],
                capture_output=True,
            )

            if result.returncode != 0:
                print("\nCommit failed. No push was attempted.")
                return result.returncode
        else:
            print("\nNo staged changes were found. Nothing to commit.")

        if not remote_exists():
            print(
                "\nNo 'origin' remote is configured."
                "\nCreate the GitHub repository, then run:"
                "\n"
                "\n  git remote add origin "
                "https://github.com/hdia/cleancharge-live.git"
                "\n  git push -u origin main"
            )
            return 1

        result = run_git(
            ["push", "-u", REMOTE, BRANCH],
            capture_output=True,
        )

        if result.returncode != 0:
            print(
                "\nPush failed. The remote branch may contain changes "
                "that are not present locally."
                "\nReview the repository status before pulling or rebasing."
            )
            return result.returncode

        print("\nDone. CleanCharge Live is synchronised with GitHub.")
        return 0

    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
