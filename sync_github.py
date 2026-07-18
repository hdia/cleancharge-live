# sync_github.py

"""
Synchronise the local CleanCharge Live repository with GitHub.

The script:
1. Checks that the configured folder is a Git repository.
2. Detects local changes.
3. Pulls remote changes before committing when the working tree is clean.
4. Stages and commits local changes when present.
5. Rebases onto the latest remote branch before pushing.
6. Stops safely if a rebase conflict occurs.
7. Pushes the synchronised branch to GitHub.

Usage:
    python sync_github.py
    python sync_github.py "Describe the update"
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_DIR = Path(r"C:\CleanCharge\cleancharge-live")
BRANCH = "main"
REMOTE = "origin"


def run_git(
    args: Sequence[str],
    *,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a Git command in the configured repository."""
    command = ["git", *args]
    print(f">> {' '.join(command)}")

    result = subprocess.run(
        command,
        cwd=REPO_DIR,
        text=True,
        capture_output=capture_output,
        shell=False,
    )

    if result.stdout:
        print(result.stdout.rstrip())

    if result.stderr:
        print(result.stderr.rstrip())

    return result


def repository_is_valid() -> bool:
    """Confirm that the configured directory exists and is a Git repository."""
    if not REPO_DIR.exists():
        print(f"\nRepository folder does not exist:\n{REPO_DIR}")
        return False

    result = run_git(["rev-parse", "--is-inside-work-tree"])
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def working_tree_status() -> str:
    """Return porcelain-format Git status output."""
    result = run_git(["status", "--short"])
    if result.returncode != 0:
        raise RuntimeError("Unable to read Git status.")
    return result.stdout.strip()


def remote_exists() -> bool:
    """Confirm that the configured remote exists."""
    result = run_git(["remote", "get-url", REMOTE])
    return result.returncode == 0


def rebase_from_remote() -> bool:
    """Fetch and rebase the local branch onto the remote branch."""
    print("\nPulling the latest GitHub changes...")
    result = run_git(["pull", "--rebase", REMOTE, BRANCH])

    if result.returncode == 0:
        print("Remote synchronisation completed successfully.")
        return True

    print(
        "\nRebase did not complete. The script has stopped safely.\n"
        "Review the conflict messages above. After resolving conflicts, run:\n\n"
        "    git add <resolved-files>\n"
        "    git rebase --continue\n\n"
        "To abandon the rebase instead, run:\n\n"
        "    git rebase --abort\n"
    )
    return False


def commit_local_changes(message: str) -> bool:
    """Stage and commit local changes. Return True if a commit was created."""
    status = working_tree_status()

    if not status:
        print("\nNo local changes to commit.")
        return False

    print("\nLocal changes detected. Staging files...")
    if run_git(["add", "--all"]).returncode != 0:
        raise RuntimeError("Unable to stage local changes.")

    result = run_git(["commit", "-m", message])

    if result.returncode == 0:
        print("Local commit created successfully.")
        return True

    staged_check = run_git(["diff", "--cached", "--quiet"])
    if staged_check.returncode == 0:
        print("No commit was required.")
        return False

    raise RuntimeError("Git could not create the commit.")


def push_to_github() -> bool:
    """Push the configured branch to GitHub."""
    print("\nPushing the synchronised branch to GitHub...")
    result = run_git(["push", "-u", REMOTE, BRANCH])

    if result.returncode == 0:
        print("\nDone. CleanCharge Live is synchronised with GitHub.")
        return True

    print(
        "\nPush failed. The remote branch may have changed again while the "
        "script was running.\n"
        "Run the script once more. It will pull and rebase before retrying."
    )
    return False


def main() -> int:
    """Synchronise the CleanCharge Live repository."""
    commit_message = (
        " ".join(sys.argv[1:]).strip()
        or f"Update CleanCharge Live {datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    print("\n==============================================")
    print(" CleanCharge Live Git synchronisation")
    print("==============================================")
    print(f"\nRepository: {REPO_DIR}")
    print(f"Branch: {BRANCH}")
    print(f"Remote: {REMOTE}")

    if not repository_is_valid():
        print("\nThe configured folder is not a valid Git repository.")
        return 1

    if not remote_exists():
        print(
            f"\nGit remote '{REMOTE}' is not configured. "
            "Add the GitHub remote before running this script."
        )
        return 1

    try:
        initial_status = working_tree_status()

        if initial_status:
            print(
                "\nUncommitted local changes are present. "
                "They will be committed before rebasing."
            )
            commit_local_changes(commit_message)

        if not rebase_from_remote():
            return 1

        # Catch files created or modified during synchronisation.
        commit_local_changes(commit_message)

        return 0 if push_to_github() else 1

    except RuntimeError as exc:
        print(f"\nSynchronisation stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
