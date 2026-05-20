import subprocess
import os
from datetime import datetime

# Path to your repo
REPO_PATH = r"C:\path\to\your\repo"

# Files you want to track (optional: leave empty to track all)
FILES = [
    "file1.txt",
    "file2.csv",
    "file3.json"
]

def run(cmd):
    result = subprocess.run(cmd, cwd=REPO_PATH, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
    return result.stdout.strip()

def main():
    os.chdir(REPO_PATH)

    # Add files (or use '.' for everything)
    if FILES:
        for f in FILES:
            run(["git", "add", f])
    else:
        run(["git", "add", "."])

    # Check if anything changed
    status = run(["git", "status", "--porcelain"])

    if not status:
        print("No changes to commit.")
        return

    # Commit with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run(["git", "commit", "-m", f"Auto update {timestamp}"])

    # Push
    run(["git", "push"])

    print("Synced to GitHub.")

if __name__ == "__main__":
    main()
