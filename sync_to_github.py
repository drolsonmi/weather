import subprocess
import os
import shutil
from datetime import datetime

# Path to your repo
REPO_PATH = r"C:\Users\GramSC\Documents\weather\data"

# Files you want to track (optional: leave empty to track all)
SOURCE_FILES = [
    r"C:\Campbellsci\LoggerNet\Snow Weather_Daily.dat",
    r"C:\Campbellsci\LoggerNet\Snow Weather_FifteenMin.dat",
    r"C:\Campbellsci\LoggerNet\Snow Weather_FiveMin.dat"
]

DEST_FILES = [
    r"C:\Users\GramSC\Documents\weather\data\Snow Weather_Daily.dat",
    r"C:\Users\GramSC\Documents\weather\data\Snow Weather_FifteenMin.dat",
    r"C:\Users\GramSC\Documents\weather\data\Snow Weather_FiveMin.dat"
]

def run(cmd):
    cmd = [GIT_PATH] + cmd[1:] if cmd[0] == "git" else cmd
    result = subprocess.run(cmd, cwd=REPO_PATH, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
    return result.stdout.strip()

def copy_files():
    for src, dest in zip(SOURCE_FILES, DEST_FILES):
        dest_path = os.path.join(REPO_PATH, dest)
        shutil.copy2(src, dest_path)
        print(f"Copied {src} to {dest_path}")

def main():
    os.chdir(REPO_PATH)

    # Step 1: Copy files into repo
    copy_files()

    # Step 2: Add files
    run(["git", "add", "."])

    # Step 3: Check for changes
    status = run(["git", "status", "--porcelain"])
    if not status:
        print("No changes to commit.")
        return

    # Step 4: Commit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run(["git", "commit", "-m", f"Auto update {timestamp}"])

    # Step 5: Push
    run(["git", "push"])
    print("Synced to GitHub.")

if __name__ == "__main__":
    main()
