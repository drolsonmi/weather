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

# Error
Copied C:\Campbellsci\LoggerNet\Snow Weather_Daily.dat to C:\Users\GramSC\Documents\weather\data\Snow Weather_Daily.dat
Copied C:\Campbellsci\LoggerNet\Snow Weather_FifteenMin.dat to C:\Users\GramSC\Documents\weather\data\Snow Weather_FifteenMin.dat
Copied C:\Campbellsci\LoggerNet\Snow Weather_FiveMin.dat to C:\Users\GramSC\Documents\weather\data\Snow Weather_FiveMin.dat
Traceback (most recent call last):
  File "C:\Users\GramSC\Documents\weather\sync_to_github.py", line 58, in <module>
    main()
    ~~~~^^
  File "C:\Users\GramSC\Documents\weather\sync_to_github.py", line 41, in main
    run(["git", "add", "."])
    ~~~^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\GramSC\Documents\weather\sync_to_github.py", line 23, in run
    result = subprocess.run(cmd, cwd=REPO_PATH, capture_output=True, text=True)
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3568.0_x64__qbz5n2kfra8p0\Lib\subprocess.py", line 554, in run
    with Popen(*popenargs, **kwargs) as process:
         ~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3568.0_x64__qbz5n2kfra8p0\Lib\subprocess.py", line 1039, in __init__
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        pass_fds, cwd, env,
                        ^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
                        ^^^^^^^^^^^^^^^^^^^^^^
                        start_new_session, process_group)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3568.0_x64__qbz5n2kfra8p0\Lib\subprocess.py", line 1554, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
                       ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
                             # no special security
                             ^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
                             cwd,
                             ^^^^
                             startupinfo)
                             ^^^^^^^^^^^^
FileNotFoundError: [WinError 2] The system cannot find the file specified
PS C:\Users\GramSC\Documents\weather> python .\sync_to_github.py
  File "C:\Users\GramSC\Documents\weather\sync_to_github.py", line 7
    REPO_PATH = r"C:\Users\GramSC\Documents\weather\data\"
                ^
SyntaxError: unterminated string literal (detected at line 7); perhaps you escaped the end quote?
PS C:\Users\GramSC\Documents\weather> python .\sync_to_github.py
Copied C:\Campbellsci\LoggerNet\Snow Weather_Daily.dat to C:\Users\GramSC\Documents\weather\data\Snow Weather_Daily.dat
Copied C:\Campbellsci\LoggerNet\Snow Weather_FifteenMin.dat to C:\Users\GramSC\Documents\weather\data\Snow Weather_FifteenMin.dat
Copied C:\Campbellsci\LoggerNet\Snow Weather_FiveMin.dat to C:\Users\GramSC\Documents\weather\data\Snow Weather_FiveMin.dat
Traceback (most recent call last):
  File "C:\Users\GramSC\Documents\weather\sync_to_github.py", line 58, in <module>
    main()
    ~~~~^^
  File "C:\Users\GramSC\Documents\weather\sync_to_github.py", line 41, in main
    run(["git", "add", "."])
    ~~~^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\GramSC\Documents\weather\sync_to_github.py", line 23, in run
    result = subprocess.run(cmd, cwd=REPO_PATH, capture_output=True, text=True)
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3568.0_x64__qbz5n2kfra8p0\Lib\subprocess.py", line 554, in run
    with Popen(*popenargs, **kwargs) as process:
         ~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3568.0_x64__qbz5n2kfra8p0\Lib\subprocess.py", line 1039, in __init__
    self._execute_child(args, executable, preexec_fn, close_fds,
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        pass_fds, cwd, env,
                        ^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
                        gid, gids, uid, umask,
                        ^^^^^^^^^^^^^^^^^^^^^^
                        start_new_session, process_group)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3568.0_x64__qbz5n2kfra8p0\Lib\subprocess.py", line 1554, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
                       ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
                             # no special security
                             ^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
                             cwd,
                             ^^^^
                             startupinfo)
                             ^^^^^^^^^^^^
FileNotFoundError: [WinError 2] The system cannot find the file specified