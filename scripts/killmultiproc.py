#!/usr/bin/env python3

import subprocess
import os
import signal
import sys

def get_ps_aux_output():
    """
    Executes the `ps -aux` command and returns its output as a list of lines.
    """
    try:
        result = subprocess.run(['ps', '-aux'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error executing ps command: {e.stderr}", file=sys.stderr)
        sys.exit(1)

def parse_ps_aux(lines):
    """
    Parses the `ps -aux` output and returns a list of PIDs for processes containing 'multiprocessing.spawn'.
    """
    pids_to_kill = []
    
    # The first line is the header
    for line in lines[1:]:
        if 'multiprocessing.spawn' in line:
            parts = line.split(None, 10)  # Split into max 11 parts
            if len(parts) < 2:
                continue  # Malformed line
            try:
                pid = int(parts[1])
                pids_to_kill.append(pid)
            except ValueError:
                continue  # PID is not an integer
    return pids_to_kill

def kill_process(pid):
    """
    Sends a SIGKILL signal to the process with the given PID.
    """
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Successfully killed process with PID: {pid}")
    except ProcessLookupError:
        print(f"No such process with PID: {pid}")
    except PermissionError:
        print(f"Permission denied when trying to kill PID: {pid}")
    except Exception as e:
        print(f"Failed to kill PID {pid}: {e}")

def main():
    lines = get_ps_aux_output()
    pids = parse_ps_aux(lines)
    
    if not pids:
        print("No processes found containing 'multiprocessing.spawn'.")
        return
    
    print(f"Found {len(pids)} process(es) to kill:")
    for pid in pids:
        print(f" - PID: {pid}")
    
    # Optionally, you can add a confirmation prompt here
    # response = input("Are you sure you want to kill these processes? (y/N): ")
    # if response.lower() != 'y':
    #     print("Operation cancelled.")
    #     return
    
    for pid in pids:
        kill_process(pid)

if __name__ == "__main__":
    main()
