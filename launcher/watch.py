#!/usr/bin/env python3
"""Watch cluster job logs in real-time.

Usage:
    # Watch specific job
    python launcher/watch.py --job-name clonex_seed1
    
    # Watch most recent job
    python launcher/watch.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str = "launcher/cluster_config.yaml") -> dict:
    """Load cluster configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def watch_log(config: dict, job_name: str = None) -> None:
    """Watch cluster job log in real-time.
    
    Args:
        config: Cluster configuration
        job_name: Job name to watch (if None, watches most recent)
    """
    cluster = config["cluster"]
    ssh_prefix = f"ssh {cluster['user']}@{cluster['host']}"
    remote_log_dir = f"{cluster['remote_dir']}/logs/cluster"
    
    if job_name:
        # Watch specific job
        log_pattern = f"{remote_log_dir}/{job_name}_*.out"
        print(f"\nWatching logs for job: {job_name}")
    else:
        # Watch most recent log file
        log_pattern = f"{remote_log_dir}/*.out"
        print(f"\nWatching most recent cluster log...")
    
    print("=" * 70)
    print("Press Ctrl+C to stop watching")
    print("=" * 70 + "\n")
    
    # Find and tail the log file
    if job_name:
        # Wait for specific job log to appear, then tail
        cmd = f"{ssh_prefix} 'while ! ls {log_pattern} 2>/dev/null; do sleep 2; done; tail -f {log_pattern}'"
    else:
        # Tail most recent log
        cmd = f"{ssh_prefix} 'tail -f $(ls -t {log_pattern} 2>/dev/null | head -1)'"
    
    try:
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        print("\n\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(
        description="Watch cluster job logs in real-time",
    )
    parser.add_argument(
        "--job-name",
        help="Job name to watch (if omitted, watches most recent)",
    )
    parser.add_argument(
        "--config",
        default="launcher/cluster_config.yaml",
        help="Path to cluster config",
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)
    
    watch_log(config, args.job_name)


if __name__ == "__main__":
    main()
