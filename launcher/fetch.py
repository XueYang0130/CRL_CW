#!/usr/bin/env python3
"""Fetch results and logs from cluster back to local Mac.

Usage:
    # Fetch everything
    python launcher/fetch.py
    
    # Fetch only specific experiment
    python launcher/fetch.py --job-name full_bc_seed0
    
    # Dry run (print commands only)
    python launcher/fetch.py --dry-run
    
    # Custom paths
    python launcher/fetch.py --paths outputs/cw10_continual logs/cluster
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


def rsync_from_cluster(
    config: dict,
    remote_path: str,
    local_path: str,
    dry_run: bool = False,
) -> bool:
    """Rsync files from cluster to local.
    
    Args:
        remote_path: Path on cluster (relative to remote_dir)
        local_path: Local destination path
        dry_run: If True, only print command without executing
    
    Returns:
        True if successful, False otherwise
    """
    cluster = config["cluster"]
    remote_full = f"{cluster['user']}@{cluster['host']}:{cluster['remote_dir']}/{remote_path}"
    
    cmd = f"rsync -avz --progress {remote_full} {local_path}"
    
    if dry_run:
        print(f"  {cmd}")
        return True
    
    print(f"Fetching {remote_path} ...")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"  ERROR: Failed to fetch {remote_path}")
        return False
    
    print(f"  ✓ Saved to {local_path}")
    return True


def fetch_results(
    config: dict,
    job_name: str = None,
    paths: list = None,
    dry_run: bool = False,
) -> bool:
    """Fetch experiment results from cluster.
    
    Args:
        config: Cluster configuration
        job_name: Specific job name to fetch (optional)
        paths: Custom paths to fetch (optional)
        dry_run: If True, only print commands
    
    Returns:
        True if all fetches successful
    """
    if dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN - Commands that would be executed:")
        print("=" * 70 + "\n")
    
    # Default paths to fetch
    if paths is None:
        if job_name:
            # Fetch specific job outputs and logs
            # Support both Slurm (job_name_*.out) and direct SSH (job_name.out) formats
            paths = [
                f"outputs/cw10_continual/*{job_name}*",
                f"logs/cluster/{job_name}*.out",
                f"logs/cluster/{job_name}*.err",
                f"logs/cluster/{job_name}.out",
                f"logs/cluster/{job_name}.err",
            ]
        else:
            # Fetch everything
            paths = [
                "outputs/",
                "logs/cluster/",
            ]
    
    success = True
    for remote_path in paths:
        # Determine local destination
        if remote_path.startswith("outputs/"):
            local_dest = "./outputs/"
        elif remote_path.startswith("logs/"):
            local_dest = "./logs/"
        else:
            local_dest = "./"
        
        # Ensure local directory exists
        if not dry_run:
            Path(local_dest).mkdir(parents=True, exist_ok=True)
        
        # Fetch
        if not rsync_from_cluster(config, remote_path, local_dest, dry_run):
            success = False
    
    if dry_run:
        print("\n" + "=" * 70 + "\n")
    
    return success


def list_remote_logs(config: dict, job_name: str = None) -> None:
    """List available log files on cluster."""
    cluster = config["cluster"]
    ssh = f"ssh {cluster['user']}@{cluster['host']}"
    
    pattern = f"{job_name}*" if job_name else "*"
    cmd = f"{ssh} 'ls -lth {cluster['remote_dir']}/logs/cluster/{pattern} 2>/dev/null | head -20'"
    
    print(f"\nRecent cluster logs:")
    print("-" * 70)
    subprocess.run(cmd, shell=True)
    print("-" * 70)


def check_job_status(config: dict, job_name: str = None) -> None:
    """Check if jobs are still running on cluster."""
    cluster = config["cluster"]
    ssh = f"ssh {cluster['user']}@{cluster['host']}"
    
    # Try squeue first (Slurm)
    pattern = job_name if job_name else ""
    cmd = f"{ssh} 'squeue -u {cluster['user']} -o \"%.10i %.9P %.30j %.8u %.10T %.10M\" | grep -i \"{pattern}\" || echo \"(no running jobs)\"'"
    
    print(f"\nCluster job status:")
    print("-" * 70)
    subprocess.run(cmd, shell=True)
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch results from cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--job-name",
        help="Specific job name to fetch (fetches related outputs and logs)",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Custom remote paths to fetch (relative to remote_dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available log files on cluster",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check job status on cluster",
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
    
    # Check status only
    if args.status:
        check_job_status(config, args.job_name)
        sys.exit(0)
    
    # List logs only
    if args.list:
        list_remote_logs(config, args.job_name)
        sys.exit(0)
    
    # Fetch results
    print("\nFetching results from cluster...")
    if args.job_name:
        print(f"Job: {args.job_name}")
    
    success = fetch_results(
        config=config,
        job_name=args.job_name,
        paths=args.paths,
        dry_run=args.dry_run,
    )
    
    if success and not args.dry_run:
        print("\n✓ All results fetched successfully")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
