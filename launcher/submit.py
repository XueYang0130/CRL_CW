#!/usr/bin/env python3
"""Simple cluster job submitter for CRL_CW.

Usage:
    # Dry run (print commands only)
    python launcher/submit.py "your command here"
    
    # Real submit
    python launcher/submit.py "your command here" --submit
    
    # With custom job name
    python launcher/submit.py "your command here" --submit --job-name my_experiment

Example:
    python launcher/submit.py \\
        "caffeinate -dimsu env PYTHONPATH=src python scripts/run.py --mode continual --method full_bc --seed 0" \\
        --submit --job-name full_bc_seed0
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str = "launcher/cluster_config.yaml") -> dict:
    """Load cluster configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_slurm_script(
    command: str,
    config: dict,
    job_name: str = "crl_cw_job",
) -> str:
    """Generate a Slurm batch script."""
    cluster = config["cluster"]
    slurm = cluster["slurm"]
    remote_dir = cluster["remote_dir"]
    python_exec = cluster.get("python_exec", "python")
    
    # Remove 'caffeinate' and other Mac-specific commands
    clean_command = command.replace("caffeinate -dimsu", "").strip()
    
    # Replace 'python' with cluster python executable
    if clean_command.startswith("env PYTHONPATH=src python"):
        clean_command = clean_command.replace(
            "env PYTHONPATH=src python",
            f"env PYTHONPATH=src {python_exec}"
        )
    elif clean_command.startswith("python"):
        clean_command = clean_command.replace("python", python_exec, 1)
    
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={slurm['partition']}
#SBATCH --time={slurm['time']}
#SBATCH --mem-per-cpu={slurm['mem_per_cpu']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --output=logs/cluster/{job_name}_%j.out
#SBATCH --error=logs/cluster/{job_name}_%j.err

echo "Job started: $(date)"
echo "Job name: {job_name}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

cd {remote_dir} || exit 1
mkdir -p logs/cluster

{clean_command}

echo ""
echo "Job finished: $(date)"
echo "Exit code: $?"
"""
    return script


def rsync_command(config: dict) -> str:
    """Generate rsync command to sync project to cluster."""
    cluster = config["cluster"]
    excludes = " ".join(f"--exclude='{e}'" for e in cluster["rsync_excludes"])
    remote = f"{cluster['user']}@{cluster['host']}:{cluster['remote_dir']}/"
    return f"rsync -avz --delete {excludes} ./ {remote}"


def submit_job(
    command: str,
    config: dict,
    job_name: str = "crl_cw_job",
    dry_run: bool = True,
) -> bool:
    """Submit job to cluster.
    
    Auto-detects submission mode:
    - If sbatch is available: use Slurm
    - Otherwise: run directly via nohup over SSH
    
    Returns:
        True if successful, False otherwise.
    """
    cluster = config["cluster"]
    ssh_prefix = f"ssh {cluster['user']}@{cluster['host']}"
    
    # Check if sbatch is available
    check_cmd = f"{ssh_prefix} 'which sbatch 2>/dev/null'"
    has_slurm = subprocess.run(check_cmd, shell=True, capture_output=True).returncode == 0
    
    if dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN - Commands that would be executed:")
        print("=" * 70)
        print("\n# Step 1: Sync project to cluster")
        print(rsync_command(config))
        
        if has_slurm:
            print(f"\n# Step 2: Slurm script would be saved to: scripts/slurm/{job_name}.sh")
            print("\n# Step 3: Slurm script content:")
            print("-" * 70)
            print(generate_slurm_script(command, config, job_name))
            print("-" * 70)
            print(f"\n# Step 4: Submit job")
            print(f"{ssh_prefix} 'cd {cluster['remote_dir']} && sbatch scripts/slurm/{job_name}.sh'")
        else:
            print(f"\n# Step 2: Run via nohup (no Slurm detected)")
            clean_cmd = command.replace("caffeinate -dimsu", "").strip()
            if clean_cmd.startswith("env PYTHONPATH=src python"):
                clean_cmd = clean_cmd.replace(
                    "env PYTHONPATH=src python",
                    f"env PYTHONPATH=src {cluster.get('python_exec', 'python')}"
                )
            log_file = f"{cluster['remote_dir']}/logs/cluster/{job_name}.out"
            print(f"\n# Command that would run:")
            print(f"{ssh_prefix} 'cd {cluster['remote_dir']} && mkdir -p logs/cluster && nohup {clean_cmd} > {log_file} 2>&1 & echo $!'")
        
        print("\n" + "=" * 70)
        return True
    
    # Real submission
    print(f"\n[1/3] Syncing project to {cluster['host']} ...")
    rsync_cmd = rsync_command(config)
    result = subprocess.run(rsync_cmd, shell=True)
    if result.returncode != 0:
        print("ERROR: rsync failed")
        return False
    print("✓ Sync complete")
    
    if has_slurm:
        # Use Slurm
        script_dir = Path("scripts/slurm")
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"{job_name}.sh"
        
        print(f"\n[2/3] Generating Slurm script: {script_path}")
        slurm_script = generate_slurm_script(command, config, job_name)
        script_path.write_text(slurm_script)
        print("✓ Script saved")
        
        # Re-sync to include the new script
        subprocess.run(rsync_cmd, shell=True, capture_output=True)
        
        print(f"\n[3/3] Submitting job: {job_name}")
        submit_cmd = f"{ssh_prefix} 'cd {cluster['remote_dir']} && mkdir -p logs/cluster && sbatch scripts/slurm/{job_name}.sh'"
        result = subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: sbatch failed")
            print(result.stderr)
            return False
        
        print("✓ Job submitted")
        print(result.stdout.strip())
    else:
        # Direct SSH execution with nohup
        print(f"\n[2/2] Starting job via nohup (no Slurm detected): {job_name}")
        
        # Clean and prepare command
        clean_cmd = command.replace("caffeinate -dimsu", "").strip()
        if clean_cmd.startswith("env PYTHONPATH=src python"):
            clean_cmd = clean_cmd.replace(
                "env PYTHONPATH=src python",
                f"env PYTHONPATH=src {cluster.get('python_exec', 'python')}"
            )
        elif clean_cmd.startswith("python"):
            clean_cmd = clean_cmd.replace("python", cluster.get('python_exec', 'python'), 1)
        
        log_file = f"{cluster['remote_dir']}/logs/cluster/{job_name}.out"
        err_file = f"{cluster['remote_dir']}/logs/cluster/{job_name}.err"
        
        # Run with nohup
        run_cmd = (
            f"{ssh_prefix} 'cd {cluster['remote_dir']} && "
            f"mkdir -p logs/cluster && "
            f"nohup {clean_cmd} > {log_file} 2> {err_file} & echo PID:$!'"
        )
        
        result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: Failed to start job")
            print(result.stderr)
            return False
        
        # Extract PID
        output = result.stdout.strip()
        import re
        pid_match = re.search(r'PID:(\d+)', output)
        pid = pid_match.group(1) if pid_match else "unknown"
        
        print(f"✓ Job started (PID: {pid})")
        print(f"  Log: {log_file}")
        print(f"  Err: {err_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Submit CRL_CW jobs to cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        help="Command to run on cluster (wrap in quotes)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit (default is dry-run)",
    )
    parser.add_argument(
        "--job-name",
        default="crl_cw_job",
        help="Slurm job name (default: crl_cw_job)",
    )
    parser.add_argument(
        "--config",
        default="launcher/cluster_config.yaml",
        help="Path to cluster config (default: launcher/cluster_config.yaml)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="After submission, tail the log file in real-time (Ctrl+C to stop watching)",
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
    
    # Submit
    success = submit_job(
        command=args.command,
        config=config,
        job_name=args.job_name,
        dry_run=not args.submit,
    )
    
    if not success:
        sys.exit(1)
    
    # Watch logs if requested
    if args.submit and args.watch:
        cluster = config["cluster"]
        ssh_prefix = f"ssh {cluster['user']}@{cluster['host']}"
        
        # Check if sbatch is available to determine log file pattern
        check_cmd = f"{ssh_prefix} 'which sbatch 2>/dev/null'"
        has_slurm = subprocess.run(check_cmd, shell=True, capture_output=True).returncode == 0
        
        if has_slurm:
            log_pattern = f"{cluster['remote_dir']}/logs/cluster/{args.job_name}_*.out"
        else:
            log_pattern = f"{cluster['remote_dir']}/logs/cluster/{args.job_name}.out"
        
        print("\n" + "=" * 70)
        print(f"Waiting for log file to appear (Ctrl+C to stop)...")
        print("=" * 70 + "\n")
        
        # Wait for log file to be created, then tail it
        if has_slurm:
            watch_cmd = f"{ssh_prefix} 'while ! ls {log_pattern} 2>/dev/null; do sleep 2; done; tail -f {log_pattern}'"
        else:
            watch_cmd = f"{ssh_prefix} 'while [ ! -f {log_pattern} ]; do sleep 2; done; tail -f {log_pattern}'"
        
        try:
            subprocess.run(watch_cmd, shell=True)
        except KeyboardInterrupt:
            print("\n\nStopped watching. Job is still running on cluster.")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
