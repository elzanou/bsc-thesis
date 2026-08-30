"""
Usage:
    # First time: sync everything + setup
    python scripts/runpod_run.py sync --all
    python scripts/runpod_run.py setup

    # Normal run (syncs code changes automatically)
    python scripts/runpod_run.py run --task mcq --provider qwen_local

    # Sync code or data independently
    python scripts/runpod_run.py sync           # code only (default)
    python scripts/runpod_run.py sync --data    # data only
    python scripts/runpod_run.py sync --all     # both

    # Re-preprocess on pod
    python scripts/runpod_run.py preprocess

    # Pod lifecycle
    python scripts/runpod_run.py start          # start + print SSH command
    python scripts/runpod_run.py stop
"""

import os
import subprocess
import time
from pathlib import Path

import click
import runpod
from dotenv import load_dotenv

# --- Constants ---

POD_WORKSPACE = "/workspace/music-evalkit"
SSH_KEY_DEFAULT = "~/.ssh/id_ed25519"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=30",
    "-o", "BatchMode=yes",
    "-o", "LogLevel=ERROR",
]

TASK_CHOICES = ["open_ended", "mcq", "pairwise", "all"]
PROVIDER_CHOICES = ["audio_flamingo", "music_flamingo", "qwen_local"]


# --- Exceptions ---


class SSHError(Exception):
    """Raised when an SSH/SCP/rsync operation fails."""


# --- Pod Lifecycle ---


class PodManager:
    """Manages RunPod pod lifecycle via the runpod Python SDK."""

    MAX_WAIT_SECONDS = 300
    POLL_INTERVAL = 5

    def __init__(self, api_key: str, pod_id: str, ssh_key: str):
        runpod.api_key = api_key
        self.pod_id = pod_id
        self.ssh_key = ssh_key

    def get_pod_info(self) -> dict:
        pod = runpod.get_pod(self.pod_id)
        if pod is None:
            raise RuntimeError(
                f"Pod '{self.pod_id}' not found. Check RUNPOD_POD_ID."
            )
        return pod

    def get_ssh_connection(self) -> tuple[str, int]:
        pod = self.get_pod_info()
        runtime = pod.get("runtime")
        if runtime is None:
            raise RuntimeError("SSH port not available yet.")
        ports = runtime.get("ports", [])
        ssh = next(
            (p for p in ports if p["privatePort"] == 22 and p["isIpPublic"]),
            None,
        )
        if not ssh:
            raise RuntimeError("SSH port not available yet.")
        return ssh["ip"], ssh["publicPort"]

    def _resume_spot_pod(self, gpu_count: int, bid_per_gpu: float) -> None:
        """Resume a spot pod using the podBidResume GraphQL mutation."""
        from runpod.api.graphql import run_graphql_query

        query = f"""
        mutation {{
            podBidResume(input: {{
                podId: "{self.pod_id}",
                bidPerGpu: {bid_per_gpu},
                gpuCount: {gpu_count}
            }}) {{
                id
                desiredStatus
            }}
        }}
        """
        run_graphql_query(query)

    def ensure_running(self) -> tuple[str, int]:
        """Start pod if stopped, poll until SSH is reachable."""
        pod = self.get_pod_info()
        desired = pod.get("desiredStatus", "")

        if desired == "EXITED":
            gpu_count = pod.get("gpuCount", 1)
            pod_type = pod.get("podType", "ON_DEMAND")
            click.echo(f"Pod is stopped ({pod_type}). Starting...")

            if pod_type != "ON_DEMAND":
                cost = pod.get("costPerHr", 0.2)
                bid = cost / max(gpu_count, 1)
                self._resume_spot_pod(gpu_count, bid)
            else:
                runpod.resume_pod(self.pod_id, gpu_count)

        elapsed = 0
        while elapsed < self.MAX_WAIT_SECONDS:
            try:
                ip, port = self.get_ssh_connection()
                if self._test_ssh(ip, port):
                    return ip, port
            except RuntimeError:
                pass

            click.echo(f"  Waiting for pod... ({elapsed}s)")
            time.sleep(self.POLL_INTERVAL)
            elapsed += self.POLL_INTERVAL

        raise RuntimeError(
            f"Pod not ready after {self.MAX_WAIT_SECONDS}s. "
            "Check the RunPod dashboard."
        )

    def stop(self) -> None:
        runpod.stop_pod(self.pod_id)

    def _test_ssh(self, ip: str, port: int) -> bool:
        """Test SSH connection. Returns True if OK, False if not ready yet.

        Raises RuntimeError immediately on auth failure (wrong/missing key).
        """
        try:
            result = subprocess.run(
                [
                    "ssh", "-p", str(port), "-i", self.ssh_key,
                    *SSH_OPTIONS, f"root@{ip}", "echo ok",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                return True
            stderr = result.stderr.lower()
            if "permission denied" in stderr or "no more authentication" in stderr:
                raise RuntimeError(
                    f"SSH authentication failed for {ip}:{port}.\n"
                    f"Ensure your public key (~/.ssh/id_ed25519.pub) is added "
                    f"at: https://www.runpod.io/console/user/settings"
                )
            return False
        except subprocess.TimeoutExpired:
            return False


# --- Remote Execution ---


class RemoteExecutor:
    """Runs commands on the pod via SSH/SCP/rsync subprocess calls."""

    def __init__(self, host: str, port: int, ssh_key: str):
        self.host = host
        self.port = port
        self.ssh_key = ssh_key

    def _ssh_base(self) -> list[str]:
        return [
            "ssh", "-p", str(self.port), "-i", self.ssh_key, *SSH_OPTIONS,
        ]

    def _scp_base(self) -> list[str]:
        return [
            "scp", "-q", "-P", str(self.port), "-i", self.ssh_key, *SSH_OPTIONS,
        ]

    def _ssh_e_flag(self) -> str:
        """Build the -e argument for rsync."""
        parts = ["ssh", f"-p {self.port}", f"-i {self.ssh_key}"]
        for i in range(0, len(SSH_OPTIONS), 2):
            parts.append(f"{SSH_OPTIONS[i]} {SSH_OPTIONS[i + 1]}")
        return " ".join(parts)

    def run(
        self, command: str, check: bool = True, timeout: int = 600,
    ) -> subprocess.CompletedProcess:
        """Run a command on the pod and capture output."""
        result = subprocess.run(
            [*self._ssh_base(), f"root@{self.host}", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            raise SSHError(
                f"Remote command failed (exit {result.returncode}):\n"
                f"  cmd: {command[:120]}\n"
                f"  stderr: {result.stderr.strip()}"
            )
        return result

    def run_streamed(self, command: str) -> int:
        """Run a command on the pod, streaming stdout/stderr to terminal."""
        proc = subprocess.Popen(
            [*self._ssh_base(), "-T", f"root@{self.host}", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in proc.stdout:
                click.echo(line.replace("\r", ""), nl=False)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
        return proc.returncode

    def upload_file(self, local: Path, remote: str) -> None:
        """SCP a single file to the pod."""
        subprocess.run(
            [*self._scp_base(), str(local), f"root@{self.host}:{remote}"],
            check=True,
        )

    def rsync_up(
        self, local: Path, remote: str, excludes: list[str] | None = None,
        *, verbose: bool = False,
    ) -> None:
        """rsync local directory to pod (code sync)."""
        flags = "-rltvz" if verbose else "-rltz"
        cmd = ["rsync", flags, "--checksum", "--delete", "--partial", "--no-owner", "--no-group"]
        for exc in excludes or []:
            cmd.extend(["--exclude", exc])
        cmd.extend(["-e", self._ssh_e_flag()])
        cmd.extend([f"{local}/", f"root@{self.host}:{remote}"])
        subprocess.run(cmd, check=True)

    def rsync_down(self, remote: str, local: Path) -> None:
        """rsync pod directory to local (results download)."""
        local.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync", "-avz", "--partial",
            "-e", self._ssh_e_flag(),
            f"root@{self.host}:{remote}", f"{local}/",
        ]
        subprocess.run(cmd, check=True)

    def file_exists(self, remote_path: str) -> bool:
        result = self.run(f"test -f {remote_path}", check=False, timeout=10)
        return result.returncode == 0


# --- High-Level Operations ---


def ensure_rsync(executor: RemoteExecutor) -> None:
    """Install rsync on the pod if not present."""
    executor.run(f"mkdir -p {POD_WORKSPACE}")
    result = executor.run("command -v rsync", check=False, timeout=10)
    if result.returncode != 0:
        click.echo("Installing rsync on pod...")
        executor.run("apt-get update -qq && apt-get install -y -q rsync", timeout=120)


CODE_EXCLUDES = [
    ".venv", "__pycache__", "*.pyc", ".git", ".env",
    "/data/", "*.tar.gz", "uv.lock", ".setup_complete",
    ".pytest_cache/", "test_downloads/", "docs/",
    "*.zip", "*.pdf", "*.md",
]


def sync_code_to_pod(
    executor: RemoteExecutor, project_root: Path, *, verbose: bool = False,
) -> None:
    """Sync code files to the pod and reinstall the project."""
    ensure_rsync(executor)
    click.echo("Syncing code...")
    executor.rsync_up(
        local=project_root,
        remote=POD_WORKSPACE,
        excludes=CODE_EXCLUDES,
        verbose=verbose,
    )
    click.echo("Re-installing project...")
    executor.run(
        f"cd {POD_WORKSPACE} && source .venv/bin/activate && "
        f'export PATH="/workspace/.local/bin:$PATH" && uv pip install -e .',
        timeout=120,
    )


def sync_data_to_pod(
    executor: RemoteExecutor, project_root: Path, *, verbose: bool = False,
) -> None:
    """Sync data files to the pod, skipping files already present."""
    ensure_rsync(executor)
    data_dirs = ["data/audios", "data/cache"]
    data_files = ["data/mcq_updated.csv", "data/Music_samples.xlsx"]

    flags = "-rltvz" if verbose else "-rltz"

    for item in data_dirs + data_files:
        local_path = project_root / item
        if not local_path.exists():
            continue
        remote_path = f"{POD_WORKSPACE}/{item}"
        if local_path.is_dir():
            executor.run(f"mkdir -p {remote_path}")
            cmd = [
                "rsync", flags, "--ignore-existing", "--no-owner", "--no-group",
                "-e", executor._ssh_e_flag(),
                f"{local_path}/", f"root@{executor.host}:{remote_path}",
            ]
        else:
            executor.run(f"mkdir -p {POD_WORKSPACE}/data")
            cmd = [
                "rsync", flags, "--ignore-existing", "--no-owner", "--no-group",
                "-e", executor._ssh_e_flag(),
                str(local_path), f"root@{executor.host}:{remote_path}",
            ]
        click.echo(f"Syncing {item} (skipping existing files)...")
        subprocess.run(cmd, check=True)


def run_setup_script(executor: RemoteExecutor) -> None:
    """Run the setup script on the pod (install deps, download models, preprocess)."""
    click.echo("Running setup (this takes several minutes)...")
    exit_code = executor.run_streamed(
        f"cd {POD_WORKSPACE} && bash scripts/runpod_setup.sh"
    )
    if exit_code != 0:
        raise SSHError(f"Setup script failed with exit code {exit_code}")
    click.echo("Setup complete!")


def run_inference(
    executor: RemoteExecutor,
    task: str,
    provider: str,
    index: str | None,
    ids: str | None,
    run_id: str | None,
    dry_run: bool,
    data_dir: str | None = None,
    output_dir: str | None = None,
) -> int:
    """Run inference.py on the pod, streaming output."""
    inference_cmd = f"python scripts/inference.py --task {task} --provider {provider}"
    if index:
        inference_cmd += f" --index {index}"
    if ids:
        inference_cmd += f" --ids {ids}"
    if run_id:
        inference_cmd += f" --run-id {run_id}"
    if dry_run:
        inference_cmd += " --dry-run"
    if data_dir:
        inference_cmd += f" --data-dir {data_dir}"
    if output_dir:
        inference_cmd += f" --output-dir {output_dir}"

    full_cmd = (
        f"cd {POD_WORKSPACE} && source .venv/bin/activate && "
        f'export PATH="/workspace/.local/bin:$PATH" && '
        f"export HF_HOME=/workspace/hf_cache && "
        f"{inference_cmd}"
    )
    return executor.run_streamed(full_cmd)


def download_results(
    executor: RemoteExecutor,
    local_dir: Path,
    task: str | None = None,
    provider: str | None = None,
) -> None:
    """Download results from pod to local machine.

    If task and provider are given, only downloads results for that
    task/provider combination. Otherwise downloads all results.
    """
    if task and provider:
        # Expand "all" to individual tasks
        tasks = ["open_ended", "mcq", "pairwise"] if task == "all" else [task]
        for t in tasks:
            remote_path = f"{POD_WORKSPACE}/data/results/{t}/{provider}/"
            result = executor.run(
                f"test -d {remote_path}", check=False, timeout=10,
            )
            if result.returncode != 0:
                click.echo(f"No results for {t}/{provider} on pod, skipping.")
                continue
            local_path = local_dir / t / provider
            local_path.mkdir(parents=True, exist_ok=True)
            executor.rsync_down(remote_path, local_path)
    else:
        remote_results = f"{POD_WORKSPACE}/data/results/"
        result = executor.run(
            f"test -d {remote_results}", check=False, timeout=10,
        )
        if result.returncode != 0:
            click.echo("No results directory found on pod.")
            return
        executor.rsync_down(remote_results, local_dir)


# --- Shared CLI Context ---


class RunPodContext:
    """Shared state passed through Click context to all subcommands."""

    def __init__(self, ssh_key: str, api_key: str, pod_id: str):
        self.ssh_key = ssh_key
        self.api_key = api_key
        self.pod_id = pod_id
        self._pod_manager: PodManager | None = None
        self._executor: RemoteExecutor | None = None

    @property
    def pod_manager(self) -> PodManager:
        if self._pod_manager is None:
            self._pod_manager = PodManager(self.api_key, self.pod_id, self.ssh_key)
        return self._pod_manager

    def ensure_executor(self) -> RemoteExecutor:
        """Start pod if needed and return a connected executor."""
        if self._executor is None:
            click.echo(f"\n{'='*60}")
            click.echo("Connecting to RunPod")
            click.echo(f"{'='*60}")
            host, port = self.pod_manager.ensure_running()
            click.echo(f"Pod ready: {host}:{port}")
            self._executor = RemoteExecutor(host, port, self.ssh_key)
        return self._executor


pass_ctx = click.make_pass_decorator(RunPodContext)


# --- CLI ---


@click.group()
@click.option(
    "--ssh-key",
    type=click.Path(path_type=Path),
    default=None,
    help=f"SSH private key path (default: {SSH_KEY_DEFAULT})",
)
@click.pass_context
def cli(ctx: click.Context, ssh_key: Path | None):
    """Run inference on RunPod end-to-end.

    \b
    First time:
        python scripts/runpod_run.py sync --all
        python scripts/runpod_run.py setup

    Run inference:
        python scripts/runpod_run.py run --task mcq --provider qwen_local

    Sync code/data:
        python scripts/runpod_run.py sync           # code (default)
        python scripts/runpod_run.py sync --data     # data
        python scripts/runpod_run.py sync --all      # both
    """
    load_dotenv()

    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not api_key or not pod_id:
        raise click.UsageError(
            "RUNPOD_API_KEY and RUNPOD_POD_ID must be set in .env or environment."
        )

    resolved_ssh_key = str(
        ssh_key if ssh_key else Path(SSH_KEY_DEFAULT).expanduser()
    )
    if not Path(resolved_ssh_key).exists():
        raise click.UsageError(f"SSH key not found: {resolved_ssh_key}")

    ctx.obj = RunPodContext(resolved_ssh_key, api_key, pod_id)


@cli.command()
@pass_ctx
def setup(ctx: RunPodContext):
    """Install deps, download model weights, and preprocess data on the pod.

    \b
    Runs scripts/runpod_setup.sh which handles:
      - System deps (ffmpeg)
      - Python env (uv, venv, PyTorch, flash-attn, accelerate)
      - Project install (uv pip install -e .)
      - Model weight download (Audio Flamingo, Music Flamingo)
      - Data preprocessing (open_ended, mcq, pairwise)

    Code and data must already be on the pod (use 'sync --all' first).
    """
    executor = ctx.ensure_executor()
    click.echo(f"\n{'='*60}")
    click.echo("Setup")
    click.echo(f"{'='*60}")
    run_setup_script(executor)
    click.echo("\nPod left running for experiments.")


@cli.command()
@click.option(
    "--task",
    type=click.Choice(TASK_CHOICES),
    required=True,
    help="Task type to run inference on",
)
@click.option(
    "--provider",
    type=click.Choice(PROVIDER_CHOICES),
    required=True,
    help="Model provider",
)
@click.option(
    "--index",
    type=str,
    default=None,
    help="Numpy-style indexing: '0:10', '5:', ':20', '3', '0:50:2'",
)
@click.option(
    "--ids",
    type=str,
    default=None,
    help="Comma-separated sample IDs to process (filters to only these)",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Run identifier (default: timestamp, set by inference.py)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate pipeline without running inference",
)
@click.option(
    "--no-stop",
    is_flag=True,
    default=False,
    help="Don't stop pod after inference (for running multiple experiments)",
)
@click.option(
    "--no-sync",
    is_flag=True,
    default=False,
    help="Skip code sync (use code already on pod)",
)
@click.option(
    "--download-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Local directory for results (default: data/results)",
)
@click.option(
    "--data-dir",
    type=str,
    default=None,
    help="Remote data directory on pod (default: data/processed)",
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Remote output directory on pod for results",
)
@pass_ctx
def run(
    ctx: RunPodContext,
    task: str,
    provider: str,
    index: str | None,
    ids: str | None,
    run_id: str | None,
    dry_run: bool,
    no_stop: bool,
    no_sync: bool,
    download_dir: Path | None,
    data_dir: str | None,
    output_dir: str | None,
):
    """Sync code, run inference, download results, stop pod.

    \b
    Examples:
        python scripts/runpod_run.py run --task mcq --provider qwen_local
        python scripts/runpod_run.py run --task all --provider audio_flamingo --index :5
        python scripts/runpod_run.py run --task mcq --provider audio_flamingo --ids id1,id2
        python scripts/runpod_run.py run --task all --provider audio_flamingo --dry-run
    """
    executor = ctx.ensure_executor()

    if download_dir is None:
        download_dir = PROJECT_ROOT / "data" / "results"

    # Check setup was done
    if not executor.file_exists(f"{POD_WORKSPACE}/.setup_complete"):
        raise click.UsageError(
            "Pod has not been set up. Run 'setup' first."
        )

    try:
        if not no_sync:
            click.echo(f"\n{'='*60}")
            click.echo("Syncing code")
            click.echo(f"{'='*60}")
            sync_code_to_pod(executor, PROJECT_ROOT)

        click.echo(f"\n{'='*60}")
        click.echo(f"Inference: task={task}, provider={provider}")
        click.echo(f"{'='*60}")
        exit_code = run_inference(
            executor, task, provider, index, ids, run_id, dry_run,
            data_dir=data_dir, output_dir=output_dir,
        )
        if exit_code != 0:
            click.echo(f"\nWARNING: Inference exited with code {exit_code}")

        if not dry_run:
            click.echo(f"\n{'='*60}")
            click.echo("Downloading results")
            click.echo(f"{'='*60}")
            download_results(executor, download_dir, task=task, provider=provider)
            click.echo(f"Results saved to: {download_dir}")

    finally:
        if not no_stop:
            click.echo(f"\n{'='*60}")
            click.echo("Stopping pod")
            click.echo(f"{'='*60}")
            ctx.pod_manager.stop()
            click.echo("Pod stopped (data preserved on disk).")
        else:
            click.echo("\nPod left running (--no-stop). Stop it manually when done.")


@cli.command()
@click.option("--code", "targets", flag_value="code", default=True,
              help="Sync code only (default)")
@click.option("--data", "targets", flag_value="data",
              help="Sync data only (audios, cache, xlsx)")
@click.option("--all", "targets", flag_value="all",
              help="Sync both code and data")
@pass_ctx
def sync(ctx: RunPodContext, targets: str):
    """Sync files to the pod.

    \b
    Sync code (default):
        python scripts/runpod_run.py sync

    Sync data only:
        python scripts/runpod_run.py sync --data

    Sync everything:
        python scripts/runpod_run.py sync --all
    """
    executor = ctx.ensure_executor()

    click.echo(f"\n{'='*60}")
    click.echo(f"Syncing ({targets})")
    click.echo(f"{'='*60}")

    if targets in ("code", "all"):
        sync_code_to_pod(executor, PROJECT_ROOT, verbose=True)
    if targets in ("data", "all"):
        sync_data_to_pod(executor, PROJECT_ROOT, verbose=True)


@cli.command()
@click.option(
    "--download-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Local directory for results (default: data/results)",
)
@pass_ctx
def download(ctx: RunPodContext, download_dir: Path | None):
    """Download results from pod to local machine."""
    executor = ctx.ensure_executor()

    if download_dir is None:
        download_dir = PROJECT_ROOT / "data" / "results"

    click.echo(f"\n{'='*60}")
    click.echo("Downloading results")
    click.echo(f"{'='*60}")
    download_results(executor, download_dir)
    click.echo(f"Results saved to: {download_dir}")


@cli.command()
@click.option(
    "--task",
    type=click.Choice(TASK_CHOICES[:-1]),  # exclude "all"
    default=None,
    help="Preprocess a specific task (default: all tasks)",
)
@click.option(
    "--reverse",
    is_flag=True,
    default=False,
    help="Pairwise only: swap audio order and flip labels (positional bias test)",
)
@pass_ctx
def preprocess(ctx: RunPodContext, task: str | None, reverse: bool):
    """Run preprocessing on the pod.

    \b
    Re-preprocess all tasks:
        python scripts/runpod_run.py preprocess

    Single task:
        python scripts/runpod_run.py preprocess --task mcq

    Pairwise with reversed audio order:
        python scripts/runpod_run.py preprocess --task pairwise --reverse
    """
    executor = ctx.ensure_executor()

    if not executor.file_exists(f"{POD_WORKSPACE}/.setup_complete"):
        raise click.UsageError(
            "Pod has not been set up. Run 'setup' first."
        )

    if reverse and task != "pairwise":
        raise click.UsageError("--reverse is only supported for the pairwise task.")

    tasks = [task] if task else ["open_ended", "mcq", "pairwise"]

    click.echo(f"\n{'='*60}")
    click.echo(f"Preprocessing: {', '.join(tasks)}{' (reversed)' if reverse else ''}")
    click.echo(f"{'='*60}")

    for t in tasks:
        cmd = f"python scripts/preprocess.py --task {t}"
        if reverse and t == "pairwise":
            cmd += " --reverse"
        exit_code = executor.run_streamed(
            f"cd {POD_WORKSPACE} && source .venv/bin/activate && "
            f'export PATH="/workspace/.local/bin:$PATH" && '
            f"{cmd}"
        )
        if exit_code != 0:
            raise SSHError(f"Preprocessing failed for task '{t}' (exit code {exit_code})")

    click.echo("Preprocessing complete.")


@cli.command()
@pass_ctx
def start(ctx: RunPodContext):
    """Start the pod and print SSH connection info."""
    executor = ctx.ensure_executor()
    host, port = executor.host, executor.port
    click.echo(f"\nSSH command:")
    click.echo(f"  ssh -p {port} -i {ctx.ssh_key} root@{host}")


@cli.command()
@pass_ctx
def stop(ctx: RunPodContext):
    """Stop the pod (data preserved on disk)."""
    ctx.pod_manager.stop()
    click.echo("Pod stopped (data preserved on disk).")


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nInterrupted. Pod left running.")
    except SSHError as e:
        click.echo(f"\nSSH error: {e}", err=True)
        raise SystemExit(1)
