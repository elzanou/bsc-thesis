#!/usr/bin/env python3
"""
Usage:
    # Run with NOOP client (testing)
    python scripts/inference.py --task open_ended --provider noop

    # Run with real provider
    python scripts/inference.py --task mcq --provider openai_audio

    # Run all tasks
    python scripts/inference.py --task all --provider gemini
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import click

from music_evalkit.data.loader import DatasetLoader
from music_evalkit.data.schema import TaskType
from music_evalkit.inference.cache import InferenceCache
from music_evalkit.inference.runner import InferenceRunner
from music_evalkit.messages.message_builder import MessageBuilder
from music_evalkit.models.base import BaseLLMClient, LLMConfig
from music_evalkit.models.config import AppConfig
from music_evalkit.prompts.templates import get_template


TASK_CHOICES = ["open_ended", "mcq", "pairwise", "all"]


def parse_index(index_str: str) -> slice | int:
    """Parse numpy-style index notation into a slice or int.

    Examples: '5' → 5, '0:10' → slice(0,10), '5:' → slice(5,None),
    ':20' → slice(None,20), '0:50:2' → slice(0,50,2)
    """
    if ":" not in index_str:
        return int(index_str)
    parts = index_str.split(":")
    args = [int(p) if p else None for p in parts]
    return slice(*args)


def apply_index(samples: list, index_str: str | None) -> list:
    """Apply index/slice notation to a list of samples."""
    if index_str is None:
        return samples
    idx = parse_index(index_str)
    if isinstance(idx, int):
        return [samples[idx]]
    return samples[idx]


def _run_dry_run(task: str, data_dir: Path, index: str | None) -> None:
    """Validate pipeline without model inference."""
    from tqdm import tqdm

    message_builder = MessageBuilder()

    # Determine which tasks to run
    if task == "all":
        tasks = [TaskType.OPEN_ENDED, TaskType.MCQ, TaskType.PAIRWISE]
    else:
        tasks = [TaskType(task)]

    total_errors = 0

    for task_type in tasks:
        click.echo(f"\n{'='*60}")
        click.echo(f"Task: {task_type.value}")
        click.echo(f"{'='*60}")

        # Load samples
        try:
            samples = DatasetLoader.load_task(task_type, data_dir)
        except FileNotFoundError as e:
            click.echo(f"ERROR: {e}")
            total_errors += 1
            continue

        samples = apply_index(samples, index)

        click.echo(f"Loaded {len(samples)} samples")

        # Validate each sample
        errors = []
        for sample in tqdm(samples, desc="Validating"):
            try:
                # Check audio file exists
                audio_path = Path(sample.audio_path)
                if not audio_path.exists():
                    errors.append(f"{sample.id}: Audio file not found: {audio_path}")
                    continue

                # Build messages (tests message builder + audio reading)
                messages = message_builder.build(sample)

                # Validate message structure
                if len(messages) != 2:
                    errors.append(f"{sample.id}: Expected 2 messages, got {len(messages)}")
                    continue

                if messages[0]["role"] != "system":
                    errors.append(f"{sample.id}: First message should be system")
                if messages[1]["role"] != "user":
                    errors.append(f"{sample.id}: Second message should be user")

                # Check audio content exists in user message
                user_content = messages[1]["content"]
                has_audio = any(
                    part.get("type") == "input_audio"
                    for part in user_content
                    if isinstance(part, dict)
                )
                if not has_audio:
                    errors.append(f"{sample.id}: No audio content in user message")

            except Exception as e:
                errors.append(f"{sample.id}: {type(e).__name__}: {e}")

        # Report results
        if errors:
            click.echo(f"\nErrors ({len(errors)}):")
            for err in errors[:10]:  # Show first 10
                click.echo(f"  - {err}")
            if len(errors) > 10:
                click.echo(f"  ... and {len(errors) - 10} more")
            total_errors += len(errors)
        else:
            click.echo(f"✓ All {len(samples)} samples validated successfully")

    click.echo(f"\n{'='*60}")
    if total_errors == 0:
        click.echo("DRY RUN PASSED: Pipeline is ready for inference")
    else:
        click.echo(f"DRY RUN FAILED: {total_errors} errors found")
        raise SystemExit(1)


def get_client(provider: str, app_config: AppConfig | None) -> BaseLLMClient:
    """Get LLM client by provider name."""
    if provider == "noop":
        from music_evalkit.models.providers.noop import NoopClient

        config = LLMConfig(model_name="noop")
        return NoopClient(config)

    if app_config is None:
        app_config = AppConfig.load()

    from music_evalkit.models.factory import get_client as factory_get_client

    return factory_get_client(provider, app_config)


@click.command()
@click.option(
    "--task",
    type=click.Choice(TASK_CHOICES),
    required=True,
    help="Task type to run inference on",
)
@click.option(
    "--provider",
    type=str,
    required=True,
    help="LLM provider (noop, openai, openai_audio, gemini, qwen, ollama, audio_flamingo, music_flamingo)",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/processed"),
    help="Directory containing processed data",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/results"),
    help="Directory to save inference results",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Run identifier (default: timestamp)",
)
@click.option(
    "--index",
    type=str,
    default=None,
    help="Numpy-style indexing: '0:10', '5:', ':20', '3', '0:50:2'",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config.yaml (default: config.yaml in cwd)",
)
@click.option(
    "--ids",
    type=str,
    default=None,
    help="Comma-separated sample IDs to process (filters to only these)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Test pipeline without model inference (validates samples, messages, audio files)",
)
def main(
    task: str,
    provider: str,
    data_dir: Path,
    output_dir: Path,
    run_id: str | None,
    index: str | None,
    config: Path | None,
    ids: str | None,
    dry_run: bool,
):
    """Run inference on music evaluation dataset."""
    # Dry-run mode: validate pipeline without model
    if dry_run:
        click.echo("=== DRY RUN MODE ===")
        click.echo("Validating pipeline without model inference...\n")
        _run_dry_run(task, data_dir, index)
        return

    # Load config if provided
    app_config = None
    if config:
        app_config = AppConfig.load(config)
    elif provider != "noop":
        app_config = AppConfig.load()

    # Create client
    client = get_client(provider, app_config)
    click.echo(f"Using provider: {provider} ({client})")

    # Create message builder and runner
    message_builder = MessageBuilder()
    cache = InferenceCache(Path(app_config.inference.cache_dir))
    runner = InferenceRunner(client, message_builder, cache=cache)

    # Determine run ID
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine which tasks to run
    if task == "all":
        tasks = [TaskType.OPEN_ENDED, TaskType.MCQ, TaskType.PAIRWISE]
    else:
        tasks = [TaskType(task)]

    # Run inference for each task
    for task_type in tasks:
        click.echo(f"\n{'='*60}")
        click.echo(f"Task: {task_type.value}")
        click.echo(f"{'='*60}")

        # Load samples
        try:
            samples = DatasetLoader.load_task(task_type, data_dir)
        except FileNotFoundError as e:
            click.echo(f"Skipping {task_type.value}: {e}")
            continue

        samples = apply_index(samples, index)
        if ids:
            id_set = {i.strip() for i in ids.split(",")}
            samples = [s for s in samples if s.id in id_set]

        click.echo(f"Loaded {len(samples)} samples")

        # Setup output path: results/{task}/{provider}/{run_id}/
        task_output_dir = output_dir / task_type.value / provider / run_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_output_dir / "inference_results.jsonl"

        # Run inference
        results = runner.run(samples, task_type, output_path=output_path)

        # Save run metadata
        _save_run_metadata(
            task_output_dir, run_id, provider, task_type,
            results.model, samples,
        )

        # Print summary
        click.echo(f"\nResults:")
        click.echo(f"  Total:  {results.total}")
        click.echo(f"  Cached: {results.cached}")
        click.echo(f"  Failed: {results.failed}")
        click.echo(f"  Output: {output_path}")

    click.echo(f"\n{'='*60}")
    click.echo(f"Run complete: {run_id}")


def _save_run_metadata(
    output_dir: Path,
    run_id: str,
    provider: str,
    task_type: TaskType,
    model: str,
    samples: list,
) -> None:
    """Save run metadata including system prompts and prompt hash."""
    # Collect system prompts actually used (based on audio_settings in samples)
    audio_settings = {getattr(s, "audio_setting", "single") for s in samples}
    prompts = {}
    for setting in sorted(audio_settings):
        prompts[setting] = get_template(task_type, setting)

    prompt_hash = hashlib.sha256(
        json.dumps(prompts, sort_keys=True).encode()
    ).hexdigest()[:8]

    metadata = {
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "task": task_type.value,
        "prompt_hash": prompt_hash,
        "system_prompts": prompts,
        "timestamp": datetime.now().isoformat(),
    }

    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
