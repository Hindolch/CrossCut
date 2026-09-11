"""W&B run identity and durable copies of experiment results/checkpoints."""
from pathlib import Path
from common import write_json


def record_run(run, output):
    write_json(Path(output) / 'wandb_run.json', {
        'id': run.id, 'url': run.url, 'entity': run.entity, 'project': run.project,
        'mode': run.settings.mode})


def upload_results(run, output, kind):
    import wandb
    output = Path(output)
    artifact = wandb.Artifact(f'{run.id}-{kind}', type=kind)
    # Explicit file types; exclude credentials, caches and W&B's internal directory.
    for path in sorted(output.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.pt', '.mp4', '.npz'):
            artifact.add_file(str(path), name=path.name)
    logged = run.log_artifact(artifact)
    if run.settings.mode == 'online':
        logged.wait()
        write_json(output / 'wandb_artifact.json', {
            'name': logged.qualified_name, 'run_url': run.url, 'uploaded': True})
