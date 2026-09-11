"""Report equally weighted evaluation summaries across independent training seeds."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        key = f'{prefix}/{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, key))
        elif isinstance(item, (int, float)) or item is None:
            result[key] = item
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('evaluations', nargs='+', help='Evaluation directories for one condition, distinct training seeds.')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if len(args.evaluations) < 2:
        raise ValueError('At least two training seeds are required for sample standard deviation.')
    directories = [Path(path) for path in args.evaluations]
    configs = [json.loads((path / 'config.json').read_text()) for path in directories]
    if len({c['seed'] for c in configs}) != len(configs):
        raise ValueError('Duplicate training seeds.')
    reference = configs[0]
    for config in configs:
        if config['evaluation']['checkpoint'] is None:
            raise ValueError('Use student evaluations, not teacher diagnostics.')
        if {k: v for k, v in config.items() if k not in ('seed', 'evaluation')} != {
                k: v for k, v in reference.items() if k not in ('seed', 'evaluation')}:
            raise ValueError('Training conditions differ beyond seed.')
        for key in ('episodes', 'seed'):
            if config['evaluation'][key] != reference['evaluation'][key]:
                raise ValueError('Evaluation protocol differs.')
    rows = [flatten(json.loads((path / 'summary.json').read_text())) for path in directories]
    result = {'directories': args.evaluations, 'training_seeds': [c['seed'] for c in configs], 'metrics': {}}
    for key in rows[0]:
        values = [row[key] for row in rows if row[key] is not None]
        result['metrics'][key] = {'mean': mean(values) if values else None,
                                 'sample_std': stdev(values) if len(values) > 1 else None,
                                 'defined_seeds': len(values), 'total_seeds': len(rows)}
    with Path(args.output).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
