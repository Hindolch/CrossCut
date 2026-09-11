"""Teacher-loss schedules, indexed by global environment actions."""


def teacher_weight(config, step, teacher_available):
    if not teacher_available or step >= config['teacher_cutoff']:
        return 0.0
    initial = config['teacher_weight']
    schedule = config.get('teacher_schedule', 'linear')
    if schedule == 'linear':
        return initial * max(0.0, 1.0 - step / max(1, config['teacher_cutoff']))
    if schedule == 'staged':
        decay_end = config['teacher_decay_end']
        floor = config['teacher_weight_floor']
        if not 0 < decay_end < config['teacher_cutoff']:
            raise ValueError('teacher_decay_end must be between 0 and teacher_cutoff')
        if not 0 <= floor <= initial:
            raise ValueError('teacher_weight_floor must be between 0 and teacher_weight')
        if step < decay_end:
            return initial + (floor - initial) * step / decay_end
        return floor
    raise ValueError(f'Unknown teacher_schedule: {schedule}')
