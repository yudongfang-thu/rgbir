"""Bind technical execution to its actual config, small inputs and loaded sources.

This records evidence, not acceptance. Call capture after the measured stage has
executed all lazy paths. Admission separately checks successful updates, measured
gradients, coefficients, review and resource receipts. No hashes are used.
"""
from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
import sys

SCHEMA = 'rgbir-execution-binding-v1'
STAGE_ENTRY = {
    'compatibility': 'verify_compatibility.py',
    'calibration': 'calibrate_independent.py',
    'canary': 'train_independent.py',
}
BASE_SOURCES = frozenset((
    'runtime.py', 'train_independent.py', 'independent_criterion.py',
    'evidence_bindings.py',
    'task_conditional_reference/legacy_oev1/train_object_evidence.py',
    'task_conditional_reference/tracked_pair_data.py',
))
DATA_KEYS = ('student_data_yaml', 'privileged_data_yaml', 'paired_train_mapping')
LOCALIZATION_INPUT_KEYS = ('geometry_contract', 'd2_receipt', 'source_geometry_scope')
COMMON_IGNORED = frozenset((
    'seed', 'method_id', 'description', 'readiness_receipt', 'protocol_status',
    'formal_training_authorized', 'calibration_receipt', 'canary_acceptance',
    'classification_coefficient', 'localization_coefficient',
))
COMPATIBILITY_IGNORED = frozenset((
    'arm', 'source', 'classification', 'classification_carrier', 'localization',
    'geometry_contract', 'geometry', 'source_geometry_scope', 'd2_receipt', 'd2',
    'evaluation_contract', 'calibration',
))


def _json_value(value):
    """Normalize JSON-compatible tuples while rejecting silent NaN identities."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _family(arm):
    if arm in ('C1', 'C1_y'):
        return 'C1'
    if arm in ('L1', 'L_GT'):
        return 'L1'
    return arm


def normalized_effective_configuration(cfg, stage):
    if stage not in STAGE_ENTRY:
        raise ValueError('Unknown execution stage: ' + str(stage))
    result = _json_value(copy.deepcopy(cfg))
    if not isinstance(result, dict):
        raise ValueError('Configuration must be an object')
    # The source default is a real execution default, not a relaxed match.
    result.setdefault('source', 'paired')
    for key in COMMON_IGNORED:
        result.pop(key, None)
    if stage == 'compatibility':
        for key in COMPATIBILITY_IGNORED:
            result.pop(key, None)
    elif stage == 'calibration':
        family = _family(result.pop('arm', None))
        if family not in ('C1', 'L1'):
            raise ValueError('Calibration must have C1 or L1 family')
        result['_calibration_family'] = family
        if family == 'C1' and isinstance(result.get('classification'), dict):
            result['classification'].pop('off_target_weight', None)
    return result


def _within(path, root):
    path = Path(path).resolve()
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _loaded_sources(module_root):
    found = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != '.py':
            continue
        relative = _within(path, module_root)
        if relative is None:
            continue
        key = relative.as_posix()
        found.setdefault(key, []).append(name)
    return {key: sorted(names) for key, names in sorted(found.items())}


def _required(stage):
    return BASE_SOURCES | {STAGE_ENTRY[stage]}


def _auxiliary_paths(cfg, stage):
    result = {}
    if stage not in ('calibration', 'canary'):
        return result
    for key in LOCALIZATION_INPUT_KEYS:
        value = cfg.get(key)
        if value is None or value == '':
            continue
        # source_geometry_scope may be a descriptive status rather than a path.
        if key == 'source_geometry_scope':
            if not isinstance(value, str):
                continue
            candidate = Path(value)
            if not candidate.is_file() and candidate.suffix.lower() not in ('.json', '.yaml', '.yml'):
                continue
        if not isinstance(value, str):
            raise ValueError('Auxiliary input must be a file path: ' + key)
        if not Path(value).is_file():
            raise ValueError('Missing auxiliary input: ' + key)
        result[key] = value
    return result


def _roster_paths(cfg):
    """Bind train/val list files referenced by the unchanged data YAMLs."""
    import yaml
    result = {}
    for data_key in ('student_data_yaml', 'privileged_data_yaml'):
        data_path = Path(cfg['paths'][data_key])
        data = yaml.safe_load(data_path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('Dataset YAML must be an object: ' + data_key)
        root = Path(data.get('path', data_path.parent))
        if not root.is_absolute():
            root = data_path.parent / root
        # Never inspect or expand any test entry. Its rejection is the training
        # entry's responsibility; binding never edits the original YAML.
        for split in ('train', 'val'):
            sources = data.get(split, [])
            if not isinstance(sources, list):
                sources = [sources]
            for index, value in enumerate(sources):
                source = Path(value)
                if not source.is_absolute():
                    source = root / source
                if source.suffix.lower() == '.txt':
                    if not source.is_file():
                        raise ValueError('Missing referenced split roster: ' + str(source))
                    result[f'{data_key}:{split}:{index}'] = str(source.resolve())
    return result


def _check_stage_identity(cfg, stage, captured=False):
    arm, source = cfg.get('arm'), cfg.get('source', 'paired')
    if arm not in ('N', 'C0', 'C1', 'C1_y', 'L1', 'L_GT'):
        raise ValueError('Invalid arm for execution binding')
    if source not in ('paired', 'shuffled', 'same_modal'):
        raise ValueError('Invalid source for execution binding')
    if captured and stage == 'compatibility' and (arm not in ('N', 'C0') or source != 'paired'):
        raise ValueError('Compatibility capture must be an actual paired N/C0 execution')
    if stage == 'calibration' and (arm not in ('C1', 'L1') if captured else _family(arm) not in ('C1', 'L1')):
        raise ValueError('Calibration capture requires the measured C1/L1 family')


def capture_execution_binding(output, cfg, stage, module_root):
    """Create immutable small evidence files; returns the binding JSON Path."""
    normalized = normalized_effective_configuration(cfg, stage)
    _check_stage_identity(cfg, stage, captured=True)
    module_root = Path(module_root).resolve()
    sources = _loaded_sources(module_root)
    missing = _required(stage) - set(sources)
    if missing:
        raise ValueError('Required execution modules were not loaded: ' + repr(sorted(missing)))
    paths = cfg.get('paths', {})
    # Read before writing any output so incomplete inputs cannot look complete.
    data = {}
    for key in DATA_KEYS:
        original = paths.get(key)
        if not isinstance(original, str) or not original:
            raise ValueError('Missing data source path: ' + key)
        data[key] = (original, Path(original).read_bytes())
    auxiliary = {key: (path, Path(path).read_bytes()) for key, path in _auxiliary_paths(cfg, stage).items()}
    rosters = {key: (path, Path(path).read_bytes()) for key, path in _roster_paths(cfg).items()}
    source_bytes = {key: (module_root / key).read_bytes() for key in sources}
    folder = Path(output).resolve() / ('execution_binding_' + stage)
    folder.mkdir(parents=True, exist_ok=False)
    source_records = []
    for relative, names in sources.items():
        destination = folder / 'sources' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_bytes[relative])
        source_records.append(dict(relative=relative, module_names=names,
                                   original=str(module_root / relative), copy=str(destination)))
    data_records = []
    for key, (original, content) in data.items():
        destination = folder / 'data' / (key + Path(original).suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        data_records.append(dict(key=key, original=original, copy=str(destination)))
    extra_records = {}
    for kind, inputs in (('auxiliary', auxiliary), ('rosters', rosters)):
        records = []
        for index, (key, (original, content)) in enumerate(sorted(inputs.items())):
            destination = folder / kind / (str(index) + Path(original).suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            records.append(dict(key=key, original=original, copy=str(destination)))
        extra_records[kind + '_files'] = records
    receipt = dict(schema=SCHEMA, stage=stage,
        captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        module_root=str(module_root), stage_entry=STAGE_ENTRY[stage],
        actual_arm=cfg['arm'], actual_source=cfg.get('source', 'paired'),
        actual_family=_family(cfg['arm']),
        effective_configuration=_json_value(cfg), normalized_configuration=normalized,
        data_files=data_records, source_files=source_records, **extra_records,
        coefficient_verification='admission_required', acceptance='NOT_ASSERTED')
    path = folder / 'binding.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def validate_execution_binding(binding_path, cfg, stage, module_root):
    """Check byte/config bindings, raise on mismatch; never assert acceptance."""
    expected = normalized_effective_configuration(cfg, stage)
    _check_stage_identity(cfg, stage)
    module_root = Path(module_root).resolve()
    binding_path = Path(binding_path).resolve()
    receipt = json.loads(binding_path.read_text(encoding='utf-8'))
    if (receipt.get('schema') != SCHEMA or receipt.get('stage') != stage or
            receipt.get('stage_entry') != STAGE_ENTRY[stage] or
            receipt.get('module_root') != str(module_root)):
        raise ValueError('Execution binding schema/stage/root mismatch')
    captured = receipt.get('effective_configuration')
    if not isinstance(captured, dict):
        raise ValueError('Missing actual effective configuration')
    _check_stage_identity(captured, stage, captured=True)
    if (receipt.get('actual_arm') != captured.get('arm') or
            receipt.get('actual_source') != captured.get('source', 'paired') or
            receipt.get('actual_family') != _family(captured.get('arm'))):
        raise ValueError('Execution stage identity was relabeled')
    actual = normalized_effective_configuration(captured, stage)
    if actual != receipt.get('normalized_configuration') or actual != expected:
        raise ValueError('Measured effective configuration differs from requested execution')
    records = receipt.get('data_files', [])
    if len(records) != len(DATA_KEYS) or {row.get('key') for row in records} != set(DATA_KEYS):
        raise ValueError('Missing or duplicate data bindings')
    for record in records:
        key = record['key']
        if record.get('original') != cfg.get('paths', {}).get(key):
            raise ValueError('Data path changed: ' + key)
        copied = Path(record['copy']).resolve()
        if _within(copied, binding_path.parent / 'data') is None:
            raise ValueError('Data copy escapes binding evidence directory')
        if Path(record['original']).read_bytes() != copied.read_bytes():
            raise ValueError('Data source bytes changed: ' + key)
    for kind, required_paths in (('auxiliary', _auxiliary_paths(cfg, stage)), ('rosters', _roster_paths(cfg))):
        records = receipt.get(kind + '_files')
        if (not isinstance(records, list) or len(records) != len(required_paths) or
                {row.get('key') for row in records} != set(required_paths)):
            raise ValueError('Missing or duplicate ' + kind + ' input bindings')
        for record in records:
            key = record['key']
            if record.get('original') != required_paths[key]:
                raise ValueError(kind + ' input path changed: ' + key)
            copied = Path(record['copy']).resolve()
            if _within(copied, binding_path.parent / kind) is None:
                raise ValueError(kind + ' copy escapes binding evidence directory')
            if Path(record['original']).read_bytes() != copied.read_bytes():
                raise ValueError(kind + ' input bytes changed: ' + key)
    records = receipt.get('source_files', [])
    relative_names = [row.get('relative') for row in records]
    if len(relative_names) != len(set(relative_names)) or not _required(stage).issubset(relative_names):
        raise ValueError('Missing or duplicate required execution source')
    for record in records:
        relative = record['relative']
        current = (module_root / relative).resolve()
        safe_relative = _within(current, module_root)
        if safe_relative is None or safe_relative.as_posix() != relative or current.suffix != '.py':
            raise ValueError('Invalid execution source relative path')
        if record.get('original') != str(current) or not record.get('module_names'):
            raise ValueError('Invalid actual loaded module identity')
        copied = Path(record['copy']).resolve()
        if _within(copied, binding_path.parent / 'sources') is None:
            raise ValueError('Source copy escapes binding evidence directory')
        if current.read_bytes() != copied.read_bytes():
            raise ValueError('Actual execution source bytes changed: ' + relative)
    return receipt
