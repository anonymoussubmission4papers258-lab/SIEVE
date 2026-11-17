"""No-LIB Stage 0: reuse independently verified raw maps, refit PCA on 98 images.

Existing raw maps are referenced in place, never copied or altered. Missing maps
alone trigger lazy loading of the unchanged teacher. No LIB feature enters PCA.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_teacher_stream_v2 as base
from sieve3.data import load_panels
from sieve3.teacher import Teacher, project


ALLOWED = {'PCB-SR', 'Chikusei-native', 'Harvard-native', 'HyTexiLa-native'}


def canonical(path):
    return str(Path(path).resolve()).replace('\\', '/').lower()


def assert_prior_producer_exited(started):
    import psutil
    identity = started.get('identity', {})
    if not identity.get('pid') or identity.get('create_time') is None:
        raise ValueError('Prior producer identity is required to prove a safe reuse boundary')
    try:
        process = psutil.Process(int(identity['pid']))
        if abs(process.create_time() - float(identity['create_time'])) < 0.01 and process.is_running():
            raise RuntimeError('Prior raw producer is still running; do not reuse an active journal')
    except psutil.NoSuchProcess:
        pass
    # AccessDenied is deliberately not suppressed: lack of host visibility is
    # not evidence of process exit.


def validate_reuse(args, manifests, samples):
    receipts = Path(args.reuse_receipts)
    started_path = receipts / 'STARTED.json'
    started_bytes = started_path.read_bytes()
    started = json.loads(started_bytes)
    assert_prior_producer_exited(started)
    for key in ('long_side', 'tile', 'dims', 'max_tokens', 'seed'):
        if started.get(key) != getattr(args, key):
            raise ValueError(f'Prior feature settings disagree: {key}')
    if canonical(started['teacher']) != canonical(args.teacher):
        raise ValueError('Prior teacher path binding disagrees')
    root = Path(__file__).resolve().parents[1]
    expected_names = {'cache_teacher_stream_v2.py': root / 'scripts/cache_teacher_stream_v2.py',
                      'teacher.py': root / 'sieve3/teacher.py',
                      'data.py': root / 'sieve3/data.py',
                      'cache_teacher.py': root / 'scripts/cache_teacher.py'}
    seen = set()
    for path, expected in started['source_sha256'].items():
        name = Path(path).name
        if name not in expected_names:
            raise ValueError(f'Unexpected prior source: {path}')
        if base.sha(path) != expected or base.sha(expected_names[name]) != expected:
            raise ValueError(f'Prior/current source SHA differs: {name}')
        seen.add(name)
    if seen != set(expected_names):
        raise ValueError('Prior source closure is incomplete')
    model_proof = json.loads(Path(args.model_proof).read_bytes())
    if model_proof.get('state') != 'completed' or canonical(model_proof['target']) != canonical(args.teacher):
        raise ValueError('Teacher model proof target/state mismatch')
    required_model_files = {'config.json', 'model.safetensors', 'preprocessor_config.json'}
    if {row['name'] for row in model_proof['files']} != required_model_files:
        raise ValueError('Teacher model proof file set mismatch')
    model_verified = []
    for row in model_proof['files']:
        path = Path(args.teacher) / row['name']
        if path.stat().st_size != row['bytes'] or base.sha(path) != row['sha256']:
            raise ValueError(f'Teacher static model SHA mismatch: {path}')
        model_verified.append({'name': row['name'], 'sha256': row['sha256'], 'bytes': row['bytes']})
    old_manifests = {row['panel']: row for row in started['manifests']}
    for row in manifests:
        old = old_manifests.get(row['panel'])
        if not old or any(old.get(key) != row.get(key) for key in ('manifest_sha256', 'protocol_sha256')):
            raise ValueError(f'Prior manifest/protocol binding mismatch: {row["panel"]}')
        if canonical(old['manifest']) != canonical(row['manifest']):
            raise ValueError('Prior manifest path binding mismatch')
        if base.sha(row['manifest']) != row['manifest_sha256']:
            raise ValueError('Fresh manifest SHA mismatch')
    old_samples = {(row['panel'], str(row['id'])): canonical(row['reference']) for row in started['samples']}
    for panel, sample in samples:
        if panel.name not in ALLOWED or old_samples.get((panel.name, str(sample.id))) != canonical(sample.reference):
            raise ValueError(f'Prior training sample binding mismatch: {panel.name}/{sample.id}')
    # Read the append-only journal with a short, closed handle, then parse.
    journal_path = receipts / 'RAW_FILES.jsonl'
    journal_bytes = journal_path.read_bytes()
    if journal_bytes and not journal_bytes.endswith(b'\n'):
        raise ValueError('Prior raw journal ends with an incomplete row; wait for writer exit')
    import hashlib
    journal_sha = hashlib.sha256(journal_bytes).hexdigest()
    indexed = {}
    for line in journal_bytes.splitlines():
        row = json.loads(line)
        key = (row['panel'], str(row['id']))
        if key in indexed:
            raise ValueError(f'Duplicate prior raw row: {key}')
        indexed[key] = row
    config = json.loads((Path(args.teacher) / 'config.json').read_text(encoding='utf-8'))
    dimension = int(config['hidden_size'])
    reusable, missing = {}, []
    for panel, sample in samples:
        key = (panel.name, str(sample.id))
        row = indexed.get(key)
        if row is None:
            missing.append(key)
            continue
        expected_path = Path(started['raw_out']) / panel.name / f'{sample.id}.npy'
        if canonical(row['path']) != canonical(expected_path):
            raise ValueError(f'Prior raw path binding mismatch: {key}')
        path = Path(row['path'])
        if path.stat().st_size != row['bytes'] or base.sha(path) != row['sha256']:
            raise ValueError(f'Fresh raw SHA/size mismatch: {key}')
        feature = np.load(path, mmap_mode='r', allow_pickle=False)
        shape, dtype = list(feature.shape), str(feature.dtype)
        del feature
        if dtype != 'float16' or len(shape) != 3 or shape[0] != dimension or shape != row['shape']:
            raise ValueError(f'Prior raw feature header mismatch: {key}')
        reusable[key] = dict(row)
    assert_prior_producer_exited(started)
    if base.sha(journal_path) != journal_sha:
        raise RuntimeError('Prior journal changed while being verified')
    binding = {'prior_started': str(started_path), 'prior_started_sha256': base.sha(started_path),
               'prior_journal': str(journal_path), 'prior_journal_sha256': journal_sha,
               'prior_journal_rows': len(indexed), 'selected_reusable': len(reusable),
               'selected_missing': len(missing), 'missing_keys': missing,
               'model_proof_sha256': base.sha(args.model_proof), 'verified_model_files': model_verified,
               'ignored_prior_rows': len(indexed) - len(reusable),
               'ignored_rows_payloads_opened': 0, 'raw_maps_copied': 0,
               'prior_producer_actual_exit_verified': True,
               'pca_must_be_refit_on_selected_training_only': True}
    return reusable, binding


def main(args):
    begin = time.time()
    out, raw_out = Path(args.out), Path(args.raw_out)
    receipt = Path(args.receipts) if args.receipts else out / 'STREAM_STATUS_V3'
    receipt.mkdir(parents=True, exist_ok=True)
    state = {'utc': base.now(), 'state': 'starting', 'pid': os.getpid()}
    try:
        if args.splits != 'train':
            raise ValueError('Only train split is admitted')
        panels = load_panels(args.panels, args.panels_used.split(',') if args.panels_used else None)
        if set(panels) != ALLOWED:
            raise ValueError('No-LIB production must include exactly the four authorised native panels')
        samples = [(panel, sample) for panel in panels.values() for sample in panel.samples
                   if sample.split == 'train' and sample.reference]
        if len(samples) != 98:
            raise ValueError(f'Expected all 98 non-LIB training samples, got {len(samples)}')
        manifests = []
        for entry in json.loads(Path(args.panels).read_text(encoding='utf-8'))['panels']:
            if entry['name'] not in panels:
                continue
            manifest = Path(entry['manifest'])
            protocol = Path(entry.get('protocol') or json.loads(manifest.read_text())['protocol'])
            manifests.append({'panel': entry['name'], 'manifest': str(manifest),
                              'manifest_sha256': base.sha(manifest), 'protocol_sha256': base.sha(protocol)})
        reusable, reuse_binding = validate_reuse(args, manifests, samples)
        root = Path(__file__).resolve().parents[1]
        source_files = [Path(__file__), root / 'scripts/cache_teacher_stream_v2.py',
                        root / 'sieve3/teacher.py', root / 'sieve3/data.py']
        identity = {'pid': os.getpid()}
        try:
            import psutil
            identity.update(create_time=psutil.Process().create_time(), argv=psutil.Process().cmdline())
        except ImportError:
            pass
        base.write_json(receipt / 'STARTED.json', {'utc': base.now(), 'state': 'started',
            'identity': identity, 'source_sha256': {str(p): base.sha(p) for p in source_files},
            'panels_config_sha256': base.sha(args.panels), 'manifests': manifests,
            'samples': [{'panel': p.name, 'id': s.id, 'reference': s.reference} for p, s in samples],
            'sample_count': len(samples), 'teacher': args.teacher, 'raw_out': str(raw_out), 'out': str(out),
            'long_side': args.long_side, 'tile': args.tile, 'dims': args.dims,
            'max_tokens': args.max_tokens, 'seed': args.seed, 'test_arrays_opened': 0,
            'deferred_panels': ['LIB-HSI-native'], 'reuse_binding': reuse_binding})

        def progress(stage, **details):
            state.clear()
            state.update(utc=base.now(), state=stage, pid=os.getpid(),
                         elapsed_seconds=time.time() - begin, **details)
            base.write_json(receipt / 'CURRENT.json', state, mutable=True)
            print(json.dumps(state, ensure_ascii=False), flush=True)

        device, teacher, raw_records = None, None, []
        for index, (panel, sample) in enumerate(samples):
            key = (panel.name, str(sample.id))
            if key in reusable:
                record = dict(reusable[key])
                record.update(reused=True, reuse_prior_journal_sha256=reuse_binding['prior_journal_sha256'])
                progress('reuse_raw', images_done=index, images_total=len(samples), panel=panel.name, sample=sample.id)
            else:
                if teacher is None:
                    progress('loading_teacher_for_missing_raw', images_done=index, images_total=len(samples))
                    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
                    teacher = Teacher(args.teacher, device)
                progress('extract_missing_raw', images_done=index, images_total=len(samples), panel=panel.name, sample=sample.id)
                rgb = base.rgb_from_memmap(panel, sample, args.rgb_row_chunk)
                feature = teacher.features(rgb, args.long_side, args.tile).cpu().half()
                record = base.save_new(raw_out / panel.name / f'{sample.id}.npy',
                    lambda stream: np.save(stream, feature.numpy(), allow_pickle=False))
                record.update(panel=panel.name, id=sample.id, shape=list(feature.shape), reused=False)
                del rgb, feature
            raw_records.append(record)
            with (receipt / 'RAW_FILES.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        del teacher
        if device is not None and device.type == 'cuda':
            torch.cuda.empty_cache()
        paths = [row['path'] for row in raw_records]
        assert all(row['panel'] in ALLOWED for row in raw_records) and len(paths) == 98
        progress('pca_mean_nonLIB_only', images_total=len(samples))
        mean, basis, explained, sampled_count = base.fit_pca_stream(
            paths, args.dims, args.max_tokens, args.seed, args.pca_chunk_tokens, progress)
        pca_record = base.save_new(out / 'pca.pt', lambda stream: torch.save({'mean': mean, 'basis': basis}, stream))
        projected = []
        for index, ((panel, sample), row) in enumerate(zip(samples, raw_records)):
            progress('project_nonLIB', images_done=index, images_total=len(samples), panel=panel.name, sample=sample.id)
            feature = torch.from_numpy(np.array(np.load(row['path'], mmap_mode='r', allow_pickle=False),
                                              dtype=np.float16, copy=True)).float()
            reduced = project(feature, mean, basis).numpy().astype(np.float16)
            record = base.save_new(out / panel.name / f'{sample.id}.npy',
                                  lambda stream: np.save(stream, reduced, allow_pickle=False))
            record.update(panel=panel.name, id=sample.id, shape=list(reduced.shape), raw_sha256=row['sha256'])
            projected.append(record)
            with (receipt / 'PROJECTED_FILES.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            del feature, reduced
        meta = {'teacher': args.teacher, 'long_side': args.long_side, 'dims': args.dims,
                'explained': explained, 'images': len(samples), 'seconds': time.time() - begin,
                'panels': sorted(panels), 'splits': ['train'], 'seed': args.seed,
                'tile': args.tile, 'max_tokens': args.max_tokens, 'sampled_tokens': sampled_count,
                'implementation': 'stream_v3_verified_raw_reuse_PCA_refit_nonLIB_only',
                'deferred_panels': ['LIB-HSI-native'], 'reuse_binding': reuse_binding,
                'teacher_model_loaded_for_missing': device is not None}
        base.write_json(out / 'meta.json', meta)
        base.write_json(receipt / 'DONE.json', {**meta, 'utc': base.now(), 'state': 'completed',
            'identity': identity, 'test_arrays_opened': 0, 'LIB_feature_arrays_opened': 0,
            'pca': pca_record, 'projected_files': projected, 'raw_files_count': len(raw_records),
            'raw_journal_sha256': base.sha(receipt / 'RAW_FILES.jsonl'),
            'projected_journal_sha256': base.sha(receipt / 'PROJECTED_FILES.jsonl'),
            'admission_sha256': base.sha(receipt / 'STARTED.json')})
        progress('completed', images_done=len(samples), images_total=len(samples))
    except BaseException as error:
        if not (receipt / 'ATTENTION.json').exists():
            base.write_json(receipt / 'ATTENTION.json', {'utc': base.now(), 'state': 'attention',
                'pid': os.getpid(), 'error_type': type(error).__name__, 'error': str(error),
                'last_state': dict(state), 'payloads_preserved': True})
        raise


def parser():
    result = base.parser()
    result.description = __doc__
    result.add_argument('--reuse-receipts', required=True)
    result.add_argument('--model-proof', required=True, help='Verified static teacher model download DONE receipt')
    return result


if __name__ == '__main__':
    main(parser().parse_args())
