"""Versioned lazy replacement for TrainStore; sampling and augmentation are unchanged.

Only selected spatial crops are copied from frozen read-only NPY maps. All train
samples remain eligible with precisely the original panel/image pixel weights.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import lcm, observable_hr, sanitise


def _shape(path):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    shape = tuple(a.shape)
    del a
    return shape


def _cube_crop(path, layout, y0, x0, side):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    if layout.upper() == 'HWC':
        crop = np.moveaxis(a[y0:y0 + side, x0:x0 + side, :], -1, 0)
    else:
        crop = a[:, y0:y0 + side, x0:x0 + side]
    result = np.array(crop, dtype=np.float32, order='C', copy=True)
    del a
    return result


def _valid_crop(path, hw, y0, x0, side):
    if not path or not Path(path).exists():
        return np.ones((side, side), bool)
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    if a.ndim == 2:
        result = np.array(a[y0:y0 + side, x0:x0 + side], dtype=bool, copy=True)
    elif tuple(a.shape[:2]) == tuple(hw):
        result = np.asarray(a[y0:y0 + side, x0:x0 + side, :]).astype(bool).all(-1)
    elif tuple(a.shape[1:]) == tuple(hw):
        result = np.asarray(a[:, y0:y0 + side, x0:x0 + side]).astype(bool).all(0)
    else:
        raise ValueError(f'{path}: valid shape {a.shape} incompatible with {hw}')
    del a
    return result


def _teacher_crop(path, hw, y0, x0, side, device):
    """Read the exact bilinear footprint of the original full-map grid sample."""
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    ht, wt = a.shape[-2:]
    h, w = hw
    ys = (torch.arange(y0, y0 + side, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1
    xs = (torch.arange(x0, x0 + side, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1
    # Convert original normalised coordinates to source pixel coordinates. Keep
    # both interpolation neighbours, including padding_mode='border' behaviour.
    yc, xc = ((ys + 1) * ht - 1) / 2, ((xs + 1) * wt - 1) / 2
    ya = max(0, min(ht - 1, int(torch.floor(yc.min()).item())))
    yb = max(ya + 1, min(ht, int(torch.ceil(yc.max()).item()) + 1))
    xa = max(0, min(wt - 1, int(torch.floor(xc.min()).item())))
    xb = max(xa + 1, min(wt, int(torch.ceil(xc.max()).item()) + 1))
    footprint = np.array(a[:, ya:yb, xa:xb], dtype=np.float16, order='C', copy=True)
    del a
    feature = torch.from_numpy(footprint).to(device).float()[None]
    gy = (yc - ya + 0.5) / (yb - ya) * 2 - 1
    gx = (xc - xa + 0.5) / (xb - xa) * 2 - 1
    gy, gx = torch.meshgrid(gy, gx, indexing='ij')
    return F.grid_sample(feature, torch.stack([gx, gy], -1)[None], mode='bilinear',
                         padding_mode='border', align_corners=False)[0]


class MemmapTrainStore:
    def __init__(self, panels, teacher_dir, device, max_gb=8.0, splits=('train',),
                 metadata_cache=None):
        self.device = torch.device(device)
        self.store_device = torch.device('cpu')
        self.groups = []
        self.storage_mode = 'read_only_npy_memmap_selected_crop_v1'
        cache = {}
        cache_path = Path(metadata_cache) if metadata_cache else Path(__file__).resolve().parents[1] / 'results/DATA_MAPPING_LOCAL_V1.json'
        if cache_path.exists():
            proof = json.loads(cache_path.read_text(encoding='utf-8'))
            for p in proof.get('panels', []):
                for row in p.get('train_val_sources', []):
                    info = row['reference']
                    cache[str(Path(info['path'])).lower()] = tuple(info['shape'])
        total, teachers = 0, []
        for panel in panels.values():
            items = []
            for sample in panel.samples:
                if sample.split not in splits or not sample.reference:
                    continue
                shape = cache.get(str(Path(sample.reference)).lower()) or _shape(sample.reference)
                if len(shape) != 3:
                    raise ValueError(f'{sample.reference}: cube must be 3-D')
                h, w, c = shape if panel.layout.upper() == 'HWC' else (shape[1], shape[2], shape[0])
                rgb = sample.rgb if panel.srf is None else None
                if panel.srf is None and not rgb:
                    raise FileNotFoundError(f'{panel.name}/{sample.id}: training needs SRF or observed RGB')
                rgb_layout = None
                if rgb:
                    rs = _shape(rgb)
                    if rs == (3, h, w):
                        rgb_layout = 'CHW'
                    elif rs == (h, w, 3):
                        rgb_layout = 'HWC'
                    else:
                        raise ValueError(f'{rgb}: RGB shape {rs} incompatible with {(h,w)}')
                teach = Path(teacher_dir) / panel.name / f'{sample.id}.npy' if teacher_dir else None
                teach = str(teach) if teach and teach.exists() else None
                teachers.append(teach is not None)
                items.append({'reference': sample.reference, 'valid_path': sample.valid,
                              'rgb_path': rgb, 'rgb_layout': rgb_layout, 'teacher_path': teach,
                              'shape': (c, h, w), 'id': sample.id})
                total += c * h * w * 4 + (3 * h * w * 4 if rgb else 0)
            if not items:
                continue
            step = lcm(panel.scale, 8)
            side = min(min(it['shape'][1:]) for it in items)
            self.groups.append({'panel': panel, 'items': items,
                                'max_side': side - side % step, 'step': step,
                                'pixels': sum(it['shape'][1] * it['shape'][2] for it in items)})
        if not self.groups:
            raise ValueError('no training samples found')
        self.has_teacher = bool(teachers) and all(teachers)
        self.total_gb = total / 2 ** 30

    def sample(self, batch, crop, generator):
        weights = torch.tensor([g['pixels'] for g in self.groups], dtype=torch.float64)
        group = self.groups[int(torch.multinomial(weights, 1, generator=generator))]
        panel, s, step = group['panel'], group['panel'].scale, group['step']
        side = min(crop, group['max_side'])
        side = max(step, side - side % step)
        areas = torch.tensor([it['shape'][1] * it['shape'][2] for it in group['items']], dtype=torch.float64)
        refs, valids, rgbs, teachers = [], [], [], []
        for _ in range(batch):
            item = group['items'][int(torch.multinomial(areas, 1, generator=generator))]
            _, h, w = item['shape']
            y0 = s * int(torch.randint(0, (h - side) // s + 1, (1,), generator=generator))
            x0 = s * int(torch.randint(0, (w - side) // s + 1, (1,), generator=generator))
            k = int(torch.randint(0, 4, (1,), generator=generator))
            flip = bool(torch.randint(0, 2, (1,), generator=generator))

            def aug(tensor):
                tensor = torch.rot90(tensor, k, (-2, -1))
                return torch.flip(tensor, (-1,)) if flip else tensor

            ref = _cube_crop(item['reference'], panel.layout, y0, x0, side)
            valid = _valid_crop(item['valid_path'], (h, w), y0, x0, side)
            ref, valid = sanitise(ref, valid)
            if item['rgb_path']:
                rgb = _cube_crop(item['rgb_path'], item['rgb_layout'], y0, x0, side)
                rgb, valid = sanitise(rgb, valid)
                rgbs.append(aug(torch.from_numpy(rgb).to(self.device)))
            refs.append(aug(torch.from_numpy(ref).to(self.device)))
            valids.append(aug(torch.from_numpy(valid).float()[None].to(self.device)))
            if item['teacher_path']:
                teachers.append(aug(_teacher_crop(item['teacher_path'], (h, w), y0, x0,
                                                  side, self.device)))
        valid = observable_hr(torch.stack(valids).float(), s)
        return {'ref': torch.stack(refs).float(), 'valid': valid, 's': s,
                'sigma': panel.sigma, 'panel': panel.name,
                'rgb': torch.stack(rgbs).float() if rgbs else None,
                'teacher': torch.stack(teachers) if len(teachers) == batch else None,
                'srf': None if panel.srf is None else torch.from_numpy(panel.srf).to(self.device)}


TrainStore = MemmapTrainStore
