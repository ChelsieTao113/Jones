"""Read-only frozen analyzer NPZ data selection and loading."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def cfg():
    return json.loads((ROOT/'configs'/'prep_v1.json').read_text(encoding='utf-8'))


def data_root():
    return (ROOT/cfg()['dataset_root']).resolve()


def manifest_rows():
    p=data_root()/'runs'/'analyzer_formal_train'/'manifest.jsonl'
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def fixed_pool():
    path=ROOT/'manifests'/'fixed16.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    rows=[r for r in manifest_rows() if r['status']=='accepted' and r['split']=='train']
    rng=np.random.default_rng(cfg()['seed'])
    # Deterministic strata by polarization bucket; source mix is checked after selection.
    quota={'canonical':5,'continuous':8,'mixed':3}
    selected=[]
    for bucket,count in quota.items():
        candidates=sorted((r for r in rows if r['polarization_bucket']==bucket),
                          key=lambda r:r['task_id'])
        indices=rng.choice(len(candidates),size=count,replace=False)
        selected.extend(candidates[int(i)] for i in indices)
    selected.sort(key=lambda r:r['task_id'])
    records=[{'task_id':r['task_id'],'path':r['path'],'sha256':r['sha256'],
              'polarization_bucket':r['polarization_bucket'],
              'source_families':r['source_families'],'per_channel':r['per_channel']}
             for r in selected]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'manifests'/'tiny8.json').write_text(
        json.dumps(records[:cfg()['tiny_count']],ensure_ascii=False,indent=2),encoding='utf-8')
    return records


def load(record):
    path=data_root()/record['path']
    if not path.resolve().is_relative_to(data_root()):raise ValueError('outside dataset')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('frozen NPZ hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        out={k:z[k].copy() for k in ('jones','polarization','analyzer','target_intensity',
                                      'reconstruction_full','beta')}
    if (out['jones'].shape!=(128,128,2,2) or out['polarization'].shape!=(3,2)
        or out['analyzer'].shape!=(3,2) or out['target_intensity'].shape!=(3,64,64)):
        raise ValueError('unexpected frozen shapes')
    out['task_id']=record['task_id']
    return out
