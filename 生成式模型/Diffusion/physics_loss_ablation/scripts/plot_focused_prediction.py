"""Render one paired pure-noise generation comparison from frozen final weights."""
import runpy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
experiment = runpy.run_path(str(HERE / 'prediction_type_ablation.py'))
_, z = experiment['task']()
train_seed = 20261124
generation_seed = 20261201
views = {}

for branch, prediction_type in experiment['BRANCHES'].items():
    model, condition, ddpm, ddim, optimizer, _ = experiment['make_model'](
        train_seed, prediction_type)
    generator = experiment['train_generator'](train_seed)
    experiment['load_checkpoint'](
        experiment['RUNS'] / f'seed_{train_seed}' / branch / 'final.pt',
        branch, train_seed, model, condition, optimizer, generator)
    matrix, _, _, _ = experiment['generate'](
        model, condition, ddim, z, generation_seed, branch, 2000)
    views[branch] = experiment['image_metrics'](matrix, z)
    del model, condition, ddpm, ddim, optimizer
    torch.cuda.empty_cache()

target = np.zeros_like(z['reconstruction_full'])
target[:, 32:96, 32:96] = z['target_intensity']
images = [target, z['reconstruction_full'],
          views['E']['projected_image'], views['V']['projected_image']]
labels = ['Ideal target', 'GS reference', 'epsilon (E)', 'v_prediction (V)']
gs_metrics = experiment['quality'](z['reconstruction_full'], z['target_intensity'])

fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.5), constrained_layout=True)
for channel in range(3):
    # Absolute intensity is comparable across all four columns in a row.
    vmax = max(float(np.percentile(im[channel], 99.9)) for im in images)
    for column, img in enumerate(images):
        ax = axes[channel, column]
        ax.imshow(img[channel], cmap='magma', vmin=0, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Channel {channel + 1}: {labels[column]}', fontsize=10)
        if column == 1:
            q = gs_metrics[channel]
        elif column >= 2:
            q = views['E' if column == 2 else 'V']['projected_metrics'][channel]
        else:
            q = None
        if q is not None:
            ax.set_xlabel(f"PCC {q['pcc']:.3f}  e_abs {q['e_abs']:.3f}  ROI/Pin {q['eta_roi']:.3f}",
                          fontsize=9)

fig.suptitle(f'Train seed {train_seed}, paired pure-noise seed {generation_seed}, DDIM100; '
             'shared absolute intensity scale per channel', fontsize=11)
path = experiment['FIGURES'] / f'seed_{generation_seed}_focused_comparison.png'
path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(path, dpi=160)
plt.close(fig)
print(path)
