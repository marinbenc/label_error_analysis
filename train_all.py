import train
import numpy as np

leps = np.linspace(0.0, 0.5, 6)
ratios = np.linspace(-3.0, 1.0, 11)

for lep in leps:
  for r in ratios:
    log_name = f'lep={lep}_r={r}'
    train.train(
        batch_size=64, 
        epochs=100, 
        lr=0.001, 
        dataset='seg_isic',
        log_name='log_name',
        device='cuda',
        folds=1,
        overwrite=True,
        workers=8, 
        label_error_percent=lep,
        ratio=r)
