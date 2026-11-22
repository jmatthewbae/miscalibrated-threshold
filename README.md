# The miscalibrated threshold

Code for "The miscalibrated threshold: Metric structure and subgroup failure in AI-based skin cancer detection" (Bae, PLOS Digital Health, submitted).

Two ResNet-50 models, trained on 1,200 Fitzpatrick17k images each (50% malignant). Run A uses FST I-II only. Run B uses 400 per band (I-II, III-IV, V-VI). Both are tested on DDI at one Youden's J threshold fit on Run A's 5-fold out-of-fold predictions.

## Data

Not included. Get both datasets from their maintainers.

```
data/fitzpatrick17k/fitzpatrick17k.csv    https://github.com/mattgroh/fitzpatrick17k
data/fitzpatrick17k/images/<md5hash>.jpg
data/ddi/ddi_metadata.csv                 Stanford AIMI (DDI)
data/ddi/images/<DDI_file>
```

`outputs/splits/` lists the exact images each run used.

## Run

```
./setup.sh
./run.sh
```

Or by stage:

```
python build_splits.py
python train.py --run run_a --fold 0     # folds 0-4
python train.py --run run_a
python train.py --run run_b
python evaluate.py
python figures.py
```

Settings are in `config.py`. Results in the paper were produced on an NVIDIA GeForce RTX 5070 with the CUDA 12.8 PyTorch build.

## License

MIT
