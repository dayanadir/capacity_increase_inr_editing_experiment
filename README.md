# Capacity Increase INR Editing Experiment

This repository reproduces the 8-run "capacity increase" INR-editing experiment  
on MNIST INRs from our paper [On the Expressive Power of Permutation-Equivariant Weight-Space Networks](https://arxiv.org/pdf/2602.01083),
co-authored with [Yam Eitan](https://github.com/yameitan) and supervised by [Haggai Maron](https://github.com/Haggaim):

```
@article{dayan2026expressive,
  title={On the Expressive Power of Permutation-Equivariant Weight-Space Networks},
  author={Dayan, Adir and Eitan, Yam and Maron, Haggai},
  journal={arXiv preprint arXiv:2602.01083},
  year={2026}
}
```

4× DWS and 4× ScaleGMN models with output feature dimension
∈ {1, 2, 4, 8}, under a parameter-count budget matched to the `feat=1`
baseline. Each model is trained to predict the gradient that edits an INR so
that its rendered image is the **dilation** of the original MNIST digit.

## Acknowledgements

The experiment code in this repository is adapted from the codebase of
**GradMetaNet: An Equivariant Architecture for Learning on Gradients**
([arxiv:2507.01649](https://arxiv.org/pdf/2507.01649)). If you build on this
work, please cite:

```bibtex
@article{gelberg2026gradmetanet,
  title   = {{GradMetaNet}: An Equivariant Architecture for Learning on Gradients},
  author  = {Gelberg, Yoav and Eitan, Yam and Navon, Aviv and Shamsian, Aviv
             and Putterman, Theo and Bronstein, Michael and Maron, Haggai},
  journal = {Advances in Neural Information Processing Systems},
  volume  = {38},
  pages   = {154115--154158},
  year    = {2026}
}
```

## Setup

```bash
conda create -n capacity-increase python=3.10
conda activate capacity-increase
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

The code expects to be run from the repo root (so that `experiments` and `nn`
resolve as top-level Python packages).

## Data

Two data sources are required.

1. **MNIST INRs** — pre-trained per-image SIRENs (one INR per MNIST training /
  test image). Download:
   [https://www.dropbox.com/scl/fo/2akm78f7ot4o52o1mrtun/ADLLU8zOj73tswlhhCR_yF8/mnist-inrs.zip?rlkey=4oj9ao6om06tgmfabyctzu2n2&e=1&dl=0](https://www.dropbox.com/scl/fo/2akm78f7ot4o52o1mrtun/ADLLU8zOj73tswlhhCR_yF8/mnist-inrs.zip?rlkey=4oj9ao6om06tgmfabyctzu2n2&e=1&dl=0)
   Unzip into a directory such as `./data/mnist_inrs/mnist-inrs`. The loader
   expects `.pth` files arranged into `training/` and `testing/` subtrees.
2. **MNIST images** — `torchvision.datasets.MNIST` will auto-download MNIST on
  first run. Point `data.image_data_path` at the parent directory containing
   `MNIST/raw/` (e.g. `./data/images/MNIST/raw`).

Either edit `experiments/inr_editing/inr_editing.yaml` or pass the paths on the
command line, as shown in the commands below.

## Reproducing the 8 runs

All runs use `train.epochs=150`. The 4× DWS and 4× GMN configurations below
match the parameter-budget design of the experiment (models with `feat>1`
have a parameter count ≤ the `feat=1` baseline). The commands can be launched
in parallel (e.g. in separate tmux panes).

Replace `./data/mnist_inrs/mnist-inrs` and `./data/images/MNIST/raw` with the
absolute paths to your local copies of the data.

### 4× DWS

```bash
# feat=1 (baseline): hidden_dim=32, n_hidden=4
python experiments/inr_editing/training.py \
  train.model=dws \
  dws_args.hidden_dim=32 dws_args.n_hidden=4 dws_args.output_features=1 \
  wandb.prefix=ie-dws-f1-b \
  data.dir=./data/mnist_inrs/mnist-inrs data.image_data_path=./data/images/MNIST/raw

# feat=2: hidden_dim=30, n_hidden=4, num_heads=10
python experiments/inr_editing/training.py \
  train.model=dws \
  dws_args.hidden_dim=30 dws_args.n_hidden=4 +dws_args.num_heads=10 dws_args.output_features=2 \
  wandb.prefix=ie-dws-f2-h30h10 \
  data.dir=./data/mnist_inrs/mnist-inrs data.image_data_path=./data/images/MNIST/raw

# feat=4
python experiments/inr_editing/training.py \
  train.model=dws \
  dws_args.hidden_dim=30 dws_args.n_hidden=4 +dws_args.num_heads=10 dws_args.output_features=4 \
  wandb.prefix=ie-dws-f4-h30h10 \
  data.dir=./data/mnist_inrs/mnist-inrs data.image_data_path=./data/images/MNIST/raw

# feat=8
python experiments/inr_editing/training.py \
  train.model=dws \
  dws_args.hidden_dim=30 dws_args.n_hidden=4 +dws_args.num_heads=10 dws_args.output_features=8 \
  wandb.prefix=ie-dws-f8-h30h10 \
  data.dir=./data/mnist_inrs/mnist-inrs data.image_data_path=./data/images/MNIST/raw
```

`num_heads=10` is used at `hidden_dim=30` so that the SAB head count divides
the hidden dim evenly.

### 4× GMN

```bash
# feat=1 (baseline): d_hid=128, num_layers=10
python experiments/inr_editing/training.py \
  train.model=scalegmn \
  scalegmn_args.symmetry=permutation \
  scalegmn_args.d_hid=128 scalegmn_args.num_layers=10 scalegmn_args.equiv_out_features=1 \
  wandb.prefix=ie-gmn-f1-b \
  data.dir=./data/mnist_inrs/mnist-inrs data.image_data_path=./data/images/MNIST/raw

# feat=2: d_hid=127, num_layers=10
python experiments/inr_editing/training.py \
  train.model=scalegmn \
  scalegmn_args.symmetry=permutation \
  scalegmn_args.d_hid=127 scalegmn_args.num_layers=10 scalegmn_args.equiv_out_features=2 \
  wandb.prefix=ie-gmn-f2-h127 \
  data.dir=./data/mnist_inrs/mnist-inrs data.image_data_path=./data/images/MNIST/raw

# feat=4
python experiments/inr_editing/training.py \
  train.model=scalegmn \
  scalegmn_args.symmetry=permutation \
  scalegmn_args.d_hid=127 scalegmn_args.num_layers=10 scalegmn_args.equiv_out_features=4 \
  wandb.prefix=ie-gmn-f4-h127 \
  data.dir=./data/mnist_inrs/mnist-inrs data.image_data_path=./data/images/MNIST/raw

# feat=8
python experiments/inr_editing/training.py \
  train.model=scalegmn \
  scalegmn_args.symmetry=permutation \
  scalegmn_args.d_hid=127 scalegmn_args.num_layers=10 scalegmn_args.equiv_out_features=8 \
  wandb.prefix=ie-gmn-f8-h127 \
  data.dir=./data/mnist_inrs/mnist-inrs data.image_data_path=./data/images/MNIST/raw
```

To run without Weights & Biases logging, add `wandb.log=false`.

## Results

Best test loss per run, taken from our W&B runs of the configurations above:


| Model | feat | Best test loss |
| ----- | ---- | -------------- |
| DWS   | 1    | 0.026429       |
| DWS   | 2    | 0.020132       |
| DWS   | 4    | 0.015818       |
| DWS   | 8    | 0.013208       |
| GMN   | 1    | 0.022444       |
| GMN   | 2    | 0.017282       |
| GMN   | 4    | 0.012189       |
| GMN   | 8    | 0.009810       |


- Increasing output feature capacity (`feat=1 → 2 → 4 → 8`) consistently
improves test loss for both DWS and GMN.
- DWS: 0.026429 → 0.013208 (~50.0% relative reduction).
- GMN: 0.022444 → 0.009810 (~56.3% relative reduction).
- At every tested `feat`, GMN achieves lower best test loss than DWS under
these parameter-controlled configurations.

## Repository layout

```
capacity_increase_inr_editing_experiment/
├── README.md
├── requirements.txt
├── experiments/
│   ├── data/                  
│   ├── utils/                 
│   └── inr_editing/
│       ├── training.py        # entry point
│       └── inr_editing.yaml   # Hydra config
└── nn/
    ├── inr.py                 # SIREN / BatchSiren / hookable INR
    ├── attention.py
    ├── equivariant_layers.py
    ├── dws/                   # Deep Weight Spaces (Navon et al.)
    └── scalegmn/              # Graph Metanetworks
```

