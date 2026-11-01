# Two-Tower Retrieval on Amazon Reviews 2023 (Movies & TV)

A TensorFlow two-tower candidate-retrieval model trained on the McAuley Lab
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) **Movies_and_TV** category,
built from the raw review dump with a leakage-free leave-last-out split.

| Notebook | What it does | Open |
|---|---|---|
| `01_data_prep.ipynb` | Parse 17M raw reviews → dedup → iterative 5-core → leave-last-out split → history sequences | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/problemme/two-twoer-movies/blob/main/notebooks/01_data_prep.ipynb) |
| `02_train_two_tower.ipynb` | Item metadata features → two-tower model → in-batch softmax training → evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/problemme/two-twoer-movies/blob/main/notebooks/02_train_two_tower.ipynb) |

## Results

Full-corpus retrieval over all 197,943 items (no candidate sampling), 50K sampled users.

| Model | Split | Recall@10 | Recall@50 | NDCG@10 |
|---|---|---|---|---|
| Popularity (top train items) | test | _TBD_ | _TBD_ | – |
| Two-tower (best epoch = 2) | valid | 0.0716 | 0.1620 | 0.0359 |
| Two-tower (best epoch = 2) | test | _TBD_ | _TBD_ | _TBD_ |

Validation by epoch (T4 GPU, ~85 s/epoch):

| Epoch | Train loss | Recall@10 | Recall@50 | NDCG@10 |
|---|---|---|---|---|
| 1 | 7.19 | 0.0654 | 0.1524 | 0.0338 |
| 2 | 6.08 | **0.0716** | **0.1620** | **0.0359** |
| 3 | 5.21 | 0.0632 | 0.1495 | 0.0304 |

Training loss keeps falling while validation peaks at epoch 2, so the model overfits after that
(likely through the per-user ID embedding, with ~9 training interactions per user).
The training notebook keeps the weights from the best validation epoch.

## Data

| Stage | Interactions | Users | Items |
|---|---|---|---|
| Raw reviews, deduplicated (user, item) | 17,158,519 | 6,503,429 | 747,764 |
| After iterative 5-core (12 rounds) | 7,441,129 | 657,203 | 197,943 |
| Train / valid / test | 6,126,723 / 657,203 / 657,203 | | |

Cold-start items (absent from train): 0.04% of valid targets, 0.14% of test targets.

## Method

**Split — leave-last-out.** Each user's interactions are sorted by timestamp (stable sort for ties);
the last one is the test target, the second-to-last the validation target, the rest are training.

**No leakage.**
- Each example's history contains only the user's items *before* that interaction
  (train sees earlier train items, valid sees all train, test sees train + valid).
- Item popularity for the logQ correction and the popularity baseline is counted on train only.

**Model.**
- *User tower*: user-ID embedding ⊕ masked mean of the last 20 history item embeddings → MLP → L2-normalize.
- *Item tower*: item-ID embedding ⊕ mean title-word embedding ⊕ mean category embedding → MLP → L2-normalize.
  Item-ID embeddings are shared with the user tower's history encoder.
- *Loss*: in-batch sampled softmax (batch 4096, temperature 0.05) with logQ popularity correction;
  duplicate items within a batch are masked out as negatives.

## Run it

### Colab (recommended)
1. Open `01_data_prep.ipynb` with the badge above, use a CPU runtime, run all cells (~30 min).
   The raw files are downloaded automatically from the McAuley Lab server, or copied from
   `MyDrive/` if you have already put them there. Outputs go to `MyDrive/two_tower_movies_tv/`.
2. Open `02_train_two_tower.ipynb`, switch to a **T4 GPU** runtime, run all cells.
   Test metrics are written to `MyDrive/two_tower_movies_tv/metrics.json`.

### Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATA_DIR=./data/amazon2023        # raw files, intermediate files and outputs
jupyter notebook notebooks/
```
Needs ~16 GB RAM. Training on CPU works but is much slower than a T4.

## Repository layout
```
notebooks/
  01_data_prep.ipynb        # cleaning, 5-core, leave-last-out split
  02_train_two_tower.ipynb  # features, model, training, evaluation
requirements.txt
```
Data, embeddings and weights are not committed (see `.gitignore`).

## Next steps
- Mask the user's already-seen items at evaluation time.
- Ablate the user-ID embedding / add embedding regularization to reduce overfitting.
- Item text embeddings from descriptions and review text (train-period reviews only).
- Mixed negative sampling (in-batch + uniform) and hard negatives.

## Citation
```bibtex
@article{hou2024bridging,
  title   = {Bridging Language and Items for Retrieval and Recommendation},
  author  = {Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal = {arXiv preprint arXiv:2403.03952},
  year    = {2024}
}
```
