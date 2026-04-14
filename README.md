# CSCLog Training — Multi-OS Log Anomaly Detection

Training pipeline for the **CSCLog** model applied to Linux, Windows, macOS, and network device logs.

---

## Project Structure

```
csclog-training/
├── conf/                   Hydra configs
│   ├── config.yaml         Root (defaults: data=linux, model=csclog, train=default)
│   ├── data/               Per-OS data configs
│   ├── model/csclog.yaml   Model hyperparameters
│   └── train/default.yaml  Training hyperparameters
├── src/
│   ├── model/              CSCLog variants + EarlyStopping
│   ├── data/
│   │   ├── parsers/        Drain + per-OS parsers (Linux, Windows, Mac, Network)
│   │   ├── embedder.py     BERT + TF-IDF sentence embeddings
│   │   ├── labeler.py      Component map + train/test split
│   │   └── sequencer.py    Sliding-window encoding for train/eval
│   ├── train.py            Hydra + MLflow training entry point
│   ├── evaluate.py         Top-K anomaly detection evaluation
│   └── serve/              FastAPI serving API
├── scripts/
│   ├── fetch_datasets.py   Download Loghub datasets
│   └── prepare_all.py      Full data preparation pipeline
├── data/
│   ├── raw/{linux,windows,mac,network}/
│   └── sequences/{linux,windows,mac,network}/
├── checkpoints/            Best model checkpoints per OS
└── docker-compose.yml
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
# For CUDA (adjust to your version):
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

### 2. Download datasets

```bash
python scripts/fetch_datasets.py --os linux windows mac network
```

This downloads the 2k-sample files from Loghub.  
For full datasets (BGL 4.7M, Thunderbird), download from
[https://zenodo.org/record/3227177](https://zenodo.org/record/3227177)
and place in `data/raw/<os_type>/`.

### 3. Prepare data

```bash
# Prepare all OS types (adjusts --bert to BERT checkpoint path)
python scripts/prepare_all.py --bert ../CSCLog/model/bert --os linux

# Or all at once:
python scripts/prepare_all.py --bert ../CSCLog/model/bert
```

Output in `data/sequences/<os_type>/`:
- `train_normal.csv`, `test_normal.csv`, `test_anomaly.csv`
- `<os>_templates.csv`, `<os>_sentences_emb.json`, `<os>_component.json`

### 4. Train

```bash
# Train Linux model
python src/train.py data=linux

# Train all OS types
python src/train.py --multirun data=linux,windows,mac,network

# Ablation study (no IREncoder)
python src/train.py data=linux model.variant=wo_ic

# Override hyperparameters
python src/train.py data=linux train.lr=1e-4 train.num_epochs=50
```

View results:
```bash
mlflow ui
# open http://localhost:5000
```

### 5. Evaluate

Evaluation runs automatically after each epoch during training.  
To run standalone:

```python
from src.evaluate import run_test
run_test(
    test_normal_csv="data/sequences/linux/test_normal.csv",
    test_anomaly_csv="data/sequences/linux/test_anomaly.csv",
    checkpoint_path="checkpoints/linux/best.pth",
    templates_csv="data/sequences/linux/linux_templates.csv",
    emb_json="data/sequences/linux/linux_sentences_emb.json",
    com_json="data/sequences/linux/linux_component.json",
    window_size=9,
    model_cfg=...,   # OmegaConf from conf/model/csclog.yaml
    num_candidates=[1, 5],
)
```

### 6. Serve

```bash
# Local
PROJECT_ROOT=$(pwd) uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

# Docker
docker compose up
```

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      {"event_id": "abc12345", "component": "sshd", "timestamp": "2024-01-01T10:00:00+00:00"},
      ...
    ],
    "os_type": "linux",
    "top_k": 1
  }'
```

---

## Configuration Reference

### Model variants (`conf/model/csclog.yaml → variant`)

| Variant | Description |
|---------|-------------|
| `full` | Full CSCLog — shared LSTM + GCN IREncoder (default) |
| `wo_ic` | No inter-component GCN (ablation) |
| `no_shared` | Per-component separate LSTMs (ablation) |
| `wo_lstm` | Mean pooling instead of LSTM (ablation) |

### OS data configs

| OS | Source | Label Strategy |
|----|--------|---------------|
| `linux` | BGL.log (Loghub) | Column-based (`-` = normal) |
| `windows` | Windows.log CSV (Loghub) | Severity: `Error`, `Critical` |
| `mac` | Mac.log (Loghub) | Severity: `error`, `fault`, `critical` |
| `network` | openstack_normal/abnormal.log | Filename-based (`abnormal` = all anomaly) |

---

## Original Paper

> CSCLog: A Component Subsequence Correlation-Aware Log Anomaly Detection Method  
> Hang Zhang et al.  
> Source: [https://github.com/Hang-Z/CSCLog](https://github.com/Hang-Z/CSCLog)
