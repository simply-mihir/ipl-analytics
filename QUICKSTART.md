# Quick Start (≤ 5 minutes)

## 1. Install
```bash
pip install -r requirements.txt
```

## 2. Get the data
Place ONE of these in `data/raw/`:

- **Combined format (preferred):** a single `IPL.csv` file (one row per ball,
  with match metadata denormalised).
- **Legacy split format:** `matches.csv` + `deliveries.csv` from the
  classic Kaggle [IPL Complete Dataset 2008-2020](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020).

The data loader auto-detects which format is present.

## 3. Build everything
```bash
# Trains both models and caches processed data:
python -c "from src.models import build_and_save_all; build_and_save_all()"
```

You'll see something like:
```
== SCORE MODEL ==
  r2: 0.15   rmse: 35   mae: 26
== WINPROB MODEL ==
  accuracy: 0.67   roc_auc: 0.70   base_rate: 0.45
== LIVE WINPROB MODEL ==   <-- the 90% model
  accuracy: 0.73 overall (~90% in final over)
  roc_auc: 0.80
```

## 4. Run the dashboard
```bash
streamlit run app/streamlit_app.py
```

Open <http://localhost:8501>.

## 5. (Optional) Re-run notebooks to regenerate the README figures
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

---

# Deployment to Streamlit Cloud (free)

## What gets committed vs not

| File | Commit to GitHub? | Why |
|------|-------------------|-----|
| `data/raw/IPL.csv` | **No** — 105 MB exceeds GitHub's 100 MB file limit | Use Git LFS if you really want it |
| `data/processed/*.parquet` | **Yes** — only ~1.6 MB total after parquet compression | The dashboard reads these at runtime |
| `models/*.joblib` (includes `live_winprob_model.joblib`) | **Yes** — ~50 KB total | The dashboard needs these to predict |
| `reports/figures/*.png` | **Yes** | Embedded in README |
| `notebooks/*.ipynb` | **Yes** | The analytical story |

## Steps

1. **Locally**: run `python -c "from src.models import build_and_save_all; build_and_save_all()"` and
   `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb` so that the parquet, joblib,
   and PNG artifacts exist on disk.
2. **Edit `.gitignore`** — comment out (or delete) these two lines so the artifacts get committed:
   ```
   # data/processed/*.parquet
   # models/*.joblib
   ```
3. **Push** to GitHub:
   ```bash
   git add . && git commit -m "Deploy: include trained artifacts" && git push
   ```
4. Visit https://streamlit.io/cloud → sign in with GitHub.
5. **Create app** → select your repo → main file: `app/streamlit_app.py`.
6. The app installs from `requirements.txt` and goes live in 3–5 minutes.

---

# Common gotchas

**`IPL.csv` is too big for git push (> 100 MB).**
That's expected. Don't commit it. The dashboard uses the pre-trained
parquet + joblib files instead. If a teammate needs the raw CSV, share
a Kaggle link.

**Streamlit Cloud build fails: "ModuleNotFoundError: No module named 'src'".**
Streamlit Cloud doesn't add the project root to `PYTHONPATH` automatically.
The included `app/streamlit_app.py` already handles this (it prepends the
repo root to `sys.path`). If you renamed folders, check those lines.

**The dashboard works locally but fails on Streamlit Cloud with "FileNotFoundError".**
The trained artifacts are missing from the repo. Re-check step 2 — both
`data/processed/*.parquet` and `models/*.joblib` must be committed.

**You want to switch from the combined `IPL.csv` format to the older
`matches.csv` + `deliveries.csv`.** Just delete one and drop in the
other. The loader auto-detects.
