# LoanLens

Interactive loan-approval estimator built with Streamlit, scikit-learn (LogisticRegression) and imbalanced-learn (SMOTENC).

> LoanLens is an educational decision-support demonstration, not a lender or a substitute for professional underwriting.

## Files
`app.py` (whole application) · `requirements.txt` · `.streamlit/config.toml` (dark FinTech theme) · `README.md`

## Run locally (Python 3.11)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
No model file is needed. On first run the app trains a model (a spinner is shown), then saves `loan_model.joblib`
plus a `.sha256` checksum. The saved file is loaded only if its checksum, scikit-learn version, feature order,
target mapping and data fingerprint all match; otherwise it is ignored and retrained. If saving fails
(read-only/ephemeral disk) the app keeps working.

## Dataset integration
Handle: `muhammadmusharraf444/loan-approval-dataset` ([Kaggle](https://www.kaggle.com/datasets/muhammadmusharraf444/loan-approval-dataset)).

The app downloads the dataset with `kagglehub.dataset_download`, lists CSVs with `pathlib` (sorted), picks the first
whose normalised header contains the target `loan_status` and all 11 predictors, then loads it with
`kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, handle, <relative path>)`.

Expected schema (after trimming/lower-casing/underscoring column names): `no_of_dependents, education, self_employed,
income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value,
luxury_assets_value, bank_asset_value, loan_status`. `loan_id` and any other columns are dropped. `loan_status` must be
`Approved`/`Rejected` (case/whitespace-insensitive) and is mapped Approved→1, Rejected→0; numeric or other target values
are rejected rather than guessed.

Assumptions you should confirm against the dataset page: `loan_term` is treated as **years**, amounts as **₹** (change
`CURRENCY` in `app.py` if different). The dataset's provenance is not documented as verified lending outcomes, and the
app does not claim that.

If download, authentication, file selection, parsing or schema validation fails, a concise warning is shown and
seeded synthetic data is used instead. The banner always states which data the active model was trained on.

### Optional Kaggle credentials
The public dataset normally needs none. If needed, set `KAGGLE_USERNAME` / `KAGGLE_KEY` as environment variables or in
Streamlit secrets (`.streamlit/secrets.toml` locally, **Settings → Secrets** on Community Cloud):
```toml
KAGGLE_USERNAME = "your-username"
KAGGLE_KEY = "your-key"
```
Never commit `secrets.toml`. The app never calls an interactive login and never prints credentials.

## Deploy to Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `.streamlit/config.toml`, `README.md` to a GitHub repo.
2. At share.streamlit.io choose **New app**, select the repo/branch, set main file to `app.py`.
3. Under **Advanced settings** choose Python **3.11** and (optionally) paste secrets.
4. Deploy. First load trains the model (a few seconds); later reruns use the cache.

## Design notes
- Calculator-only inputs (existing monthly debt, interest rate) affect EMI/DTI only, never the model.
- Local explanations are logistic-regression **log-odds contributions**, aggregated per original feature; not SHAP.
- Uploaded CSVs are processed in memory, never cached or saved; invalid rows are reported and excluded, not scored as declines.
- Gender and other protected characteristics are not model inputs. No fairness or compliance evaluation was done.

## Verification checklist
Run on 2026-10-11 in a sandbox (Python 3.12.3, streamlit 1.65, pandas 2.3.3, scikit-learn 1.7.2, imbalanced-learn 0.14.2,
kagglehub 1.0.2) using Streamlit's headless `AppTest` plus direct function tests.

**Performed and passed**
- [x] First-run training with no `loan_model.joblib` (SMOTENC applied, artifact + checksum saved, reload accepted)
- [x] Kaggle unavailable (sandbox network blocked): warning shown, synthetic data used, no crash
- [x] Corrupt artifact ignored and replaced by automatic retraining
- [x] Live prediction: changing credit score 850 → 320 flips APPROVED → DECLINED with no button
- [x] Zero-interest EMI (₹6M income, 15M/10y → ₹125,000/month); invalid term / NaN EMI → N/A
- [x] Zero income: prediction blocked with a message, DTI and loan-to-income show N/A, no exception
- [x] Unseen categories at inference give finite probabilities
- [x] Explanation reconstruction: intercept + summed contributions reproduces predict_proba (to 1e-9)
- [x] CSV validation: valid file, missing column, non-numeric, blank, unknown category, mixed good/bad rows (only good scored; row numbers reported)
- [x] "Generate 10" gives a new batch each click and persists across unrelated reruns
- [x] Single-class donut charts (all approved / all declined) render without error
- [x] Kaggle discovery/cleaning logic using a **mocked** kagglehub (nested CSV, decoy CSV, padded headers/values, negative asset clipped, ambiguous numeric target refused)

**Not verified**
- [ ] A real download from Kaggle (the sandbox could not reach it): the actual file name, columns, units and class balance are unconfirmed
- [ ] Python 3.11 and the pinned lower bounds of dependencies (tested on 3.12 with latest in-range versions)
- [ ] Deployment on Streamlit Community Cloud, and visual rendering in a real browser/mobile (theme, CSS, charts)
- [ ] Actual file upload through the browser widget (validation was tested on DataFrames, not through `st.file_uploader`)
- [ ] Model quality on real data; synthetic metrics only show the pipeline functions
