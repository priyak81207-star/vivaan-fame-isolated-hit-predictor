# Fame-Isolated Hit Predictor

A Streamlit web app that lets anyone upload an unreleased song and get the same kind of analysis used in my research: a real audio-feature extraction, a prediction from my actual trained classifier, an artist fame lookup, and a verdict sorted into the same four categories (Pure Sound Hit / Mega-Hit / Coattail Ride / Genre Flop) used for the 127 songs in the study.

## What it does

1. Runs the real feature extractor (`song_features.py`) on the uploaded audio, pulling out the 14 features used in the actual analysis.
2. Scores the song with the actual trained classifier (`model.joblib`) — the same Random Forest reported in the paper (61.5% held-out accuracy) — instead of a simple rule of thumb.
3. Looks up the artist's real fame on Last.fm (or lets you estimate it, if you skip the Last.fm setup) and combines it with the song's sound score into the same four categories the original 127-track study used, so a fresh upload gets sorted the same way the training data was.

## Files

- `app.py` — the Streamlit app
- `song_features.py` — the feature-extraction script used to build the whole dataset
- `train_and_save_model.py` — retrains the classifier from `features_editable.xlsx` and saves `model.joblib`, `reference_stats.json`. Already run once — those files are included, so it doesn't need to be re-run unless the dataset changes.
- `lastfm_fame.py` — looks up an artist's Last.fm listener count (fame signal)
- `features_editable.xlsx` — the 127-song dataset the whole analysis is built on
- `requirements.txt` — for `pip install -r requirements.txt` or Streamlit Community Cloud

## Getting a free Last.fm API key (free forever, about a minute, no premium account)

1. Go to <https://www.last.fm/api/account/create> and log in (or create a free Last.fm account).
2. Fill in the short form (an app name and a contact email are all that's required).
3. Submit it — the API key is shown immediately.
4. Paste that into the app's "Last.fm lookup settings" box when running it, or add it as a Secret in Streamlit Community Cloud so it doesn't need to be entered every time:

```toml
   # .streamlit/secrets.toml (don't commit this file)
   LASTFM_API_KEY = "..."
```

If this step is skipped, the app still works — it just asks for the artist's fame level to be described directly (emerging / rising / established / major) instead of pulling a real Last.fm listener count.

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying

Push this folder to a GitHub repo and deploy it on <https://share.streamlit.io>, pointing it at `app.py`. Add the Last.fm key there under the app's Settings → Secrets if using the `st.secrets` approach above.

## Testing notes

I tested the full pipeline end-to-end — upload, feature extraction, and prediction — using a generated test tone to confirm every step runs without errors. Before demoing, I'd recommend running it on 2–3 real songs that are already in the dataset, so the app's prediction can be checked against the `Success_Label` that song actually got during training.

## Honesty notes for judges / write-up

- The "sound score" percentage is the trained model's own `predict_proba` output — a genuine model prediction, not a distance-from-target heuristic.
- The "why" section is ranked by the model's actual `feature_importances_`, not an arbitrary pick of features.
- The fame/sound combination reuses the same four labels as the original 127-track analysis, applied to a new song — it's the same idea, not a new metric invented for the app.
- Every one of the 14 features is genuinely extracted from the uploaded audio, and fame is either a real Last.fm listener count or an honest self-estimate — never a hidden default.
