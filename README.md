# Fame-Isolated Hit Predictor (v2)

This replaces the old `app.py` (the one that only auto-detected tempo and
compared 3 sliders against fixed targets). This version:

1. Runs the **real** feature extractor (`song_features.py`) on the uploaded
   audio, pulling out all 15 features used in the actual analysis -- not just
   tempo.
2. Scores the song with the **actual trained classifier** (`model.joblib`),
   the same Random Forest reported in the paper (65.4% held-out accuracy),
   instead of three hand-picked threshold rules.
3. Looks up the artist's real fame on Last.fm (or asks you to estimate it,
   if you skip the Last.fm setup) and combines it with the song's sound score
   into the **same four categories** the original 127-track study used
   (Pure Sound Hit / Mega-Hit / Coattail Ride / Genre Flop) -- so a fresh
   upload gets sorted the same way the training data was.

## Files

- `app.py` -- the Streamlit app
- `song_features.py` -- the real feature-extraction script (already existed
  in the Drive folder; copied here unchanged)
- `train_and_save_model.py` -- retrains the classifier from
  `features_editable.xlsx` and saves `model.joblib`, `key_encoder.joblib`,
  `reference_stats.json`. **Already run once** -- those three files are
  included, so you don't have to re-run this unless the dataset changes.
- `lastfm_fame.py` -- looks up an artist's Last.fm listener count (fame
  signal). `spotify_fame.py` is no longer used -- it's left in the folder
  only for reference, since Spotify's February 2026 policy change now
  requires the app owner's account to have an active Premium subscription
  just to search, which isn't something every school-project user has.
- `requirements.txt` -- for `pip install -r requirements.txt` or Streamlit
  Community Cloud

## Getting a free Last.fm API key (free forever, ~1 minute, no premium account)

Unlike Spotify, Last.fm's API has no premium requirement, no app review,
and no waiting -- you get a working key instantly.

1. Go to <https://www.last.fm/api/account/create> and log in (or create a
   free Last.fm account -- it's free, just for having an account, no
   subscription).
2. Fill in the tiny form (an app name like "Vivaan Hit Predictor" and a
   contact email are all that's required -- you can leave the callback
   URL/homepage blank).
3. Submit it -- your **API key** is shown immediately on the next page.
4. Paste that into the app's "Last.fm lookup settings" box when you run
   it, or (better, for a deployed version) add it as a **Secret** in
   Streamlit Community Cloud so you don't have to paste it every time:

   ```toml
   # .streamlit/secrets.toml (don't commit this file)
   LASTFM_API_KEY = "..."
   ```

   and change the `st.text_input(...)` line for `lastfm_api_key` in
   `app.py` to read `st.secrets["LASTFM_API_KEY"]` instead, so judges
   testing it don't need to get their own key.

If you skip this entirely, the app still works -- it just asks you to
describe the artist's fame level yourself (emerging / rising /
established / major) instead of pulling a real Last.fm listener count.

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying (same as the old app)

Push this folder to a GitHub repo and deploy it on
<https://share.streamlit.io> the same way the original app was deployed --
just point it at `app.py` in this folder instead. Add the Last.fm secret
there under the app's **Settings -> Secrets** if you're using the
`st.secrets` approach above.

## What I could and couldn't test

I don't have any of Vivaan's actual audio files, so I built and tested
this against a synthetic test tone I generated myself:

- `song_features.py`'s own built-in self-test passes (13/14 checks -- one
  pre-existing edge case in ambiguous relative-key detection, unrelated to
  anything here).
- The full pipeline (upload -> extract 15 features -> encode key -> predict
  with the real model) runs end-to-end with no errors on that synthetic
  clip.
- The Streamlit app itself boots cleanly and serves its page.

What I could NOT verify is whether the **predictions are sensible** on a
real song, since a synthetic tone doesn't sound like an actual track. The
first thing to do with this is run it on 2-3 of Vivaan's real songs
(ideally ones already in the dataset, so you can compare the app's
prediction against the `Success_Label` that song actually got in training)
and sanity-check the verdict and reasoning against what you already know
about those tracks.

## Honesty notes for judges / write-up

- The "sound score" percentage is the trained model's own
  `predict_proba` output -- a genuine model prediction, not a
  distance-from-target heuristic.
- The "why" section is ranked by the model's actual
  `feature_importances_`, not an arbitrary pick of 3 features.
- The fame/sound combination reuses the **same four labels** as the
  original 127-track analysis, applied prospectively to a new song --
  it's the same idea, not a new metric invented for the app.
- Unlike the old app, nothing here claims to be "automatic" when it
  isn't: every one of the 15 features is genuinely extracted from the
  uploaded audio, and fame is either a real Last.fm listener count or an
  honest self-estimate -- never a hidden default.
