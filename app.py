"""
Fame-Isolated Hit Predictor
============================

Upload an unreleased track + tell it who the artist is -> it extracts the
real 15-feature audio profile (song_features.py, the same extractor used to
build the whole dataset), scores it with the actual trained classifier
(model.joblib, same one reported in the research: ~65.4% held-out accuracy),
looks the artist up on Last.fm for a real fame signal, and combines both into
one of the same four categories the original study sorted its 127 training
tracks into -- so the verdict is a genuine extension of the analysis, not a
new, disconnected feature.
"""
import base64
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from song_features import Config, extract_features, true_duration
from lastfm_fame import get_artist_fame, search_artists


def _decode_if_missing(binary_name: str, b64_name: str) -> None:
    """GitHub's web editor (used to deploy this repo) can only accept text,
    so the two small binary model files are checked into the repo as base64
    text and decoded back to real binary once here at startup, the first
    time the app boots. Harmless no-op on every later rerun."""
    if not os.path.exists(binary_name) and os.path.exists(b64_name):
        with open(b64_name, "r") as f:
            data = base64.b64decode(f.read())
        with open(binary_name, "wb") as f:
            f.write(data)


_decode_if_missing("model.joblib", "model_b64.txt")
_decode_if_missing("key_encoder.joblib", "key_encoder_b64.txt")

AUDIO_FEATURES = [
    "Song Length", "Intro Length", "Chorus Length", "Repetition",
    "Time to First Hook", "Chord Complexity", "Melodic Range",
    "Tempo", "Syncopation", "Danceability", "Loudness",
    "Dynamic Range", "Distortion", "Spectral Brightness", "Key_Encoded",
]

# Cap how much of the track the expensive per-frame analysis (melody
# tracking, chroma/CQT, structure detection) actually looks at. Melody
# tracking especially can take minutes on a full song on a slow CPU. The
# track's REAL length is looked up separately (see true_duration below) and
# is always what gets fed to the model -- this is a speed change, not an
# accuracy one.
ANALYSIS_WINDOW_SECONDS = 150.0

st.set_page_config(page_title="Fame-Isolated Hit Predictor", page_icon="music", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    key_encoder = joblib.load("key_encoder.joblib")
    with open("reference_stats.json") as f:
        reference = json.load(f)
    return model, key_encoder, reference


model, key_encoder, reference = load_model()
meta = reference["_meta"]

st.title("Fame-Isolated Hit Predictor")
st.caption(
    "Built on the same 127-track analysis as the research paper -- a real "
    f"classifier trained on {meta['n_tracks']} tracks "
    f"({meta['held_out_accuracy']*100:.1f}% held-out accuracy), not a rule-of-thumb."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Step 1: inputs
# ---------------------------------------------------------------------------
st.markdown("### Step 1 -- Upload the track and name the artist")
audio_file = st.file_uploader("Unreleased song (MP3, WAV, or M4A)", type=["mp3", "wav", "m4a"])

with st.expander("Last.fm lookup settings (one-time setup -- see README)"):
    st.caption(
        "This looks the artist up on Last.fm's free public API to get a real "
        "listener count, instead of guessing or Googling by hand. It needs a "
        "free Last.fm API key (no premium account, no payment, ever) -- "
        "get one at last.fm/api/account/create, see README.md."
    )
    lastfm_api_key = st.text_input("Last.fm API key", type="password", key="lastfm_key")

manual_fame_tier = None
selected_artist = None

if lastfm_api_key:
    # With a key, search Last.fm and make the user pick the exact artist
    # from a dropdown, instead of typing a free-text name and hoping the
    # single best-guess exact-match logic lands on the right one -- this is
    # what "can we drop-down from a search list" is asking for. (This used
    # to query Spotify, but Spotify's Feb 2026 policy change now requires the
    # app owner to have Premium just to search -- Last.fm has no such
    # requirement, so it's a free drop-in replacement for the same idea.)
    st.caption(
        "Search Last.fm and pick the exact artist/band from the results -- this "
        "avoids the lookup matching the wrong same-named artist, or missing on a typo."
    )
    search_col, button_col = st.columns([4, 1])
    with search_col:
        search_query = st.text_input(
            "Search for the artist/band on Last.fm",
            placeholder="e.g. The Local Train",
            key="artist_search_query",
            label_visibility="collapsed",
        )
    with button_col:
        search_clicked = st.button("Search", use_container_width=True)

    if search_clicked and search_query.strip():
        with st.spinner("Searching Last.fm..."):
            results, search_error = search_artists(search_query, lastfm_api_key)
        if search_error:
            st.warning(f"Last.fm search didn't work: {search_error}")
            st.session_state["lastfm_search_results"] = []
        else:
            st.session_state["lastfm_search_results"] = results
        st.session_state["lastfm_selected_artist"] = None  # a fresh search clears any earlier pick

    search_results = st.session_state.get("lastfm_search_results", [])
    if search_results:
        options = {
            f"{a.name} -- {a.listeners:,} listeners": a
            for a in search_results
        }
        choice_label = st.selectbox("Select the exact artist/band", list(options.keys()), key="artist_choice")
        selected_artist = options[choice_label]
        st.session_state["lastfm_selected_artist"] = selected_artist
    else:
        selected_artist = st.session_state.get("lastfm_selected_artist")

    artist_name = selected_artist.name if selected_artist else ""
    if not selected_artist:
        st.caption("Search above and pick the artist from the dropdown before analyzing.")
else:
    artist_name = st.text_input("Artist / band name", placeholder="e.g. The Local Train")
    st.info(
        "No Last.fm API key entered -- you can still get a prediction by "
        "describing the artist's current reach yourself below."
    )
    manual_fame_tier = st.select_slider(
        "Roughly how established is this artist right now?",
        options=["emerging (little to no following yet)", "rising (a real but modest following)",
                 "established (well-known within the genre)", "major (broad mainstream reach)"],
    )

run = st.button("Analyze track", type="primary", disabled=(audio_file is None or not artist_name.strip()))

# ---------------------------------------------------------------------------
# Step 2: run the analysis
# ---------------------------------------------------------------------------
if run:
    with st.spinner(
        f"Extracting audio features (analyzing the first {ANALYSIS_WINDOW_SECONDS/60:.1f} "
        "minutes of the track for speed -- usually 15-40s, not minutes)..."
    ):
        suffix = Path(audio_file.name).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        try:
            feats = extract_features(tmp_path, cfg=Config(duration=ANALYSIS_WINDOW_SECONDS))
            real_length = true_duration(tmp_path)  # cheap: doesn't decode the whole file
            if real_length:
                feats["Song Length"] = round(real_length, 2)
        except Exception as e:
            st.error(f"Couldn't extract features from this file: {e}")
            st.stop()

    # Some songs build slowly and don't reach their chorus until after the
    # analyzed window (the training data itself has hooks arriving past 3
    # minutes in slower-building tracks) -- if that happens here, the
    # structure features below default to 0, which could look like "no
    # chorus" when it really means "chorus wasn't in the part we analyzed."
    if real_length and real_length > ANALYSIS_WINDOW_SECONDS and not feats.get("Chorus Length"):
        st.caption(
            "No clear chorus/hook was found in the analyzed portion of this track. "
            "If this song builds slowly and its hook lands later, the Chorus Length and "
            "Time to First Hook figures below may understate it -- for a slow-build song, "
            "trimming and uploading a clip that starts near the hook will give a truer read."
        )

    # ---- Sound score: the real trained classifier, not a threshold rule ----
    key_str = str(feats.get("Key") or "")
    if key_str in key_encoder.classes_:
        key_encoded = int(key_encoder.transform([key_str])[0])
    else:
        # -1 used to be passed straight into the model here. That's not a
        # graceful "skip this feature" -- the classifier never saw -1 during
        # training, and since it's lower than every real key code, a
        # threshold split like "Key_Encoded <= 0.5" routes -1 exactly like it
        # would route the key that happens to sit first alphabetically ("A
        # major"). So an unseen key wasn't being ignored, it was being
        # silently swapped for a specific WRONG one, which could tilt the
        # verdict for no real reason. A neutral, in-range fallback (the
        # middle key code) can't push the prediction toward any one key.
        key_encoded = int(np.median(np.arange(len(key_encoder.classes_))))
        st.warning(
            f"Key '{key_str}' wasn't one of the keys in the training set "
            f"({', '.join(meta['known_keys'])}), so this one input is a neutral "
            "placeholder rather than a real reading -- treat this result with a "
            "little more caution than usual, since it's built on 14 solid "
            "features instead of 15."
        )

    row = {**{f: feats.get(f) for f in AUDIO_FEATURES[:-1]}, "Key_Encoded": key_encoded}
    X_new = pd.DataFrame([row])[AUDIO_FEATURES]
    sound_prob = float(model.predict_proba(X_new)[0][1])  # P(beats its own artist's baseline)

    # ---- Fame score: the exact artist picked from the Last.fm dropdown, if ----
    # ---- we're in that mode; otherwise the user's own manual estimate.      ----
    fame = selected_artist
    if fame is None and lastfm_api_key and artist_name:
        # Shouldn't normally happen -- the Analyze button requires a dropdown
        # pick whenever a key is set -- but fall back to a live name-match
        # lookup rather than silently having no fame data at all.
        fame, fame_error = get_artist_fame(artist_name, lastfm_api_key)
        if fame_error:
            st.warning(f"Last.fm lookup didn't work, so falling back: {fame_error}")
    if fame:
        fame_tier = fame.fame_tier
        fame_label = f"{fame.name} -- {fame.listeners:,} Last.fm listeners"
    elif manual_fame_tier:
        fame_tier = manual_fame_tier.split(" ")[0]
        fame_label = f"{artist_name} -- self-described as \"{manual_fame_tier}\""
    else:
        fame_tier = "unknown"
        fame_label = f"{artist_name} -- fame level not determined"

    st.markdown("---")
    st.markdown("## Result")

    # ---- The verdict: reuse the SAME four categories as the original study ----
    high_fame = fame_tier in ("established", "major")
    high_sound = sound_prob >= 0.5

    if fame_tier == "unknown":
        # This used to fall straight into the "not high_fame" branches below,
        # which don't just skip a fame claim -- they actively ASSERT one
        # ("this artist doesn't have a large existing audience yet"). Not
        # knowing an artist's fame is not the same as knowing they're
        # unfamous, and for a genuinely well-known artist (Last.fm lookup
        # failed, or the fame question just wasn't answered) that assertion
        # would be flatly wrong, not merely cautious. This is its own state,
        # not a 5th coin-flip default onto "low fame."
        quadrant = "Sound-Only Read (fame not determined)"
        color = "info"
        verdict = (
            "**This isn't one of the four fame/sound categories** -- fame wasn't determined "
            "for this artist, so calling it 'low-fame' or 'high-fame' would be a guess, not "
            "a finding. What's still solid is the sound score below: it's based only on this "
            "track's own audio structure and doesn't depend on fame either way. For the full "
            "quadrant verdict, add a Last.fm API key above or use the fame slider to "
            "describe the artist yourself, then re-run."
        )
    elif not high_fame and high_sound:
        quadrant = "Pure Sound Hit candidate"
        color = "success"
        verdict = (
            "**Not fame-biased.** This artist doesn't have a large existing audience yet, "
            "but the song's own structure closely matches the pattern of tracks that beat "
            "their artist's own baseline in the training data. If this song does well, the "
            "evidence points to the *song itself*, not pre-existing fame."
        )
    elif high_fame and high_sound:
        quadrant = "Mega-Hit candidate"
        color = "success"
        verdict = (
            "**Some fame bias likely, but the song holds up on its own too.** This artist "
            "already has a real audience, which will help regardless -- but the song's "
            "structure *also* matches the pattern of genuinely overperforming tracks, so "
            "this isn't just riding on the artist's name."
        )
    elif high_fame and not high_sound:
        quadrant = "Coattail Ride risk"
        color = "warning"
        verdict = (
            "**Likely fame-biased.** This artist already has a substantial following, so "
            "the song may still do fine in raw numbers -- but its own structure does *not* "
            "closely resemble tracks that outperformed expectations. If it succeeds, that "
            "success would probably trace mainly to the artist's existing fame, not this "
            "song's own structure."
        )
    else:
        quadrant = "Genre Flop risk"
        color = "warning"
        verdict = (
            "**Not fame-biased, but not favored either.** This artist doesn't have a large "
            "audience yet, and the song's structure doesn't closely match the pattern of "
            "tracks that beat expectations. On the current evidence this specific track "
            "looks like an uphill climb -- see the reasoning below for what's working against it."
        )

    getattr(st, color)(f"**{quadrant}** -- {fame_label}")
    st.markdown(verdict)
    st.metric("Sound score -- matches the pattern of tracks that outperformed", f"{sound_prob*100:.0f}%")
    st.caption(
        "This is NOT literally \"the odds this exact song beats a baseline this artist "
        "doesn't have yet\" -- for a new or unreleased artist there's no history to beat. "
        "It means: of the 127 training tracks, this is how closely this song's *audio "
        "structure* resembles the ones whose numbers beat their own artist's typical release."
    )

    # ---- Reasoning: the actual top features the model leans on, compared to what "beat the baseline" tracks look like ----
    st.markdown("### Why -- the features driving this")
    st.caption(
        "Ranked by how much weight the trained model actually puts on each feature "
        "(not just the 3 the earlier prototype checked)."
    )
    importances = meta["feature_importances"]
    plain_feats_ranked = sorted(
        [f for f in importances if f != "Key_Encoded"], key=lambda f: -importances[f]
    )[:6]

    for feat in plain_feats_ranked:
        val = feats.get(feat)
        ref = reference[feat]
        if val is None:
            continue

        # A checkmark/warning used to be shown for every one of these, even when
        # "successful" tracks and the dataset as a whole land on basically
        # the same number (Danceability's two medians are identical to 3
        # decimal places, for instance) -- on a 127-song dataset that gap is
        # noise, not a real pattern, so the icon was reading as confident
        # evidence when there wasn't any. Now a feature only gets a
        # directional icon when the gap between the two medians is at least
        # 15% of the successful tracks' own spread (P75-P25); otherwise it's
        # shown as neutral instead of a fabricated positive or negative icon.
        iqr = ref["successful_p75"] - ref["successful_p25"]
        gap = ref["successful_median"] - ref["all_median"]
        has_signal = iqr > 0 and abs(gap) >= 0.15 * iqr

        if has_signal:
            direction_good = ref["direction"] == "higher_is_typical_of_success"
            above = val >= ref["successful_median"]
            aligned = above == direction_good
            icon = "[+]" if aligned else "[!]"
            note = (
                f"vs **{ref['successful_median']:.3g}** typical for tracks that beat "
                "their artist's baseline"
            )
        else:
            icon = "[-]"
            note = "close to typical either way in this dataset -- not a strong individual signal"

        st.markdown(
            f"{icon} **{feat}**: your track is **{val:.3g}**, {note} "
            f"(model weight {importances[feat]*100:.0f}%)."
        )

    with st.expander("Full extracted feature profile"):
        st.json({k: v for k, v in feats.items() if not k.startswith("_")})

st.markdown("---")
st.caption(
    "Sound score comes from a Random Forest trained only on audio features (no fame signal). "
    "Fame comes from Last.fm or your own estimate, kept completely separate, then combined at "
    "the end -- the same separation of \"song\" from \"artist\" the whole research project is about."
)
