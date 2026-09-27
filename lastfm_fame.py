"""
lastfm_fame.py
================

Looks up an artist's real, current fame signal on Last.fm: how many unique
listeners they have logged there. This replaces the earlier Spotify-based
lookup -- as of Spotify's February 2026 Web API changes, a Developer app's
owner now needs an ACTIVE PREMIUM SUBSCRIPTION just to use the search
endpoint in Development Mode (see
https://developer.spotify.com/documentation/web-api/tutorials/february-2026-migration-guide),
which isn't something every school-project user has.

Last.fm's API has no such requirement: it's free forever, no payment tier,
no premium account needed, and the artist.search / artist.getInfo endpoints
work with just a personal API key from https://www.last.fm/api/account/create
(instant, no app review, no waiting).

The idea is unchanged from the original research: separate "how famous is
this artist right now" from "how good is this specific song," so the app can
tell the two apart instead of one masquerading as the other. Last.fm's
"listeners" count (unique people who've scrobbled/played the artist) plays
the same role Spotify's 0-100 popularity score played before -- it's just a
different, free data source for the same idea.

If no API key is configured, get_artist_fame() and search_artists() return
an error and the app falls back to asking the user to describe the artist's
rough fame level themselves -- exactly like before.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import requests

API_URL = "https://ws.audioscrobbler.com/2.0/"


@dataclass
class ArtistFame:
    name: str
    listeners: int           # unique Last.fm listeners -- this source's fame signal
    playcount: Optional[int]  # total scrobbles, when available (search results don't include it)
    url: str
    tags: list = field(default_factory=list)
    fame_tier: str = "unknown"  # "emerging" | "rising" | "established" | "major"


def _fame_tier(listeners: int) -> str:
    # Judgment-call cut points, not a Last.fm standard -- tuned so a genuinely
    # unsigned/independent artist typically lands in "emerging" (a few
    # thousand listeners or fewer) and a real mainstream act lands in
    # "established" or "major" (hundreds of thousands to tens of millions).
    # Last.fm listener counts span a much wider range than Spotify's 0-100
    # popularity score, so the cuts are log-scale, not linear.
    if listeners < 5_000:
        return "emerging"
    if listeners < 50_000:
        return "rising"
    if listeners < 500_000:
        return "established"
    return "major"


def _request(params: dict) -> tuple[Optional[dict], Optional[str]]:
    """Shared GET + error handling. Last.fm returns HTTP 200 even on API
    errors, with an {"error": <code>, "message": "..."} body, so status_code
    alone can't be trusted -- the body has to be checked either way."""
    try:
        resp = requests.get(API_URL, params={**params, "format": "json"}, timeout=10)
    except requests.RequestException as e:
        return None, f"Couldn't reach Last.fm's API: {e}"

    try:
        data = resp.json()
    except Exception:
        return None, f"Last.fm returned something unexpected ({resp.status_code}): {resp.text[:200]}"

    if "error" in data:
        return None, f"Last.fm rejected the request: {data.get('message', data['error'])}"
    if resp.status_code != 200:
        return None, f"Last.fm request failed ({resp.status_code}): {resp.text[:200]}"
    return data, None


def search_artists(query: str, api_key: str, limit: int = 8) -> tuple[list[ArtistFame], Optional[str]]:
    """Search Last.fm for artists matching `query`, for a picker UI -- lets
    the user pick the exact artist/band they mean, the same way the earlier
    Spotify dropdown did, just backed by a free data source instead of one
    that now requires the app owner to have Spotify Premium."""
    if not api_key:
        return [], "No Last.fm API key entered"
    if not query.strip():
        return [], "Type something to search for"

    data, err = _request({
        "method": "artist.search", "artist": query, "api_key": api_key, "limit": limit,
    })
    if err:
        return [], err

    items = data.get("results", {}).get("artistmatches", {}).get("artist", [])
    if isinstance(items, dict):  # Last.fm returns a bare dict, not a list, for a single match
        items = [items]
    if not items:
        return [], f"Last.fm has no artist matching '{query}' -- check spelling/spacing"

    results = []
    for item in items:
        try:
            listeners = int(item.get("listeners", 0))
        except (TypeError, ValueError):
            listeners = 0
        results.append(ArtistFame(
            name=item.get("name", query),
            listeners=listeners,
            playcount=None,  # not included in search results
            url=item.get("url", ""),
            fame_tier=_fame_tier(listeners),
        ))
    return results, None


def get_artist_fame(artist_name: str, api_key: str) -> tuple[Optional[ArtistFame], Optional[str]]:
    """Look up one artist by (near-exact) name -- the fallback path for when
    a dropdown pick isn't available. Returns (ArtistFame, None) on success,
    or (None, error_message) on any failure, same contract as before: the
    caller shows error_message instead of silently guessing."""
    if not api_key:
        return None, "No Last.fm API key entered"
    if not artist_name.strip():
        return None, "No artist name entered"

    data, err = _request({"method": "artist.getinfo", "artist": artist_name, "api_key": api_key})
    if err:
        return None, err

    artist = data.get("artist")
    if not artist:
        return None, f"Last.fm has no artist matching '{artist_name}' -- check spelling/spacing"

    stats = artist.get("stats", {})
    try:
        listeners = int(stats.get("listeners", 0))
    except (TypeError, ValueError):
        listeners = 0
    try:
        playcount = int(stats.get("playcount", 0))
    except (TypeError, ValueError):
        playcount = None

    tags = [t["name"] for t in artist.get("tags", {}).get("tag", [])][:3]

    return ArtistFame(
        name=artist.get("name", artist_name),
        listeners=listeners,
        playcount=playcount,
        url=artist.get("url", ""),
        tags=tags,
        fame_tier=_fame_tier(listeners),
    ), None
