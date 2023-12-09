"""
This file was created by referencing the following:
    musicbrainz api documentation: <https://musicbrainz.org/doc/MusicBrainz_API>
    YouTubeBrainz: <https://github.com/Flowrey/youtube-bz>
"""

import json
import urllib.parse
import urllib.request
from typing import Any, TypedDict, cast


class ArtistCredit(TypedDict):
    """A MusicBrainz ArtistCredit."""

    name: str


Recording = TypedDict(
    "Recording", {"artist-credit": list[ArtistCredit], "title": str,
                  "length": int} # "isrcs":list[str]
)



class Client:
    """MusicBrainz API client."""

    _base: str

    def __init__(self, base: str = "https://musicbrainz.org"):
        """Create a new MusicBrainz client."""
        self._base = base

    def _lookup(self, entity_type: str, mbid: str) -> dict[str, Any]:
        url = urllib.parse.urljoin(self._base, f"/ws/2/{entity_type}/{mbid}")
        url += "?inc=artist-credits+isrcs+releases&fmt=json"
        with urllib.request.urlopen(url) as response:
            html = response.read()
            data = json.loads(html)
        return data

    def lookup_recording(self, mbid: str) -> Recording:
        """Lookup for a release with it's MBID."""
        return cast(Recording, self._lookup("recording", mbid))
