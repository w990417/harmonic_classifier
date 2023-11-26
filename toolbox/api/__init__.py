"""This subpackage contains modules that are used to call the MusicBrainz and YouTube APIs.

Modules
=======
mb_api
    A module for calling the MusicBrainz API.
yt_api
    A module for calling the YouTube API.

Usage
=====
>>> from toolbox.api import MusicBrainzAPI, YouTubeAPI
"""

from toolbox.api import mb_api as MusicBrainzAPI
from toolbox.api import yt_api as YouTubeAPI

__all__ = ['MusicBrainzAPI', 'YouTubeAPI']
