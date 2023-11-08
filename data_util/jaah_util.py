"""This module contains functions for downloading audio from YouTube given a MusicBrainz ID.

It is heavily inspired by the YouTubeBrainz project: <https://github.com/Flowrey/youtube-bz>,
and is modified under the GPL-3.0 License.

use:
    from data_util import mbid

    mbid.download('MBID', 'path/to/download')
"""

from dataclasses import dataclass
from typing import Any, Generator, Optional
from urllib.error import URLError

import os
import json
import pytube

from .api import mb_api as MusicBrainzAPI
from .api import yt_api as YouTubeAPI

DOWNLOAD_INFO = 'd:\\repos\\harmonic_classifier\\JAAH\\audio\\download_info.json'

@dataclass
class VideoMatch:
    title: str
    video_id: str
    levenshtein: int


def match_duration(video: VideoMatch, length: int) -> bool:
    """Check if the video duration matches up with the required length."""
    diff = abs(pytube.YouTube(f'http://youtube.com/watch?v={video.video_id}').length - (length/1000))
    return diff <= 5


def get_best_match(recording: MusicBrainzAPI.Recording) -> Optional[VideoMatch]:
    """Get YouTube videos corresponding to MusicBrainz tracks.
    
    Returns error message (str) if no video is found.
    Otherwise, returns VideoMatch object.
    """
    search_query = generate_search_query(recording)

    youtube_client = YouTubeAPI.Client()
    search_results = youtube_client.get_search_results(search_query) # returns html
    yt_initial_data = YouTubeAPI.get_initial_data(search_results) # returns dict

    contents = get_contents(yt_initial_data) # returns generator
    videos = get_video_distance(recording['title'], contents) # returns list of VideoMatch objects
    best_matches = sorted(videos, key=lambda d: d.levenshtein)

    if len(best_matches) == 0:
        return f'No video found for {recording["title"]}'

    for video in best_matches:
        if match_duration(video, recording['length']):
            return video # return the first video that matches the duration
    
    return f'No duration match for {recording["title"]}'


def get_contents(yt_initial_data: dict[str, Any]):
    contents = (
        itemSectionRenderer["videoRenderer"]
        for itemSectionRenderer in yt_initial_data["contents"][
            "twoColumnSearchResultsRenderer"
        ]["primaryContents"]["sectionListRenderer"]["contents"][0][
            "itemSectionRenderer"
        ][
            "contents"
        ]
        if "videoRenderer" in itemSectionRenderer
    )

    return contents


def get_video_distance(title: str, contents: Generator[Any, None, None]):
    videos = [
        VideoMatch(
            title=videoRenderer["title"]["runs"][0]["text"],
            video_id=videoRenderer["videoId"],
            levenshtein=levenshtein_distance(
                title.lower(), videoRenderer["title"]["runs"][0]["text"]
            ),
        )
        for videoRenderer in contents
    ]

    return videos


def download_video_audio(video_id: str, filename:str, destination: Optional[str] = None):
    if stream := pytube.YouTube(
        f"http://youtube.com/watch?v={video_id}",
    ).streams.get_audio_only():
        stream.download(output_path=destination, filename=filename)


def generate_search_query(recording: MusicBrainzAPI.Recording) -> str:
    """Generate a {artist name}{title}{"Auto-generated"} search query."""
    return f'"{recording["artist-credit"][0]["name"]}" "{recording["title"]}" "Auto-generated"'


def download_best_match(
    recording: MusicBrainzAPI.Recording,
    filename: str,
    destination: Optional[str],
):
    res = get_best_match(recording)
    if isinstance(res, str): # error message (str) or VideoMatch object
        return res
    
    download_video_audio(res.video_id, filename, destination)
    return f'Downloaded video: youtube.com/watch?v={res.video_id} as {filename}'


def download(mbid: str, filename: str, destination: Optional[str] = None):
    musicbrainz_client = MusicBrainzAPI.Client()

    try:
        recording = musicbrainz_client.lookup_recording(mbid)
    except URLError:
        return "Failed to get musicbrainz recording. Verify the MBID provided."

    result = download_best_match(recording, filename, destination)
    
    return result


def get_mbid(json_path:str ,title:list[str]='all'):
    """Returns mbid from .json annotation files included in the JAAH dataset
    
    Parameters
    ==========
    json_path: str
        path to the annotations/ directory of the JAAH dataset
    title: list[str]
        title of the songs to retrieve mbid from. If 'all', returns all mbids.
        title should match the name of the json file in the annotations/ directory
        rather than the "title" as listed in the json file.
    """

    mbids = []
    
    if title == 'all':
        title = os.listdir(json_path)
    else:
        title = [t + '.json' for t in title]

    for t in title:
        with open(os.path.join(json_path, t)) as f:
            data = json.load(f)
            mbids.append(data['mbid'])

    return mbids


def levenshtein_distance(first_word: str, second_word: str) -> int:
    """Implementation of the levenshtein distance in Python.
    This code came from: https://github.com/TheAlgorithms/Python.

    :param first_word: the first word to measure the difference.
    :param second_word: the second word to measure the difference.
    :return: the levenshtein distance between the two words.
    Examples:
    >>> levenshtein_distance("planet", "planetary")
    3
    >>> levenshtein_distance("", "test")
    4
    >>> levenshtein_distance("book", "back")
    2
    >>> levenshtein_distance("book", "book")
    0
    >>> levenshtein_distance("test", "")
    4
    >>> levenshtein_distance("", "")
    0
    >>> levenshtein_distance("orchestration", "container")
    10
    """
    # The longer word should come first
    if len(first_word) < len(second_word):
        return levenshtein_distance(second_word, first_word)

    if len(second_word) == 0:
        return len(first_word)

    previous_row = list(range(len(second_word) + 1))

    for i, c1 in enumerate(first_word):
        current_row = [i + 1]

        for j, c2 in enumerate(second_word):
            # Calculate insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)

            # Get the minimum to append to the current row
            current_row.append(min(insertions, deletions, substitutions))

        # Store the previous row
        previous_row = current_row

    # Returns the last element (distance)
    return previous_row[-1]