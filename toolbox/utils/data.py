"""Create custom dataset to be used with PyTorch."""
import os
import json
import numpy as np
import torch
import torchaudio
import librosa
import pandas as pd
from torch.utils.data import Dataset

from toolbox.utils import downloader


chord_to_degree = {'maj': ['3', '5', None],
                    'min': ['b3', '5', None],
                    'dim': ['b3', 'b5', None],
                    'aug': ['3', '#5', None],
                    'maj7': ['3', '5', '7'],
                    'min7': ['b3', '5', 'b7'],
                    '7': ['3', '5', 'b7'],
                    'dim7': ['b3', 'b5', 'bb7'],
                    'hdim7': ['b3', 'b5', 'b7'],
                    'minmaj7': ['b3', '5', '7'],
                    'maj6': ['3', '5', '6'],
                    'min6': ['b3', '5', '6'],
                    'sus2': ['2', '5', None],
                    'sus4': ['4', '5', None]}

# used for the chord degrees
degree_to_idx = {'1': 0,
                 'b2': 1,
                 '2': 2,
                 '#2': 3, 'b3': 3,
                 '3': 4,
                 '#3': 5, '4': 5,
                 '#4': 6, 'b5': 6,
                 '5': 7,
                 '#5': 8, 'b6': 8,
                 '6': 9, 'bb7': 9,
                 '#6': 10, 'b7': 10,
                 '7': 11,
                 None: 12}

# used for the root note
note_to_idx = {'C': 0,
                'C#': 1, 'Db': 1,
                'D': 2,
                'D#': 3, 'Eb': 3,
                'E': 4,
                'F': 5,
                'F#': 6, 'Gb': 6,
                'G': 7,
                'G#': 8, 'Ab': 8,
                'A': 9,
                'A#': 10, 'Bb': 10,
                'B': 11}

class JAAHDataset(Dataset):
    """JAAH dataset for PyTorch.

        Parameters
        ==========
        data_home: str
            path to the JAAH dataset root directory
        drop_N: bool
            If True, drop 'N' notations from the lab file.
            
        Attributes
        ==========
        data_home: str
            path to the JAAH dataset
        audio_path: str
            path to the audio/ directory of the JAAH dataset
        anno_path: str
            path to the annotations/ directory of the JAAH dataset
        track_list: list[str]
            list of track_id for audio files available in audio_path
        drop_N: bool
            If True, drop 'N' notations from the lab file.
        sr: int (default=None)
            Sample rate to be used if the audio file is to be resampled.
            If None, use the sample rate of the audio file.
        to_mono: bool (default=True)
            If True, convert the audio file to mono.
            IT IS ALWAYS TRUE FOR NOW.
        audio_format: str (default='.mp4')
            audio file format to be used.
        
        Returns
        =======
        __getitem__() returns a list of Segment objects for the given track_id.
            
        """
    def __init__(self, data_home, drop_N:bool, use_librosa:bool=True,
                 sr:int=None, audio_format:str='.mp4'):

        self.data_home = data_home # ./data/JAAH/
        self.audio_path = os.path.join(data_home, 'audio')
        self.anno_path = os.path.join(data_home, 'annotations')
        self.labs_path = os.path.join(data_home, 'labs')

        self.drop_N = drop_N
        self.use_librosa = use_librosa
        self.sr = sr
        self.audio_format = audio_format
        
        self.track_list = self._get_track_list()

    def __len__(self):
        return len(self.track_list)


    def __getitem__(self, track_id):
        return self._segment_track(track_id)

    
    def _load_audio_torch(self, track_id:str):
        
        # load audio file
        audio_path = os.path.join(self.audio_path, track_id + self.audio_format)
        waveform, sample_rate = torchaudio.load(audio_path)

        # resample if necessary
        if self.sr is not None:
            waveform = torchaudio.transforms.Resample(sample_rate, self.sr)(waveform)
        else:
            self.sr = sample_rate

        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def _load_audio_librosa(self, track_id:str):
        audio_path = os.path.join(self.audio_path, track_id + self.audio_format)
        waveform, sample_rate = librosa.load(audio_path, sr=self.sr)
        self.sr = sample_rate
        return waveform
        

    def _segment_track(self, track_id:str):
        """Returns a list of Segment objects for the given track_id.
        Each audio segment will correspond to a line in the lab file.

        Parameters
        ==========
        track_id: str
            track_id of the audio file to segment

        drop_N: bool
            If True, drop 'N' notations from the lab file.
            If False,
        """
        segments = []
        # read lab file
        lines = self._read_lab(track_id)

        # load audio file
        if self.use_librosa:
            waveform = self._load_audio_librosa(track_id) # np.ndarray
        else:
            waveform = self._load_audio_torch(track_id) # torch.tensor

        # segment audio file
        for i, line in enumerate(lines):
            start_time, end_time, notation = line
            start_frame = int(start_time * self.sr)
            end_frame = int(end_time * self.sr)
            seg_audio = waveform[start_frame:end_frame] # didn't account for torch.tensor yet
            chord_frame = self._components_to_frame(self._notation_to_components(notation))
            segments.append(Segment(track_id, i, start_time, end_time, seg_audio, chord_frame))
        
        return segments


    def _components_to_frame(self, components:list[str]|None):
        """Returns a stack of 13-dim vectors representing the chord frame
        as an numpy.ndarray.
        
                0   1   2   3   4   5   6   7   8   9   10  11  12
                C   C#  D   D#  E   F   F#  G   G#  A   A#  B   None
        =================================================================
        0 root
        -----------------------------------------------------------------
        1 third
        -----------------------------------------------------------------
        2 fifth
        -----------------------------------------------------------------
        3 seventh
        =================================================================

        Float values represent the probability of the note being present and
        functioning as the corresponding chord degree.
            For example, if the 'third' vector is [0.1, 0.45, 0.01, ...],
            there is a .10, .45, .01, ... probability that C, C#, D, ... is 
            present and functioning as the third of the chord, respectively.

        The last column of each vector is reserved for the probability that
        the corresponding chord degree is not present.
            For example, a row vector with similarly low values for all the
            elements may represent uncertainty in the chord label, whereas
            a row vector with low values for all the elements except the last
            may indicate the absence of the corresponding chord degree.

        Parameters
        ==========
        components: list[str]|None
            List of notes (components) in the chord.
            root is in note format, rest of the components are in degree format.
            If None, returns a frame with the last column set to 1.0.
        """
        frame = np.zeros((4, 13))

        if components is None:  # 'N' notation
            frame[:, -1] = 1.0
            return frame
            
        for i, comp in enumerate(components):
            if i == 0: # root in note format
                frame[i, note_to_idx[comp]] = 1.0
            else: # rest of the components in degree format
                frame[i, degree_to_idx[comp]] = 1.0
        
        return frame


    def _notation_to_components(self, notation:str):
        """Returns a list of notes (components) from .lab notation
        
        Returns
        =======
        [root, third, fitfh, seventh] if drop_extended is True
        [root, third, fitfh, seventh, extended] if drop_extended is False
        None if notation is 'N'
        """
        if notation == 'N':
            return None
        
        root, rest = notation.split(':')
        if rest[0] == '(':
            components = [root] + rest[1:-1].split(',')
            return components[:-1]
        
        components = [root] + chord_to_degree[rest]
        return components

        

    def _read_lab(self, track_id:str):
        """Reads a .lab file and returns a list of tuples (start_time, end_time, notation)
        for each chord in the lab file."""
        
        lab_path = os.path.join(self.labs_path, track_id + '.lab')
        with open(lab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        lines = [line.strip().split('\t') for line in lines]
        lines = [(float(line[0]), float(line[1]), line[2]) for line in lines]
        
        return lines


    def _get_track_list(self):
        """Returns a list of track_id for audio files available in audio_path"""
        track_list = [f.split('.')[0] for f in os.listdir(self.audio_path) if f.endswith(self.audio_format)]
        
        assert 'download_info' not in track_list, "download_info.json file was included in the track_list."

        return track_list
    
    def _get_mbids(self, which:list[str]='all'):
        """Returns {title: mbid} from .json annotation files included in the JAAH dataset
        
        Parameters
        ==========
        which: list[str], default='all'
            list of titles of the songs to retrieve mbid for. Must be given as
            the name of the json file (i.e. snake_case), rather than one given by "title" of the json file.
            If 'all', return mbids for all songs in the JAAH dataset.
        """
        mbids = {}
        
        if which == 'all':
            which = os.listdir(self.anno_path)
        else:
            which = [t + '.json' for t in which]

        for title in which:
            with open(os.path.join(self.anno_path, title), encoding='utf-8') as f:
                data = json.load(f)
                mbids[title.split('.')[0]] = data['mbid']

        return mbids

    def download(self, which:list[str]='all', suffix:str='.mp4', save_download_info:str=None):
        """Download audio files into audio/ directory from YouTube.
        Download information is saved in the 'audio/download_info.json' file.
        
        Parameters
        ==========
        which: list[str], default='all'
            List of titles of the songs to retrieve mbid for given as 
            the name of the json file (i.e. snake_case).
            If 'all', download all songs in the JAAH dataset.
        suffix: str (default='.mp4')
            suffix to be added to the audio file name
        """
        mbids = self._get_mbids(which)

        download_info = {} # download_info.json file

        for title, mbid in mbids.items():
            filename = title + suffix
            download_info[title] = downloader.download(mbid, filename, self.audio_path)

        if save_download_info is not None:
            with open(os.path.join(self.audio_path, 'download_info.json'), 'w', encoding='utf-8') as f:
                json.dump(download_info, f, indent=4)


class Segment:
    """Segment of a track with relavant information.
    
    If the JAAHDataset was initialised with use_librosa=True, the audio attribute
    will be a numpy.ndarray. Otherwise, it will be a torch.tensor.
    """

    def __init__(self, track_id:str, seg_id:int, start_time:float, end_time:float,
                 seg_audio, chord_frame):
        self.track_id = track_id
        self.seg_id = seg_id
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.audio = seg_audio
        self.chord_frame = chord_frame


    def _as_df(self):
        """Returns segment.chord_frame as a pandas.DataFrame."""

        df = pd.DataFrame(self.chord_frame, columns=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',
                                        'G', 'G#', 'A', 'A#', 'B', 'None'])
        df.index = ['root', '3rd', '5th', '7th']
        return df.style.background_gradient(cmap="YlGnBu", axis=1, low=0.0, high=1.0)
