"""Create custom dataset to be used with PyTorch."""
import os
import re
import json
import numpy as np
import torch
import librosa
import pandas as pd
from torch.utils.data import Dataset

from toolbox.utils import downloader

# pylint: disable=trailing-whitespace

chord_to_degree = {'maj': ['3', '5'],
                    'min': ['b3', '5'],
                    'dim': ['b3', 'b5'],
                    'aug': ['3', '#5'],
                    'maj7': ['3', '5', '7'],
                    'min7': ['b3', '5', 'b7'],
                    '7': ['3', '5', 'b7'],
                    'dim7': ['b3', 'b5', 'bb7'],
                    'hdim7': ['b3', 'b5', 'b7'],
                    'minmaj7': ['b3', '5', '7'],
                    'maj6': ['3', '5', '6'],
                    'min6': ['b3', '5', '6'],
                    'sus2': ['2', '5'],
                    'sus4': ['4', '5'],
                    '9': ['3', '5', 'b7'], # 9th is not included
                    'maj9': ['3', '5', '7'], # 9th is not included
                    'min9': ['b3', '5', 'b7'],} # 9th is not included

# used for the chord degrees
degree_to_interval = {'1': 0,
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
# note_to_idx = {'C': 0,
#                 'C#': 1, 'Db': 1,
#                 'D': 2,
#                 'D#': 3, 'Eb': 3,
#                 'E': 4,
#                 'F': 5,
#                 'F#': 6, 'Gb': 6,
#                 'G': 7,
#                 'G#': 8, 'Ab': 8,
#                 'A': 9,
#                 'A#': 10, 'Bb': 10,
#                 'B': 11, 'Cb': 11,
#                 'N': 12}
note_to_idx = {'Cb': 11, 'C': 0, 'C#': 1,
                'Db': 1, 'D': 2, 'D#': 3,
                'Eb': 3, 'E': 4,
                'F': 5, 'F#': 6,
                'Gb': 6, 'G': 7, 'G#': 8,
                'Ab': 8, 'A': 9, 'A#': 10,
                'Bb': 10, 'B': 11,
                'N': 12}

# class Segment:
#     """Segment of a track with relavant information.
    
#     If the JAAHDataset was initialised with use_librosa=True, the audio attribute
#     will be a numpy.ndarray. Otherwise, it will be a torch.tensor.
#     """

#     def __init__(self, track_id:str, seg_id:int, start_time:float, end_time:float,
#                  audio, spectrogram, chord_frame):
#         self.track_id = track_id
#         self.seg_id = seg_id
#         self.start_time = start_time
#         self.end_time = end_time
#         self.duration = end_time - start_time
#         self.audio = audio
#         self.chord_frame = chord_frame
#         self.spectrogram = spectrogram


#     def _as_df(self):
#         """Returns segment.chord_frame as a pandas.DataFrame."""

#         df = pd.DataFrame(self.chord_frame, columns=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',
#                                         'G', 'G#', 'A', 'A#', 'B', 'None'])
#         df.index = ['root', '3rd', '5th', '7th']
#         return df.style.background_gradient(cmap="YlGnBu", axis=1, low=0.0, high=1.0)
    

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
    def __init__(self, data_home, drop_N:bool, sr:int=44100, audio_format:str='.mp4'):

        self.data_home = data_home # ./data/JAAH/
        self.audio_dir = os.path.join(data_home, 'audio')
        self.anno_dir = os.path.join(data_home, 'annotations')
        self.labs_dir = os.path.join(data_home, 'labs')
        self.segs_dir = os.path.join(data_home, 'segs')
        self.sr = sr
        self.drop_N = drop_N
        self.audio_format = audio_format
        
        self.track_list = self._get_track_list()

    def __len__(self):
        return len(self.track_list)


    def __getitem__(self, track_id:str|int, seg_id:list[int]=None):
        """Loads segments from .npz files and return them.
        
        Parameters
        ==========
        track_id: str|int
            Track_id of the audio, spectrogram, and chord_frame to be loaded.
            If int, track_id will be self.track_list[track_id].
        seg_id: list[int] (default=None)
            List of segment ids to be loaded. If None, all segments for the given
            track_id will be loaded.

        Returns
        =======
        -> list[(spectrogram, chord_frame)]
        """
        if isinstance(track_id, int):
            track_id = self.track_list[track_id]

        seg_path = os.path.join(self.segs_dir, track_id)

        if seg_id is None:
            seg_list = [os.path.join(seg_path, f) for f in os.listdir(seg_path) if f.endswith('.npz')]
        else:
            seg_list = [os.path.join(seg_path, f'{track_id}_{i}.npz') for i in seg_id]
        
        segments = []
        for seg in seg_list:
            with np.load(seg) as data:
                    segments.append((data['spec'], data['chord']))
        
        return segments


    def _load_audio(self, track_id:str):
        audio_path = os.path.join(self.audio_dir, track_id + self.audio_format)
        waveform, sample_rate = librosa.load(audio_path, sr=self.sr)
        self.sr = sample_rate
        return waveform


    def segment_track(self, track_id:str, savez:bool=True, spec_h:int=84, spec_w:int=168):
        """

        Parameters
        ==========
        track_id: str
            track_id of the audio file to segment
        spec_h, spec_w: int
            target height and width of the spectrogram
        """
        # read lab file
        lines = self._read_lab(track_id)

        # load audio file
        audio_path = os.path.join(self.audio_dir, track_id + self.audio_format)
        waveform, _ = librosa.load(audio_path, sr=self.sr)

        # segment audio file
        segments = []
        for i, line in enumerate(lines):
            start_time, end_time, notation = line
            start_frame = int(start_time * self.sr)
            end_frame = int(end_time * self.sr)
            seg_audio = waveform[start_frame:end_frame]
            seg_spectrogram = self._preprocess_audio(seg_audio, spec_h, spec_w)
            chord_frame = self.get_chord_frame(notation)
            segments.append((seg_spectrogram, chord_frame))

            if savez:
                savez_dir = os.path.join(self.segs_dir, track_id)
                os.makedirs(savez_dir, exist_ok=True)
                savez_path = os.path.join(savez_dir, f'{track_id}_{i}.npz')
                np.savez(savez_path, spec=seg_spectrogram, chord=chord_frame)
        
        return segments
        

    def _preprocess_audio(self, audio, spec_h:int, spec_w:int):
        """Prepares segmented audio for training by applying a number of transformations.

        Constant-Q transform, np.abs, amplitude_to_db is applied to the audio.
        The resulting spectrogram is converted to a torch.tensor and returned.

        Constant-Q Transform
        ====================
        n_bins is determined so that the resulting spectrogram will have the height of `spec_h`.
        hop_length is determined so that the resulting spectrogram will have the width of `spec_w`.
        """

        # n_bins = spec_h
        # spec_w = ceil(len(audio) / hop_length)
        hop_length = int((len(audio) / (spec_w-1)))

        cqt = librosa.cqt(audio, sr=self.sr, n_bins=spec_h, hop_length=hop_length)
        cqt = np.abs(cqt)
        cqt = librosa.amplitude_to_db(cqt, ref=np.max)
        cqt = torch.from_numpy(cqt)

        return cqt
    
    def get_idx(self, this, root):
        """Get the row, col index for 'this' in chord_frame when the root note is 'root'"""
        if this in degree_to_interval: # 1, b2 ...
            interval = degree_to_interval[this]
            row = note_to_idx[root] + interval
            row = row % 12
        elif this in note_to_idx: # C, C#, Db ...
            interval = np.abs(note_to_idx[root] - note_to_idx[this])
            row = note_to_idx[root]
        else: # extended degrees: 9, 11, 13 etc.
            return None

        if interval in [1, 2, 3, 4, 5]:
            col = 1 # third
        elif interval in [6, 7, 8]:
            col = 2 # fifth
        elif interval in [9, 10, 11]:
            col = 3 # seventh
        else:
            raise ValueError(f'get_idx({this},{root}) error: interval={interval}')
        return col, row


    def get_chord_frame(self, notation):
        """
        Cases:
        1. single note
        2. single note with bass (e.g. 'C/E')
        3. inversion/slash (e.g. 'G/b5')
        4. components (e.g. 'Eb:(b3,5,b7,11)')
        5. chord type (e.g. 'C:maj7')
        
        """
        frame = np.zeros((4,13))
        
        # single note (includes 'N')
        if len(re.split(r'[:/]', notation)) == 1:
            frame[0, note_to_idx[notation]] = 1
            return frame

        root = re.split(r'[:/]', notation)[0]
        rest = re.split(r'[:/]', notation)[1]
              
        if '/' in rest: # discard after slash
            rest = rest.split('/')[0]

        if rest[0] == '(':
            frame[0, note_to_idx[root]] = 1.0
            rest = rest[1:-1]
            for this in rest.split(',')[:3]:
                if idx:= self.get_idx(this, root):
                    frame[idx] = 1.0
            return frame
        
        if rest in chord_to_degree: # chord type
            frame[0, note_to_idx[root]] = 1.0
            rest = chord_to_degree[rest]
            for this in rest:
                if idx:= self.get_idx(this, root):
                    frame[idx] = 1.0
            return frame

        if rest in degree_to_interval:
            frame[0, note_to_idx[root]] = 1.0
            idx = self.get_idx(rest, root)
            frame[idx] = 1.0
            return frame

        raise ValueError(f'get_chord_frame({notation}) error: notation={notation}')



    def _read_lab(self, track_id:str):
        """Reads a .lab file and returns a list of tuples (start_time, end_time, notation)
        for each chord in the lab file."""
        
        lab_path = os.path.join(self.labs_dir, track_id + '.lab')
        with open(lab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        lines = [line.strip().split('\t') for line in lines]
        lines = [(float(line[0]), float(line[1]), line[2]) for line in lines]
        
        return lines


    def _get_track_list(self):
        """Returns a list of track_id for audio files available in audio_path"""
        track_list = [f.split('.')[0] for f in os.listdir(self.audio_dir) if f.endswith(self.audio_format)]
        
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
            which = os.listdir(self.anno_dir)
        else:
            which = [t + '.json' for t in which]

        for title in which:
            with open(os.path.join(self.anno_dir, title), encoding='utf-8') as f:
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
            download_info[title] = downloader.download(mbid, filename, self.audio_dir)

        if save_download_info is not None:
            with open(os.path.join(self.audio_dir, 'download_info.json'), 'w', encoding='utf-8') as f:
                json.dump(download_info, f, indent=4)


def prep_audio(audio:np.ndarray, sample_rate:int, hop_length=512):
    """Prepares segmented audio for training by applying a number of transformations.
    
    Constant-Q transform, np.abs, amplitude_to_db is applied to the audio.
    The resulting spectrogram is converted to a torch.tensor and returned.
    """

    cqt = librosa.cqt(audio, sr=sample_rate, hop_length=hop_length)
    cqt = np.abs(cqt)
    cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    cqt = torch.from_numpy(cqt)

    return cqt
