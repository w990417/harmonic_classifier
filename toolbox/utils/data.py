"""Create custom dataset to be used with PyTorch."""
import os
import random
import re
import json
import numpy as np
import torch
import librosa
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import Sampler

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


note_to_idx = {'Cb': 11, 'C': 0, 'C#': 1,
                'Db': 1, 'D': 2, 'D#': 3,
                'Eb': 3, 'E': 4,
                'F': 5, 'F#': 6,
                'Gb': 6, 'G': 7, 'G#': 8,
                'Ab': 8, 'A': 9, 'A#': 10,
                'Bb': 10, 'B': 11,
                'N': 12}


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
    def __init__(self, data_home, drop_N:bool, sr:int=44100):

        self.data_home = data_home # ./data/JAAH/
        self.audio_dir = os.path.join(data_home, 'audio')
        self.anno_dir = os.path.join(data_home, 'annotations')
        self.labs_dir = os.path.join(data_home, 'labs')
        self.spec_dir = os.path.join(data_home, 'spectrograms')
        self.cf_dir = os.path.join(data_home, 'chord_frames')
        self.sr = sr
        self.drop_N = drop_N
        self.spec_data = None
        self.cf_data = None
        
        self.track_list = self._get_track_list()

    def __len__(self):
        if self.spec_data is None:
            self.load_data()
        return len(self.spec_data)

        
    def __getitem__(self, index): 
        if isinstance(index, int):
            return self.spec_data[index], self.cf_data[index]
        
        # index is a list of indices
        return self.spec_data[index], self.cf_data[index]

    def load_data(self):
        for track_id in self.track_list:
            spec_path = os.path.join(self.spec_dir, f'{track_id}_spec.pt')
            cf_path = os.path.join(self.cf_dir, f'{track_id}_cf.pt')

            spec_batch = torch.load(spec_path)
            cf_batch = torch.load(cf_path)

            if self.spec_data is None:
                self.spec_data = spec_batch
                self.cf_data = cf_batch
            else:
                self.spec_data = torch.cat((self.spec_data, spec_batch))
                self.cf_data = torch.cat((self.cf_data, cf_batch))




    def _load_audio(self, track_id:str):
        audio_path = os.path.join(self.audio_dir, track_id + '.mp4')
        waveform, sample_rate = librosa.load(audio_path, sr=self.sr)
        self.sr = sample_rate
        return waveform


    def segment_track(self, track_id:str|int, savez:bool, spec_h:int=84, spec_w:int=168):
        """Segment an audio track into spectrograms and chord-frames.

        Parameters
        ==========
        track_id: str|int
            track_id of the audio file to segment.
            If int, track_id will be self.track_list[track_id].
        savez: bool
            If True, save the spectrogram and chord_frame tensors as .pt files.
        spec_h, spec_w: int
            target height and width of the spectrogram.
        """
        if isinstance(track_id, int):
            track_id = self.track_list[track_id]

        # read lab file
        lines = self._read_lab(track_id)

        # load audio file
        audio_path = os.path.join(self.audio_dir, track_id + '.mp4')
        waveform, _ = librosa.load(audio_path, sr=self.sr) # waveform: np.ndarray

        # segment audio file
        spec_batch = torch.empty((len(lines), spec_h, spec_w))
        chord_batch = torch.empty((len(lines), 4, 13))

        for i, line in enumerate(lines):
            start_time, end_time, notation = line
            start_frame = int(start_time * self.sr)
            end_frame = int(end_time * self.sr)
            seg_audio = waveform[start_frame:end_frame]

            # get chord_frame for the segment
            seg_chord_frame = self.get_chord_frame(notation)
            if seg_chord_frame is None:
                continue # skip the invalid notation
            chord_batch[i] = torch.from_numpy(seg_chord_frame)

            # segment spectrogram
            spec_batch[i] = self._preprocess_audio(seg_audio, spec_h, spec_w)

        if savez:
            torch.save(spec_batch, os.path.join(self.spec_dir, f'{track_id}_spec.pt'))
            torch.save(chord_batch, os.path.join(self.cf_dir, f'{track_id}_cf.pt'))
        
        return spec_batch, chord_batch
        

    def _preprocess_audio(self, audio:np.ndarray, spec_h:int, spec_w:int) -> torch.tensor:
        """Prepares segmented audio for training by applying a number of transformations.

        Constant-Q transform, np.abs, amplitude_to_db is applied to the audio.
        The resulting spectrogram is converted to a torch.tensor and returned.


        Parameters
        ==========
        audio: np.ndarray
            Segmented audio
        spec_h, spec_w: int (default=84, 168)
            Target height and width of the spectrogram.
            See <Constant-Q Transform> for more details.


        Constant-Q Transform
        ====================
        n_bins: number of frequency bins in the spectrogram.
            n_bins will be set to spec_h (default=84)
        hop_length: number of samples between successive CQT columns.
            hop_length will be set so that the resulting spectrogram will have the width as
            close to spec_w (default=168) as possible.
        """

        # n_bins = spec_h
        # spec_w = ceil(len(audio) / hop_length)
        hop_length = int((len(audio) / (spec_w-1)))

        seg_spectrogram = librosa.cqt(audio, sr=self.sr, n_bins=spec_h, hop_length=hop_length)
        seg_spectrogram = np.abs(seg_spectrogram)
        seg_spectrogram = librosa.amplitude_to_db(seg_spectrogram, ref=np.max)

        # pad or crop spectrogram to have the width of `spec_w`
        if seg_spectrogram.shape[1] < spec_w: # pad
            n_pad = spec_w - seg_spectrogram.shape[1]
            seg_spectrogram = np.pad(seg_spectrogram, ((0,0),(0,n_pad)), 'edge')
        elif seg_spectrogram.shape[1] > spec_w: # crop
            seg_spectrogram = seg_spectrogram[:, :spec_w]

        seg_spectrogram = torch.from_numpy(seg_spectrogram)

        return seg_spectrogram
    
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


    def get_chord_frame(self, notation) -> np.ndarray:
        """Returns a chord_frame for the given notation.
        Returns None if the notation cannot be parsed into a chord_frame.

        Chord-Frame
        ===========
        chord_frame is a 4x13 matrix where each row vector represents the
        chord's ['root', 'third', 'fifth', 'seventh'] component/degree.

        The first 12 columns represent the 12 notes in the chromatic scale,
        and the last column represents 'absence' of any note functioning as
        the corresponding row's chord degree.

        For example,
        the chord_frame for Cmaj7 with the 'fifth' omitted (C, E, N, B) is:

                C    C#   D    D#   E    F    F#   G    G#   A    A#   B    N
        ======================================================================
        root   1.0
        ----------------------------------------------------------------------
        third                      1.0
        ----------------------------------------------------------------------
        fifth                                                              1.0
        ----------------------------------------------------------------------
        seventh                                                       1.0
        ======================================================================
        where the rest of the entries are 0.0.
        """
        frame = np.zeros((4,13))
        
        # single note (includes 'N')
        if len(re.split(r'[:/]', notation)) == 1:
            frame[0, note_to_idx[notation]] = 1
            return frame

        root = re.split(r'[:/]', notation)[0]
        rest = re.split(r'[:/]', notation)[1]

            

        if rest[0] == '(': # C:(3,5,b7,#9)
            frame[0, note_to_idx[root]] = 1.0
            rest = rest[1:-1]
            for this in rest.split(',')[:3]:
                if idx:= self.get_idx(this, root):
                    frame[idx] = 1.0
            return frame
        
        # Bb:min7 -> (root=Bb, rest=min7)
        # Bb:7 -> (root=Bb, rest=7) (i.e. 7 is treated as a chord name rather than a degree)
        if rest in chord_to_degree:
            frame[0, note_to_idx[root]] = 1.0
            rest = chord_to_degree[rest]
            for this in rest:
                if idx:= self.get_idx(this, root):
                    frame[idx] = 1.0
            return frame

        # C/G -> (root=C, rest=G)
        if rest in note_to_idx:
            frame[0, note_to_idx[root]] = 1.0
            frame[self.get_idx(rest, root)] = 1.0
            return frame

        # C/b5 -> (root=C, rest=b5)
        if rest in degree_to_interval:
            frame[0, note_to_idx[root]] = 1.0
            frame[self.get_idx(rest, root)] = 1.0
            return frame

        # B:min/7 -> (root=B, rest=min/7) -> (root=B, chord=min, inversion=7)
        if '/' in rest:
            try:
                chord, inversion = rest.split('/')
                frame[0, note_to_idx[root]] = 1.0
                chord = chord_to_degree[chord]
                for this in chord:
                    if idx:= self.get_idx(this, root):
                        frame[idx] = 1.0
                
                # inversion=7 not in chord=[3, 5] -> add 7 to chord_frame
                if inversion in note_to_idx and inversion not in rest:
                    frame[self.get_idx(inversion, root)] = 1.0

                return frame

            except ValueError:
                raise ValueError(f'get_chord_frame({notation}) error: rest={rest}')
        
        return None



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
        track_list = [f.split('.')[0] for f in os.listdir(self.audio_dir) if f.endswith('.mp4')]
        
        assert 'download_info' not in track_list, "download_info.json file was included in the track_list."

        return track_list
    
    def _read_anno(self, key:str, track_id:str|int='all'):
        """Returns {title: mbid} from .json annotation files included in the JAAH dataset
        
        Parameters
        ==========
        track_id: str|int, default='all'
            Track_id to read annotation file for.
            If 'all', read all .json files in the annotations/ directory.
        key: str from ['mbid', 'key', 'duration']
            Which relevant information to retrieve from the .json file.
        """
        if track_id == 'all':
            list_filename = [track_id + '.json' for track_id in self.track_list]
            data = [None] * len(self.track_list)

            for i, filename in enumerate(list_filename):
                with open(os.path.join(self.anno_dir, filename), encoding='utf-8') as f:
                    data[i] = json.load(f)[key]
            return data
                
        if isinstance(track_id, int):
            filename = self.track_list[track_id] + '.json'
        else:
            filename = track_id + '.json'

        with open(os.path.join(self.anno_dir, filename), encoding='utf-8') as f:
            data = json.load(f)[key]
            return data


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
        mbids = self._read_anno(which, 'mbid')

        download_info = {} # download_info.json file

        for title, mbid in mbids.items():
            filename = title + suffix
            download_info[title] = downloader.download(mbid, filename, self.audio_dir)

        if save_download_info is not None:
            with open(os.path.join(self.audio_dir, 'download_info.json'), 'w', encoding='utf-8') as f:
                json.dump(download_info, f, indent=4)


class JAAHSampler:
    def __init__(self, dataset, batch_size, shuffle, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # initialise indices
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        batch_start = 0
        while batch_start < len(self.indices):
            batch_end = min(batch_start + self.batch_size, len(self.indices))
            batch = self.indices[batch_start:batch_end]
            yield self.dataset.__getitem__(batch)
            batch_start = batch_end
            
        # If drop_last is False and there are remaining samples, yield the last batch
        if not self.drop_last and batch_start != len(self.indices):
            batch = self.indices[batch_start:]
            yield self.dataset.__getitem__(batch)

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size
        

def cf_to_df(chord_frame:np.ndarray, styled:bool=True):
    """Displays chord frame as pandas DataFrame"""
    df = pd.DataFrame(chord_frame, columns=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',
                                    'G', 'G#', 'A', 'A#', 'B', 'None'])
    df.index = ['root', '3rd', '5th', '7th']
    if styled:
        return df.style.background_gradient(cmap="YlGnBu", axis=1, low=0.0, high=1.0)
    else:
        return df