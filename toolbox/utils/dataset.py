"""Create custom dataset to be used with PyTorch."""
import os
import json
import pandas as pd
from torch.utils.data import Dataset
from toolbox.utils import downloader

class JAAHDataset(Dataset):
    """JAAH dataset for PyTorch
    
    Parameters
    ==========
    data_home: str
        path to the JAAH dataset
        
    Attributes
    ==========
    data_home: str
        path to the JAAH dataset
    audio_path: str
        path to the audio/ directory of the JAAH dataset
    anno_path: str
        path to the annotations/ directory of the JAAH dataset
    track_list: dict
        dictionary of track_id: (audio_path, anno_path)
        If audio file is not found, it is not included in the track_list.
    download_info: path to download_info.json
    """
    def __init__(self, data_home, track_list_as_df=True):
        """Initialise JAAHDataset
        
        Parameters
        ==========
        data_home: str
            path to the JAAH dataset
        track_list_as_df: bool, default=True
            if True, track_list is returned as a pandas DataFrame
            otherwise, track_list is returned as a dictionary

        Attributes
        ==========
        self.download_info: path to download_info.json
            Contains information about the downloaded audio files and their
            source videos.
        self.track_list: DataFrame or dict
            DataFrame or dict of track_id: (title, audio, lab, anno)
            If audio file is not found, it is not included in the track_list.
        """

        self.data_home = data_home # ./data/JAAH/
        self.audio_path = os.path.join(data_home, 'audio')
        self.anno_path = os.path.join(data_home, 'annotations')
        self.labs_path = os.path.join(data_home, 'labs')

        self.download_info = self.get_download_info()
        self.track_list = self.get_track_list(track_list_as_df)

    def __len__(self):
        return len(self.track_list)


    def __getitem__(self, track_id):
        return self.track_list[track_id]

    def get_mbids(self, which:list[str]='all'):
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
            with open(os.path.join(self.anno_path, title)) as f:
                data = json.load(f)
                mbids[title.split('.')[0]] = data['mbid']

        return mbids


    def get_track_list(self, as_df):
        """return a dictionary of track_id: (title, audio, lab, anno)
        If audio file is not found, it is not included in the track_list.
        """
        track_list = {}
        track_count = 0 # as id / index
        for track in os.listdir(self.audio_path):
            track_list[track_count] = (track.split('.')[0],
                                       os.path.join(self.audio_path, track),
                                       os.path.join(self.labs_path, track.split('.')[0]+'.lab'),
                                       os.path.join(self.anno_path, track.split('.')[0]+'.json'))
            track_count += 1

        if as_df:
            return pd.DataFrame.from_dict(track_list, orient='index', columns=['title','audio','lab','anno'])
        
        return track_list
    

    def get_download_info(self):
        """return a download_info.json path if it exists, otherwise return None"""
        download_info_path = os.path.join(self.audio_path, 'download_info.json')
        if os.path.exists(download_info_path):
            return download_info_path
        else:
            return None


    def download(self, which:list[str]='all', suffix:str='.mp4'):
        """Download audio files into audio/ directory from YouTube.
        Download information is saved in the 'audio/download_info.json' file.
        
        Parameters
        ==========
        which: list[str], default='all'
            list of titles of the songs to retrieve mbid for. Must be given as
            the name of the json file (i.e. snake_case), rather than one given by "title" of the json file.
            If 'all', download all songs in the JAAH dataset.
        """
        mbids = self.get_mbids(which)

        download_info = {} # download_info.json file

        for title, mbid in mbids.items():
            filename = title + suffix
            download_info[title] = downloader.download(mbid, filename, self.audio_path)

        with open(os.path.join(self.audio_path, 'download_info.json'), 'w') as f:
            json.dump(download_info, f, indent=4)
        
        self.download_info = os.path.join(self.audio_path, 'download_info.json')
