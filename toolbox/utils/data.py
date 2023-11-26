"""This file is for defining classes and functions related to musical data"""
import numpy as np
import pandas as pd

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


class ChordFrame:
    def __init__(self, lab_path:str, start_time:float, end_time:float, drop_N:bool, drop_extended:bool=True):
        self.lab_path = lab_path
        self.track_id = lab_path.split('/')[-1].split('.')[0]
        self.start_time = start_time
        self.end_time = end_time
        self.drop_N = drop_N
        self.drop_extended = drop_extended
        self.time_frame, self.chord_frame = self._get_frame()

    def _get_frame(self):
        lines = self._read_lab()
        time_frame = np.zeros((len(lines), 2))
        chord_frame = np.zeros((len(lines), 4 if self.drop_extended else 5, 13))
        for i, line in enumerate(lines):
            time_frame[i, :] = line[:2]
            chord_frame[i] = self._components_to_frame(self._notation_to_components(line[2]))
            
        return time_frame, chord_frame
    
    def __len__(self):
        return len(self.time_frame)


    def _read_lab(self):
        """Reads a .lab file and returns a list of tuples (start_time, end_time, notation)."""
        
        with open(self.lab_path, 'r') as f:
            lines = f.readlines()
        
        lines = [line.strip().split('\t') for line in lines]
        lines = [(float(line[0]), float(line[1]), line[2]) for line in lines]

        lines = [line for line in lines if line[0] >= self.start_time and line[1] <= self.end_time]
        if self.drop_N:
            lines = [line for line in lines if line[2] != 'N']
        
        return lines


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
            return components[:-1] if self.drop_extended else components

        components = [root] + chord_to_degree[rest]
        return components if self.drop_extended else components + [None]
        

    def _components_to_frame(self, components:list[str]|None, drop_extended:bool=True):
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
        -----------------------------------------------------------------
        extended (NOT IMPLEMENTED)
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
        num_components = 4 if drop_extended else 5
        frame = np.zeros((num_components, 13))

        if components is None:
            frame[:, -1] = 1.0
            return frame
            
        for i, comp in enumerate(components):
            if i == 0: # root in note format
                frame[i, note_to_idx[comp]] = 1.0
            else: # rest of the components in degree format
                frame[i, degree_to_idx[comp]] = 1.0
        
        return frame
    
    def as_df(self, index:int, color=True):
        """Render the chord-frame as a pd.DataFrame.
        
        Parameters
        ==========
        index: int
            index of the chord-frame to be rendered
        color: bool
            If True, use gradient color to represent the probability values.
        """
        df = pd.DataFrame(self.chord_frame[index], columns=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',
                                                            'G', 'G#', 'A', 'A#', 'B', 'None'])
        df.index = ['root', '3rd', '5th', '7th']

        if color:
            return df.style.background_gradient(cmap="YlGnBu", axis=1, low=0.0, high=1.0)

        return df


def notation_to_components(notation, drop_extended=True):
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
        return components[:-1] if drop_extended else components
    else:
        components = [root] + chord_to_degree[rest]
        return components if drop_extended else components + [None]


def components_to_frame(components:list[str]|None, drop_extended:bool=True):
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
    -----------------------------------------------------------------
    extended (NOT IMPLEMENTED)
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
    num_components = 4 if drop_extended else 5
    frame = np.zeros((num_components, 13))

    if components is None:
        frame[:, -1] = 1.0
        return frame
        
    for i, comp in enumerate(components):
        if i == 0: # root in note format
            frame[i, note_to_idx[comp]] = 1.0
        else: # rest of the components in degree format
            frame[i, degree_to_idx[comp]] = 1.0
    
    return frame


def print_frame(data):
    """Render the frame as a human-readable table
    
    Parameters
    ==========
    frame: numpy.ndarray or Chord
    """
    frame = data if isinstance(data, np.ndarray) else data.frame

    label = ['root', '3rd', '5th', '7th']
    print()
    print('\t C\t C#\t D\t D#\t E\t F\t F#\t G\t G#\t A\t A#\t B\tNone')
    print('='*100)
    for row in frame:
        print(f"{label.pop(0)}\t", end='')
        for val in row:
            print(f"{val:.2f}\t", end='')
        print()
    print('='*100)


def read_lab(lab_path, drop_N:bool, start_time=0.0, end_time=None):
    """Reads a .lab file and returns a list of tuples (start_time, end_time, notation)
    
    Parameters
    ==========
    lab_path: str
        path to the .lab file
    drop_N: bool
        If True, drops the 'N' notation from the output.
        Otherwise, 'N' notation is kept.
    start_time: float
        start time of the firt segment of .lab file to be read
    end_time: float
        end time of the last segment of .lab file to be read
    """
    with open(lab_path, 'r') as f:
        lines = f.readlines()
    
    lines = [line.strip().split('\t') for line in lines]
    lines = [(float(line[0]), float(line[1]), line[2]) for line in lines]
    if end_time is None:
        end_time = lines[-1][1]
    lines = [line for line in lines if line[0] >= start_time and line[1] <= end_time]
    if drop_N:
        lines = [line for line in lines if line[2] != 'N']
    
    return lines
