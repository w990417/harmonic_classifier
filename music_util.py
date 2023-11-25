"""This file is for defining classes and functions related to musical data"""
import numpy as np

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


def to_Chord(start_time, end_time, notation, drop_extended=True):
    """Factory function for Chord class.
    Returns None if notation is 'N'
    Otherwise, returns a Chord object

    TODO: MIGHT ADD CHORD CONSTRUCTION WITHOUT START_TIME AND END_TIME
    """
    return None if notation == 'N' else Chord(start_time, end_time, notation, drop_extended)


class Chord(object):
    """
    Attributes
    ==========
    notation: str
        Chord notation as read from the .lab file
    components: list[str]
        List of notes (components) of the chord. (e.g. ['C', '3', '5', '7'] to represent a Cmaj7 chord)
        The first element (root) is in note name format (e.g. 'C', 'C#', 'Db').
        The rest of the elements are in degree format (e.g. '3', 'b3', '#5').
    root, third, fifth, seventh: str or None
        The root, third, fifth, seventh note of the chord.
        If the chord does not have the corresponding note, the attribute is None.
    extended: NOT IMPLEMENTED YET (drop_extended is always True at the moment)
        Set to None at the moment.
    """

    def __init__(self, start_time, end_time, notation, drop_extended=True):
        self.notation = notation
        self.components = self._get_components(notation, drop_extended)
        self.frame = self._get_frame(self.components)
        self.start_time = start_time
        self.end_time = end_time

    def describe(self, only_frame=False):
        """Prints the chord's notation, components and frame"""
        
        if not only_frame:
            print('Notation:', self.notation)
            print('-'*22)
            print('Components:', self.components)
            print('-'*22)
            print('Duration:', self.start_time, '~', self.end_time)
            print('-'*22)
        print_frame(self.frame)


    def _get_frame(self, components):
        """Returns a stack of 13-dim vectors representing the chord frame
        as an numpy.ndarray.
        
                0   1   2   3   4   5   6   7   8   9   10  11  12
                C   C#  D   D#  E   F   F#  G   G#  A   A#  B   None
        =================================================================
        root
        -----------------------------------------------------------------
        third
        -----------------------------------------------------------------
        fifth
        -----------------------------------------------------------------
        seventh
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
        """
        frame = np.zeros((4, 13))
        for i, comp in enumerate(components):
            if i == 0: # root in note format
                frame[i, note_to_idx[comp]] = 1.0
            else: # rest of the components in degree format
                frame[i, degree_to_idx[comp]] = 1.0
        
        return frame


    def _get_components(self, lab_notation, drop_extended=True):
        """Returns a list of notes (components) from lab file's chord notation
        
        Returns
        =======
        [root, third, fitfh, seventh] if drop_extended is True
        [root, third, fitfh, seventh, extended] if drop_extended is False
        """

        root, rest = lab_notation.split(':')
        if rest[0] == '(':
            components = [root] + rest[1:-1].split(',')
            return components[:-1] if drop_extended else components
        else:
            components = [root] + chord_to_degree[rest]
            return components if drop_extended else components + [None]



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
