""""""

def read_lab(file_path):
    """Reads a .lab file and returns a list of tuples (start_time, end_time, label)
    
    Parameters
    ==========
    file_path: str
        path to the .lab file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    lines = [line.strip().split('\t') for line in lines]
    lines = [(float(line[0]), float(line[1]), line[2]) for line in lines]
    
    return lines