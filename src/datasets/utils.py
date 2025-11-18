# Common utility functions for datasets

def load_file_list(path):
    # Load a list of file names from a text file.
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]