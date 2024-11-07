import os


def check_file_size(file):
    max_file_size = 5 * 1024 * 1024 * 1024
    if os.fstat(file.fileno()).st_size <= max_file_size:
        return
    else:
        raise IOError("file too large to read")
