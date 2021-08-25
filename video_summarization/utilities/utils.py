"""Helper funbctions"""
import errno
import os
from shutil import move

def crawl_directory(directory: str) -> list:
    """
    Crawling DATA directory

    Args:
        directory (str): The directory path to crawl

    Returns
        tree (list): A list with all the filepaths

    """
    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            tree.append(os.path.join(subdir, _file))

    return tree


def is_wav(filename) -> bool:
    """
    Checks if file's extension is wav
    Parameters
    ----------
    filename: File name

    Returns
    -------

    """
    if filename.endswith('.wav'):
        return True
    else:
        return False


# def is_mp4(filename) -> bool:
#     """
#     Checks if file's extension is mp4
#     Parameters
#     ----------
#     filename: File name
#
#     Returns
#     -------
#
#     """
#     if filename.endswith('.mp4'):
#         return True
#     else:
#         return False
#
#
# def read_data(tree) -> list:
#     """
#
#     Parameters
#     ----------
#     tree: List of file paths to read
#
#     Returns
#     -------
#     audio: List of Sampling rate and data of wav files
#
#     """
#
#     audio = []
#
#     for filename in tree:
#         if is_wav(filename):
#             audio.append(wavfile.read(filename))
#
#     return audio

def move_npys(tree, dst_dir):
    for filename in tree:
        if filename.endswith(".npy"):
            destination = os.path.join(dst_dir, filename.split(os.sep)[-2])
            if not os.path.isdir(destination):
                os.makedirs(destination)
            move(filename, destination)


def is_dir(path: str) -> None:
    try:
        return os.path.isdir(path)
    except:
        assert f'{dir} is not a directory. Start again the process '


def init_directory(directory: str):
    """
    Initializing directories to store the features
    Args:
        directory (str): Directory to create in the filesystem

    Returns:
        None
    """
    try:
        print(f'Trying to create {directory}, in order to store aural features ')
        os.makedirs(directory)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print(f'Skipping {directory} creation. Directory already exists.')
        else:
            raise
    except:
        assert f'Cannot create {directory} directory'
