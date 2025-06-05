"""Helper funbctions"""

import errno
import os
from shutil import move

from video_summarization.config import ALLOWED_EXTENSIONS


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


def is_audio_file(filename: str) -> bool:
    """
    Checks if file's extension is wav

    Args:
        filename (str):

    Returns:
        (bool): True if file in expected audio format, False otherwise
    """

    return filename.split(".")[-1] in ALLOWED_EXTENSIONS


def move_npys(tree: list, dst_dir: str) -> None:
    """
    Move .npys files from one directory to an other directory.
    FYI: Multimodal movie analysis by default stores the feature extracted features files (.npy)
    on the same folder with the processed video.
    Args:
        tree (list): List with all the videos path
        dst_dir (str): Destination directory to store the .npys. Destination directory will simulate the structure of
                       the source directory

    Returns:
        None

    """
    for filename in tree:
        if filename.endswith(".npy"):
            destination = os.path.join(dst_dir, filename.split(os.sep)[-2])
            if not os.path.isdir(destination):
                os.makedirs(destination)
            move(filename, destination)


def is_dir(path: str) -> bool:
    """
    Checks if the given path argument is directory or not
    Args:
        path (str): Path of the directory

    Returns:
        (bool): True if the directory exists, False otherwise

    """

    return os.path.isdir(path)


def init_directory(directory: str):
    """
    Initializing directories to store the features
    Args:
        directory (str): Directory to create in the filesystem

    Returns:
        None
    """
    try:
        print(f"Trying to create {directory} ")
        os.makedirs(directory)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print(f"Skipping {directory} creation. Directory already exists.")
        else:
            raise
    except:
        assert f"Cannot create {directory} directory"
