"""Helper funbctions"""
import argparse
import os

import scipy.io.wavfile as wavfile


def crawl_directory(directory: str) -> list:
    """
    Crawling data directory
        Args:
            directory (str) : The directory to crawl
        Returns:
            tree (list)     : A list with all the filepaths
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


def is_mp4(filename) -> bool:
    """
    Checks if file's extension is mp4
    Parameters
    ----------
    filename: File name

    Returns
    -------

    """
    if filename.endswith('.mp4'):
        return True
    else:
        return False


def read_data(tree) -> list:
    """

    Parameters
    ----------
    tree: List of file paths to read

    Returns
    -------
    audio: List of Sampling rate and data of wav files

    """

    audio = []

    for filename in tree:
        if is_wav(filename):
            audio.append(wavfile.read(filename))

    return audio


def is_dir(path: str) -> None:
    return os.path.isdir(path)


def make_directory(path: str, name: str):
    try:
        os.mkdir(path)
    except:
        assert f"Cannot create {path}"


