"""Helper funbctions"""
import os
import scipy.io.wavfile as wavfile


def crawl_directory(directory) -> list:
    """Crawling data directory
        Args:
            directory (str) : Audio files directory to crawl
        Returns:
            A generator with all the filepaths
    """
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            yield os.path.join(subdir, _file)



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

def split_data():
    pass
