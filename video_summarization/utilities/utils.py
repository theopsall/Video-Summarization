"""Helper funbctions"""
import argparse
import os
import requests
import scipy.io.wavfile as wavfile
from video_summarization.config import MODEL_DIR

def crawl_directory(directory: str) -> list:
    """Crawling data directory
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


def split_data():
    pass


def is_dir(path: str) -> None:
    pass


def make_directory(path: str, name: str):
    try:
        os.mkdir(path)
    except:
        assert f"Cannot create {path}"


def save_model(content):
    open(os.path.join(MODEL_DIR, 'rf_model.pt'), 'wb').write(content)


def download_model(url):
    r = requests.get(url, allow_redirects=True)
    save_model(r.content)

def parse_arguments() -> argparse.Namespace:
    '''
    Command Line Argument Parser
    '''
    epilog = """python3 train -v -l -o  in order to train the classifier 
                python3 predict -v -o  in order to export the summary of a video file"""
    parser = argparse.ArgumentParser(description="Video Summarization application",
                                     formatter_class=argparse.RawTextHelpFormatter, epilog=epilog)

    tasks = parser.add_subparsers(title="subcommands", description="available tasks", dest="task", metavar="")

    train = tasks.add_parser("train", help="Train video summarization classifier")
    train.add_argument("-v", "--videos", required=True, help="Video Input Directory")
    train.add_argument("-l", "--labels", required=True, help="Label Input Directory")
    train.add_argument("-o", "--output", required=False, help="Output Folder")

    train = tasks.add_parser("extractAndTrain", help="Train video summarization classifier")
    train.add_argument("-v", "--videos", required=True, help="Video Input Directory")
    train.add_argument("-l", "--labels", required=True, help="Label Input Directory")
    train.add_argument("-o", "--output", required=False, help="Output Folder")

    predict = tasks.add_parser("predict",
                               help="Export the summary of a video")
    predict.add_argument("-v", "--video", required=True, help="Video Input File")
    predict.add_argument("-o", "--output", required=False, help="Output Directory")

    return parser.parse_args()


if __name__ == "__main__":
    parse_arguments()
