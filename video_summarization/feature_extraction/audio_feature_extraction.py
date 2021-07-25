import os
import argparse
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import ShortTermFeatures as sF
from pyAudioAnalysis import MidTermFeatures as mF


def crawl_directory(directory):
    """Crawling data directory
        Args:
            directory (str) : Audio files directory to crawl
        Returns:
            A generator with all the filepaths
    """

    subdirs = (folder[0] for folder in os.walk(directory))

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            yield os.path.join(subdir, _file)


def extract_audio(video, output):
    destination_name = os.path.join(output, video.split(os.sep)[-1] + '.wav')
    command = "ffmpeg -i '{0}' -q:a 0 -ac 1 -ar 16000  -map a '{1}'".format(video,
                                                                            destination_name)

    # ffmpeg -i short_supra.mkv -q:a 0 -ac 1 -ar 16000 -map a sample.wav
    os.system(command)


def get_audio_features(audio_file, output_file):
    mid_window, mid_step, short_window, short_step = 1, 1, 0.1, 0.1
    store_csv = False
    store_short_features = False
    plot = False
    mF.mid_feature_extraction_to_file(
        audio_file, mid_window, mid_step, short_window, short_step, output_file, store_short_features, store_csv, plot)


def main(tree):

    audio_dir = os.path.join(os.getcwd(), 'audio_data')
    if not os.path.isdir(audio_dir):
        os.mkdir(audio_dir)
    audio_features = os.path.join(os.getcwd(), 'audio_features')
    if not os.path.isdir(audio_features):
        os.mkdir(audio_features)
    for filename in tree:
        destination = os.path.join(audio_dir, filename.split(os.sep)[-2])
        feature_destination = os.path.join(
            audio_features, filename.split(os.sep)[-2])
        if not os.path.isdir(destination):
            os.makedirs(destination)
        if not os.path.isdir(feature_destination):
            os.makedirs(feature_destination)
        extract_audio(filename, destination)
        destination_name = os.path.join(
            destination, filename.split(os.sep)[-1] + '.wav')
        feature_destination = os.path.join(
            feature_destination, filename.split(os.sep)[-1] + '.wav')
        get_audio_features(destination_name, feature_destination)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input folder with Videos")

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_arguments()
    intput_directory = parser.input
    tree = crawl_directory(intput_directory)
    main(tree)
