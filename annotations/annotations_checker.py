import os
import argparse
import pandas as pd
import numpy as np
import cv2
import sys
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from utilities.utils import crawl_directory

def unique_files(files_path: str) -> set:
    """
    Returns the uniques file names from a directory. It will be quiet usefull when you deal with same files seperated in
    different directories, e.g multi label dataset split in subdirectories
    Args:
        files_path: List contains the path of every file

    Returns:
        The unique file names, including the file extension, of a given directory

    """
    return set([filename.split(os.sep)[-1] for filename in files_path])


def get_video_duration(filename: str) -> tuple:
    """
    Getting the video frame count and frame per second.

    Args:
        filename (str): Video filepath

    Returns:
        tuple: Tuple containing frame count and fps
    """
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count, fps


def agreement(files_path, filenames, data_set) -> dict:
    """
    [summary]

    Args:
        files_path ([type]): [description]
        filenames ([type]): [description]
        data_set ([type]): [description]

    Returns:
        dict: [description]
    """
    new_dataset = data_set.copy()
    for csv_file in filenames:
        if csv_file.endswith('.txt'):
            continue
        for user_annotation in files_path:
            if csv_file in user_annotation:
                data = pd.read_csv(user_annotation )
                key = user_annotation.split(
                    os.sep)[-2] + str(os.sep) + user_annotation.split(os.sep)[-1]
                key = os.path.splitext(key)[0]
                for row in range(data.shape[0]):
                    start = data.iloc[row][0] * 60 + data.iloc[row][1]
                    end = data.iloc[row][2] * 60 + data.iloc[row][3]
                    # print(data.iloc[row][0], data.iloc[row][1],
                    #       ':', data.iloc[row][2], data.iloc[row][3])
                    new_dataset[key][start:end] += 1

    return new_dataset


def initialize_dataset(directory: str) -> dict:
    """
    Initializing the dictionary of the video segment importance.
    Key is the video name and value a zero numpy array, in video duration size

    Args:
        directory (str): Video directory path

    Returns:
        dict: Initialized video dictionary
    """
    dataset = {}
    videos_path = crawl_directory(directory)
    for video in videos_path:
        frame_count, fps = get_video_duration(video)
        if fps != 0:
            duration = int(np.ceil(frame_count / fps))

            dataset[video.split(os.sep)[-2] + str(os.sep) +
                    video.split(os.sep)[-1]] = np.zeros((duration,), dtype=int)

    return dataset


def check_directory(directory_name: str) -> bool:
    """
    Checking if a directory exists.

    Args:
        directory_name (str): Directory path

    Returns:
        bool: True is directory exists, False otherwise
    """

    if not os.path.isdir(directory_name):
        return False
    return True


def save_npys(dataset: dict, dest_dir: str) -> None:
    """
    Function saving the numpy arrays of the summed annotations for each video.

    Args:
        dataset (dict): Dictinary containing the numpy array for each video
        dest_dir (str): Destination directory to save the numpy files
    """

    for key, value in dataset.items():
        dst_class = os.path.join(dest_dir, key.split(os.sep)[0])
        if not check_directory(dst_class):
            os.makedirs(dst_class)
        np.save( os.path.join(dst_class, key.split(os.sep)[-1]), value)


def parse_arguments() -> argparse.Namespace:
    """
    Handling and parsing the command line arguments

    Returns:
        [argparse.Namespace]: The parsed arguments
    """

    usage_example = """Example of use
        python3 annotations_checker.py -i /media/theo/Hard\ Disk\ 2/PyCharm/Video-Summarization/DATA/Videos -l Annotated_31_12_21 -o labels"""
    parser = argparse.ArgumentParser(description="Create annotated dataset",
                                     epilog=usage_example,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-i", "--input", required=True, help="Videos directory")
    parser.add_argument("-l", "--labels", required=True, help="Timestamps directory")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory for annotations matrices")

    return parser.parse_args()


def main() -> None:
    """
    Main function exporting the annotations numpy files aggregating annotations from all users.

    Raises:
        Exception: If Video directory does not exists
        Exception: If Annotations directory does not exists
    """
    args = parse_arguments()

    videos = args.input
    if not check_directory(videos):
        raise Exception("Input path with Videos not found!")
    annotations = args.labels
    if not check_directory(annotations):
        raise Exception("Input path with Annotations not found!")
    output = args.output
    if not check_directory(output):
        print("Output directory does not exists!\n Attempting to create it!")
        os.mkdir(output)
    annotations_dir = crawl_directory(annotations)
    unique = unique_files(annotations_dir)
    dataset = initialize_dataset(videos)
    labels = agreement(annotations_dir, unique, dataset)
    save_npys(labels, output)


if __name__ == "__main__":
    main()
