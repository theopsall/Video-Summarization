"""
dummy script to cut videos bigger than 15minutes
"""

import cv2
import argparse
import shutil
import os


original_dataset = "/media/theo/Data/PycharmProjects/Video-Summarization/DATA/Video/"
new_dataset_name = "/Video_small/"


# TODO use crawl directory from utilities directory


def crawl_directory(directory):
    """Crawling data directory
    Args:
        directory (str) : The directory to crawl
    Returns:
        tree (list)     : A list with all the filepaths
    """
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            yield os.path.join(subdir, _file)


def get_duration(filename):
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration = frame_count/fps

    return frame_count, fps


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input folder with Videos")

    return parser.parse_args()


def main():
    original_tree = crawl_directory(original_dataset)
    for filename in original_tree:
        frame_count, fps = get_duration(filename)
        if fps != 0:
            duration = frame_count / fps
            if duration <= 900:
                destination = filename.replace("/Video/", new_dataset_name)
                dst_dir = os.path.join(*destination.split(os.sep)[-3:-1])
                if not os.path.exists(dst_dir):
                    print("{0}Creation".format(dst_dir))
                    os.makedirs(dst_dir)
                # print(filename)
                shutil.copy(filename, destination)


# 15 minutes = 900 seconds
if __name__ == "__main__":
    main()
