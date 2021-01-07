import os
import argparse
import pandas as pd
import numpy as np
import cv2


def crawl_directory(directory):
    """
    Crawling data directory
        Args:
            directory (str) : The directory to crawl
        Returns:
            tree (list)     : A list with all the filepath
    """
    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            tree.append(os.path.join(subdir, _file))
    return tree


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
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count, fps


def agreement(files_path, filenames, data_set) -> dict:
    new_dataset = data_set.copy()
    for csv_file in filenames:
        print("=====",csv_file)
        if csv_file.endswith('.txt'):
            print(csv_file)
            continue
        for user_annotation in files_path:
            if csv_file in user_annotation:
                print(">>>>>>",type(user_annotation), "=",csv_file)
                data = pd.read_csv(user_annotation)
                key = user_annotation.split(os.sep)[-2] + str(os.sep) + user_annotation.split(os.sep)[-1]
                for row in range(data.shape[0]):
                    start = data.iloc[row][0] * 60 + data.iloc[row][1]
                    end = data.iloc[row][2] * 60 + data.iloc[row][3]
                    print(csv_file)
                    print(data.iloc[row][0], data.iloc[row][1],
                          ':', data.iloc[row][2], data.iloc[row][3])
                    new_dataset[key][start:end] += 1
                    # print(new_dataset[csv_file])

    return new_dataset


def initialize_dataset(directory: str) -> dict:
    dataset = {}
    videos_path = crawl_directory(directory)
    for video in videos_path:
        frame_count, fps = get_video_duration(video)
        if fps != 0:
            duration = int(np.ceil(frame_count / fps))

            dataset[video.split(os.sep)[-2] + str(os.sep) +  video.split(os.sep)[-1] +
                    '.csv'] = np.zeros((duration,), dtype=int)

    return dataset


def check_directory(directory_name: str) -> bool:

    if not os.path.isdir(directory_name):
        return False
    return True


def save_npys(dataset: dict) -> bool:

    return True


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create annotated dataset")

    parser.add_argument("-i", "--input", required=True, help="Video file")
    parser.add_argument("-l", "--labels", required=True, help="Timestamp file")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory for annotations matrices")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    videos = args.input
    annotations = args.labels
    print(annotations)
    output = args.output
    annotations_dir = crawl_directory(annotations)
    unique = unique_files(annotations_dir)
    dataset = initialize_dataset(videos)
    new_data = agreement(annotations_dir, unique, dataset)
    np.save('teliko', new_data)


if __name__ == "__main__":
    main()
