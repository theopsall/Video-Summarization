import os
import argparse
import sys
import pandas as pd
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../"))
from utilities.utils import crawl_directory

def parser() -> argparse.Namespace
    """
    Handling and parsing the command line arguments

    Returns:
        [argparse.Namespace]: The parsed arguments
    """
    usage_example = """Example of use
        python3 user_statistics.py -l Annotated_31_12_21"""

    parser = argparse.ArgumentParser(
        description="Extract Video statistics",
        epilog=usage_example,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-l", "--labels", required=True, help="Labels directory")

    return parser.parse_args()


def main(annotations: str) -> None:
    """
    Parsing annotations files counting the number of annotated videos

    Args:
        annotations (str): The input directory path containing the annotation files

    Returns:
        None
    """
    videos_stats = {}
    annotations = crawl_directory(annotations)
    for annotation_file in annotations:
        if annotation_file.endswith(".txt"):
            with open(annotation_file, 'r') as f:
                user_videos = f.read().split('\n')
            for user_video in user_videos:
                if not user_video in videos_stats:
                    videos_stats[user_video] = 1
                else:
                    videos_stats[user_video] += 1

    data = pd.DataFrame({'Video Name':list(videos_stats.keys()), 'Number of Annotations': list(videos_stats.values())})
    data.to_csv("video_stats.csv", index=False)


    return videos_stats
if __name__ == "__main__":
    parser = parser()
    labels = parser.labels
    if not os.path.isdir(labels):
        raise Exception("Input path not found!")
    main(labels)
