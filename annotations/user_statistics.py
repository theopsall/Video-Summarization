import os
import argparse
import sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from utilities.utils import crawl_directory

def parser() -> argparse.Namespace:
    """
    Handling and parsing the command line arguments

    Returns:
        [argparse.Namespace]: The parsed arguments
    """
    usage_example = """Example of use
        python3 user_statistics.py -l Annotated_31_12_21"""

    parser = argparse.ArgumentParser(
        description="Extract video statistics for each user",
        epilog=usage_example,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-l", "--labels", required=True,
                        help="Labels directory")

    return parser.parse_args()


def main(annotations: str) -> None:
    """
    Parsing annotations files counting the number of annotated videos for each user

    Args:
        annotations (str): The input directory path containing the annotation files

    Returns:
        None
    """
    user_stats = {}
    annotations = crawl_directory(annotations)
    for annotation_file in annotations:
        if annotation_file.endswith(".txt"):
            username = os.path.splitext(annotation_file)[0].split(os.sep)[-1]
            with open(annotation_file, 'r') as f:
                user_videos = f.read().split('\n')
            for line in user_videos:
                if len(line) > 1:
                    if not username in user_stats :
                        user_stats[username] = 1
                    else:
                        user_stats[username] += 1

    data = pd.DataFrame({'User Name': list(
        user_stats.keys()), 'Number of Annotations': list(user_stats.values())})
    data.to_csv("user_stats.csv", index=False)


if __name__ == "__main__":
    parser = parser()
    labels = parser.labels
    if not os.path.isdir(labels):
        raise Exception("Input path not found!")
    main(labels)
