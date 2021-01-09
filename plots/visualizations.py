'''
    Exporting visualizations for users and videos statistics.
'''

import os
import argparse
import pandas as pd
from bokeh.layouts import column, row
from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.autompg import autompg
from bokeh.transform import jitter



def parser() -> argparse.Namespace:
    """
    Handling and parsing the command line arguments

    Returns:
        [argparse.Namespace]: The parsed arguments
    """
    usage_example = """Example of use
        python3 visualization.py -u annotations/user_stats.csv -v annotations/video_stats.csv"""

    parser = argparse.ArgumentParser(
        description="Extract video statistics for each user",
        epilog=usage_example,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-i", "--input", required=True, help="Video statistics file)

    return parser.parse_args()

def main():
    pass


if __name__ == "__main__":
    parser = parser()
    users = parser.users
    videos = parser.videos
    if not os.path.isfile(users):
        raise Exception(f"File {users} path not found!")
    if not os.path.isfile(videos):
        raise Exception(f"File {users} path not found!")
    main()
