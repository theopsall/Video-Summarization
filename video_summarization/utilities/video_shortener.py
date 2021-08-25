'''
cutting the video depending the prediction.
'''
import argparse
import os
import sys

import pandas as pd

from video_summarization.utilities.utils import crawl_directory


def file_shorten(input_file, timestamps_file, output_file):
    if not os.path.isfile(input_file):
        raise Exception("{0} not found!".format(input_file))
    if not os.path.isfile(timestamps_file):
        raise Exception("{0} not found!".format(timestamps_file))
    if os.path.isfile(output_file):
        print("Output file {0} already exists!\n Please change output file name or delete the".format(output_file))
        print("Exiting")
        return
    annotation = pd.read_csv(timestamps_file)
    total_timestamps = annotation.shape[0]
    temp_files = []
    for i in range(total_timestamps):
        start_minute = str(annotation.iloc[i, 0]).zfill(2)
        start_sec = str(annotation.iloc[i, 1]).zfill(2)
        end_minute = str(annotation.iloc[i, 2]).zfill(2)
        end_sec = str(annotation.iloc[i, 3]).zfill(2)
        tmp_file = 'output_{0}.mp4'.format(i)
        temp_files.append(tmp_file)
        os.system("echo file {0} >> temporary_list.txt".format(tmp_file))
        sys.stdout = open(os.devnull, 'w')
        cmd = "ffmpeg -i '{0}' -ss {1} -to {2} -c copy {3}".format(input_file,
                                                                   '00:' + start_minute + ':' + start_sec,
                                                                   '00:' + end_minute + ':' + end_sec,
                                                                   tmp_file)
        sys.stdout = sys.__stdout__
        print("=> {0} <=\n \n ".format(cmd))
        os.system(cmd)

    sys.stdout = open(os.devnull, 'w')

    cmd = "ffmpeg -f concat -i {0} -c copy {1}".format('temporary_list.txt',
                                                       "'{0}'".format(output_file))
    sys.stdout = sys.__stdout__
    os.system(cmd)
    for tmp_file in temp_files:
        os.system('rm {0}'.format(tmp_file))
    os.system('rm temporary_list.txt')


def dir_shorten(input_directory, timestamps_directory, output_directory):
    if not os.path.isdir(input_directory):
        raise Exception("{0} not found!".format(input_directory))
    if not os.path.isdir(timestamps_directory):
        raise Exception("{0} not found!".format(timestamps_directory))
    if not os.path.isdir(output_directory):
        print("Output {0} directory not found! \n Resolving by creating the directory".format(output_directory))
        os.mkdir(output_directory)
    videos = crawl_directory(input_directory)
    timestamps = crawl_directory(timestamps_directory)
    for video_file, timestamp in zip(videos, timestamps):
        print(video_file, 'with', timestamp)
        class_name = video_file.split(os.sep)[-2]
        output_folder = os.path.join(output_directory, class_name)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        output_file = os.path.join(output_folder, "short_{0}".format(video_file.split(os.sep)[-1]))
        print(output_file)
        file_shorten(video_file, timestamp, output_file)


def parse_arguments():
    parser = argparse.ArgumentParser(description="A demonstration script "
                                                 "for video shorting")
    tasks = parser.add_subparsers(
        title="subcommands",
        description="available tasks",
        dest="task", metavar="")

    single_file = tasks.add_parser("fileShorten",
                                   help="Shorten a video file")
    single_file.add_argument("-i", "--input", required=True, help="Video file")
    single_file.add_argument("-l", "--labels", required=True, help="Timestamp file")
    single_file.add_argument("-o", "--output", required=True, help="Output directory for shorten video file")

    video_dir = tasks.add_parser("dirShorten",
                                 help="Shorten all videos on the directory")
    video_dir.add_argument("-i", "--input", required=True, help="Videos input directory")
    video_dir.add_argument("-l", "--labels", required=True, help="Timestamps directory")
    video_dir.add_argument("-o", "--output", required=True, help="Output directory for shorten videos")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # python3 video_shortener.py dirShorten -i Video -l Video -o shorted
    # python3 video_shortener.py fileShorten -i "TK Hinshaw - 14-way Hybrid -- Skydive Hawaii.mp4" -l 'TK Hinshaw - 14-way Hybrid -- Skydive Hawaii.mp4.csv' -o theo.mp4
    if args.task == "fileShorten":
        # Convert mp3 to wav (batch - folder)
        file_shorten(args.input, args.labels, args.output)
    elif args.task == "dirShorten":
        # Convert fs for a list of wavs, stored in a folder
        dir_shorten(args.input, args.labels, args.output)
