import os
import argparse
from multimodal_movie_analysis.analyze_visual import analyze_visual as aV

def parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input folder with Videos")

    return parser.parse_args()



def crawl_class_directory(directory):
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        cmd = f"python3 analyze_visual.py -d '{class_dir}'"
        os.system(cmd)



if __name__ == "__main__":
    videos = "/media/theo/Hard Disk 2/PyCharm/Video-Summarization/DATA/Video_smaller"
    print("Crawling Dir started")
    crawl_class_directory(videos)
    print("END")