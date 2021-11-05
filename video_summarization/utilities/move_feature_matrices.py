import os
import shutil
from glob import glob


def crawl_directory(directory):
    """Crawling data directory
        Args:
            directory (str) : The directory to crawl
        Returns:
            A list with all the filepaths
    """
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            yield os.path.join(subdir, _file)


def move_npys(tree, dst_dir):
    for filename in tree:
        if filename.endswith(".npy"):
            destination = os.path.join(dst_dir, filename.split(os.sep)[-2])
            if not os.path.isdir(destination):
                os.makedirs(destination)
            shutil.copy(filename, destination)


if __name__ == '__main__':
    videos = "/media/theo/Hard Disk 2/PyCharm/Video-Summarization/DATA/Video_smaller"
    features_directory = "/media/theo/Hard Disk 2/PyCharm/Video-Summarization/DATA/features"
    tree = crawl_directory(videos)
    move_npys(tree, features_directory)
