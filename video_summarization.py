import argparse
import os

from video_summarization.config import MODEL_URL
from video_summarization.libs.lib import (classify,
                                          extract_and_make_classification,
                                          features_extraction,
                                          make_classification)
from video_summarization.libs.utils import (download_dataset, download_model,
                                            save_prediction)
from video_summarization.utilities.rename import rename


def parse_arguments() -> argparse.Namespace:
    """
    Argument Parser For Video Summarization tasks

    Returns:
        (argparse.Namespace): Returns the parsed args of the parser
    """
    epilog = """
        python3 video_summarization.py extractAndTrain -v /home/theo/VIDEOS -l /home/theo/LABELS -o /home/theo/videoSummary  
        python3 video_summarization.py train -v /home/theo/visual_features -a /home/theo/aural_features -l /home/theo/LABELS -o /home/theo/videoSummary 
        python3 video_summarization.py predict -v /home/theo/sample.mp4   
        python3 video_summarization.py featureExtraction -v /home/theo/VIDEOS
        """
    parser = argparse.ArgumentParser(description="Video Summarization application",
                                     formatter_class=argparse.RawTextHelpFormatter, epilog=epilog)

    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks", dest="task", metavar="")

    _train = tasks.add_parser(
        "train", help="Train video summarization classifier")
    _train.add_argument(
        "-v", "--visual", required=True, help="Directory of visual features")
    _train.add_argument(
        "-a", "--aural", required=True, help="Directory of aural features")
    _train.add_argument("-l", "--labels", required=True,
                        help="Label Input Directory")
    _train.add_argument("-o", "--output", required=False, help="Output Folder")

    extract_train = tasks.add_parser(
        "extractAndTrain", help="Extract and Train video summarization classifier")
    extract_train.add_argument(
        "-v", "--videos", required=True, help="Videos Directory")
    extract_train.add_argument(
        "-l", "--labels", required=True, help="Label Input Directory")
    extract_train.add_argument(
        "-o", "--output", required=True, help="Output Folder")
    extract_train.add_argument(
        "-d", "--download", action='store_true', help="Download Youtube Videos")

    _predict = tasks.add_parser("predict",
                                help="Export the summary of a video")
    _predict.add_argument("-v", "--video", required=True,
                          help="Video Input File")

    _feature_extraction = tasks.add_parser("featureExtraction",
                                           help="Export the audiovisual features from videos directory")
    _feature_extraction.add_argument(
        "-v", "--videos", required=True, help="Videos Input Directory")

    return parser.parse_args()


def train(visual_dir: str, aural_dir: str, labels_dir: str, destination: str) -> None:
    """
    Classification of video summarization using the already extracted features and labels
    Args:
        aural_dir (str): Aural directory with npys files
        visual_dir (str): Visual directory with npys files
        labels_dir (str): Labels directory with npys files
        destination (Str): Destination directory to save the model

    Returns:
        None

    """
    if not os.path.isdir(visual_dir):
        raise Exception("Visual directory not found!")
    if not os.path.isdir(aural_dir):
        raise Exception("Aural directory not found!")
    if not os.path.isdir(labels_dir):
        raise Exception("Labels directory not found!")
    if not os.path.isdir(destination):
        print("Output directory not found!\n \t Trying to create it!")
        try:
            os.mkdir(destination)
        except:
            assert f"Cannot create output directory {destination}"

    print('Training video summarization classifier')
    make_classification(aural_dir, visual_dir, labels_dir, destination)
    print(
        f"Training process completed, random forest is located at {destination}")


def extract_and_train(videos_dir: str, labels_dir: str, destination: str) -> None:
    """
    Extract Both aural and visual features and train the random forest.

    Args:
        videos_dir (str): Path to the videos directory
        labels_dir (str): Path to the labels directory
        destination (str): PAth to save the model

    Returns:
        None
    """
    if not os.path.isdir(videos_dir):
        raise Exception("Videos directory not found!")
    if not os.path.isdir(labels_dir):
        raise Exception("Labels directory not found!")
    if not os.path.isdir(destination):
        print("Output directory not found!\n \t Trying to create it!")
        try:
            os.mkdir(destination)
        except:
            assert f"Cannot create output directory {destination}"

    print('Extracting data and Training new  video summarization classifier')
    extract_and_make_classification(videos_dir, labels_dir, destination)


def predict(video: str) -> None:
    """
    Predicts the significant seconds of a video
    Args:
        video (str): Path of the video to make prediction

    Returns:

    """
    if not os.path.isfile(video):
        assert f"Video  {video} does not exists"
    download_model(MODEL_URL)
    prediction = classify(video)
    save_prediction(prediction)


def extract_features(videos_dir: str) -> None:
    """
    Use video summarization as feature extraction tool
    Args:
        videos_dir (path): Path of the videos directory to extract features

    Returns:

    """
    features_extraction(videos_dir)


def main() -> None:
    """
    Main functionality of the video summarization as command tool
    Returns:

    """
    args = parse_arguments()
    if args.task == "train":
        train(args.visual, args.aural, args.labels, args.output)
    elif args.task == "extractAndTrain":
        _videos_dir = args.videos
        if args.download:
            print(
                f"Given videos directory  {args.videos} ignored, starting downloading the proposed youtube videos")
            _videos_dir = download_dataset()
            rename(_videos_dir)
        extract_and_train(_videos_dir, args.labels, args.output)
    elif args.task == "predict":
        predict(args.video, args.output)
    elif args.task == "extract_features":
        extract_features(args.videos, args.output)
    else:
        print(f"You have not choose any video summarization task.\n\t Video summarization exiting ")


if __name__ == "__main__":
    main()
