import os
import argparse

from video_summarization.config import  MODEL_URL
from video_summarization.libs.lib import make_classification, classify, extract_and_make_classification
from video_summarization.libs.utils import download_model, download_dataset


def parse_arguments() -> argparse.Namespace:
    """
    Argument Parser For Video Summarization tasks
    Returns:
        (argparse.Namespace): Returns the parsed args of the parser
    """
    epilog = """python3 train -v -l -o  in order to train the classifier
                python3 extractAndTrain -v -l -o  in order to extract the features from new videos and train the
                        classifier, you have the option to download our dataset from youtube using the parameter -d
                python3 predict -v -o  in order to export the summary of a video file
                python3 featureExtraction -v -o  in order to use the video_summarization package as audioVisual feature
                        Extractor
                """
    parser = argparse.ArgumentParser(description="Video Summarization application",
                                     formatter_class=argparse.RawTextHelpFormatter, epilog=epilog)

    tasks = parser.add_subparsers(title="subcommands", description="available tasks", dest="task", metavar="")

    _train = tasks.add_parser("train", help="Train video summarization classifier")
    _train.add_argument("-l", "--labels", required=True, help="Label Input Directory")
    _train.add_argument("-v", "--videos", required=True, help="Video Input Directory")
    _train.add_argument("-o", "--output", required=False, help="Output Folder")

    extract_train = tasks.add_parser("extractAndTrain", help="Train video summarization classifier")
    extract_train.add_argument("-v", "--videos", required=True, help="Video Input Directory")
    extract_train.add_argument("-l", "--labels", required=True, help="Label Input Directory")
    extract_train.add_argument("-o", "--output", required=False, help="Output Folder")
    extract_train.add_argument("-d", "--download", action='store_true', help="Download Youtube Videos")

    _predict = tasks.add_parser("predict",
                                help="Export the summary of a video")
    _predict.add_argument("-v", "--video", required=True, help="Video Input File")
    _predict.add_argument("-o", "--output", required=False, help="Output Directory")

    _feature_extraction = tasks.add_parser("featureExtraction",
                                help="Export the summary of a video")
    _feature_extraction.add_argument("-v", "--video", required=True, help="Video Input File")
    _feature_extraction.add_argument("-o", "--output", required=False, help="Output Directory")

    return parser.parse_args()


def train(aural_dir: str, visual_dir: str, labels_dir: str, destination: str):
    if not os.path.isdir(aural_dir):
        raise Exception("Aural directory not found!")
    if not os.path.isdir(visual_dir):
        raise Exception("Visual directory not found!")
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
    print(f"Training process completed, random forest is located at {destination}")


def extract_and_train(videos_dir: str, labels_dir: str, destination: str):
    """
    Extract Both aural and visual features and train the random forest.

    Args:
        videos_dir:
        labels_dir:
        destination:

    Returns:

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


def predict(video: str, dst: str):
    """

    Args:
        video:
        dst:

    Returns:

    """
    if not os.path.isdir(dst):
        print("Output directory not found!\n \t Trying to create it!")
        try:
            os.mkdir(dst)
        except:
            assert f"Cannot create destination directory {dst}"
    download_model(MODEL_URL)
    prediction = classify(video)
    if not os.path.isdir(dst):
        assert f"Destination {dst} directory not found. Please give a proper destination folder to store the prediction"
    utils.save_prediction(prediction, dst)


def extract_features(videos, output):
    """

    Args:
        videos:
        output:

    Returns:

    """
    pass


def main() -> None:
    """

    Returns:

    """
    args = parse_arguments()
    if args.task == "train":
        train(args.videos, args.labels, args.output)
    elif args.task == "extractAndTrain":
        _videos_dir = args.videos
        if args.d:
            print(f"Given video directory  {args.videos} ignored, starting downloading the proposed youtube videos")
            _videos_dir = download_dataset()
        extract_and_train(_videos_dir, args.labels, args.output)
    elif args.task == "predict":
        predict(args.video, args.output)
    elif args.task == "extract_features":
        extract_features(args.videos, args.output)
    else:
        print(f"You have not choose any video summarization task.\n\t Video summarization exiting ")


if __name__ == "__main__":
    main()
