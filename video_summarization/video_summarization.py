import os

from video_summarization.libs.config import MODEL_URL
from video_summarization.libs.lib import make_classification, classify, extract_and_make_classification
from video_summarization.libs.utils import download_model
from video_summarization.utilities.utils import parse_arguments


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
    # save_prediction(prediction, dst)


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
        extract_and_train(args.videos, args.labels, args.output)
    elif args.task == "predict":
        predict(args.video, args.output)
    elif args.task == "extract_features":
        extract_features(args.videos, args.output)
    else:
        print(f"You have not choose any video summarization task.\n\t Video summarization exiting ")


if __name__ == "__main__":
    main()
