import os

from video_summarization.libs.config import MODEL_URL
from video_summarization.libs.lib import make_classification, classify, extract_and_make_classification
from video_summarization.utilities.utils import parse_arguments


def train(features: str, labels: str, output: str):
    if not os.path.isdir(features):
        raise Exception("Features directory not found!")
    if not os.path.isdir(labels):
        raise Exception("Labels directory not found!")
    if not os.path.isdir(output):
        print("Output directory not found!\n \t Trying to create it!")
        try:
            os.mkdir(output)
        except:
            assert f"Cannot create output directory {output}"

    print('Training video summarization classifier')
    make_classification(features, labels, output)
    pass


def extract_and_train(videos: str, labels: str, output: str):
    if not os.path.isdir(videos):
        raise Exception("Videos directory not found!")
    if not os.path.isdir(labels):
        raise Exception("Labels directory not found!")
    if not os.path.isdir(output):
        print("Output directory not found!\n \t Trying to create it!")
        try:
            os.mkdir(output)
        except:
            assert f"Cannot create output directory {output}"

    print('Extracting data and Training new  video summarization classifier')
    extract_and_make_classification(videos, labels, output)


def predict(video: str, output: str):
    if not os.path.isdir(output):
        print("Output directory not found!\n \t Trying to create it!")
        try:
            os.mkdir(output)
        except:
            assert f"Cannot create output directory {output}"
    download_model(MODEL_URL)
    classify(video, output)


def extract_features(videos, output):
    pass


def main() -> None:
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
