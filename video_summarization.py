import os

from video_summarization.libs.config import MODEL_URL
from video_summarization.libs.lib import make_classification, classify, extract_and_make_classification
from video_summarization.utilities.utils import parse_arguments, download_model


def train(videos: str, labels: str, output: str):
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

    print('Training video summarization classifier')
    make_classification()
    pass


def extractAndTrain(videos: str, labels: str, output: str):
    print('Extracting data and Training new  video summarization classifier')
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


def main() -> None:
    args = parse_arguments()
    if args.task == "train":
        train(args.videos, args.labels, args.output)
    elif args.task == "extractAndTrain":
        extractAndTrain(args.videos, args.labels, args.output)
    elif args.task == "predict":
        predict(args.video, args.output)
    else:
        print(f"You have not choose any video summarization task. \t Video summarization exiting ")


if __name__ == "__main__":
    main()
