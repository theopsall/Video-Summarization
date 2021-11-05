import os
from pickle import dump as pdump
from pickle import load as pload

import numpy as np
import requests
from numpy.random import shuffle
from pyAudioAnalysis import MidTermFeatures as mF
from pyAudioAnalysis import audioBasicIO as iO
from scipy.signal import medfilt

from video_summarization.config import AUDIO_SCALER, VISUAL_FEATURES_DIR, VISUAL_SCALER, MODEL_DIR, DATASET, AUDIO_DATA_DIR, \
    AURAL_FEATURES_DIR, VIDEOS_DATA_DIR
from video_summarization.libs.multimodal_movie_analysis.analyze_visual.analyze_visual import process_video, \
    dir_process_video
from video_summarization.utilities.utils import is_dir, init_directory, crawl_directory, move_npys


def shuffle_lists(labels: list, aural: list, visual: list) -> zip:
    """
    Shuffling the lists from all modalities and labels all together to keep the correct order

    Args:
        labels (list): List of the video labels
        audio (list): List of the audio features
        visual (list): List of the visual features

    Returns
        (tuple): The shuffled lists as tuple
    """
    np.random.seed(47)
    zipped_list = list(zip(labels, aural, visual))
    shuffle(zipped_list, )
    return zip(*zipped_list)


def save_model(content) -> None:
    """
    Storing the Random Forest model pickle to the local machine

    Args:
        content (): The content from the request

    Returns
        None
    """
    print('Saving RF Model')
    open(os.path.join(MODEL_DIR, 'rf_model.pt'), 'wb').write(content)


def download_model(url: str) -> None:
    """
    Requesting and saving the Random Forest model from the cloud

    Args:
        url (str): Model's url

    Returns

    """
    print('Downloading RF Model from Cloud')
    r = requests.get(url, allow_redirects=True)
    try:
        save_model(r.content)
    except:
        assert f"Cannot save the model to the local machine"


def video_exists(video_path: str) -> bool:
    """
    Checking the existence of the video file

    Args:
        video_path (str): Video's path in disk

    Returns
        (bool): True if video exists, False otherwise

    """
    if os.path.isfile(video_path):
        return True
    return False


def audio_isolation(video: str) -> bool:
    """
    Isolate the audio signal from a video stream
    in wav file with sampling rate= 1600 in mono channel

    Args:
        video (str): Video's path in disk

    Returns
        (bool): True if audio isolated successfully, False otherwise
    """
    command = "ffmpeg -i '{0}' -q:a 0 -ac 1 -ar 16000  -map a '{1}'".format(
        video, 'isolated_audio.wav')
    try:
        os.system(command)
        return True
    except:
        print("Audio isolation failed")
        return False


def extract_audio(video: str, output: str):
    """
    Extract audio from every video file
    Args:
        video (str): Video path in disk
        output (str): Destination folder to be stored
    Returns

    """
    destination_name = os.path.join(output, video.split(os.sep)[-1] + '.wav')
    command = "ffmpeg -i '{0}' -q:a 0 -ac 1 -ar 16000  -map a '{1}'".format(video,
                                                                            destination_name)

    # ffmpeg -i short_supra.mkv -q:a 0 -ac 1 -ar 16000 -map a sample.wav
    os.system(command)


def get_audio_features(audio_file: str, output_file: str):
    """
    Extract and store audio features using pyAudioAnalysis
    Args:
        audio_file (str): Audio path in disk
        output_file (str): Destination name to store the extraction result

    Returns

    """
    mid_window, mid_step, short_window, short_step = 1, 1, 0.1, 0.1
    store_csv = False
    store_short_features = False
    plot = False
    mF.mid_feature_extraction_to_file(
        audio_file, mid_window, mid_step, short_window, short_step, output_file, store_short_features, store_csv, plot)


def store_audio_features(video_dir: str):
    tree = crawl_directory(video_dir)
    is_dir(AUDIO_DATA_DIR)

    is_dir(AURAL_FEATURES_DIR)
    for filename in tree:
        destination = os.path.join(AUDIO_DATA_DIR, filename.split(os.sep)[-2])
        feature_destination = os.path.join(
            AURAL_FEATURES_DIR, filename.split(os.sep)[-2])
        init_directory(destination)
        init_directory(feature_destination)
        extract_audio(filename, destination)
        destination_name = os.path.join(
            destination, filename.split(os.sep)[-1] + '.wav')
        feature_destination = os.path.join(
            feature_destination, filename.split(os.sep)[-1] + '.wav')
        get_audio_features(destination_name, feature_destination)


def extract_audio_features(audio: str):
    """
    Extracting Mid Term audio features

    Args:
        audio (str): Audio filepath on disk

    Returns
        (list): A list of midTerm features
    """

    [sampling_rate, x] = iO.read_audio_file(audio)
    mid_window, mid_step, short_window, short_step = 1, 1, 0.1, 0.1

    # sampling_rate = 16000
    # keeping short_features, mid_feature_names := for future use

    mid_features, short_features, mid_feature_names = mF.mid_feature_extraction(x, sampling_rate,
                                                                                mid_window * sampling_rate,
                                                                                mid_step * sampling_rate,
                                                                                short_window * sampling_rate,
                                                                                short_step * sampling_rate)
    return mid_features


def extract_video_features(video_path: str,
                           process_mode: int = 2,
                           print_flag: bool = False,
                           online_display: bool = False,
                           save_results: bool = True) -> list:
    """
    Extracting and storing the visual features of a video using the multimodal_movie_analysis
    Args:
        video_path (str): Video's path in disk
        process_mode (int): Process mode, default 2
        print_flag (bool): default False,
        online_display (bool): default False, disable commandline messages
        save_results (bool): default `True`, store the features

    """
    if not video_exists(video_path):
        raise Exception(f'{video_path} Not Found')

    features_stats, f_names_stats, feature_matrix, f_names, \
        shot_change_times = process_video(video_path, process_mode, print_flag,
                                          online_display, save_results)
    return feature_matrix


def extract_video_dir_features(videos_dir: str,
                               process_mode: int = 2,
                               print_flag: bool = False,
                               online_display: bool = False,
                               save_results: bool = True) -> list:
    """
    Extracting and storing the visual features of a video using the multimodal_movie_analysis
    Args:
        videos_dir (list): Directory of videos in disk
        process_mode (int): Process mode, default 2
        print_flag (bool): default False,
        online_display (bool): default False, disable commandline messages
        save_results (bool): default `True`, store the features

    """
    if not is_dir(videos_dir):
        raise Exception(f'{videos_dir} Not Found')

    for class_name in os.listdir(videos_dir):
        class_dir = os.path.join(videos_dir, class_name)
        features_all, video_files_list, f_names = dir_process_video(class_dir, process_mode, print_flag,
                                                                    online_display, save_results)


def store_visual_features(videos_dir):
    tree = crawl_directory(videos_dir)
    extract_video_dir_features(videos_dir)
    tree = [name + '.npy' for name in tree]
    move_npys(tree, VISUAL_FEATURES_DIR)


def reshape_features(audio: np.ndarray, visual: np.ndarray) -> tuple:
    """
    Reshaping the features matrices. Visual features are 5 times the audio features
    caused by the 0.2 step in processing video by multimodal_movie_analysis

    Args:
        audio (np.ndarray): Audial features matrix. 136 features in total
        visual (np.ndarray): Visual features matrix. 88 in total

    Returns:
        tuple: A tuple of the reshape audial and visual feature matrices
    """
    audio = audio.transpose()

    v_r, v_c = visual.shape
    a_r, a_c = audio.shape

    v_r = v_r // 5
    min_seconds = min(v_r, a_r)

    audio = audio[:min_seconds]
    visual = visual[:min_seconds * 5]
    # averaging visual every 5 (Because we have analyze video with .2 step)
    visual = visual.transpose().reshape(-1, 5).mean(1).reshape(v_c, -1).transpose()

    return audio, visual


def scale_features(aural: np.ndarray, visual: np.ndarray) -> tuple:
    """
    Scaling the features from both modalities
    Args:
        aural (nd.array): The aural features
        visual (nd.array): The visual features

    Returns:
        (aural, visual) rescaled features using the already pretrained scalers from our experiment
    """
    visual_scaler = pload(VISUAL_SCALER)
    aural_scaler = pload(AUDIO_SCALER)

    return visual_scaler.transform(visual), aural_scaler.transform(aural)


def fused_features(audio: list, visual: list) -> np.ndarray:
    """
    Fusing (concatenating) audial and visual features

    Args:
        audio (list): Audial features matrix. 136 features in total
        visual (list): Visual features matrix. 88 in total

    Returns:
        list: With the concatenated features
    """
    audio = scale_features(audio, 'aural')
    visual = scale_features(visual, 'visual')

    return np.concatenate((audio, visual), axis=1)


def smooth_prediction(prediction: list, hard_thres: int = 3) -> list:
    """
    Smoothing predictions. If the number of ones 'interesting' in the row are less than hard_thres
    discard them, by changing them to zeros 'non-interesting'. With smoothing we
    are making the predicted video summary smaller in duration.

    Args:
        prediction (list): Predicted one segment values
        hard_thres (int, optional): The minimum number of 'one' in a row. Defaults to 3.

    Returns:
        list: Smoothed prediction.
    """
    data = prediction.copy()
    count = 0
    start = -1
    length = len(data)
    for idx, item in enumerate(data):
        if item == 1:
            count += 1
            if start == -1:
                start = idx
            if count < hard_thres:
                if idx == length - 1:
                    return data
                elif data[idx + 1] == 0:
                    count = 0
                    data[start:idx + 1] = 0
        if item == 0:
            start = -1
            count = 0
    return data


def median_filtering_prediction(prediction: np.ndarray, med_thres: int = 5) -> np.ndarray:
    """
    Median filtering on predictions in order to change the zeros to ones,
    between of ones in a med_thres window

    Args:
        prediction (np.ndarray): Predicted video summary
        med_thres (int, optional): Window size. Defaults to 5.

    Returns:
        (np.ndarray): Medfilted predictions
    """

    return medfilt(prediction, med_thres)


def get_model(model_path: str = MODEL_DIR):
    """
    Loading Random Forest model, implemented with the imblearn module

    Args:
        model_path (str, optional): The path to the stored model.
        Defaults to const: RF_MODEL from models directory.

    Returns:
        model [type]: The imblearn Random Forest model
    """
    try:
        with open(model_path, 'rb') as saved_model:
            model = pload(saved_model)
    except:
        print("Failed to load model")
        exit()
    return model


def load_npys_to_matrices(labels: list,  audio: list, videos: list) -> tuple:
    """
    Loading the numpy files. Visual and audio will be averaged every 5 and 10 rows respectively.
    DISCLAIMER i keep the minimum number of samples between the same video file from label, video and audio features matrices.
    """
    print("Loading numpy files!")
    files_sizes = []
    labels_matrix = []
    visual_matrix = []
    audio_matrix = []
    if not len(labels) == len(videos) == len(audio):
        raise Exception(
            "Labels, visual features and audio have not the same size")
    for idx in range(len(labels)):

        # load labels, visual and audio in temporary variables
        try:

            tmp_label = np.load(labels[idx])

            # 1 if the timestamp have been annotated at least from the half annotators of that specific file
            max_annotators = int(np.ceil(np.amax(tmp_label) / 2))

            tmp_label[tmp_label < max_annotators] = 0
            tmp_label[tmp_label >= max_annotators] = 1

            tmp_visual = np.load(videos[idx])
            tmp_audio = np.load(audio[
                idx]).transpose()
            # transposed to the same format of visual features (rows = samplles, columns = features)
        except ValueError:
            print(
                f'File in index {idx} with name {videos[idx]} Failed to load')
            continue

        # get min seconds from the same label, visual, audio np file
        l_r = tmp_label.shape[0]
        v_r, v_c = tmp_visual.shape
        a_r, a_c = tmp_audio.shape

        v_r = v_r // 5
        min_seconds = min(l_r, v_r, a_r)

        files_sizes.append(min_seconds)

        labels_matrix.append(tmp_label[:min_seconds])
        # VISUAL
        # keep number of samples divisible with 5
        tmp_visual = tmp_visual[:min_seconds * 5]
        # averaging visual every 5 (Because we have analyze video with .2 step)
        visual_matrix.append(tmp_visual.transpose(
        ).reshape(-1, 5).mean(1).reshape(v_c, -1).transpose())

        tmp_audio = tmp_audio[:min_seconds]
        audio_matrix.append(tmp_audio)

        del tmp_label
        del tmp_visual
        del tmp_audio

    return files_sizes, labels_matrix, visual_matrix, audio_matrix


def split(labels: list, videos: list, audio: list, split_size: float = 0.8) -> tuple:
    """
    Splitting the data to training and testing data
    Args:
        labels (list): List of the labels
        videos (list): List of the visual features
        audio (list): List of the aural features
        split_size (float): default 0.8, The split size for training size

    Returns:
        (tuple): Training and testing tuples of labels accompanied with visual and aural features
    """
    if not len(labels) == len(videos) == len(audio):
        raise Exception(
            "Labels, visual features and audio have not the same size")
    if split_size >= 1.0 or split_size <= 0.0:
        raise Exception("Split size is out of bound")
    training_size = int(split_size * len(labels))

    return np.hstack([label for label in labels[:training_size]]), np.vstack(
        [video for video in videos[:training_size]]), np.vstack([audio for audio in audio[:training_size]]), \
        np.hstack([label for label in labels[training_size:]]), np.vstack(
        [video for video in videos[training_size:]]), np.vstack([audio for audio in audio[training_size:]])


def download_dataset():
    if not os.path.isdir(VIDEOS_DATA_DIR):
        print(f'{VIDEOS_DATA_DIR} does not exist, trying to create it')
        try:
            os.mkdir()
        except:
            assert f'An error occurred when creating the directory {VIDEOS_DATA_DIR} '

    dataset_tree = crawl_directory(DATASET)
    for classname in dataset_tree:
        try:
            print(
                f'Attempting to create {classname} directory in {VIDEOS_DATA_DIR} directory')
            video_class = os.path.join(VIDEOS_DATA_DIR, classname[-1])
            os.mkdir(video_class)
        except:
            assert f'An error occurred in {classname} directory creation'
        print(f'Downloading videos for class {classname}')
        try:
            os.system("youtube-dl -o '" + video_class + os.sep +
                      "%(uploader)s - %(title)s' -a " + classname)
        except:
            print(f'Cannot download video from {classname} class')

    return VIDEOS_DATA_DIR


def save_prediction(prediction: np.ndarray, dst: str):
    """
    Saving the prediction result from classification
    Args:
        prediction (np.ndarray): The prediction array with the significant seconds of video duration as classified
                                 from the video summarization
        dst (str): The destination directory to store the prediction array in .npy extension

    Returns:
        (bool): True If the prediction saved successfully on disk, False otherwise
    """
    return np.save(prediction, os.path.join(dst, 'prediction.npy'))


def save_model(model, dst: str):
    """
    Saving the video summarization model
    Args:
        model (): The trained model for video summarization
        dst (str): The destination directory to store the model in .pt extension

    Returns:
        (bool): True If the model saved successfully on disk, False otherwise
    """
    return pdump(model, open(os.path.join(dst, 'model.pt'), 'wb'))
