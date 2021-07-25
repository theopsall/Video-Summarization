import os
from pickle import load as pload

import numpy as np
from scipy.signal import medfilt

# from video_summarization.libs.multimodal_movie_analysis.analyze_visual.analyze_visual import process_video
from pyAudioAnalysis import MidTermFeatures as mF
from pyAudioAnalysis import audioBasicIO as iO
from video_summarization.libs.config import AUDIO_SCALER, VISUAL_SCALER, RF_MODEL, MODEL_DIR


def video_exists(video_path) -> bool:
    """
    Checking the existance of the video file

    Args:
        video_path ([type]): [description]

    Returns:
        bool: [description]
    """
    if os.path.isfile(video_path):
        return True
    return False


def audio_isolation(video: str) -> bool:
    """
    Isolate the audio signal from a video stream
    in wav file with sampling rate= 1600 in mono channel

    Args:
        video (str): videopath

    Returns:
        bool: True if audio isolated successfully, False otherwise
    """
    command = "ffmpeg -i '{0}' -q:a 0 -ac 1 -ar 16000  -map a '{1}'".format(
        video, 'isolated_audio.wav')
    try:
        os.system(command)
        return True
    except:
        print("Audio isolation failed")
        return False


def extract_audio_features(audio: str):
    """
    Extracting Mid Term audio features

    Args:
        audio (str): Audio filepath

    Returns:
        [list]: mifTerm features
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
    if not video_exists(video_path):
        raise Exception(f'{video_path} Not Found')

    features_stats, f_names_stats, feature_matrix, f_names, \
    shot_change_times = process_video(video_path, process_mode, print_flag,
                                      online_display, save_results)
    return feature_matrix


def reshape_features(audio: list, visual: list) -> tuple:
    """
    Reshaping the features matrices. Visual features are 5 times the audio features
    caused by the 0.2 step in processing video by multimodal_movie_analysis

    Args:
        audio (list): Audial features matrix. 136 features in total
        visual (list): Visual features matrix. 88 in total

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


def scale_features(data: list, modality: str) -> list:
    """
    Scaling features

    Args:
        data (list): Fused features

    Returns:
        list: scaled features
    """
    if modality == "visual":
        scaler = pload(VISUAL_SCALER)

    if modality == "aural":
        scaler = pload(AUDIO_SCALER)

    scaled_features = scaler.transform(data)
    return scaled_features


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


def median_filtering_prediction(prediction: list, med_thres: int = 5) -> list:
    """
    Median filtering on predictions in order to change the zeros to ones,
    between of ones in a med_thres window

    Args:
        prediction (list): Predicted video summary
        med_thres (int, optional): Window size. Defaults to 5.

    Returns:
        list: Medfilted predictions
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


def make_classification():
    pass


def extract_and_make_classification():
    pass


def classify(video: str):
    isolated = audio_isolation(video)
    audial_features = extract_audio_features("isolated_audio.wav")
    visual_features = extract_video_features(video)
    reshaped_audial, reshaped_visual = reshape_features(audial_features, visual_features)

    fusion_features = fused_features(reshaped_audial, reshaped_visual)

    model = get_model()
    # scaled_features = scale_features(fusion_features)

    prediction = model.predict(fusion_features)

    medfilted = median_filtering_prediction(prediction)
    # smoothed_prediction = utils.smooth_prediction(medfilted)

    return medfilted

def save_result(result, output):
    pass