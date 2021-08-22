import os
import pickle

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from video_summarization.libs import utils


def make_classification(aural_dir: str, visual_dir: str, labels_dir: str, destination):
    """
    Classification of video summarization using already extracted features and labels
    Args:
        aural_dir (str): Aural directory with npys files
        visual_dir (str): Visual directory with npys files
        labels_dir (str): Labels directory with npys files
        destination:

    Returns:

    """
    aural_tree = utils.crawl_directory(aural_dir)
    visual_tree = utils.crawl_directory(visual_dir)
    labels_tree = utils.crawl_directory(labels_dir)

    aural_tree.sort()
    visual_tree.sort()
    labels_tree.sort()

    labels_tree, aural_tree, visual_tree = utils.shuffle_lists(labels_tree, aural_tree, visual_tree)
    files_sizes, labels_matrix, visual_matrix, audio_matrix = utils.load_npys_to_matrices(labels_tree, aural_tree,
                                                                                          visual_tree)
    train_labels, train_visual, train_audio, test_labels, test_visual, test_audio = utils.split(labels_matrix,
                                                                                                visual_matrix,
                                                                                                audio_matrix, 0.8)
    audio_scaler = StandardScaler()
    train_audio = audio_scaler.fit_transform(train_audio)

    test_audio = audio_scaler.transform(test_audio)

    visual_scaler = StandardScaler()
    train_visual = visual_scaler.fit_transform(train_visual)

    test_visual = visual_scaler.transform(test_visual)

    train_union = np.concatenate((train_audio, train_visual), axis=1)
    test_union = np.concatenate((test_audio, test_visual), axis=1)
    fusion_model = BalancedRandomForestClassifier(criterion='gini', n_estimators=400, class_weight="balanced_subsample",
                                                  random_state=42)
    fusion_model.fit(train_union, train_labels)
    preds = fusion_model.predict(test_union)
    print(f"--> F1: Random Forest on Fused features is: {f1_score(test_labels, preds, average='macro') * 100:.2f} %")
    preds_prob = fusion_model.predict_proba(test_union)
    print(
        f"--> ROC AUC: Random Forest on Fused features is: {roc_auc_score(test_labels, preds_prob[:, 1]) * 100:.2f} %")

    pickle.dump(fusion_model, open(os.path.join(destination, "fusion_RF.pt"), 'wb'))


def features_extraction(videos_dir: str, destination:str):
    """
    Feature Extraction
    Args:
        videos_dir (str): Videos directory
        destination (str): Destination directory to store the features

    Returns:

    """
    print("Feature Extraction process Completed")



def extract_and_make_classification(videos_dir: str, labels_dir: str, destination: str):
    # features_extraction()
    # make_classification(aural_dir, visual_dir, labels_dir, destination)
    print("Feature Extraction and Classification processes Completed")


def classify(video: str) -> np.ndarray:
    """
    Make prediction of the interesting seconds of one video
    Args:
        video (str): Video's path

    Returns:

    """
    if not utils.video_exists(video):
        assert f"Video: {video} not Found"

    _audio = utils.audio_isolation(video)
    audial_features = utils.extract_audio_features("isolated_audio.wav")
    visual_features = utils.extract_video_features(video)
    reshaped_audial, reshaped_visual = utils.reshape_features(audial_features, visual_features)

    fusion_features = utils.fused_features(reshaped_audial, reshaped_visual)

    model = utils.get_model()
    # scaled_features = scale_features(fusion_features)

    prediction = model.predict(fusion_features)

    medfilted = utils.median_filtering_prediction(prediction)
    smoothed_prediction = utils.smooth_prediction(medfilted)

    return smoothed_prediction
