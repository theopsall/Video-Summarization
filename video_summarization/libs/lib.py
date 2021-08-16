import numpy as np
import pickle
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from video_summarization.libs import utils
from video_summarization.libs.config import MODEL_URL


def make_classification(audio_features, visual_features, labels, output):
    labels_tree = utils.crawl_directory(labels)
    videos_tree = utils.crawl_directory(visual_features)
    audio_tree = utils.crawl_directory(audio_features)
    labels_tree.sort()
    videos_tree.sort()
    audio_tree.sort()
    labels_tree, audio_tree, videos_tree = utils.shuffle_lists(labels_tree, audio_tree, videos_tree)
    files_sizes, labels_matrix, visual_matrix, audio_matrix = utils.load_npys_to_matrices()
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

    pickle.dump(fusion_model, open("fusion_RF.pt", 'wb'))


def extract_and_make_classification(videos, labels, output):
    pass


def classify(video: str):
    utils.download_model(MODEL_URL)
    isolated = utils.audio_isolation(video)
    audial_features = utils.extract_audio_features("isolated_audio.wav")
    visual_features = utils.extract_video_features(video)
    reshaped_audial, reshaped_visual = utils.reshape_features(audial_features, visual_features)

    fusion_features = utils.fused_features(reshaped_audial, reshaped_visual)

    model = utils.get_model()
    # scaled_features = scale_features(fusion_features)

    prediction = model.predict(fusion_features)

    medfilted = utils.median_filtering_prediction(prediction)
    # smoothed_prediction = utils.smooth_prediction(medfilted)

    return medfilted
