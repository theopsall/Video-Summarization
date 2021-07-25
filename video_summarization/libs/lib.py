from video_summarization.libs.utils import download_model, audio_isolation, extract_audio_features, \
    extract_video_features, fused_features, get_model, median_filtering_prediction, reshape_features
from video_summarization.libs.config import MODEL_URL


def make_classification():
    pass


def extract_and_make_classification():
    pass


def classify(video: str):
    download_model(MODEL_URL)
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
