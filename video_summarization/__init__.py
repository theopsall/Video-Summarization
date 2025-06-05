from video_summarization.config import (
    AURAL_FEATURES_DIR,
    VISUAL_FEATURES_DIR,
    VIDEOS_DATA_DIR,
    AUDIO_DATA_DIR,
)
from video_summarization.utilities.utils import init_directory

if __name__ == "__main__":
    print(f"Initializing video summarization")
    init_directory(VIDEOS_DATA_DIR)
    init_directory(AUDIO_DATA_DIR)
    init_directory(VISUAL_FEATURES_DIR)
    init_directory(AURAL_FEATURES_DIR)
