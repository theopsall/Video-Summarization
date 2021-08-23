import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'webm'}
MODEL_URL = "https://drive.google.com/file/d/1Dwp4jYz3rLt6hgCClROjrf2ddSO9Xz0h/view?usp=sharing"
MODEL_DIR = os.path.join(PACKAGE_DIR, 'model')
RF_MODEL = os.path.join(PACKAGE_DIR, 'model', 'rf_model.pt')
AUDIO_SCALER = os.path.join(PACKAGE_DIR, 'scalers', 'aural_scaler.p')
VISUAL_SCALER = os.path.join(PACKAGE_DIR, 'scalers', 'visual_scaler.p')
DATASET = os.path.join(PACKAGE_DIR, 'dataset')
