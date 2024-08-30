MENU_ITEMS = {
    "Get help": "https://github.com/SayanoAI/RVC-Studio/discussions",
    "Report a Bug": "https://github.com/SayanoAI/RVC-Studio/issues",
    "About": """This project provides a comprehensive platform for training RVC models and generating AI voice covers.
    Check out this github for more info: https://github.com/SayanoAI/RVC-Studio
    """
}

DEVICE_OPTIONS = ["cpu","cuda"]
PITCH_EXTRACTION_OPTIONS = ["crepe","mangio-crepe","rmvpe","rmvpe+"]
MERGE_OPTIONS=["median","mean","min","max"]
TTS_MODELS = ["edge","speecht5"]
N_THREADS_OPTIONS=[1,2,4,8,12,16]
SUPPORTED_LANGUAGES = ['en', 'fr', 'es', "ja", "zh"]