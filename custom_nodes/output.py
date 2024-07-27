import os
import shutil

from ..lib.audio import SUPPORTED_AUDIO, audio_to_bytes, bytes_to_audio, save_input_audio
from ..lib.utils import get_hash
from .utils import increment_filename_no_overwrite
import folder_paths
  
temp_path = folder_paths.get_temp_directory()

class PreviewAudio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("VHS_AUDIO",),
                "filename": ("STRING",{"default": "test"}),
                "save_format": (SUPPORTED_AUDIO,{"default": "flac"},),
                "save_channels": ([1,2],{"default": 1}),
                "overwrite_existing": ("BOOLEAN", {"default": True}),
                "autoplay": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING","VHS_AUDIO")
    RETURN_NAMES = ("output_path","vhs_audio")

    OUTPUT_NODE = True

    CATEGORY = "🌺RVC-Studio/output"

    FUNCTION = "save_audio"

    def save_audio(self, audio, filename, save_format, save_channels, overwrite_existing, autoplay):

        filename = filename.strip()
        assert filename, "Filename cannot be empty"

        base_output_dir = folder_paths.get_output_directory()
        assert os.path.exists(base_output_dir), f"Output directory {base_output_dir} does not exist"
        output_dir = os.path.join(base_output_dir, 'audio')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{filename}.{save_format}")

        if os.path.isfile(output_path) and not overwrite_existing:
            output_path = increment_filename_no_overwrite(output_path)
        
        input_audio = bytes_to_audio(audio())
        print(save_input_audio(output_path,input_audio,to_int16=True,to_stereo=save_channels==2))

        tempdir = os.path.join(temp_path,"preview")
        os.makedirs(tempdir, exist_ok=True)
        widgetId = get_hash(output_path,save_channels)
        audio_name = f"{widgetId}.{save_format}"
        preview_file = os.path.join(tempdir,audio_name)
        shutil.copyfile(output_path,preview_file)
        return {"ui": {"preview": [{
            "filename": audio_name, "type": "temp", "subfolder": "preview", "widgetId": widgetId, "autoplay": autoplay
            }]}, "result": (output_path, lambda:audio_to_bytes(*input_audio))}
    