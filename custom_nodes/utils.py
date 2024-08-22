import os
import re
import folder_paths
from .settings.downloader import RVC_DOWNLOAD_LINK, download_file
from ..lib import BASE_MODELS_DIR

temp_path = folder_paths.get_temp_directory()

def model_downloader(model):
    filename = os.path.basename(model)
    subfolder = os.path.dirname(model)
    model_path = os.path.join(BASE_MODELS_DIR,subfolder,filename)
    if not os.path.isfile(model_path):
        download_link = f"{RVC_DOWNLOAD_LINK}{model}"
        params = model_path, download_link
        if download_file(params): print(f"successfully downloaded: {model_path}")
    return model_path

def increment_filename_no_overwrite(proposed_path):
    output_dir, filename_ext = os.path.split(proposed_path)
    filename, file_format = os.splitext(filename_ext)
    files = os.listdir(output_dir)
    files = filter(lambda f: os.path.isfile(os.path.join(output_dir, f)), files)
    files = filter(lambda f: f.startswith(filename), files)
    files = filter(lambda f: f.endswith('.' + file_format), files)
    file_numbers = [re.search(r'_(\d+)\.', f) for f in files]
    file_numbers = [int(f.group(1)) for f in file_numbers if f]
    this_file_number = max(file_numbers) + 1 if file_numbers else 1
    output_path = os.path.join(output_dir, filename + f'_{this_file_number}.' + file_format)
    return output_path

class MultipleTypeProxy(str):
    def __eq__(self, other: str):
        for o in other.split(","):
            if o in self: return True
        for s in self.split(","):
            if s in other: return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
# FROM: https://github.com/theUpsider/ComfyUI-Logic/blob/master/nodes.py#L11
class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False