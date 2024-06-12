
from .custom_nodes.settings.downloader import download_ffmpeg
import subprocess

if __name__=="__main__":
    try:
        subprocess.check_call("ffmpeg -version")
    except Exception as error:
        print(f"Unexpected error occured: {error}")
        # downloads ffmpeg
        download_ffmpeg()