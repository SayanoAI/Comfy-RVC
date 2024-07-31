from .custom_nodes.stt import AudioTranscriptionNode, BatchedTranscriptionEncoderNode
from .custom_nodes.uvr import UVR5Node
from .custom_nodes.rvc import RVCNode
from .custom_nodes.loaders import DownloadAudio, LoadAudio, LoadWhisperModelNode, LoadRVCModelNode, LoadHubertModel, LoadPitchExtractionParams
from .custom_nodes.output import PreviewAudio
from .custom_nodes.utils import Any2ListNode, AudioBatchValueNode, MergeImageBatches, MergeLatentBatches, ImageRepeatInterleavedNode, LatentRepeatInterleavedNode, MergeAudioNode, SimpleMathNode, SliceNode, ZipImagesNode

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./web"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "UVR5Node": UVR5Node,
    "RVC-Studio.LoadAudio": LoadAudio,
    "RVC-Studio.PreviewAudio": PreviewAudio,
    "MergeAudioNode": MergeAudioNode,
    "AudioTranscriptionNode": AudioTranscriptionNode,
    "LoadWhisperModelNode": LoadWhisperModelNode,
    "LoadRVCModelNode": LoadRVCModelNode,
    "RVCNode": RVCNode,
    "LoadHubertModel": LoadHubertModel,
    "LoadPitchExtractionParams": LoadPitchExtractionParams,
    "AudioBatchValueNode": AudioBatchValueNode,
    "MergeImageBatches": MergeImageBatches,
    "MergeLatentBatches": MergeLatentBatches,
    "ImageRepeatInterleavedNode": ImageRepeatInterleavedNode,
    "LatentRepeatInterleavedNode": LatentRepeatInterleavedNode,
    "DownloadAudio": DownloadAudio,
    "BatchedTranscriptionEncoderNode": BatchedTranscriptionEncoderNode,
    "SimpleMathNode": SimpleMathNode,
    "SliceNode": SliceNode,
    "ZipNode": ZipImagesNode,
    "Any2ListNode": Any2ListNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "UVR5Node": "🌺Vocal Removal",
    "RVC-Studio.LoadAudio": "🌺Load Audio",
    "DownloadAudio": "🌺Youtube Downloader",
    "RVC-Studio.PreviewAudio": "🌺Preview Audio",
    "AudioTranscriptionNode": "🌺Transcribe Audio",
    "LoadWhisperModelNode": "🌺Load Whisper Model",
    "LoadRVCModelNode": "🌺Load RVC Model",
    "RVCNode": "🌺Voice Changer",
    "LoadHubertModel": "🌺Load Hubert Model",
    "LoadPitchExtractionParams": "🌺Load Pitch Extraction Params",
    "MergeAudioNode": "🌺Merge Audio",
    "AudioBatchValueNode": "🌺Audio RMS Batch Values",
    "MergeImageBatches": "🌺Merge Image Batches",
    "MergeLatentBatches": "🌺Merge Latent Batches",
    "ImageRepeatInterleavedNode": "🌺Image Repeat Interleaved",
    "LatentRepeatInterleavedNode": "🌺Latent Repeat Interleaved",
    "BatchedTranscriptionEncoderNode": "🌺Batched CLIP Transcription Encode (Prompt)",
    "SimpleMathNode": "🌺Simple Math Operations",
    "SliceNode": "🌺Slice Array",
    "ZipNode": "🌺Zip Images",
    "Any2ListNode": "🌺Any to List"
}