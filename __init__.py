from .custom_nodes.audio_nodes import NODE_CLASS_MAPPINGS as audio_nodes, NODE_DISPLAY_NAME_MAPPINGS as audio_nodes_name
from .custom_nodes.stt import AudioTranscriptionNode, BatchedTranscriptionEncoderNode, LoadWhisperModelNode
from .custom_nodes.uvr import UVR5Node
from .custom_nodes.rvc import LoadHubertModel, LoadPitchExtractionParams, LoadRVCModelNode, RVCNode
from .custom_nodes.utility_nodes import NODE_CLASS_MAPPINGS as utility_nodes, NODE_DISPLAY_NAME_MAPPINGS as utility_nodes_name
from .custom_nodes.musetalk_nodes import NODE_CLASS_MAPPINGS as musetalk_nodes, NODE_DISPLAY_NAME_MAPPINGS as musetalk_nodes_name
# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./web"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "UVR5Node": UVR5Node,
    "AudioTranscriptionNode": AudioTranscriptionNode,
    "LoadWhisperModelNode": LoadWhisperModelNode,
    "LoadRVCModelNode": LoadRVCModelNode,
    "RVCNode": RVCNode,
    "LoadHubertModel": LoadHubertModel,
    "LoadPitchExtractionParams": LoadPitchExtractionParams,
    "BatchedTranscriptionEncoderNode": BatchedTranscriptionEncoderNode,
    **audio_nodes,
    **musetalk_nodes,
    **utility_nodes
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "UVR5Node": "🌺Vocal Removal",
    "AudioTranscriptionNode": "🌺Transcribe Audio",
    "LoadWhisperModelNode": "🌺Load Whisper Model",
    "LoadRVCModelNode": "🌺Load RVC Model",
    "RVCNode": "🌺Voice Changer",
    "LoadHubertModel": "🌺Load Hubert Model",
    "LoadPitchExtractionParams": "🌺Load Pitch Extraction Params",
    "BatchedTranscriptionEncoderNode": "🌺Batched CLIP Transcription Encode (Prompt)",
    **audio_nodes_name,
    **musetalk_nodes_name,
    **utility_nodes_name
}