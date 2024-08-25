from .custom_nodes.audio_nodes import NODE_CLASS_MAPPINGS as audio_nodes, NODE_DISPLAY_NAME_MAPPINGS as audio_nodes_name
from .custom_nodes.stt import NODE_CLASS_MAPPINGS as stt_nodes, NODE_DISPLAY_NAME_MAPPINGS as stt_nodes_name
from .custom_nodes.uvr import UVR5Node
from .custom_nodes.rvc_nodes import NODE_CLASS_MAPPINGS as rvc_nodes, NODE_DISPLAY_NAME_MAPPINGS as rvc_nodes_name
from .custom_nodes.utility_nodes import NODE_CLASS_MAPPINGS as utility_nodes, NODE_DISPLAY_NAME_MAPPINGS as utility_nodes_name
from .custom_nodes.musetalk_nodes import NODE_CLASS_MAPPINGS as musetalk_nodes, NODE_DISPLAY_NAME_MAPPINGS as musetalk_nodes_name
# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./web"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "UVR5Node": UVR5Node,
    **rvc_nodes,
    **stt_nodes,
    **audio_nodes,
    **musetalk_nodes,
    **utility_nodes
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "UVR5Node": "🌺Vocal Removal",
    **rvc_nodes_name,
    **stt_nodes_name,
    **audio_nodes_name,
    **musetalk_nodes_name,
    **utility_nodes_name
}