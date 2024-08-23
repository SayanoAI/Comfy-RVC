import json
import os
from pprint import pprint
import subprocess
import sys
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import torch

from .settings import SUPPORTED_LANGUAGES

from ..lib import BASE_CACHE_DIR, BASE_MODELS_DIR
from .utils import increment_filename_no_overwrite
from ..lib.utils import get_hash, get_optimal_torch_device
import folder_paths
from ..lib.audio import bytes_to_audio, remix_audio
import spacy

SPACY_LANGUAGE_MODEL_MAP = {
    "en": "en_core_web_md",
    "fr": "fr_core_news_md",
    "es": "es_core_news_md",
    "ja": "ja_core_news_md",
    "zh": "zh_core_web_md",
}

temp_path = folder_paths.get_temp_directory()
CATEGORY = "🌺RVC-Studio/stt"

def extract_keywords(text: str, max_words: int, spacy_model, prefix="", suffix="", weights=1., **kwargs):
    
    doc = spacy_model(text)
    text = doc._.text.strip().replace('"','')
    sentiment = doc._.sentiment
    topn = int(max_words) if max_words>0 else len(text)
    include_pos = ["NOUN","ADJ","PROPN","VERB","NUM","ADP"]
    ngrams = [1,2]
    tags = []
    try:
        from textacy.extract import keyterms as kt
        terms = kt.sgrank(doc, ngrams=ngrams, normalize="lower", topn=topn, include_pos=include_pos)
        tags = list(map(lambda v:v[0],sorted(terms,key=lambda v:v[1],reverse=True)[:topn]))
    except Exception as error:
        print(f"{text=} {error=}")

    tags = ", ".join(tags)
    if len(tags) and weights!=1.: tags = f"({tags}:{weights:.3f})"
    return ", ".join(filter(None,[prefix,tags,sentiment,suffix])).strip()

def limit_sentence(text: str, max_words: int, spacy_model, prefix="", suffix="", weights=1., **kwargs):
    doc = spacy_model(text)
    text = doc._.text.strip().replace('"','')
    sentiment = doc._.sentiment
    topn = int(max_words) if max_words>0 else len(text)
    if topn>0: text = " ".join(text.split()[:topn])
    if len(text) and weights!=1.: text = f"({text}:{weights:.3f})"
        
    return ", ".join(filter(None,[prefix,text,sentiment,suffix])).strip()

def init_spacy_model(language="en",use_sentiment=False):
    model_name = SPACY_LANGUAGE_MODEL_MAP.get(language,"en_core_web_md")
    model_path = os.path.join(BASE_MODELS_DIR,model_name)
    
    if not os.path.exists(model_path):
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        model = spacy.load(model_name)
        model.to_disk(model_path)

    model = spacy.load(model_path)

    if language=="en": # default text
        @spacy.Language.component("default_transformer")
        def _(doc):
            doc._.text = doc.text
            return doc
        model.add_pipe("default_transformer", last=True)
    else: # google translated text
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='en')
        @spacy.Language.component("GoogleTranslator")
        def _(doc):
            doc._.text = translator.translate(doc.text) if doc.text else ""
            return doc
        model.add_pipe("GoogleTranslator", last=True)
    if use_sentiment:
        from spacytextblob.spacytextblob import SpacyTextBlob
        @spacy.Language.component("SpacyTextBlobSentiment")
        def _(doc):
            polarity = SpacyTextBlob(model).get_polarity(doc)
            if polarity<-.5: sentiment="sad, tears, crying"
            elif polarity<-.05: sentiment="sad, tears"
            elif polarity>.5: sentiment="happy, smile, laughing"
            elif polarity>.05: sentiment="slight smile"
            else: sentiment=""
            doc._.sentiment = sentiment
            return doc
        model.add_pipe("SpacyTextBlobSentiment", last=True)

    # Register the custom attributes with spaCy
    spacy.tokens.Doc.set_extension("text", default="", force=True)
    spacy.tokens.Doc.set_extension("sentiment", default="", force=True)

    return model

   
class LoadWhisperModelNode:

    @classmethod
    def INPUT_TYPES(cls):
        model_ids = [
            'openai/whisper-large-v3',
            'openai/whisper-large-v2',
            'openai/whisper-large',
            'openai/whisper-medium',
            'openai/whisper-small',
            'openai/whisper-base',
            'openai/whisper-tiny',
            'openai/whisper-medium.en',
            'openai/whisper-small.en',
            'openai/whisper-base.en',
            'openai/whisper-tiny.en',
        ]
        return {
            'required': {
                'model_id': (model_ids,{"default": "openai/whisper-base.en"}),
            },
            "optional": {
                "max_new_tokens": ("INT", {"default": 128, "min": 16, "max": 1024, "display": "slider"}),
                "chunk_length_s": ("INT", {"default": 30, "min": 15, "max": 60, "display": "slider"}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 128, "display": "slider"}),
            }
        }

    RETURN_TYPES = ('TRANSCRIPTION_MODEL', )
    RETURN_NAMES = ('model', )

    CATEGORY = CATEGORY

    FUNCTION = 'load_model'


    def load_model(self, model_id, max_new_tokens=128, chunk_length_s=12, batch_size=16):
        device = get_optimal_torch_device()
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model.to(device)

        # generate_kwargs = {}

        def pipe(): return pipeline(
                'automatic-speech-recognition',
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=max_new_tokens,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                return_timestamps=True,
                torch_dtype=torch_dtype,
                device=device,
            )
        return ([pipe,model_id], )
    
    @classmethod
    def IS_CHANGED(cls, model_id, max_new_tokens, chunk_length_s, batch_size):
        return get_hash(model_id, max_new_tokens, chunk_length_s, batch_size)

class AudioTranscriptionNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ('TRANSCRIPTION_MODEL',),
                "audio": ('VHS_AUDIO',)
            },
            "optional": {
                "save_filename": ('STRING', {'default': ''}),
                "overwrite_existing": ('BOOLEAN', {'default': True}),
                "print_output": ('BOOLEAN', {'default': True}),
                "use_cache": ("BOOLEAN",{"default": True})
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("TRANSCRIPTION", "INT")
    RETURN_NAMES = ("transcription", "audio_frames")

    FUNCTION = "transcribe"

    CATEGORY = CATEGORY

    @staticmethod
    def save_output(filename,text,overwrite_existing=False):
        base_output_dir = folder_paths.get_output_directory()
        assert os.path.exists(base_output_dir), f"Output directory {base_output_dir} does not exist"
        output_dir = os.path.join(base_output_dir, 'transcriptions')
        
        file_path = os.path.join(output_dir, filename)
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        if os.path.exists(file_path) and not overwrite_existing:
            file_path = increment_filename_no_overwrite(file_path)
        with open(file_path, 'w') as f:
            f.write(text)
    
    @staticmethod
    def load_cache(*args,save_filename=""):
        results = None
        if not save_filename: save_filename=get_hash(*args)
        cache_name = os.path.join(BASE_CACHE_DIR,"stt",f"{save_filename}.json")
        
        if os.path.isfile(cache_name):
            with open(cache_name,"r") as f:
                results = json.load(f)
        return results, cache_name

    def transcribe(self, pipeline, audio,save_filename="", overwrite_existing=True,print_output=True,use_cache=True):
        print('Starting Transcription')
        audio_data = bytes_to_audio(audio())
        whisper_model,model_id=pipeline
        if use_cache: transcription, cache_name = self.load_cache(model_id,audio(),save_filename=save_filename)
        else: transcription = None

        audio_frames = int(np.ceil(len(audio_data[0])/audio_data[1]))

        if transcription is None:
            audio,_ = remix_audio(audio_data,target_sr=16000,norm=True)
            transcription = whisper_model()(audio)

        if use_cache and transcription is not None:
            self.save_output(cache_name, json.dumps(transcription, indent=2), overwrite_existing=overwrite_existing)

        print('Transcription Done')
        

        if print_output:
            pprint(transcription)
            print(f"{audio_frames=}s")

        return (transcription, audio_frames)

class BatchedTranscriptionEncoderNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "transcription": ('TRANSCRIPTION',),
                "clip": ('CLIP',)
            },
            "optional": {
                "loop": ('BOOLEAN', {'default': False}),
                "use_tags": ('BOOLEAN', {'default': False}),
                "use_sentiment": ('BOOLEAN', {'default': False}),
                'language': (SUPPORTED_LANGUAGES,{"default": "en"}),
                "max_chunks": ('INT', {"min": 2, "default": None}),
                "max_words": ('INT', {'default': 16, "min": 0, "max": 32, "display": "slider"}),
                "frame_interpolation": ("INT", {"default": 0, "min": 0, "max": 120, "hidden": True}), # needs more testing
                "prefix": ("STRING", {"default": "masterpiece, best quality", "multiline": True, "forceInput": True}),
                "suffix": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "print_output": ('BOOLEAN', {'default': True}),
                "weights": ("FLOAT",{"default": 1., "step": .01}),
                "pad_frames": ("INT", {"default": 0})
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("CONDITIONING", "STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "batch_prompt_text", "duration_list", "num_chunks", "num_frames", "prompt_text_list")
    OUTPUT_IS_LIST = (False, False, False, False, False, True)

    FUNCTION = "get_prompt"

    CATEGORY = CATEGORY

    @staticmethod
    def process_text_chunks(ichunk,frame_interpolation,clip,*,use_tags,**kwargs):
        i,chunk=ichunk
        frame_interpolation=int(frame_interpolation)
        textProcessor = extract_keywords if use_tags else limit_sentence
        text = textProcessor(chunk["text"],**kwargs)
        timestamp = np.nan_to_num(np.array(chunk["timestamp"],dtype=float),nan=i*frame_interpolation)
        duration = max(timestamp[1]-timestamp[0],1) # at least 1 sec duration
        if frame_interpolation>1: duration*=frame_interpolation
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        return text, duration, cond, pooled

    def get_prompt(
        self, transcription, clip, language="en", loop=False, use_tags=False, use_sentiment=False,
        max_words=16,max_chunks=None, frame_interpolation=0, print_output=True, prefix="", suffix="",
        weights=1., pad_frames=0
    ):

        if not max_chunks: max_chunks = len(transcription['chunks'])
        total_chunks = transcription['chunks'][:max_chunks]
        max_frames = max(max_chunks, *filter(None,np.array([chunk["timestamp"] for chunk in total_chunks]).flatten())) + pad_frames

        last_chunk = None
        for i in range(len(total_chunks)):
            last_chunk = total_chunks[-1-i]

            if "timestamp" in last_chunk and len(last_chunk["timestamp"])>=1:
                last_chunk["timestamp"] = np.nan_to_num(np.array(last_chunk["timestamp"],dtype=float),nan=max_frames)
                break

        if last_chunk is not None:
            timestamp = last_chunk["timestamp"]
            start_time = timestamp[-1 if loop else 0]
            end_time = start_time + max(max_frames - start_time,0)
            timestamp = (start_time,end_time)
            last_chunk = dict(timestamp=timestamp,text=total_chunks[0 if loop else -1]["text"])

            if loop: # append first frame to chunk stack
                total_chunks.append(last_chunk)
            else: # replace final frame
                total_chunks[-1] = last_chunk

        spacy_model = init_spacy_model(language,use_sentiment=use_sentiment)

        # split transcript into prompt based on timestamp
        # find length of each frame using timestamp
        text_list = []
        duration_list = []
        cond = []
        pooled = []
        for ichunk in enumerate(total_chunks):
            t, d, c, pc = self.process_text_chunks(ichunk,frame_interpolation,clip,
                                                   use_tags=use_tags,
                                                   weights=weights,
                                                   prefix=prefix,
                                                   suffix=suffix,
                                                   spacy_model=spacy_model,
                                                   max_words=max_words)
            text_list.append(t)
            duration_list.append(d)
            cond.append(c.squeeze())
            pooled.append(pc.squeeze())
        
        num_chunks = len(total_chunks)
        duration_list = np.round(duration_list)
        num_frames = int(np.sum(duration_list))
        final_pooled_output = torch.nested.to_padded_tensor(torch.nested.nested_tensor(pooled, dtype=torch.float32),0)
        final_conditioning = torch.nested.to_padded_tensor(torch.nested.nested_tensor(cond, dtype=torch.float32),0)
        print(f"{final_conditioning.shape=} {final_pooled_output.shape=}")
        conditioning = [[final_conditioning,{"pooled_output": final_pooled_output}]]
        cumsum = [0,*np.cumsum(duration_list)]
        batch_prompt_text = [f'"{int(cumsum[i])}": "{text}"' for i,text in enumerate(text_list)]
        batch_prompt_text = ",\n".join(batch_prompt_text)
        del pooled, cond, spacy_model

        if print_output:
            print(f"{batch_prompt_text=}")
            print(f"{duration_list=}")
            print(f"{num_chunks=}, {max_chunks=}, {num_frames=}")

        return (conditioning, batch_prompt_text, list(map(int,duration_list)), num_chunks, num_frames, text_list)
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "AudioTranscriptionNode": AudioTranscriptionNode,
    "LoadWhisperModelNode": LoadWhisperModelNode,
    "BatchedTranscriptionEncoderNode": BatchedTranscriptionEncoderNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioTranscriptionNode": "🌺Transcribe Audio",
    "LoadWhisperModelNode": "🌺Load Whisper Model",
    "BatchedTranscriptionEncoderNode": "🌺Batched CLIP Transcription Encode (Prompt)",
}