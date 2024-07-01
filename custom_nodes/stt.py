import json
import os
from pprint import pprint
import subprocess
import sys

import numpy as np
import torch

from .settings import SUPPORTED_LANGUAGES

from ..lib import BASE_CACHE_DIR, BASE_MODELS_DIR
from .utils import increment_filename_no_overwrite
from ..lib.utils import get_hash
import folder_paths
from ..lib.audio import bytes_to_audio, remix_audio
import textacy
from textacy.extract import keyterms as kt

temp_path = folder_paths.get_temp_directory()
CATEGORY = "🌺RVC-Studio/stt"

def extract_keywords(text: str, max_words: int, spacy_model, use_sentiment=False, prefix="", suffix="", **kwargs):
    text = text.strip()
    sentiment = ""
    if use_sentiment:
        from spacytextblob.spacytextblob import SpacyTextBlob
        doc = spacy_model(text)
        polarity = SpacyTextBlob(spacy_model).get_polarity(doc)
        if polarity<-.5: sentiment="sad, tears"
        elif polarity<-.2: sentiment="sad"
        elif polarity>.5: sentiment="happy, smile"
        elif polarity>.2: sentiment="slight smile"
    doc = textacy.make_spacy_doc(text, lang=spacy_model)
    topn = int(max_words) if max_words>0 else len(text)
    include_pos = ["NOUN","ADJ","PROPN","VERB","NUM","ADP"]
    ngrams = [1,2]
    tags = []
    try:
        terms = kt.sgrank(doc, ngrams=ngrams, normalize="lower", topn=topn, include_pos=include_pos)
        tags = list(map(lambda v:v[0],sorted(terms,key=lambda v:v[1],reverse=True)[:topn]))
    except Exception as error:
        print(f"{text=} {error=}")

    return ", ".join(filter(None,[prefix,*tags,sentiment,suffix])).strip()

def limit_sentence(text: str, max_words: int, spacy_model, use_sentiment=False, prefix="", suffix="", **kwargs):
    text = text.strip()
    sentiment = ""
    if use_sentiment:
        from spacytextblob.spacytextblob import SpacyTextBlob
        doc = spacy_model(text)
        polarity = SpacyTextBlob(spacy_model).get_polarity(doc)
        if polarity<-.5: sentiment="sad, tears"
        elif polarity<-.2: sentiment="sad"
        elif polarity>.5: sentiment="happy, smile"
        elif polarity>.2: sentiment="slight smile"
    topn = int(max_words) if max_words>0 else len(text)
    if topn>0: text = " ".join(text.split()[:topn])
        
    return ", ".join(filter(None,[prefix,text,sentiment,suffix])).strip()

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
        if save_filename: save_filename=get_hash(*args)
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
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("CONDITIONING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("conditioning", "batch_prompt_text", "duration_list", "num_chunks", "num_frames")

    FUNCTION = "get_prompt"

    CATEGORY = CATEGORY

    @staticmethod
    def process_text_chunks(ichunk,frame_interpolation,clip,*,use_tags,**kwargs):
        i,chunk=ichunk
        frame_interpolation=int(frame_interpolation)
        index = i*frame_interpolation if frame_interpolation!=0 else i
        text_processor = extract_keywords if use_tags else limit_sentence
        text = text_processor(chunk["text"],**kwargs)
        timestamp = np.nan_to_num(np.array(chunk["timestamp"],dtype=float),nan=i)
        duration = max(abs(int(np.round(timestamp[1]-timestamp[0]))),1) # at least 1 sec duration
        if frame_interpolation>1: duration*=frame_interpolation
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        return f'"{index}": "{text}"', duration, cond, pooled
    
    @staticmethod
    def get_spacy_model(language="en"):
        import spacy
        spacy_model_map = {
            "en": "en_core_web_md",
            "fr": "fr_core_news_md",
            "es": "es_core_news_md",
            "ja": "ja_core_news_md",
            "zh": "zh_core_web_md",
        }
        model_name = spacy_model_map[language]
        model_path = os.path.join(BASE_MODELS_DIR,model_name)
        
        if not os.path.exists(model_path):
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            model = spacy.load(model_name)
            model.to_disk(model_path)

        model = spacy.load(model_path)

        return model

    def get_prompt(
        self, transcription, clip, language="en", loop=False, use_tags=False, use_sentiment=False,
        max_words=16,max_chunks=None, frame_interpolation=0, print_output=True, prefix="", suffix=""
    ):

        if not max_chunks: max_chunks = len(transcription['chunks'])
        total_chunks = transcription['chunks'][:max_chunks]
        num_frames = max(max_chunks, *filter(None,np.array([chunk["timestamp"] for chunk in transcription['chunks']]).flatten()))

        if loop: # append first frame to chunk stack
            last_chunk = None
            for i in range(len(total_chunks)):
                last_chunk = total_chunks[-1-i]

                if "timestamp" in last_chunk and len(last_chunk["timestamp"])>=1:
                    last_chunk["timestamp"] = np.nan_to_num(np.array(last_chunk["timestamp"],dtype=float),nan=num_frames)
                    break

            if last_chunk is not None:
                timestamp = last_chunk["timestamp"]
                if len(timestamp)==1:
                    start_time = timestamp[0]
                    end_time = start_time + abs(num_frames - start_time)+1
                else:
                    start_time = timestamp[-1]
                    end_time = start_time+1
                total_chunks.append({
                    "text": total_chunks[0]["text"],
                    "timestamp": (start_time,end_time)
                })
        else: # adds 1s frame to prevent audio from stopping early
            num_frames+=1

        spacy_model = self.get_spacy_model(language)

        # split transcript into prompt based on timestamp
        # find length of each frame using timestamp
        batch_prompt_text = []
        duration_list = []
        cond = []
        pooled = []
        for ichunk in enumerate(total_chunks):
            t, d, c, pc = self.process_text_chunks(ichunk,frame_interpolation,clip,
                                                   use_tags=use_tags,
                                                   use_sentiment=use_sentiment,
                                                   prefix=prefix,
                                                   suffix=suffix,
                                                   spacy_model=spacy_model,
                                                   max_words=max_words)
            batch_prompt_text.append(t)
            duration_list.append(d)
            cond.append(c.squeeze())
            pooled.append(pc.squeeze())
        
        num_chunks = len(total_chunks)
        final_pooled_output = torch.nested.to_padded_tensor(torch.nested.nested_tensor(pooled, dtype=torch.float32),0)
        final_conditioning = torch.nested.to_padded_tensor(torch.nested.nested_tensor(cond, dtype=torch.float32),0)
        print(f"{final_conditioning.shape=} {final_pooled_output.shape=}")
        conditioning = [[final_conditioning,{"pooled_output": final_pooled_output}]]
        batch_prompt_text = ",\n".join(batch_prompt_text)
        del pooled, cond

        if print_output:
            print(f"{batch_prompt_text=}")
            print(f"{duration_list=}")
            print(f"{num_chunks=}, {max_chunks=}, {num_frames=}")

        return (conditioning, batch_prompt_text, duration_list, num_chunks, num_frames)