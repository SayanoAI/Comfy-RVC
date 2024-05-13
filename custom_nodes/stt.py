import json
import os
from pprint import pprint

import numpy as np

from ..lib import BASE_CACHE_DIR
from .utils import increment_filename_no_overwrite
from ..lib.utils import get_hash
import folder_paths
from ..lib.audio import bytes_to_audio, remix_audio
import textacy
from textacy.extract import keyterms as kt

temp_path = folder_paths.get_temp_directory()

def extract_keywords(text: str, max_words: int, spacy_model, **kwargs):
    text = text.strip()
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

    return ", ".join(tags).strip()

def limit_sentence(text: str, max_words: int, **kwargs):
    text = text.strip()
    topn = int(max_words) if max_words>0 else len(text)
    if topn>0: return " ".join(text.split()[:topn])
    else: return text

class AudioTranscriptionNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ('TRANSCRIPTION_MODEL',),
                "audio": ('VHS_AUDIO',),
                "format_newlines_on_punctuation": ('BOOLEAN', {'default': True}),
                "loop": ('BOOLEAN', {'default': True}),
                "use_tags": ('BOOLEAN', {'default': True}),
                "max_words": ('INT', {'default': 16, "min": 0, "max": 32, "display": "slider"}),
                "frame_interpolation": ("INT", {"default": 0, "min": 0, "max": 120})
            },
            "optional": {
                "save_transcription": ('BOOLEAN', {'default': False}),
                "save_chunks": ('BOOLEAN', {'default': False}),
                "save_filename": ('STRING', {'default': 'transcription'}),
                "overwrite_existing": ('BOOLEAN', {'default': True}),
                "print_output": ('BOOLEAN', {'default': True}),
                "use_cache": ("BOOLEAN",{"default": True})
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("transcription", "batch_prompt_text", "batch_values_text", "num_chunks","num_frames")

    FUNCTION = "transcribe"

    CATEGORY = "🌺RVC-Studio/stt"

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
    def process_text_chunks(ichunk,text_processor,frame_interpolation,**kwargs):
        i,chunk=ichunk
        frame_interpolation=int(frame_interpolation)
        text = text_processor(chunk["text"],**kwargs)
        timestamp = np.nan_to_num(np.array(chunk["timestamp"],dtype=float),nan=i)
        index = int(timestamp[0]*frame_interpolation) if frame_interpolation>0 else i
        prompt = f'"{index}":"{text}"' if len(text) else f'"{index}":""'
        duration = max(abs(int(np.round(timestamp[1]-timestamp[0]))),1) # at least 1 sec duration
        if frame_interpolation>1: duration*=frame_interpolation
        value = f'"{index}":"({duration})"'
        return prompt,value
    
    @staticmethod
    def load_cache(*args):
        results = None
        cache_name = os.path.join(BASE_CACHE_DIR,"stt",f"{get_hash(*args)}.json")
        os.makedirs(os.path.dirname(cache_name), exist_ok=True)
        if os.path.isfile(cache_name): results = json.load(cache_name)
        return results, cache_name

    def transcribe(
        self, pipeline, audio,
        format_newlines_on_punctuation,loop,
        use_tags,max_words,frame_interpolation,
        save_transcription=False, save_chunks=False,
        save_filename="transcription", overwrite_existing=True,
        print_output=True,use_cache=True
    ):
        print('Starting Transcription')

        audio_data = bytes_to_audio(audio())
        if use_cache: result, cache_name = self.load_cache(pipeline,audio(),loop,use_tags,max_words,frame_interpolation)
        else: result = None

        audio_frames = int(np.ceil(len(audio_data[0])/audio_data[1]))
        whisper_model,get_spacy_model=pipeline()

        if result is None:
            audio,_ = remix_audio(audio_data,target_sr=16000,norm=True)
            result = whisper_model(audio)

        if use_cache and result:
            with open(cache_name, "w") as f:
                json.dump(result, f)

        print('Transcription Done')
        text = result["text"]
        total_chunks = result['chunks']
        if format_newlines_on_punctuation:
            punct = ['.', '?', '!']
            for p in punct:
                text = text.replace(f'{p} ', f'{p}\n')

        if loop: # append first frame to chunk stack
            last_chunk = None
            for i in range(len(total_chunks)):
                last_chunk = total_chunks[-1-i]

                if "timestamp" in last_chunk and len(last_chunk["timestamp"])>=1:
                    last_chunk["timestamp"] = np.nan_to_num(np.array(last_chunk["timestamp"],dtype=float),nan=audio_frames)
                    break
            if last_chunk is not None:
                timestamp = last_chunk["timestamp"]
                start_time = timestamp[-1]
                duration = max(1,audio_frames - start_time)
                total_chunks.append({
                    "text": total_chunks[0]["text"],
                    "timestamp": (start_time,start_time+duration)
                })
        else: # adds 1s frame to prevent audio from stopping early
            num_frames+=1

        if use_tags:
            spacy_model,text_processor = get_spacy_model(),extract_keywords
        else:
            spacy_model,text_processor = None,limit_sentence

        # split transcript into prompt based on timestamp
        # find length of each frame using timestamp
        prompts = []
        values = []
        for ichunk in enumerate(total_chunks):
            p, v = self.process_text_chunks(ichunk,text_processor,frame_interpolation,spacy_model=spacy_model,max_words=max_words)
            prompts.append(p)
            values.append(v)
        
        num_chunks = len(total_chunks)
        batch_prompt = ',\n'.join(prompts)
        batch_values = ',\n'.join(values)
        num_frames = int(audio_frames*frame_interpolation if frame_interpolation>0 else audio_frames)

        if print_output:
            pprint(total_chunks)
            print(batch_prompt)
            print(batch_values)
            print(f"{num_chunks=}, {audio_frames=}s, {num_frames=}")
            
        if save_transcription:
            self.save_output(save_filename + '.txt', text, overwrite_existing=overwrite_existing)

        if save_chunks:
            self.save_output(
                save_filename + '_prompt.json',
                json.dumps(json.loads("{"+batch_prompt+"}"),indent=2), # formats json with indent
                overwrite_existing=overwrite_existing)
            
            self.save_output(
                save_filename + '_values.json',
                json.dumps(json.loads("{"+batch_values+"}"),indent=2), # formats json with indent
                overwrite_existing=overwrite_existing)

        text = text_processor(text,max_words=max_words,spacy_model=spacy_model)
        return (text, batch_prompt, batch_values, num_chunks, num_frames)
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        print(f"{args=} {kwargs=}")
        return get_hash(*args, *kwargs.items())