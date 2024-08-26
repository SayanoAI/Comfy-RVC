import base64
import io
import os
import zlib
from .utils import get_merge_func
import numpy as np
import librosa
import soundfile as sf
import ffmpeg

MAX_INT16 = 32768
SUPPORTED_AUDIO = ["mp3","flac","wav"] # ogg breaks soundfile
OUTPUT_CHANNELS = ["mono", "stereo"]
AUTOTUNE_NOTES = np.array([
    65.41, 69.30, 73.42, 77.78, 82.41, 87.31,
    92.50, 98.00, 103.83, 110.00, 116.54, 123.47,
    130.81, 138.59, 146.83, 155.56, 164.81, 174.61,
    185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23,
    369.99, 392.00, 415.30, 440.00, 466.16, 493.88,
    523.25, 554.37, 587.33, 622.25, 659.25, 698.46,
    739.99, 783.99, 830.61, 880.00, 932.33, 987.77,
    1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91,
    1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53,
    2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83,
    2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07
])

def load_audio(file, sr, **kwargs):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as error:
        raise RuntimeError(f"Failed to load audio: {error=} {file=}")

    return remix_audio((np.frombuffer(out, np.float32).flatten(), sr),**kwargs)

def remix_audio(input_audio,target_sr=None,norm=False,to_int16=False,resample=False,axis=0,merge_type=None,max_volume=.99,**kwargs):
    audio = np.array(input_audio[0],dtype="float32")
    if target_sr is None: target_sr=input_audio[1]

    print(f"before remix: shape={audio.shape}, max={audio.max()}, min={audio.min()}, mean={audio.mean()} sr={input_audio[1]}")
    if resample or input_audio[1]!=target_sr:
        audio = librosa.resample(np.array(input_audio[0],dtype="float32"),orig_sr=input_audio[1],target_sr=target_sr,**kwargs)
    
    if audio.ndim>1:
        merge_func = get_merge_func(merge_type)
        audio=merge_func(audio,axis=axis)
    if norm: audio = librosa.util.normalize(audio,axis=axis)

    audio_max = np.abs(audio).max()/max_volume
    if audio_max > 1: audio = audio / audio_max
        
    if to_int16: audio = np.clip(audio * MAX_INT16, a_min=1-MAX_INT16, a_max=MAX_INT16-1).astype("int16")
    print(f"after remix: shape={audio.shape}, max={audio.max()}, min={audio.min()}, mean={audio.mean()}, sr={target_sr}")

    return audio, target_sr

def load_input_audio(fname,sr=None,**kwargs):
    if sr is None: sr=44100
    audio, sr = load_audio(fname, sr, **kwargs)
    print(f"loading sound {fname=} {audio.ndim=} {audio.max()=} {audio.min()=} {audio.dtype=} {sr=}")
    return audio, sr
   
def save_input_audio(fname,input_audio,sr=None,to_int16=False,to_stereo=False,max_volume=.99):
    print(f"saving sound to {fname}")
    os.makedirs(os.path.dirname(fname),exist_ok=True)
    audio=np.array(input_audio[0],dtype="float32")

    audio_max = np.abs(audio).max()/max_volume
    if audio_max > 1: audio = audio / audio_max
    if to_int16: audio = np.clip(audio * MAX_INT16, a_min=1-MAX_INT16, a_max=MAX_INT16-1)
    if to_stereo and audio.ndim<2: audio=np.stack([audio,audio],axis=-1)
    
    try:
        if audio.ndim>1 and audio.shape[0]<audio.shape[1]: audio=audio.T # soundfile expects data in frames x channels
        print(f"{audio.shape=}")
        sf.write(fname, audio.astype("int16" if np.abs(audio).max()>1 else "float32"), sr if sr else input_audio[1])
        return f"File saved to ${fname}"
    except Exception as e:
        return f"failed to save audio: {e}"
    
def audio_to_bytes(audio,sr,target_sr=None,to_int16=False,to_stereo=False,format="WAV"):
    
    with io.BytesIO() as bytes_io:
        audio=np.array(audio,dtype="float32")

        if to_int16:
            audio_max = np.abs(audio).max()/.99
            if audio_max > 1: audio = audio / audio_max
            audio = np.clip(audio * MAX_INT16, a_min=-MAX_INT16+1, a_max=MAX_INT16-1)

        if to_stereo and audio.ndim<2: audio=np.stack([audio,audio],axis=-1)

        if audio.ndim>1 and audio.shape[0]<audio.shape[1]: audio=audio.T # soundfile expects data in frames x channels
        print(f"{audio.shape=}")
        samplerate = sr if target_sr is None else target_sr
        sf.write(bytes_io, audio.astype("int16" if np.abs(audio).max()>1 else "float32"), samplerate=samplerate,format=format)
        bytes_io.seek(0)
        return bytes_io.read()

def bytes_to_audio(data: bytes,**kwargs):
    with io.BytesIO(data) as bytes_io:
        audio, sr = sf.read(bytes_io,**kwargs)
        if audio.ndim>1:
            if audio.shape[-1]<audio.shape[0]: # is channel-last format
                audio = audio.T # transpose to channels-first
        return audio, sr

def bytes2audio(data: str):
    try:
        # Split the suffixed data by the colon
        dtype,data,shape,sr = data.split(":")

        # Get the data, the dtype, and the shape from the split data
        shape = tuple(map(int, shape.split(",")))
        sr=int(sr)

        # Decode the data using base64
        decoded_data = base64.b64decode(data)

        # Decompress the decoded data using zlib
        decompressed_data = zlib.decompress(decoded_data)

        # Convert the decompressed data to a numpy array with the given dtype
        arr = np.frombuffer(decompressed_data, dtype=dtype)

        # Reshape the array to the original shape
        arr = arr.reshape(shape)
        return arr, sr
    except Exception as e:
        print(e)
    return None

def audio2bytes(audio: np.array, sr: int):
    try:
        # Get the dtype, the shape, and the data of the array
        dtype = audio.dtype.name
        shape = audio.shape
        data = audio.tobytes()

        # Compress the data using zlib
        compressed_data = zlib.compress(data)

        # Encode the compressed data using base64
        encoded_data = base64.b64encode(compressed_data)

        # Add a suffix with the dtype and the shape to the encoded data
        suffixed_data = ":".join([dtype,encoded_data.decode(),",".join(map(str, shape)),str(sr)])
        return suffixed_data
    except Exception as e:
        print(e)
    return ""

def pad_audio(*audios,axis=0):
    maxlen = max(len(a) if a is not None else 0 for a in audios)
    if maxlen>0:
        stack = librosa.util.stack([librosa.util.fix_length(a,size=maxlen) for a in audios if a is not None],axis=axis)
        return stack
    else: return np.stack(audios,axis=axis)

def merge_audio(audio1,audio2,sr=40000,**kwargs):
    print(f"merging audio audio1={audio1[0].shape,audio1[1]} audio2={audio2[0].shape,audio2[1]} sr={sr}")
    if sr is None: sr=min(audio1[-1],audio2[-1])
    m1,_=remix_audio(audio1,target_sr=sr,axis=0,**kwargs)
    m2,_=remix_audio(audio2,target_sr=sr,axis=0,**kwargs)
    
    mixed = pad_audio(m1,m2,axis=0)

    return remix_audio((mixed,sr),axis=0,**kwargs)

def autotune_f0(f0, threshold=0.):
    # autotuned_f0 = []
    # for freq in f0:
        # closest_notes = [x for x in self.note_dict if abs(x - freq) == min(abs(n - freq) for n in self.note_dict)]
        # autotuned_f0.append(random.choice(closest_notes))
    # for note in self.note_dict:
    #     closest_notes = np.where((f0 - note)/note<.05,f0,note)
    print("autotuning f0 using note_dict...")

    autotuned_f0 = []
    # Loop through each value in array1
    for freq in f0:
        # Find the absolute difference between x and each value in array2
        diff = np.abs(AUTOTUNE_NOTES - freq)
        # Find the index of the minimum difference
        idx = np.argmin(diff)
        # Find the corresponding value in array2
        y = AUTOTUNE_NOTES[idx]
        # Check if the difference is less than threshold
        if diff[idx] < threshold:
            # Keep the value in array1
            autotuned_f0.append(freq)
        else:
            # Use the nearest value in array2
            autotuned_f0.append(y)
    # Return the result as a numpy array
    return np.array(autotuned_f0, dtype="float32")