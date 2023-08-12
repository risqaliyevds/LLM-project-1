import subprocess

import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import WHITESPACE_HANDLER
from transformers import pipeline
from settings import settings
import torchaudio


async def create_wav(audio_file):
    wav_audio_path = audio_file.replace(audio_file.split(".")[-1], '.wav')
    subprocess.run(['ffmpeg', '-i', audio_file, wav_audio_path])
    return wav_audio_path


async def speech2text(audio_file):
    if not audio_file.endswith(".wav"):
        audio_file = await create_wav()

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as audio_file:
        audio = recognizer.record(audio_file)
        aligned_transcript = recognizer.recognize_google(audio, language=settings.LANGUAGE)

    # waveform, sample_rate = torchaudio.load(audio_path)
    # pipe = pipeline(settings.STT_PROCESSOR, model=settings.SUMMARIZER_MODEL)
    # numpy_waveform = waveform.numpy()
    # aligned_transcript = pipe(numpy_waveform)

    return aligned_transcript


async def summerizer(aligned_transcript):
    model_name = settings.SUMMARIZER_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(
        [WHITESPACE_HANDLER(aligned_transcript)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512)["input_ids"]
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]
    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return summary


async def STT_with_Summary(audio_file):
    aligned_transcript = await speech2text(audio_file)
    summary = await summerizer(aligned_transcript)
    return aligned_transcript, summary
