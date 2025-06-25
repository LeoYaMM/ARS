# infer.py
import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# 1) Cargar modelo y procesador
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-finetuned")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-finetuned").to("cuda")

# 2) Parámetros de audio
SAMPLE_RATE = 16000  # Hz
CHUNK = int(0.5 * SAMPLE_RATE)  # 0.5 s de audio

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio = indata.squeeze().copy()
    input_vals = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to("cuda")
    with torch.no_grad():
        logits = model(input_vals).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    # Borra línea anterior y muestra transcripción
    print(f"\r{text}", end="", flush=True)

if __name__ == "__main__":
    print("Hablá ahora...")
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK, callback=callback):
        sd.sleep(int(60e3))  # mantiene abierto 60 000 ms = 1 min
