# train.py

import os
import torch
from datasets import Dataset, Audio
from evaluate import load as load_metric
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments
)

# ----------------------------
# 1) Parámetros de usuario
# ----------------------------
DATA_ROOT = "/ruta/a/LibriSpeech"      # carpeta que contiene train-clean-100, dev-clean, ...
OUTPUT_DIR = "./wav2vec2-finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hiperparámetros adaptados a RTX 2060 6GB
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE  = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE    = 3e-5
NUM_EPOCHS       = 3
WARMUP_STEPS     = 500

# ----------------------------
# 2) Función para cargar LibriSpeech local
# ----------------------------
def load_librispeech_split(split: str):
    """
    Crea un Dataset Hugging Face para el split indicado:
      - 'train': usa train-clean-100
      - 'validation': usa dev-clean
    Recorre los archivos .txt de transcripciones y empareja
    cada ID con su .flac correspondiente.
    """
    folder = "train-clean-100" if split == "train" else "dev-clean"
    base = os.path.join(DATA_ROOT, folder)

    audio_paths = []
    transcripts = []

    # Cada archivo .txt contiene líneas: "<utt_id> <texto>"
    for dirpath, _, filenames in os.walk(base):
        for fname in filenames:
            if not fname.endswith(".txt"):
                continue
            txt_path = os.path.join(dirpath, fname)
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    utt_id, text = line.strip().split(" ", 1)
                    # Construye ruta al .flac
                    flac_path = os.path.join(base, *utt_id.split("-")[:2], f"{utt_id}.flac")
                    if os.path.isfile(flac_path):
                        audio_paths.append(flac_path)
                        transcripts.append(text.lower())

    ds = Dataset.from_dict({
        "path": audio_paths,
        "transcription": transcripts
    })
    # Cast a Audio para decodificar y resamplear on-the-fly :contentReference[oaicite:0]{index=0}
    ds = ds.cast_column("path", Audio(sampling_rate=16_000))  # rechaza streaming, mantiene __len__

    return ds

# ----------------------------
# 3) Preprocesado
# ----------------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def preprocess(batch):
    # batch["path"]["array"] es un np.array de forma (n_muestras,)
    audio = batch["path"]["array"]
    # Extrae features de entrada
    input_vals = processor(audio, sampling_rate=16_000).input_values[0]
    # Tokeniza la transcripción a IDs
    labels = processor.tokenizer(batch["transcription"]).input_ids
    return {
        "input_values": input_vals,
        "labels": labels
    }

# ----------------------------
# 4) Métrica
# ----------------------------
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_ids = torch.argmax(torch.tensor(pred.predictions), dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ----------------------------
# 5) Carga y mapeo de datasets
# ----------------------------
print("Cargando y preprocesando datos…")

train_ds = load_librispeech_split("train")
eval_ds  = load_librispeech_split("validation")

# Map en paralelo con num_proc=4 (dataset estático) :contentReference[oaicite:1]{index=1}
train_ds = train_ds.map(
    preprocess,
    remove_columns=["path", "transcription"],
    batched=False,
    num_proc=4
)
eval_ds = eval_ds.map(
    preprocess,
    remove_columns=["path", "transcription"],
    batched=False,
    num_proc=4
)

# ----------------------------
# 6) Configuración de entrenamiento
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    eval_steps=1000,
    save_steps=1000,
    logging_steps=200,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    fp16=True,
    save_total_limit=2,
    push_to_hub=False,
)

# ----------------------------
# 7) Inicializar modelo y Trainer
# ----------------------------
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics
)

# ----------------------------
# 8) Entrenamiento
# ----------------------------
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Modelo guardado en {OUTPUT_DIR}")
    print("Entrenamiento completado.")