import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm


def generate_speaker_embedding(speaker, speaker_encoder, minimum_duration: float = 5.):

    with os.scandir(speaker.path) as sessions:
        sessions = list(sessions)

        for element in sessions:
            if element.name == ".DS_Store":
                sessions.remove(element)

        # select a random session
        session = sessions[random.randint(0, len(sessions)-1)]

        # get the filepaths for the current speaker
        files = glob(session.path + '/*.flac')
        n_files = len(files)

        waveforms_preprocessed = []
        current_duration = 0.
        while current_duration < minimum_duration:
            random_file = files[random.randint(0, n_files - 1)]
            waveform, sample_rate = torchaudio.load(random_file)
            current_duration += waveform.size(-1) / sample_rate
            waveform_preprocessed = preprocess_wav(waveform[0].numpy())
            waveforms_preprocessed.append(waveform_preprocessed)
            if len(waveforms_preprocessed) >= n_files:
                break

        embedding = speaker_encoder.embed_speaker(waveforms_preprocessed)
    return torch.from_numpy(embedding)


def generate_embeddings(libri_root: str, split: str = "dev-clean", save_dir: str = "embeddings"):
    speaker_encoder = VoiceEncoder()
    Path(save_dir).mkdir(parents=True, exist_ok=False)
    with os.scandir(libri_root + split) as speakers:
        speakers_list = list(speakers)
        n_speakers = len(speakers_list)
        embeddings = {}
        for speaker in tqdm(speakers_list, unit=" embeddings", total=n_speakers):
            if os.path.isdir(speaker.path) and not speaker.name.startswith("."):
                embedding = generate_speaker_embedding(speaker, speaker_encoder)
                embeddings[speaker.name] = embedding

    torch.save(embeddings, f=save_dir + "/speaker_embeddings.pt")


def load_speaker_embedding(speaker_id: int, embeddings_dir: str, split: str):
    return torch.load(embeddings_dir + split + "/" + str(speaker_id) + ".pt")


def compute_similarity(path, speaker_embedding, speaker_encoder):
    waveform, sr = torchaudio.load(path)
    wav_preprocessed = preprocess_wav(waveform[0].numpy())

    _, partial_embeddings, _ = speaker_encoder.embed_utterance(wav_preprocessed,
                                                               return_partials=True,
                                                               min_coverage=0.5,
                                                               rate=2.5)

    similarity = torch.nn.functional.cosine_similarity(speaker_embedding, torch.from_numpy(partial_embeddings),
                                                       dim=-1).numpy()
    if np.isnan(similarity).any():
        print("found NaN in similarity scores")
    torch.cuda.empty_cache()
    return similarity


def generate_similarity_scores(libri_concat_root: str, split: str = "dev-clean", waveforms_dir: str = "Waveforms",
                               speaker_embedding_dir: str = "SpeakerEmbeddings", save_dir: str = "SpeakerEmbeddings"):
    speaker_encoder = VoiceEncoder(device="cpu")
    waveform_paths = sorted(str(p) for p in Path(libri_concat_root + split + "/" + waveforms_dir).glob("**/*.flac"))
    embeddings_path = libri_concat_root + split + "/" + speaker_embedding_dir + "/speaker_embeddings.pt"
    metadata = pd.read_csv(libri_concat_root + split + "/metadata.csv").set_index('identifier')
    speaker_embeddings = torch.load(embeddings_path)
    for path in tqdm(waveform_paths, unit="utterances", total=len(waveform_paths)):
        identifier = Path(path).stem
        out_name = "/" + identifier + "_similarity"
        # if Path(libri_concat_root + split + "/" + save_dir + out_name + ".pt").exists():
        #     continue

        target_speaker_id = int(metadata.loc[identifier]["target_speaker_id"])
        speaker_embedding = speaker_embeddings[str(target_speaker_id)]
        similarity = compute_similarity(path, speaker_embedding, speaker_encoder=speaker_encoder)
        
        torch.save(similarity, f=libri_concat_root + split + "/" + save_dir + out_name + ".pt")


if __name__ == "__main__":
    #generate_embeddings(libri_root="/Users/JG96XG/Desktop/data_sets/LibriSpeech/")
    splits = ["10h", "1h", "10min"] # ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "test-clean"]
    for split in splits:
        print(f"generating similarity scores for split = {split}")
        generate_similarity_scores("data/LibriLightConcat/", split=split)
