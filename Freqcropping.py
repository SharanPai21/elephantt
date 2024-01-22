import librosa
import numpy as np
import soundfile as sf
import os


def extract_segment(audio_file_path, frequency_threshold, amplitude_threshold):
    audio, sr = librosa.load(audio_file_path)
    hop_length = 512
    n_fft = 2048
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    freq_mask = mag > frequency_threshold
    amp_mask = mag > amplitude_threshold
    mask = freq_mask & amp_mask
    start_index = np.where(mask.any(axis=0))[0][0]
    end_index = np.where(mask.any(axis=0))[0][-1] + 1
    time_step = hop_length / sr
    start_time = start_index * time_step
    end_time = end_index * time_step
    segment = audio[int(start_time * sr):int(end_time * sr)]
    return segment, sr


def process_files(file_list, frequency_threshold, amplitude_threshold):

    for file_path in file_list:
        segment, sr = extract_segment(file_path, frequency_threshold, amplitude_threshold)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        new_file_name = "processed_" + file_name + ".wav"
        sf.write(new_file_name, segment, sr)
        print(f"Processed {file_path} and saved as {new_file_name}")


file_list = ["t1.wav", "t2.wav", "t3.wav", "t4.wav", "t5.wav", "t6.wav", "t7.wav", "t8.wav",]
frequency_threshold = 20
amplitude_threshold = 20
process_files(file_list, frequency_threshold, amplitude_threshold)
