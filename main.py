# IMC Coursework

# Has to involve at least one of the following:
# 1. Fourier / Inverse Fourier transform
# 2. Mathematical representations of scales, temparement and/or tuning
# 3. Computational models of beat, tempo and/or meter
# 4. Geometric models of harmony and/or voice-leading (e.g. Tonnetz)
# 5. Sequential models of music data (harmony, melody and/or rhythm)
# 6. Hierarchical models of music data
# 7. Neural models of music data

# Current ideas:
# - Synysthesia: analysing audio and sscore to produce some visualisation: audiovisual output with colours [2, 3]

import librosa as lib
import librosa.display
import pygame
import pygame.mixer as pgm
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pydub
import argparse

parser = argparse.ArgumentParser(prog='PSvis', description='Programmatic Synaesthesia Visualisation algorithm')
parser.add_argument('infile')
# parser.add_argument('outfile')
parser.add_argument('-live', '--live_video', action='store_true') # Optionally shows the MP4 file in a window
args = parser.parse_args()

aud_path = f'./mp3s/{args.infile}.mp3' # Only works with MP3 files!
y, sr = lib.load(aud_path, sr=44100) # y = waveform | SR = sample rate

spec = np.abs(librosa.stft(y))
spec_db = lib.amplitude_to_db(spec, ref=np.max) # Convert the spectrogram to decibels

plt.figure(figsize=(10,5))
librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(label='dB')
plt.title('Spectrogram')
plt.show()