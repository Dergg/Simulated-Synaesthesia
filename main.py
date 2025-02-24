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

spec = np.abs(librosa.stft(y, hop_length = 512, n_fft=2048*4)) # Frequency visualisation of sounds
spec_db = lib.amplitude_to_db(spec, ref=np.max) # Convert the spectrogram to decibels

# plt.figure(figsize=(10,5))
# librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='log') 
# plt.colorbar(label='dB')
# plt.title('Spectrogram')
# plt.show() # Spectrogram works fine.

tempo, beats = librosa.beat.beat_track(y=y, sr=sr) # Pretty self-explanatory here
# print(f'Tempo estimate: {tempo} BPM')
# Actual tempo is 124 BPM, algorithm says 123, sub 1% error is good.

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # Sound timbre
# plt.figure(figsize=(10,4))
# librosa.display.specshow(mfccs, x_axis='time', sr=sr)
# plt.colorbar(label='MFCC')
# plt.title('MFCCs')
# plt.show() # MFCC works fine.

rms = librosa.feature.rms(y=y)[0] # Loudness / Intensity
times = librosa.times_like(rms, sr=sr)
# plt.plot(times, rms)
# plt.title('RMS energy')
# plt.show() # RMS shows correctly over time

chromagram = librosa.feature.chroma_stft(y=y, sr=sr) # Show musical notes
# plt.figure(figsize=(10,4))
# librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', sr=sr)
# plt.colorbar(label='Chroma Intensity')
# plt.title('Chromagram')
# plt.show()
# Chromagram seems correct

def clamp(min, max, val):
    if val < min:
        return min
    
    if val > max:
        return max

    return val

class AudioBar:
    def __init__(self, x, y, freq, colour, width=50, min_height=10, max_height=100, min_decibel=-80, max_decibel=0):
        self.x, self.y, self.freq = x, y, freq

        self.colour = colour
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height

        self.min_decibel, self.max_decibel = min_decibel, max_decibel

        self.__decibel_height_ratio = (self.max_height - self.min_height) / (self.max_decibel - self.min_decibel)


    def update(self, dt, decibel):
        height = decibel * self.__decibel_height_ratio + self.max_height

        speed = (height / self.height)/0.1

        self.height += speed * dt

        self.height = clamp(self.min_height, self.max_height, self.height)

        self.colour = (clamp(0, 255, self.height), 0, clamp(0, 255, self.height))

    def render(self, screen):
        pygame.draw.rect(screen, self.colour, (self.x, self.y + self.max_height - self.height, self.width, self.height))


n_fft = 2048 * 4

freqs = librosa.core.fft_frequencies(n_fft=n_fft)
times = librosa.core.frames_to_time(np.arange(spec_db.shape[1]), sr=sr, hop_length=512, n_fft = n_fft)

tir = len(times) / times[len(times) - 1]
fir = len(freqs) / freqs[len(freqs) - 1]

def get_dec(target_time, freq):
    return spec_db[int(freq * fir)][int(target_time * tir)]

pygame.init()

infoObject = pygame.display.Info()

screen_size = int(infoObject.current_w/2)
screen = pygame.display.set_mode([screen_size, screen_size])
pygame.display.set_caption('Audio Visualiser')

bars = []

freqs = np.arange(100, 5000, 100)
r = len(freqs)
bar_width = screen_size/r
x = (screen_size - bar_width * r) / 2

for c in freqs:
    bars.append(AudioBar(x, 300, c, (255, 0, 0), max_height=400, width=bar_width))
    x += bar_width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(aud_path)
pygame.mixer.music.play(0)

running = True
while running:

    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.quit:
            running = False

    screen.fill((0,0,0))
    for b in bars:
        b.update(deltaTime, get_dec(pygame.mixer.music.get_pos()/1000, b.freq))
        b.render(screen)

    pygame.display.flip()

pygame.quit()