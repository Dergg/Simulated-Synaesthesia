import librosa as lib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import ffmpeg
import os
import argparse

parser = argparse.ArgumentParser(prog='PSvis', description='Programmatic Synaesthesia Visualisation algorithm')
parser.add_argument('infile')
args = parser.parse_args()

aud_path = f'./wavs/{args.infile}.wav' # Only works with WAV files!
y, sr = lib.load(aud_path, sr=44100) # y = waveform | SR = sample rate

spec = np.abs(lib.stft(y, hop_length=512, n_fft=2048*4)) # Frequency visualisation of sounds
spec_db = lib.amplitude_to_db(spec, ref=np.max) # Convert the spectrogram to decibels

n_fft = 2048 * 4
freqs = lib.fft_frequencies(n_fft=n_fft)
times = lib.frames_to_time(np.arange(spec_db.shape[1]), sr=sr, hop_length=512, n_fft=n_fft)

def get_dec(target_time, freq):
    freq_idx = min(len(freqs) - 1, max(0, int(freq / freqs[-1] * len(freqs))))
    time_idx = min(len(times) - 1, max(0, int(target_time / times[-1] * len(times))))
    return spec_db[freq_idx, time_idx]

def clamp(min_val, max_val, val):
    return max(min_val, min(max_val, val))

class AudioBar:
    def __init__(self, x, freq, width=10, min_height=10, max_height=100, min_decibel=-80, max_decibel=0):
        self.x, self.freq = x, freq
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.decibel_height_ratio = (self.max_height - self.min_height) / (self.max_decibel - self.min_decibel)
    
    def update(self, decibel):
        self.height = clamp(self.min_height, self.max_height, decibel * self.decibel_height_ratio + self.max_height)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, len(freqs))
ax.set_ylim(0, 100)
ax.set_xlabel("Frequency Bins")
ax.set_ylabel("Amplitude (dB)")

bars = [AudioBar(x, f, width=5, max_height=100) for x, f in enumerate(freqs)]
bar_patches = ax.bar([b.x for b in bars], [b.height for b in bars], width=5, color='r')

def update(frame):
    time_pos = frame / 30  # Assuming 30 FPS
    for b, patch in zip(bars, bar_patches):
        b.update(get_dec(time_pos, b.freq))
        patch.set_height(b.height)
    return bar_patches
print("Creating animation...")
ani = ani.FuncAnimation(fig, update, frames=len(times), interval=1000/30, blit=True)
print("Animation made. Saving...")
#output_video = "output.mp4"
ani.save('./frames/frames%04d.png', writer='pillow', fps=30)
print("Animation saved. Using FFMPEG output.")
final_output = "final_output.mp4"
ffmpeg.input('./frames/frames%04d.png', framerate=30).output(final_output, audio=aud_path, vcodec='mpeg4', acodec='aac').override_output().run(quiet=True)
print("Done.")
# os.remove(output_video)
# print("MP4 file saved as", final_output)
import shutil
shutil.rmtree('frames')