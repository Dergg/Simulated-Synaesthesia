# Original code: https://github.com/chaosqueenbee/music_visualizer/blob/main/main.py

# Adapted to save to MP4 file as Pygame + WSL = Major headache

import librosa as lib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import subprocess
import argparse

# Argument parser
parser = argparse.ArgumentParser(prog='PSvis', description='Programmatic Synaesthesia Visualisation algorithm')
parser.add_argument('infile')
args = parser.parse_args()

# Load audio
aud_path = f'./wavs/{args.infile}.wav'  # Only works with WAV files!
y, sr = lib.load(aud_path, sr=44100)  # y = waveform | SR = sample rate

# Compute spectrogram
spec = np.abs(lib.stft(y, hop_length=512, n_fft=2048*4))
spec_db = lib.amplitude_to_db(spec, ref=np.max)

# FFT settings
n_fft = 2048 * 4
freqs = lib.fft_frequencies(n_fft=n_fft)
times = lib.frames_to_time(np.arange(spec_db.shape[1]), sr=sr, hop_length=512, n_fft=n_fft)

# Get decibel at a given time and frequency
def get_dec(target_time, freq):
    freq_idx = min(len(freqs) - 1, max(0, int(freq / freqs[-1] * len(freqs))))
    time_idx = min(len(times) - 1, max(0, int(target_time / times[-1] * len(times))))
    return spec_db[freq_idx, time_idx]

# Clamp function
def clamp(min_val, max_val, val):
    return max(min_val, min(max_val, val))

# Audio bar class
class AudioBar:
    def __init__(self, x, freq, width=10, min_height=0, max_height=100, min_decibel=-80, max_decibel=0):
        self.x, self.freq = x, freq
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.decibel_height_ratio = (self.max_height - self.min_height) / (self.max_decibel - self.min_decibel)
    
    def update(self, decibel):
        self.height = clamp(self.min_height, self.max_height, decibel * self.decibel_height_ratio + self.max_height)

# Matplotlib setup
WIDTH, HEIGHT, FPS = 1280, 720, 30
fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
ax.set_xlim(0, len(freqs))
ax.set_ylim(0, 100)
#ax.set_xlabel("Frequency Bins")
ax.set_ylabel("Amplitude (dB)")


## Please note: having wider bars does not mean it'll take less time to make
bars = [AudioBar(x, f, width=2, max_height=100) for x, f in enumerate(freqs)]
bar_patches = ax.bar([b.x for b in bars], [b.height for b in bars], width=5, color='r')

# FFmpeg subprocess for direct encoding
output_video = f"./mp4s/{args.infile}-vis.mp4"
ffmpeg_cmd = [
    "ffmpeg",
    "-y",  # Overwrite output file
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "rgb24",
    "-s", f"{WIDTH}x{HEIGHT}",
    "-r", str(FPS),
    "-i", "-",  # Input from stdin
    "-i", aud_path,  # Audio input
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-c:a", "aac",
    "-b:a", "192k",
    "-shortest",
    output_video
]
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Frame update function
def update(frame):
    time_pos = frame / FPS
    for b, patch in zip(bars, bar_patches):
        b.update(get_dec(time_pos, b.freq))
        patch.set_height(b.height)
    fig.canvas.draw()
    
    # Convert frame to raw RGB data and send to FFmpeg
    frame_data = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    process.stdin.write(frame_data.tobytes())

# Generate animation frames
print("Encoding video with FFmpeg...")
for frame_idx in range(int(times[-1] * FPS)):  # Process full track
    update(frame_idx)

# Finalize FFmpeg process
process.stdin.close()
process.wait()
print(f"MP4 file saved as {output_video}")
