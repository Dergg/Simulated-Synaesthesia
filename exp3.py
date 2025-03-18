import librosa as lib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import subprocess
import argparse
import random
import os

# Argument parser
parser = argparse.ArgumentParser(prog='PSvis', description='Programmatic Synaesthesia Visualisation algorithm')
parser.add_argument('infile')
args = parser.parse_args()

# Create output directory if it doesn't exist
if not os.path.exists('./mp4s'):
    os.makedirs('./mp4s')

# Load audio
aud_path = f'./wavs/{args.infile}.wav'
y, sr = lib.load(aud_path, sr=44100)

# Extract features
spec = np.abs(lib.stft(y, hop_length=512, n_fft=2048*4))
spec_db = lib.amplitude_to_db(spec, ref=np.max)
tempo, beats = lib.beat.beat_track(y=y, sr=sr)
mfcc = lib.feature.mfcc(y=y, sr=sr, n_mfcc=13)
rms = lib.feature.rms(y=y)[0]
chromagram = lib.feature.chroma_stft(y=y, sr=sr)

times = lib.frames_to_time(np.arange(spec_db.shape[1]), sr=sr, hop_length=512)

# Initialize visualization
WIDTH, HEIGHT, FPS = 1280, 720, 30
fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)

class Particle:
    def __init__(self, x, y, color, size, speed, lifetime=3):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.speed = speed
        self.start_time = 0
        self.lifetime = lifetime
        self.alpha = 1.0

    def update(self, time_pos):
        elapsed_time = time_pos - self.start_time
        self.alpha = max(0, 1 - elapsed_time / self.lifetime)

        # Movement logic improved to ensure particles stay visible
        self.y -= self.speed
        self.x = max(0, min(WIDTH, self.x + np.sin(time_pos * 2) * 2))

    def delete(self):
        pass

class ParticleList:
    def __init__(self):
        self.particles = []

    def add(self, particle):
        self.particles.append(particle)

    def remove(self, particle):
        if particle in self.particles:
            self.particles.remove(particle)

    def zero_a_cleanup(self):
        self.particles = [p for p in self.particles if p.alpha > 0]

    def update_all(self, time_pos):
        for particle in self.particles:
            particle.update(time_pos)

particles = ParticleList()

if os.path.isfile(f'./mp4s/{args.infile}-vis.mp4'): # Remove the file if it already exists
    os.remove(f'./mp4s/{args.infile}-vis.mp4')

# FFmpeg subprocess for direct encoding
output_video = f"./mp4s/{args.infile}-vis.mp4"
ffmpeg_cmd = [
    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
    "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
    "-i", "-", "-i", aud_path, "-c:v", "libx264",
    "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "192k",
    "-shortest", output_video
]
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Frame update function
def update(frame):
    time_pos = frame / FPS
    ax.clear()  # Clears previous frame content
    ax.set_facecolor('black')
    ax.axis('on')  # Enable axes temporarily   
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axhline(HEIGHT // 2, color='white', linestyle='--', linewidth=1)
    ax.axvline(WIDTH // 2, color='white', linestyle='--', linewidth=1)

    # --- Particle Generation Logic ---
    # MAX_PARTICLES = 1000 # Max of 1000 particles to avoid clutter
    
    # if len(particles) < MAX_PARTICLES:
    
    rms_energy = rms[int(time_pos * sr // 512)] if int(time_pos * sr // 512) < len(rms) else 0
    num_particles = int(rms_energy * 50)  # Scales particle count to loudness

    # for _ in range(max(10, num_particles)):  # Ensures a base number of particles
    #     x = random.randint(0, WIDTH)
    #     y = random.randint(0, HEIGHT)
    #     color = (random.random(), random.random(), random.random()) # Change this later
    #     size = random.randint(10, 40)
    #     speed = random.uniform(2, 6)
    #     particles.add(Particle(x, y, color, size, speed))

    # Generate particles on RMS peaks (loudness spikes)
    rms_value = rms[frame % len(rms)]
    if rms_value > np.percentile(rms, 90):  # Top 10% of loudness
        for _ in range(30):  # Less intense than beat-triggered particles
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            color = (random.random(), random.random(), 0.5)  # Blue-tinted particles for loudness
            size = random.randint(5, 20)
            speed = random.uniform(1, 4)
            particles.add(Particle(x, y, color, size, speed))

    # # Generate particles from chromagram activity
    # chroma_values = chromagram[:, frame % chromagram.shape[1]]
    # if np.max(chroma_values) > 0.8:  # High chroma intensity
    #     for _ in range(20):  # Medium intensity response
    #         x = random.randint(0, WIDTH)
    #         y = HEIGHT // 2
    #         color = (random.random(), 0.5, random.random())  # Purple-tinted particles for notes
    #         size = random.randint(10, 30)
    #         speed = random.uniform(2, 6)
    #         particles.append(Particle(x, y, color, size, speed))

    # Generate particles from beats (still included for impact moments)
    if any(abs(time_pos - beat_time) < 0.5 for beat_time in beats):
        for _ in range(50):  # More dramatic effect for beats
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            color = (1.0, random.random(), 0.5)  # Warm colors for beats
            size = random.randint(15, 50)
            speed = random.uniform(3, 7)
            particles.add(Particle(x, y, color, size, speed))

    # --- Particle Management ---

    particles.update_all()

    for particle in particles.particles:
        if particle.alpha > 0:
            circle = patches.Circle((particle.x, particle.y),
                                    particle.size,
                                    color=particle.colour,
                                    alpha=particle.alpha,
                                    edgecolor='white')
            ax.add_patch(circle)
    
    particles.cleanup()

    #print(f"Frame {frame}: {len(particles)} particles active")  # Debugging particle count

    # Send frame to FFmpeg
    fig.canvas.draw()
    plt.pause(0.001)
    plt.gcf().canvas.flush_events()
    frame_data = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3] # :3
    frame_data = np.flip(frame_data, axis=0)  # Correct orientation for FFmpeg
    process.stdin.write(frame_data.tobytes())

import glob

files = glob.glob('./debug_frames/*')
for f in files: # Clear all debug frames
    os.remove(f)


total_frames = int(len(y) / sr * FPS)
print("Encoding video with FFmpeg...")
for frame_idx in range(total_frames):
    update(frame_idx)
    if frame_idx % 100 == 0:
        plt.savefig(f'./debug_frames/frame_{frame_idx}.png')

# Finalize FFmpeg process
process.stdin.close()
# stderr_output = process.stderr.read()
# if stderr_output:
#     print(f"FFmpeg stderr: {stderr_output.decode()}")
process.wait()
print(f"MP4 file saved as {output_video}")
