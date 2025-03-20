import librosa as lib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import subprocess
import argparse
import random
import os
import logging


# Argument parser
parser = argparse.ArgumentParser(prog='PSvis', description='Programmatic Synaesthesia Visualisation algorithm')
parser.add_argument('infile')
parser.add_argument('-d', '--debug', action='store_true')
args = parser.parse_args()

if os.path.isfile('./debug.log') and args.debug == True: # Fresh debug log every time we debug to ensure no crossover.
    os.remove('./debug.log')

if args.debug == True:
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='debug.log', encoding='utf-8', level=logging.INFO) # Use this to calm down the print statements.

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
    def __init__(self, x, y, colour, size, speed, lifetime=3):
        self.x = x
        self.y = y
        self.colour = colour
        self.size = size
        self.speed = speed
        self.start_time = None
        self.lifetime = lifetime
        self.alpha = 1.0

    def update(self, time_pos):
        if self.start_time == None:
            self.start_time = time_pos
        elapsed_time = time_pos - self.start_time
        self.alpha = max(0, np.exp(-elapsed_time / self.lifetime))

        if elapsed_time > self.lifetime or self.alpha < 0.01:
            self.alpha = 0 # Stop it from displaying if the lifetime is exceeded.

        # Movement logic improved to ensure particles stay visible
        self.y -= self.speed
        self.x = max(0, min(WIDTH, self.x + np.sin(time_pos * 2) * 2))

        # if self.alpha == 0 and args.debug == True: # Particles *do* reach zero alpha!
        #     logger.info("Zero alpha found.")


class ParticleList:
    def __init__(self):
        self.particles = []

    def add(self, particle):
        self.particles.append(particle)

    def remove(self, particle):
        if particle in self.particles:
            self.particles.remove(particle)

    def zero_a_cleanup(self):
        if args.debug == True:
            b4 = len(self.particles)
            zero_a_parts = 0
            for p in self.particles:
                if p.alpha == 0:
                    zero_a_parts += 1
            logger.info('There are %s zero-alpha particles.', zero_a_parts)
        self.particles = [p for p in self.particles if p.alpha > 0.001]
        if args.debug == True:
            logger.info('Removed %s particles.', (b4 - len(self.particles)))

    def get_particle_num(self):
        return len(self.particles)

    def update_all(self, time_pos):
        for particle in self.particles:
            particle.update(time_pos)
        
        self.particles.sort(key=lambda p: p.alpha, reverse=True) # Sort by alpha, descending.
        self.particles = self.particles[:1200]

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
    ax.clear()
    ax.set_facecolor('black')
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    
    # --- Particle Generation Logic ---
    rms_energy = rms[int(time_pos * sr // 512)] if int(time_pos * sr // 512) < len(rms) else 0
    chroma_values = chromagram[:, min(frame % chromagram.shape[1], chromagram.shape[1] - 1)]
    
    # Generate particles on **beats** for stronger synchronization
    if any(abs(time_pos - beat_time) < 0.1 for beat_time in beats):  # Small window around beats
        for _ in range(30):
            x, y = random.randint(0, WIDTH), random.randint(HEIGHT // 2 - 50, HEIGHT // 2 + 50)
            color = (1.0, random.random(), 0.5)  # Warm colors for beats
            size, speed = random.randint(15, 50), random.uniform(3, 7)
            particles.add(Particle(x, y, color, size, speed))
    
    # Generate particles **for musical notes (chromagram peaks)**
    if np.max(chroma_values) > 0.8:
        for _ in range(10):  # Moderate response to strong notes
            x, y = random.randint(0, WIDTH), random.randint(0, HEIGHT)
            color = (random.random(), 0.5, random.random())  # Purple-tinted
            size, speed = random.randint(10, 30), random.uniform(2, 6)
            particles.add(Particle(x, y, color, size, speed))
    
    # Generate particles **from RMS loudness spikes**
    if rms_energy > np.percentile(rms, 90):  # Only on high-energy sections
        for _ in range(20):
            x, y = random.randint(0, WIDTH), random.randint(0, HEIGHT)
            color = (random.random(), random.random(), 0.5)  # Blue-tinted
            size, speed = random.randint(5, 20), random.uniform(1, 4)
            particles.add(Particle(x, y, color, size, speed))

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

    particles.update_all(time_pos)

    for particle in particles.particles:
        if particle.alpha > 0:
            circle = patches.Circle((particle.x, particle.y),
                                    particle.size,
                                    color=particle.colour,
                                    alpha=particle.alpha,
                                    edgecolor='white')
            ax.add_patch(circle)
    
    if args.debug == True:
        logger.info('Frame %s: %s particles', frame, len(particles.particles))

    particles.zero_a_cleanup()

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


print("Encoding video with FFmpeg...")
logging.info('Frames to encode: %s', times[-1] * FPS)
for frame_idx in range(int(times[-1] * FPS)):
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
