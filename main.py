# Imports
import librosa as lib # For music data analysis and extraction
import numpy as np # Mathsy stuff
import matplotlib.pyplot as plt       ## Using these two to plot each frame individually
import matplotlib.patches as patches  ## by using MatPlotLib like a scatter graph
import subprocess # For running FFMPEG encoding directly through a pipe
import argparse # Parsing arguments
import random # For random particle placement
import os # For doing various tasks like making/deleting files/directories
import logging # For debugging purposes


# Parse arguments (this is my new favourite Python thing)
parser = argparse.ArgumentParser(prog='PSvis', description='Programmatic Synaesthesia Visualisation algorithm')
parser.add_argument('infile') # The music file you want to read; should be in the /wavs folder, stored as 'filename.wav'
parser.add_argument('-d', '--debug', action='store_true') # Enable debugging to a debug.log file (among other things)
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
y, sr = lib.load(aud_path, sr=44100) # Basic librosa stuff

# LIBROSA EXTRACTED FEATURES: DO ONCE, NEVER AGAIN! #
spec = np.abs(lib.stft(y, hop_length=512, n_fft=2048*4))
spec_db = lib.amplitude_to_db(spec, ref=np.max) # Spectrogram
tempo, beats = lib.beat.beat_track(y=y, sr=sr) # Beat detection
mfcc = lib.feature.mfcc(y=y, sr=sr, n_mfcc=13)
rms = lib.feature.rms(y=y)[0] # Loudness detection
chromagram = lib.feature.chroma_stft(y=y, sr=sr) # Note detection
onset_env = lib.onset.onset_strength(y=y, sr=sr, hop_length=512) # Onset detection for percussion
onset_frames = lib.onset.onset_detect(y=y, sr=sr, hop_length=512, units='time')
spec_con = lib.feature.spectral_contrast(y=y, sr=sr) # Calculate spectral contrast
major = [0, 2, 4, 5, 7, 9, 11] # C, D, E, F, G, A, B
# Leaving out minor for memory efficiency: if it's not major, it's minor

n_fft = 2048 * 4 # Length of signal after padding with 0s. Default value (2048) set to power of 2 (4) to optimise speed of FFT.
times = lib.frames_to_time(np.arange(spec_db.shape[1]), sr=sr, hop_length=512, n_fft=n_fft) # Convert the frames to time to access data

# Initialize visualization
WIDTH, HEIGHT, FPS = 1280, 720, 30 # 720p for quicker rendering
fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)

class Particle: # This is the fun part!
    def __init__(self, x, y, colour, size, speed, lifetime=3):
        """Initialise a particle with a location, colour, size and speed."""
        self.x = x
        self.y = y
        self.colour = colour
        self.size = size
        self.speed = speed
        self.start_time = None
        self.lifetime = lifetime
        self.alpha = 1.0

    def update(self, time_pos):
        """Update the particle at a given time, moving it and changing transparency."""
        if self.start_time == None:
            self.start_time = time_pos # Ensure all start times are rendered properly
        elapsed_time = time_pos - self.start_time
        self.alpha = max(0, np.exp(-elapsed_time / self.lifetime))

        if elapsed_time > self.lifetime or self.alpha < 0.01: # Don't allow particles to linger for too long
            self.alpha = 0 # Stop it from displaying if the lifetime is exceeded.

        # Movement logic improved to ensure particles stay visible
        self.y -= self.speed
        self.x = max(0, min(WIDTH, self.x + np.sin(time_pos * 2) * 2))

        # if self.alpha == 0 and args.debug == True: # Particles *do* reach zero alpha!
        #     logger.info("Zero alpha found.")


class ParticleList: # Class for the particle list for easier management (and prettiness)
    def __init__(self):
        """Define a list of particles."""
        self.particles = []

    def add(self, particle):
        """Add a particle to the list."""
        self.particles.append(particle)

    def remove(self, particle):
        """Remove a particle from the list, if it exists."""
        if particle in self.particles:
            self.particles.remove(particle)

    def zero_a_cleanup(self):
        """Clean up all particles with a low transparency to reduce computational strain."""
        if args.debug == True:
            b4 = len(self.particles) # Because typing 'before' is so 2020
            zero_a_parts = 0
            for p in self.particles:
                if p.alpha == 0:
                    zero_a_parts += 1
            logger.info('There are %s zero-alpha particles.', zero_a_parts)
        self.particles = [p for p in self.particles if p.alpha > 0.001] # Remove particles that haven't been removed yet
        if args.debug == True:
            logger.info('Removed %s particles.', (b4 - len(self.particles)))

    def get_particle_num(self):
        """Get the number of particles in the list."""
        return len(self.particles)

    def update_all(self, time_pos):
        """Update all particles and limit the number of particles to 500 to speed up rendering and limit clutter.
        Removes the particles with the lowest alpha, in case they're not picked up by other functions."""
        for particle in self.particles:
            particle.update(time_pos)
        
        self.particles.sort(key=lambda p: p.alpha, reverse=True) # Sort by alpha, descending.
        self.particles = self.particles[:500] # Limit the number of particles to speed up rendering and limit clutter

particles = ParticleList() # Use a specialised ParticleList for easier functionality.

if os.path.isfile(f'./mp4s/{args.infile}-vis.mp4'): # Remove the file if it already exists
    os.remove(f'./mp4s/{args.infile}-vis.mp4')

# FFmpeg subprocess for direct encoding
output_video = f"./mp4s/{args.infile}-vis.mp4"

ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-pix_fmt", "rgb24",
    "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
    "-i", "-",  # Use stdin instead of a file
    "-i", aud_path, "-c:v", "libx264",
    "-preset", "fast", "-crf", "23",
    "-c:a", "aac", "-b:a", "192k",
    "-shortest", output_video
]
# Big old FFMPEG command: basically, take 2 inputs (stdin, audio) and encode them together, synchronised with the
# audio that is provided. Encode with libx264 and a rawvideo video codec, plus an rgb24 pixel format.
# It is for these reasons the video will not show on mobile, though the created YouTube video should (hopefully) show 
# anywhere and everywhere. Check the README!
ffmpeg_p = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Frame update function
def update(frame):
    time_pos = frame / FPS
    ax.cla() # Clear all; more efficient
    ax.set_facecolor('black') # Set background to black for nicer visualisation.
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    
    # --- Particle Generation Logic ---
    rms_norm = rms / np.max(rms) # Normalised loudness
    l_factor = rms_norm[int(time_pos * sr // 512)]
    chroma_values = chromagram[:, min(frame % chromagram.shape[1], chromagram.shape[1] - 1)] 
    # Grab the chromagram values at this frame
    
    sc_norm = spec_con[:, min(frame % spec_con.shape[1], spec_con.shape[1] - 1)]

    # Drum hit detection
    if any(abs(time_pos - onset) < 0.05 for onset in onset_frames):
        for _ in range(10):
            x, y = random.randint(0, WIDTH), random.randint((HEIGHT // 2) - 50, (HEIGHT // 2) + 50)
            size, speed = random.randint(20, 50), random.uniform(3, 7)
            particles.add(Particle(x, y, (0.6, 0.6, 0.6), size, speed)) # Grey particles for drum hits

    dominant_key = np.argmax(chroma_values) # Find the most "forward" note

    if dominant_key in major: # Major is pre-defined earlier in the code
        # Warm colour if it's a major key
        R = 1.0
        G = 0.5
        B = 0.2
    else: 
        # Cool colour if it's a minor key
        R = 0.2
        G = 0.5
        B = 1.0

    # Chroma based particle drawing
    for _, intensity in enumerate(chroma_values):
        if intensity > 0.7: # Only have the stronger notes trigger a particle
            x = int((dominant_key / 12) * WIDTH) # Map note to horizontal position
            y = int((1 - l_factor) * HEIGHT) # Higher intensity = lower y = higher up
            brightness = intensity
            R = min(1.0, R * brightness)
            G = min(1.0, G * brightness)
            B = min(1.0, B * brightness)
            # Adjust brightness depending on pitch
            size = int(10 + (30 * intensity))
            speed = 2 + (4 * intensity) # More intense notes = bigger, faster-moving notes
            particles.add(Particle(x, y, (R, G, B), size, speed))

    # RMS / Loudness-based particle drawing
    if l_factor > 0.8:
        for _ in range(10):
            x, y = random.randint(0, WIDTH), random.randint(0, HEIGHT)
            size, speed = random.randint(5, 25), random.uniform(1, 4)
            particles.add(Particle(x, y, (1.0, 0.5, 0.5), size, speed))  # Soft red burst

    # --- Particle Management ---

    particles.update_all(time_pos) # Update all particles

    for particle in particles.particles:
        if particle.alpha > 0: # Draw all visible particles
            circle = patches.Circle((particle.x, particle.y),
                                    particle.size,
                                    color=particle.colour,
                                    alpha=particle.alpha,
                                    edgecolor='white')
            ax.add_patch(circle)
    
    if args.debug == True:
        logger.info('Frame %s: %s particles', frame, len(particles.particles)) # Print debug information

    particles.zero_a_cleanup() # Clean up particles with zero alpha (invisible)

    # For direct to FFMPEG stuff
    fig.canvas.draw()
    plt.pause(0.001) # Slight pause to allow catchup (I don't know if this works but I'm too scared to remove it)
    plt.gcf().canvas.flush_events()
    frame_data = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3] # :3
    frame_data = np.flip(frame_data, axis=0)  # Correct orientation for FFmpeg
    ffmpeg_p.stdin.write(frame_data.tobytes()) # Write the data to the STDIN pipe to be encoded by FFMPEG.

    

import glob

files = glob.glob('./debug_frames/*')
for f in files: # Clear all debug frames
    os.remove(f)

import gc # Other imports, specifically for this part
import time

total_frames = int(times[-1] * FPS)

logging.info('Frames to encode: %s', str(total_frames))
for frame_idx in range(total_frames):
    fdat = update(frame_idx)
    if frame_idx % 100 == 0:
        plt.savefig(f'./debug_frames/frame_{frame_idx}.png')
        gc.collect() # Collect the garbage
        ffmpeg_p.stdin.flush() # Flush the pipe once every so often to keep it all running nice
        time.sleep(0.01) # Give the processor a tiny moment to catch up (probably not needed but I'm scared to remove it)

ffmpeg_p.stdin.close()
ffmpeg_p.wait()
print(f"MP4 file saved as {output_video}")
