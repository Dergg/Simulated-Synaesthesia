# Simulated Synaethesia Audio Visualisation

Synesthesia is a neurological phenomenon in which the brain "crosses the wires" between different sensory experiences. The most common instance of synesthesia is known as chromasthesia, in which a person may experience visual sensations when listening to music.

This program, known as **PSVis** (Programmatic Synesthesia Visualisation) aims to provide a glimpse into what someone experiencing chromasthesia may see when they listen to pieces of music. Chromasthesia is idiosyncratic, so everyone will experience it differently. **PSVis** aims to use musical data to provide a unique audiovisual experience based on a musical input.

Below is a video example of what this program has created (which can be found in the mp4s folder).

[![Youtube Video Thumbnail; Headrush Visualiser](https://img.youtube.com/vi/7uXCbUfj4QM/0.jpg)](https://www.youtube.com/watch?v=7uXCbUfj4QM)


How to run:
- Ensure that Librosa, NumPy and MatPlotLib are all installed, along with FFMPEG!\
Use your favourite method of installing Python packages; I recommend `pip install`.\
FFMPEG cannot be installed through `pip`, so if you're using a Linux terminal, run `sudo apt-get install ffmpeg`!
We use Librosa v0.11.0, MatPlotLib 3.9.4 and NumPy 1.26.4 -- compatibility issues may exist, so your mileage may vary.
- Also ensure you have a ./wavs folder wherever you have the code, featuring a Wave file of the audio you want\
This repository has an example song (Headrush by MEDUZA), which you can replace with another `.wav` file.
- Run this in your command line by doing `python3 main.py (yourmusicnamehere)` (e.g. `python3 main.py headrush`)
- You can optionally add '-d' at the end to create a new debug.log file to ensure everything is running smoothly
- Wait! This process will take a while, so maybe make yourself some food, a drink, take a little break while it runs
- When the program finishes, it'll create a visualisation file in a newly created mp4s folder.

NOTE:
This will only work with Linux / WSL builds. We have not tested on any Windows or Mac machines, though it may still work (let us know!)
