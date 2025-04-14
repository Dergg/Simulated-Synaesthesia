# IMC-Coursework

Synesthesia is a neurological phenomenon in which the brain "crosses the wires" between different sensory experiences. The most common instance of synesthesia is known as chromasthesia, in which a person may experience visual sensations when listening to music.

This program, known as **PSVis** (Programmatic Synesthesia Visualisation) aims to provide a glimpse into what someone experiencing chromasthesia may see when they listen to pieces of music. Chromasthesia is idiosyncratic, so everyone will experience it differently. **PSVis** aims to use musical data to provide a unique audiovisual experience based on a musical input.

Below is a video example of what this program has created (which can be found in the mp4s folder).

<iframe width="560" height="315" src="https://www.youtube.com/embed/7uXCbUfj4QM?si=Pm8_UiSldcIpN8gO" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

How to run:
- Ensure that Librosa, NumPy and MatPlotLib are all installed, along with FFMPEG (pip install whatever you don't have)
- Also ensure you have a ./wavs folder wherever you have the code, featuring a Wave file of the audio you want
- Run this in your command line by doing 'python3 main.py (yourmusicnamehere)' i.e. python3 main.py headrush
- You can optionally add '-d' at the end to create a new debug.log file to ensure everything is running smoothly
- Wait! This process will take a while, so maybe make yourself some food, a drink, take a little break while it runs
- When the program finishes, it'll create a visualisation file in a newly created mp4s folder.