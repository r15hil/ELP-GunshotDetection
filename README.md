# ELP-GunshotDetection
 
## Requirements
Python:
- pip install pydub
- pip install numpy
- pip install pandas
- pip install librosa
- pip install matplotlib
- pip install Pillow
- pip install opencv-python
- pip install tensorflow
- pip install ffmpeg
- pip install ffprobe

## How to run

1. Make a folder called 'sounds' and put the soundfile you want to detect in there.
2. Run the command 'python python/predict.py sounds'
3. It will put detected gunshots inside cache/images/ with the bounding box
