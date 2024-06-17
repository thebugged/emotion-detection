<div align="center">
  <br />
    <a href="https://emotion--detection.streamlit.app" target="_blank">
      <img src="https://github.com/thebugged/emotion-detection/assets/74977495/a3ff9d3f-581d-45d8-8209-2e39d707f045" alt="Banner">
    </a>
  <br />

  <div>
    <img src="https://img.shields.io/badge/-Python-black?style=for-the-badge&logoColor=white&logo=python&color=3776AB" alt="python" />
    <img src="https://img.shields.io/badge/-TensorFlow-black?style=for-the-badge&logoColor=white&logo=tensorflow&color=FF6F00" alt="tensorflow" />
    <img src="https://img.shields.io/badge/-scikit_learn-black?style=for-the-badge&logoColor=white&logo=scikitlearn&color=F7931E" alt="scikit-learn" />
    <img src="https://img.shields.io/badge/-Streamlit-black?style=for-the-badge&logoColor=white&logo=streamlit&color=FF4B4B" alt="streamlit" />
</div>


  <h3 align="center">Emotion Detection</h3>

   <div align="center">
     This application detects emotions from either uploads of an audio, image, video or through real-time processing. 
    </div>
</div>
<br/>

**Datasets** 🗃️
- [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- [RAVDESS Emotional speech audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) 
- [Toronto emotional speech set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)


## Setup & Installation
**Prerequisites**

Ensure the following are installed
- [Git](https://git-scm.com/)
- [Python](https://www.python.org/downloads/)
- [FFmpeg](https://ffmpeg.org/)
- [Jupter Notebook](https://jupyter.org/install) (or install the Jupyter extension on [Visual Studio Code](https://code.visualstudio.com/)).
  
To set up this project locally, follow these steps:

1. Clone the repository:
```shell
git clone https://github.com/thebugged/emotion-detection.git
```

2. Change into the project directory: 
```shell
cd emotion-detection
```

3. Install the required dependencies: 
```shell
pip install -r requirements.txt
```
<br/>

## Running the application
1. Run the command: 
```shell
streamlit run main.py
```
2. Alternatively, you can run the `face.ipynb` and `audio.ipynb` notebooks to get their respective models then run the command in 1.

The application will be available in your browser at http://localhost:8501.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](emotion--detection.streamlit.app)


