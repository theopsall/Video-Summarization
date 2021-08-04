# Multimodal summarization of user-generated videos from wearable cameras.

![APM](https://img.shields.io/apm/l/vim-mode?style=plastic)
[![Generic badge](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)](https://shields.io/)
![GitHub issues](https://img.shields.io/github/issues/theopsall/Video-Summarization?style=plastic)

This repository contains the source code of my Thesis in [MSc Data Science](http://msc-data-science.iit.demokritos.gr/), entitled: <ins>"Multimodal summarization of user-generated videos from wearable cameras"</ins>

## Intro
&nbsp;&nbsp;&nbsp;&nbsp; The proposed video summarization technique is based on the audio and visual features extracted using [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) and [multimodal_movie_analysis](https://github.com/tyiannak/multimodal_movie_analysis) respectively.

&nbsp;&nbsp;&nbsp;&nbsp; For the purpose of my thesis, i also created a dataset, provided it [here](https://drive.google.com/drive/folders/1-nBp2zJKXsUe2xa9DtxonNdZ6frwWkMp?usp=sharing), which contains the audio and visual features accompanied with the ground truth annotation files. In order to construct the ground truth for the videos, user-created video summaries was collected using the [video annotator tool](https://github.com/theopsall/video_annotator) and then with the aggregation process we build the final labels.

# Structure

## Download the dataset
&nbsp;&nbsp;&nbsp;&nbsp; In order to run from the experiments and train the model from the begining you have to download the aformentioned [dataset](https://drive.google.com/drive/folders/1-nBp2zJKXsUe2xa9DtxonNdZ6frwWkMp?usp=sharing), otherwise you can use a video collection of your own. In case you want to extract your own features, a superset (unfiltered) of the videos (youtube urls) is provided [here](https://github.com/theopsall/Video-Summarization/tree/refactor/video_summarization/dataset). 


## Clone the repository

```bash 
https://github.com/theopsall/video-summarization.git
```
## Install dependencies
```bash
cd Video-summarization
pip install -r requirements.txt
```


## Running Video Summarization
```bash
python3 main.py 
```


# Citation
```
@article{psallidas2021multimodal,
  title={Multimodal summarization of user-generated videos from wearable cameras},
  author={Psallidas, Theodoros},
  year={2021}
}
```

Enjoy the video summarization tool & feel free to bother me in case you need help. You can reach me at
[Theo Psallidas](mailto:theopsall@gmail.com.com?subject=[GitHub]%20Mutlimodal%20Video%20Summarization)


**DISCLAIMER**

I have made all the feature extraction scripts, as command line executables, in case you want to use some tools arbitrary out of the main pipeline, you are able to call them from the command line.


