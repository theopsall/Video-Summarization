# Multimodal summarization of user-generated videos from wearable cameras.

![APM](https://img.shields.io/apm/l/vim-mode?style=plastic)
[![Generic badge](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)](https://shields.io/)
![GitHub issues](https://img.shields.io/github/issues/theopsall/Video-Summarization?style=plastic)

This repository contains the source code of my Thesis in [MSc Data Science](http://msc-data-science.iit.demokritos.gr/), entitled: <ins>"Multimodal summarization of user-generated videos from wearable cameras"</ins>

## Intro
&nbsp;&nbsp;&nbsp;&nbsp; The proposed video summarization technique is based on the audio and visual features extracted using [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) and [multimodal_movie_analysis](https://github.com/tyiannak/multimodal_movie_analysis) respectively.

&nbsp;&nbsp;&nbsp;&nbsp; For the purpose of my thesis, I also created a dataset, provided it [here](https://drive.google.com/drive/folders/1-nBp2zJKXsUe2xa9DtxonNdZ6frwWkMp?usp=sharing), which contains the audio and visual features accompanied with the ground truth annotation files. In order to construct the ground truth for the videos, user-created video summaries was collected using the [video annotator tool](https://github.com/theopsall/video_annotator) and then with the aggregation process we build the final labels.

# Structure

## Download the Annotations files
&nbsp;&nbsp;&nbsp;&nbsp; In order to run from the experiments and train the model from the beginning you have to download the aformentioned [dataset](https://drive.google.com/drive/folders/1-nBp2zJKXsUe2xa9DtxonNdZ6frwWkMp?usp=sharing), otherwise you can use a video collection of your own.


## Clone the repository

```bash 
https://github.com/theopsall/video-summarization.git
```
## Installation 
```bash
cd Video-summarization
chmod -x install.sh
./install.sh
```


## Usage
###1. To extract and train the classifier
```python
python3 video_summarization.py extractAndTrain -v /home/theo/VIDEOS -l /home/theo/LABELS -o /home/theo/videoSummary  
```
`-v`: The directory containing the video files.

`-l`: The directory containing the annotations files.

`-o`: The directory  to store the final model.

`(-d)`:  Optional, in case you want to download and use the video files from the experiment. 

###2. To train the classifier (assuming you already have extracted the features)
```python
python3 video_summarization.py train -v /home/theo/visual_features -a /home/theo/aural_features -l /home/theo/LABELS -o /home/theo/videoSummary 
```
`-v`: The directory with the visual features.

`-a`: The directory with the aural features.

`-l`: The directory containing the annotations files. 

`-o`: The directory  to store the final model.

###3. To classify a video from the model
```python
python3 video_summarization.py predict -v /home/theo/sample.mp4 -o /home/theo/prediction 
```
`-v`: The path of the video file.

`-o`: The destination directory to store the prediction file.

###4. To extract the features used in video summarization
```python
python3 video_summarization.py featureExtraction -v /home/theo/VIDEOS -o /home/theo/FEATURES
```
`-v`: The directory containing the video files.

`-o`: The destination directory to store the features files for both modalities.


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


