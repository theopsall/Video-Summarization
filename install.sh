echo 'Installing requirements'
pip install -r requirements.txt
cd video_summarization/libs
echo 'Cloning multimodal_movie_analysis'
git clone https://github.com/tyiannak/multimodal_movie_analysis.git
echo 'Installing multimodal_movie_analysis dependencies'
pip install -r multimodal_movie_analysis/requirements.txt
echo 'install.sh Finished!'