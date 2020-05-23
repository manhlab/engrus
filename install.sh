pip3 install pytorch_lightning
pip3 install transformers
pip3 install textwrap
pip3 install --upgrade pip --user
sudo apt-get install openjdk-11-jdk
pip install torch torchtext torchvision sentencepiece psutil future
pip install torchserve torch-model-archiver


brew install wget
wget https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-ru.zipporah0-dedup-clean.tgz
tar -xzvf paracrawl-release1.en-ru.zipporah0-dedup-clean.tgz
mkdir t5_segtiment

mkdir data
mkdir templates

mv paracrawl-release1.en-ru.zipporah0-dedup-clean.en data
mv paracrawl-release1.en-ru.zipporah0-dedup-clean.ru data