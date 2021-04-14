# CoQA Chatbot
Question and Answer bot powered by BERT + CoQA built in Pytorch. This project was done as course project of CMPT 629 

## Installation
### Requirements
	cd CoQAbot
	mkdir data
	mkdir bert-base-uncased
	mkdir output
	pip install requirements.txt

### Download Data
	wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json -O data/coqa-train-v1.0.json 
	wget https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json -O data/coqa-dev-v1.0.json 
### Download BERT model
	wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin -O bert-base-uncased/pytorch_model.bin
	wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json -O bert-base-uncased/berconfig.json
	wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -O bert-base-uncased/vocab.txt
## Train
	sh run.sh
## Predict 
	sh predict.sh
## Evaluate
	sh evaluate.sh
## Chatbot Running
	python3 server.py

