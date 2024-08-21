# Gradio ASR streaming application

This is project consists of gradio application written by
Jan Piotrowski for the interview coding tasks.

It uses Sherpa
Onnx streaming ASR server with it pretrained models.
https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/streaming_server.py

## Installation

Download pretrained models [(link)](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2) and unpack them in the root project folder.

Install Python3 with the following packages
- gradio
- scipy
- sherpa-onnx
- websocket

It's possible to use `./install.sh` to install the packages with pip.

## How to run

**To start the server**
```
./start-server.sh
```

**To start the client**
```
python3 client.py
```



