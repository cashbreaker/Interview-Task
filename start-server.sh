#!/bin/sh

python3 python-api-examples/streaming_server.py \
  --encoder ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
  --decoder ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
  --joiner ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
  --tokens ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt
