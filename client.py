import json
import time
import gradio as gr
import numpy as np
import scipy.signal as ss
import threading

import websockets.exceptions
from websockets.sync.client import connect

# Text is a global variable that stores transcribed results from the serverd
text = dict()


# Resample chunk to target sampling rate
def transcode(chunk: np.array, sr_original, sr_target):
    return ss.resample(chunk, int(chunk.shape[0] / sr_original * sr_target))


# Handle new chunk of audio
# https://www.gradio.app/guides/real-time-speech-recognition#3-create-a-streaming-asr-demo-with-transformers
def handle_audio_chunk(stream, new_chunk, sequence, ws, sample_size, server_sr):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    y = transcode(y, sr, server_sr)

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    seq = sequence
    if stream.shape[0] > (sequence + 1) * server_sr * sample_size:
        ws.send(stream[seq * server_sr * sample_size:(seq + 1) * server_sr * sample_size].tobytes())
        seq = seq + 1

    return stream, seq, get_current_text()


# Formats returned text from dictionary to string
def get_current_text():
    result = ""
    for x in sorted(text.keys()):
        result += text.get(x) + '\n'
    return result


# Handler of incoming data from the server
def handler(ws):
    while True:
        try:
            msg = json.loads(ws.recv())
            text.update({msg["segment"]: msg["text"]})
        except websockets.exceptions.ConnectionClosedOK:
            break


# Function called when record of audio is started
def start(server_url, port, ws):
    websocket = ws
    if ws is None:
        websocket = connect(f"ws://{server_url}:{port}")
        threading.Thread(target=handler, args=(websocket,)).start()  # Start Handler in separate thread
    text.clear()
    return websocket


# Function called when recording of audio is stopped
def stop(audio, ws, server_sr, interval):
    slice_size = -(audio.shape[0] % (interval * server_sr))
    if slice_size != 0:  # Managed to crash it once here
        ws.send(audio[slice_size:].tobytes())


def clear_func(ws):
    text.clear()
    if ws:
        ws.send("Done")
        time.sleep(0.5)  # Ensure the server acknowledges the Done message
        ws.close()
    return None, 0, None, "\n"


with gr.Blocks() as demo:
    with gr.Row():
        audio = gr.Audio(label="Input", sources=["microphone"], streaming=True)
        textbox = gr.Textbox(label="Transcribed")
    with gr.Row():
        interval_duration = gr.Number(label="Audio interval duration", minimum=1, maximum=10, value=5)
        server_sr = gr.Number(label="Server Sampling Rate", minimum=0, maximum=96000, value=16000)
        server_address = gr.Textbox(label="Server address", value="localhost")
        server_port = gr.Number(label="Port", minimum=1, maximum=65536, value=6006)
        sync = gr.Button("Sync")
        clear = gr.Button("Clear")
    seq = gr.State(0)
    stream = gr.State()
    ws = gr.State()
    audio.start_recording(start, inputs=[server_address, server_port, ws], outputs=ws)
    audio.stop_recording(stop, inputs=[stream, ws, server_sr, interval_duration])
    audio.input(handle_audio_chunk, inputs=[stream, audio, seq, ws, interval_duration, server_sr], outputs=[stream, seq, textbox])
    sync.click(get_current_text, outputs=[textbox])
    clear.click(clear_func, inputs=[ws], outputs=[ws, seq, stream, textbox])

demo.launch()
