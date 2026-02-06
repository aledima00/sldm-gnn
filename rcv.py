import os
import click
import json
import pandas as pd
from pathlib import Path
from collections import deque
import threading

from src.gbuilder import GraphOnlineCreator
import torch

MAX_JSON_CHUNK_SIZE = 32 * 1024  # 32KB ~ about 300 vehicles in each frame

def pipeout_producer(fd: int, pack_queue: deque, pack_size:int,condition: threading.Condition, terminate_event: threading.Event):
    buffer = ""
    while not terminate_event.is_set():
        chunk = os.read(fd, MAX_JSON_CHUNK_SIZE).decode()
        #print(f"CHUNKSIZE: {len(chunk)}")
        if not chunk:
            print("Writer has closed the FIFO. Exiting.")
            terminate_event.set()
            with condition:
                condition.notify_all()  # Notify the consumer to exit if waiting
            break
        
        buffer += chunk
        while '\n' in buffer:
            # cycle over the lines/frames
            line, buffer = buffer.split('\n', 1)
            if line.strip():
                data = json.loads(line)
                df = pd.DataFrame(data)
                #print("received df:\n",df)
                with condition:
                    pack_queue.append(df)
                    if len(pack_queue) >= pack_size:
                        condition.notify_all()  # Notify the consumer that a pack is ready

def infer_consumer(pack_queue: deque, pack_size:int, condition: threading.Condition,stride:int, terminate_event: threading.Event):
    gc = GraphOnlineCreator(frames_num=pack_size, m_radius=25, active_labels=None, rscToCenter=True, removeDims=False, heading_enc=True, has_label=False)
    
    # =============== instantiate/configure model here ===============
    # model = Model.loadPTH("model.pth").eval()
    # ...
    # =============== end ===============

    while not terminate_event.is_set():
        with condition:
            while (len(pack_queue) < pack_size) and not terminate_event.is_set():
                condition.wait()  # Wait until enough frames are available
            if not terminate_event.is_set():
                packDf = pd.concat(list(pack_queue)[:pack_size], keys=range(0, pack_size), names=['FrameId']).reset_index(level=0)
        if not terminate_event.is_set():

            # =============== forward model here ===============
            gdata = gc(packDf)
            print(f"new graph:", gdata)
            with torch.inference_mode():
                pass #.... # out = model(gdata)
            #print(f"model output: {out}")
            # =============== end ===============
            
            with condition:
                for _ in range(stride):
                    if pack_queue:
                        pack_queue.popleft()  # Remove frames based on stride

@click.command()
@click.option('-f', '--fifo-path', 'fifo_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), required=True, help='Path to the FIFO (named pipe) to read from.')
@click.option('-p', '--pack-size', 'pack_size', type=int, required=True, help='Number of frames to pack together before processing.')
@click.option('-s', '--stride', 'stride', type=int, default=1, help='Number of frames to stride after each pack processing.')
def main(fifo_path: Path, pack_size: int, stride: int):
    # apre la fifo in lettura (bloccante finché un writer non si connette)
    fd = os.open(fifo_path.resolve(),  os.O_RDONLY)
    pack_queue = deque()
    lock = threading.Lock()
    condition = threading.Condition(lock)
    terminate_event = threading.Event()

    try:
        t1 = threading.Thread(target=pipeout_producer, args=(fd, pack_queue, pack_size, condition, terminate_event))
        t2 = threading.Thread(target=infer_consumer, args=(pack_queue, pack_size, condition, stride, terminate_event))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    finally:
        os.close(fd)
        print("Bye!")

if __name__ == "__main__":
    main()