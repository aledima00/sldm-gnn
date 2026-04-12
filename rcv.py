import os
import click
import json
import pandas as pd
from pathlib import Path
from collections import deque
import threading

from src.models.grusage import GruSage
from src.gbuilder import GraphOnlineCreator
from src.utils import loadSnapshot
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

def infer_consumer(pack_queue: deque, pack_size:int, condition: threading.Condition,stride:int, terminate_event: threading.Event, snapshot_path: Path, output_csv_file: Path):
    snap = loadSnapshot(snapshot_path)
    gc = GraphOnlineCreator(frames_num=pack_size, m_radius=25, active_labels=None, has_label=False, norm_stats=snap['norm_stat_dict'])
    
    # =============== instantiate/configure model here ===============
    # model = Model.loadPTH("model.pth").eval()
    model = GruSage(**snap['ip_dict']).cuda().eval()
    model.load_state_dict(snap['state_dict'])
    
    # ...
    # =============== end ===============

    with open(output_csv_file, "w") as logfile:
        logfile.write("PredictionLabels,Scores\n")

    while not terminate_event.is_set():
        with condition:
            while (len(pack_queue) < pack_size) and not terminate_event.is_set():
                condition.wait()  # Wait until enough frames are available
            if not terminate_event.is_set():
                packDf = pd.concat(list(pack_queue)[:pack_size], keys=range(0, pack_size), names=['FrameId']).reset_index(level=0)
        if not terminate_event.is_set():
            trigger_detected = False

            # =============== forward model here ===============
            gdata = gc(packDf).cuda()
            with open(output_csv_file, "a") as logfile:
                if gdata.x.shape[0] != 0:
                    with torch.inference_mode():
                        out = model(gdata)
                        scores = torch.sigmoid(out)
                        preds = (scores >= 0.5).int()
                        trigger_detected = bool((scores > 0.5).item())
                        print(f"prediction: {preds.item()}, score: {scores.item()}")
                        logfile.write(f"{preds.item()},{scores.item():.6f}\n")
                else:
                    print(".")  # No nodes in the graph, skip inference but print a dot to show we're alive
                    logfile.write(".,.\n")
            # =============== end ===============
            
            with condition:
                if trigger_detected:
                    pack_queue.clear()
                else:
                    for _ in range(stride):
                        if pack_queue:
                            pack_queue.popleft()  # Remove frames based on stride
                #print(f"Queue size after stride: {len(pack_queue)}")

@click.command()
@click.option('-f', '--fifo-path', 'fifo_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), required=True, help='Path to the FIFO (named pipe) to read from.')
@click.option('-p', '--pack-size', 'pack_size', type=int, required=True, help='Number of frames to pack together before processing.')
@click.option('--stride', 'stride', type=int, default=1, help='Number of frames to stride after each pack processing.')
@click.option('-s','--snapshot-path', 'snapshot_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), required=True, help='Path to the model weights file (.pth).')
@click.option('-O', '--output-csv-file', 'output_csv_file', type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path), default="out.csv", help='Path to the output CSV file for predictions.')
def main(fifo_path: Path, pack_size: int, stride: int, snapshot_path: Path, output_csv_file: Path):
    # apre la fifo in lettura (bloccante finché un writer non si connette)
    fd = os.open(fifo_path.resolve(),  os.O_RDONLY)
    pack_queue = deque()
    lock = threading.Lock()
    condition = threading.Condition(lock)
    terminate_event = threading.Event()

    try:
        t1 = threading.Thread(target=pipeout_producer, args=(fd, pack_queue, pack_size, condition, terminate_event))
        t2 = threading.Thread(target=infer_consumer, args=(pack_queue, pack_size, condition, stride, terminate_event, snapshot_path, output_csv_file))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    finally:
        os.close(fd)
        print("Bye!")

if __name__ == "__main__":
    main()