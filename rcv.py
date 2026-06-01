import os
import click
import json
import pandas as pd
from pathlib import Path
from collections import deque
import threading

from src.models.grusage import GruSage
from src.gbuilder import GraphOnlineCreator
from src.utils import loadSnapshot, bayesPriorShift
import torch

MAX_JSON_CHUNK_SIZE = 32 * 1024  # 32KB ~ about 300 vehicles in each frame

def signal_termination(condition: threading.Condition, terminate_event: threading.Event, reason: str | None = None):
    if reason:
        print(reason)
    terminate_event.set()
    with condition:
        condition.notify_all()  # Wake up waiting threads so they can exit cleanly


def pipeout_producer(fd: int, pack_queue: deque, pack_size:int,condition: threading.Condition, terminate_event: threading.Event):
    buffer = ""
    try:
        while not terminate_event.is_set():
            try:
                chunk = os.read(fd, MAX_JSON_CHUNK_SIZE).decode()
            except OSError as e:
                print(f"Error reading from Named Pipe: {e}. Exiting producer thread.")
                signal_termination(condition, terminate_event, "Producer thread terminating due to read error.")
                break
            #print(f"CHUNKSIZE: {len(chunk)}")
            if not chunk:
                signal_termination(condition, terminate_event, "Writer has closed the Named Pipe. Exiting producer thread.")
                break
            
            buffer += chunk
            while '\n' in buffer:
                # cycle over the lines/frames
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        signal_termination(condition, terminate_event, f"Malformed JSON from Named Pipe: {e}. Exiting producer thread.")
                        return
                    df = pd.DataFrame(data)
                    #print("received df:\n",df)
                    with condition:
                        pack_queue.append(df)
                        if len(pack_queue) >= pack_size:
                            condition.notify_all()  # Notify the consumer that a pack is ready
    finally:
        signal_termination(condition, terminate_event, "Producer thread terminating.")

def infer_consumer(pack_queue: deque, pack_size:int, condition: threading.Condition,stride:int, terminate_event: threading.Event, snapshot_path: Path, output_csv_file: Path, threshold: float, calibrate_priors: bool, test_prior: float|None):
    snap = loadSnapshot(snapshot_path)
    gc = GraphOnlineCreator(frames_num=pack_size, m_radius=25, active_labels=None, has_label=False, norm_stats=snap['norm_stat_dict'])

    train_prior = snap.get('train_prior')
    if calibrate_priors:
        if train_prior is None:
            raise ValueError("--calibrate-priors requires train_prior in snapshot (re-train with updated main.py)")
        if test_prior is None:
            raise ValueError("--calibrate-priors requires --test-prior")
        # Warm up CUDA with a dummy tensor so that the first real torch.tensor() allocation
        # inside bayesPriorShift doesn't cause a CUDA-side fork error in the consumer thread
        _ = torch.zeros(1, device='cuda')

    model = GruSage(**snap['ip_dict']).cuda().eval()
    model.load_state_dict(snap['state_dict'])

    header = "PredictionLabels,Score,RawScore" if calibrate_priors else "PredictionLabels,Scores"
    with open(output_csv_file, "w") as logfile:
        logfile.write(header + "\n")

    while not terminate_event.is_set():
        with condition:
            while (len(pack_queue) < pack_size) and not terminate_event.is_set():
                condition.wait()
            if not terminate_event.is_set():
                packDf = pd.concat(list(pack_queue)[:pack_size], keys=range(0, pack_size), names=['FrameId']).reset_index(level=0)
        if not terminate_event.is_set():

            gdata = gc(packDf).cuda()
            with open(output_csv_file, "a") as logfile:
                if gdata.x.shape[0] != 0:
                    with torch.inference_mode():
                        out = model(gdata)
                        raw_score = torch.sigmoid(out)

                        if calibrate_priors:
                            calibrated, _ = bayesPriorShift(raw_score.cpu().numpy(), train_prior, test_prior)
                            score = torch.tensor(calibrated, device='cuda')
                        else:
                            score = raw_score

                        preds = (score >= threshold).int()
                        if calibrate_priors:
                            print(f"prediction: {preds.item()}, calibrated_score: {score.item():.6f}, raw_score: {raw_score.item():.6f}")
                            logfile.write(f"{preds.item()},{score.item():.6f},{raw_score.item():.6f}\n")
                        else:
                            print(f"prediction: {preds.item()}, score: {score.item():.6f}")
                            logfile.write(f"{preds.item()},{score.item():.6f}\n")
                else:
                    print(".")
                    logfile.write(".,.,.\n" if calibrate_priors else ".,.\n")

            with condition:
                for _ in range(stride):
                    if pack_queue:
                        pack_queue.popleft()
                #print(f"Queue size after stride: {len(pack_queue)}")

@click.command()
@click.option('-f', '--fifo-path', 'fifo_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), required=True, help='Path to the FIFO (named pipe) to read from.')
@click.option('-p', '--pack-size', 'pack_size', type=int, required=True, help='Number of frames to pack together before processing.')
@click.option('--stride', 'stride', type=int, default=1, help='Number of frames to stride after each pack processing.')
@click.option('-s','--snapshot-path', 'snapshot_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), required=True, help='Path to the model weights file (.pth).')
@click.option('-O', '--output-csv-file', 'output_csv_file', type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path), default="out.csv", help='Path to the output CSV file for predictions.')
@click.option('--threshold', type=float, default=0.5, show_default=True, help='Threshold for binary prediction from scores.')
@click.option('--calibrate-priors', is_flag=True, default=False, help='Apply Bayes prior-shift calibration. Requires train_prior in snapshot and --test-prior.')
@click.option('--test-prior', type=float, default=None, help='Deployment P(y=1) for prior-shift calibration (e.g. 0.00356 for TURN).')
def main(fifo_path: Path, pack_size: int, stride: int, snapshot_path: Path, output_csv_file: Path, threshold: float, calibrate_priors: bool, test_prior: float|None):
    if calibrate_priors and test_prior is None:
        raise click.ClickException("--calibrate-priors requires --test-prior (deployment P(y=1))")
    fd = os.open(fifo_path.resolve(),  os.O_RDONLY)
    pack_queue = deque()
    lock = threading.Lock()
    condition = threading.Condition(lock)
    terminate_event = threading.Event()

    try:
        t1 = threading.Thread(target=pipeout_producer, args=(fd, pack_queue, pack_size, condition, terminate_event))
        t2 = threading.Thread(target=infer_consumer, args=(pack_queue, pack_size, condition, stride, terminate_event, snapshot_path, output_csv_file, threshold, calibrate_priors, test_prior))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    finally:
        os.close(fd)
        print("Bye!")

if __name__ == "__main__":
    main()