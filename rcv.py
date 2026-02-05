import os
import click
import json
import pandas as pd
from pathlib import Path

@click.command()
@click.option('-f', '--fifo-path', 'fifo_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, type=Path), required=True, help='Path to the FIFO (named pipe) to read from.')
def main(fifo_path: Path):
    # apre la fifo in lettura (bloccante finché un writer non si connette)
    fd = os.open(fifo_path.resolve(),  os.O_RDONLY)
    buffer = ""
    try:
        while True:
            chunk = os.read(fd, 4096).decode() #TODO: decide Bytes num (4096?)
            if not chunk:
                print("Writer has closed the FIFO. Exiting.")
                break
            
            buffer += chunk
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    data = json.loads(line)
                    df = pd.DataFrame(data)
                    #print("received df:\n",df)
    finally:
        os.close(fd)

if __name__ == "__main__":
    main()