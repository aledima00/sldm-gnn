import os
import click
from pathlib import Path

@click.command()
@click.option('-f', '--fifo-path', 'fifo_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), required=True, help='Path to the FIFO (named pipe) to read from.')
def main(fifo_path):
    # apre la fifo in lettura (bloccante finché un writer non si connette)
    with open(fifo_path, "r") as fifo:
        while True:
            line = fifo.readline()

            if line == "":
                # writer ha chiuso la pipe
                break

            line = line.rstrip("\n")

            # ---- elaborazione ----
            result = line.upper()   # esempio
            # ----------------------

            print(f"Received: {line} -> Processed: {result}")

if __name__ == "__main__":
    main()