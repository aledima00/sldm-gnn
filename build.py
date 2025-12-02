import click
from src.gbuilder import GraphsBuilder
from pathlib import Path
from typing import Literal as Lit

DEF_FNUM = 20
DEF_RADIUS = 30.0

@click.command()
@click.argument('data_path', type=click.Path(exists=True), required=True)
@click.option('-r', '--radius-threshold', 'radius_threshold', type=float, default=DEF_RADIUS, help=f'Radius threshold for graph building (default: {DEF_RADIUS}).')
@click.option('-l', '--active-label', 'active_label', type=int, help='Active label (one only) to consider during graph building.', required=True, prompt="Choose index of the active label:")
@click.option('-f', '--frames-num', 'frames_num', type=int, default=DEF_FNUM, help=f'Number of frames to process (default: {DEF_FNUM}).')
@click.option('--no-rescaling', 'no_rescaling', is_flag=True, default=False, help='Disable rescaling of input data.')
@click.option('--remove-dims', 'remove_dims', is_flag=True, default=False, help='Remove dimensions features from the data (after eventual rescaling).')
@click.option('--time-sc-enc', 'addSinCosTimeEnc', is_flag=True, default=False, help='Add sine-cosine time encoding to the data.')
@click.option('-F', '--flatten-time', 'flatten_time', is_flag=True, default=False, help='Flatten time dimension into features.')
@click.option('--no-heading-enc', 'no_heading_enc', is_flag=True, default=False, help='Disable heading encoding in sin+cos coordinates.')
def main(data_path, radius_threshold, active_label, frames_num, no_rescaling, remove_dims, addSinCosTimeEnc, flatten_time, no_heading_enc):
    builder = GraphsBuilder(
        Path(data_path),
        frames_num=frames_num,
        m_radius=radius_threshold,
        addSinCosTimeEnc=addSinCosTimeEnc,
        rscToCenter= not no_rescaling,
        removeDims=remove_dims,
        heading_enc = not no_heading_enc,
        flatten_time=flatten_time,
        active_labels=[active_label],
    )
    builder.save()

if __name__ == '__main__':
    main()