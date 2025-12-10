import click
from src.gbuilder import GraphsBuilder
from pathlib import Path

DEF_FNUM = 20
DEF_RADIUS = 30.0

@click.command()
@click.argument('data_path', type=click.Path(exists=True), required=True)
@click.option('-r', '--radius-threshold', 'radius_threshold', type=float, default=DEF_RADIUS, help=f'Radius threshold for graph building (default: {DEF_RADIUS}).')
@click.option('-l', '--active-label', 'active_label', type=int, help='Active label (one only) to consider during graph building.', required=True, prompt="Choose index of the active label:")
@click.option('-f', '--frames-num', 'frames_num', type=int, default=DEF_FNUM, help=f'Number of frames to process (default: {DEF_FNUM}).')
@click.option('--no-rescaling', 'no_rescaling', is_flag=True, default=False, help='Disable rescaling of input data.')
@click.option('--remove-dims', 'remove_dims', is_flag=True, default=False, help='Remove dimensions features from the data (after eventual rescaling).')
@click.option('--no-heading-enc', 'no_heading_enc', is_flag=True, default=False, help='Disable heading encoding in sin+cos coordinates.')
@click.option('-F','--flatten-time-as-graphs', is_flag=True, default=False, help='Flatten time dimension as graphs (build many disconnected graphs, one per timeframe). This forces --no-aggregate-edges even if specified otherwise!')
@click.option('--aggregate-edges/--no-aggregate-edges', 'aggregate_edges', default=True, help='Enable or disable edge feature aggregation when building graphs.')
def main(data_path, radius_threshold, active_label, frames_num, no_rescaling, remove_dims, no_heading_enc, flatten_time_as_graphs, aggregate_edges):
    builder = GraphsBuilder(
        Path(data_path),
        frames_num=frames_num,
        m_radius=radius_threshold,
        rscToCenter= not no_rescaling,
        removeDims=remove_dims,
        heading_enc = not no_heading_enc,
        flatten_time_as_graphs=flatten_time_as_graphs,
        active_labels=[active_label],
        aggregate_edges=aggregate_edges
    )
    builder.save()

if __name__ == '__main__':
    main()