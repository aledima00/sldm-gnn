import click
from src.gbuilder import GraphsBuilder, MapBuilder
from pathlib import Path

DEF_FNUM = 20
DEF_RADIUS = 30.0
DEF_MAP_LAT_CONN_MAX_ANGLE = 30.0
DEF_MAP_LAT_CONN_PROXIMITY_THRESHOLD = 1.0

@click.command()
@click.argument('data_path', type=click.Path(exists=True), required=True)
@click.option('-r', '--radius-threshold', 'radius_threshold', type=float, default=DEF_RADIUS, help=f'Radius threshold for graph building (default: {DEF_RADIUS}).')
@click.option('-l', '--active-label', 'active_label', type=int, help='Active label (one only) to consider during graph building.', required=True, prompt="Choose index of the active label:")
@click.option('-f', '--frames-num', 'frames_num', type=int, default=DEF_FNUM, help=f'Number of frames to process (default: {DEF_FNUM}).')
@click.option('--map-only', is_flag=True, default=False, help='If specified, only build the map without building graphs.')
@click.option('--map.lat-conn.max-angle', 'map_lat_conn_max_angle', type=float, default=DEF_MAP_LAT_CONN_MAX_ANGLE, help=f'Maximum angle (in degrees) for lateral connections in the map building step (default: {DEF_MAP_LAT_CONN_MAX_ANGLE}).')
@click.option('--map.lat-conn.proximity-threshold', 'map_lat_conn_proximity_threshold', type=float, default=DEF_MAP_LAT_CONN_PROXIMITY_THRESHOLD, help=f'Proximity threshold (in meters) for lateral connections in the map building step (default: {DEF_MAP_LAT_CONN_PROXIMITY_THRESHOLD}).')
def main(data_path, radius_threshold, active_label, frames_num, map_only, map_lat_conn_max_angle, map_lat_conn_proximity_threshold):
    dp = Path(data_path).resolve()
    map_filepath = dp / 'vmap.parquet'

    click.echo(f"Building common map...")
    map_builder = MapBuilder(
        map_filepath,
        lat_conn_max_angle_deg=map_lat_conn_max_angle,
        lat_conn_proximity_threshold=map_lat_conn_proximity_threshold
    )
    map_builder.save()
    if map_only:
        return
    
    train_dirpath = dp / 'train'
    eval_dirpath = dp / 'eval'

    click.echo(f'Building train split graphs...')
    train_builder = GraphsBuilder(
        train_dirpath,
        frames_num=frames_num,
        m_radius=radius_threshold,
        active_labels=[active_label],
    )
    train_builder.save()

    click.echo(f'Building eval split graphs...')
    eval_builder = GraphsBuilder(
        eval_dirpath,
        frames_num=frames_num,
        m_radius=radius_threshold,
        active_labels=[active_label],
    )
    eval_builder.save()

if __name__ == '__main__':
    main()