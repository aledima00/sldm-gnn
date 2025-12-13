import torch
from torch_geometric.loader import DataLoader as GDL
import torch_geometric.transforms as T
from pathlib import Path
import colorama
from colorama import Fore, Style
import numpy as np
from typing import Literal as Lit
from matplotlib import pyplot as plt
import click
import re

from src.dataset import MapGraph
from src.models.grusage import GruSage
from src.models.sagegru import SageGru
from src.models.grugat import GRUGAT
from src.models.grufc import GruFC
import src.transforms as TFs
from src.utils import train_model, split_tr_ev_3to1, MetaData

colorama.init(autoreset=True)

ModelOptsType = Lit['grusage','sagegru','grugat', 'grufc']


# general params
EMB_DIM = 8
NUM_POSSIBLE_STATION_TYPES = 256
VERBOSE = False
PROGRESS_LOGGING = 'clilog'  # options: 'clilog', 'tqdm', 'none'
DF_ACTIVE_LABELS = [0,1,2,3,4,5,6,7,8]

# training params
EPOCHS = 200
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 5e-5

# ------------------- Data augmentation params -------------------
TF_ROTATE = False
TF_POS_NOISE = False
POS_NOISE_STD = 0.2

# ------------------- GRUSAGE parameters -------------------
GS_GRU_HIDDEN_SIZE = 48
GS_GRU_NUM_LAYERS=1
GS_FC1_DIMS = [48]
GS_SAGE_HIDDEN_DIMS = [48, 48]
GS_FC2_DIMS = [16]
GS_DROPOUT = 0.1
GS_NEGSLOPE = None
GS_GPOOLING = 'double'

# ------------------- SAGEGRU parameters -------------------
SG_SAGE_HIDDEN_DIMS = [32, 32]
SG_FC1_DIMS = [64]
SG_GRU_HIDDEN_SIZE = 64
SG_GRU_NUM_LAYERS=1
SG_FC2_DIMS = [32,16]
SG_DROPOUT = 0.1
SG_NEGSLOPE = 0.01

# ------------------- GRUGAT parameters -------------------
GG_GRU_HIDDEN_SIZE = 48
GG_GRU_NUM_LAYERS = 1
GG_FCDIMS1 = [48]
GG_GAT_HIDDEN_DIMS = [48, 48]
GG_GAT_NHEADS = 2
GG_GAT_HEADS_CONCAT = True
GG_FCDIMS2 = [16]
GG_DROPOUT = 0.1
GG_NEGSLOPE = None
GG_GPOOLING = 'double'

# ------------------- GRUFC parameters -------------------
GF_GRU_HIDDEN_SIZE = 48
GF_GRU_NUM_LAYERS = 1
GF_FCDIMS = [48, 16]
GF_DROPOUT = 0.1
GF_NEGSLOPE = None
GF_GPOOLING = 'double'

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    

def psplit(ds:MapGraph):
    d_train,d_eval = split_tr_ev_3to1(ds)
    print(f"{Style.DIM}Train set length: {len(d_train)}{Style.RESET_ALL}")
    print(f"{Style.DIM}Validation set length: {len(d_eval)}{Style.RESET_ALL}")
    return d_train, d_eval

def stripnum(match:re.Match)->str:
    sign = match.group(1).replace('+','')
    num = match.group(2)
    if int(num) == 0:
        return ''
    else:
        return f"E{sign}{num}"

def getPlotFname(model:ModelOptsType, outdir:Path)->str:
    #TODO: launch checks at the beginning of the main script
    fnamebase = f"{model.upper()}_RUN_"
    for i in range(1,1001):
        fname = f"{fnamebase}{i:03d}.png"
        if not (outdir / fname).exists():
            return fname
        
def getParams(model:ModelOptsType) -> str:
    #TODO: improve formatting
    """ Parameters as string for plot text box """
    match model:
        case 'grusage':
            params = f"GRUSAGE model parameters:\n - GRU hidden size: {GS_GRU_HIDDEN_SIZE}\n - GRU num layers: {GS_GRU_NUM_LAYERS}\n - FC1 dims: {GS_FC1_DIMS}\n - SAGE hidden dims: {GS_SAGE_HIDDEN_DIMS}\n - FC2 dims: {GS_FC2_DIMS}\n - Dropout: {GS_DROPOUT}\n - ReLU Neg. slope: {GS_NEGSLOPE}\n"
        case 'sagegru':
            params = f"SAGEGRU model parameters:\n - SAGE hidden dims: {SG_SAGE_HIDDEN_DIMS}\n - FC1 dims: {SG_FC1_DIMS}\n - GRU hidden size: {SG_GRU_HIDDEN_SIZE}\n - GRU num layers: {SG_GRU_NUM_LAYERS}\n - FC2 dims: {SG_FC2_DIMS}\n - Dropout: {SG_DROPOUT}\n - ReLU Neg. slope: {SG_NEGSLOPE}\n"
        case 'grugat':
            params = f"GRUGAT model parameters:\n - GRU hidden size: {GG_GRU_HIDDEN_SIZE}\n - GRU num layers: {GG_GRU_NUM_LAYERS}\n - FC1 dims: {GG_FCDIMS1}\n - GAT hidden dims: {GG_GAT_HIDDEN_DIMS}\n - GAT nheads: {GG_GAT_NHEADS}\n - FC2 dims: {GG_FCDIMS2}\n - Dropout: {GG_DROPOUT}\n - ReLU Neg. slope: {GG_NEGSLOPE}\n"
        case 'grufc':
            params = f"GRUFC model parameters:\n - GRU hidden size: {GF_GRU_HIDDEN_SIZE}\n - GRU num layers: {GF_GRU_NUM_LAYERS}\n - FC dims: {GF_FCDIMS}\n - Dropout: {GF_DROPOUT}\n - ReLU Neg. slope: {GF_NEGSLOPE}\n"
        case _:
            raise ValueError(f"Unknown model type: {model}")
        
    params += f"Embedding size for station types: {EMB_DIM}\n"
    params += f"Training Parameters:\n - Epochs: {EPOCHS}\n - Batch size: {BATCH_SIZE}\n - Learning rate: {LR}\n - Weight decay: {WEIGHT_DECAY}\n"
    params += "Data Augmentation:\n"
    if TF_ROTATE:
        params += " - Random Rotate\n"
    if TF_POS_NOISE:
        params += f" - Add Noise on Positions (X,Y) with std: {POS_NOISE_STD}\n"
        
    return params
    
@click.command()
@click.argument('inputdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.argument('outdir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.option('-l', '--label-num', 'lbnum', type=int, required=True, prompt='Label number to train the model on')
@click.option('-m', '--model', 'model', type=click.Choice(ModelOptsType.__args__, case_sensitive=False), required=True, prompt='Choose model', help='Model to use')
def main(inputdir,outdir,lbnum:int, model:str):

    path = inputdir.resolve()
    gpath = path / '.graphs'
    metapath = gpath / 'metadata.json'
    metadata = MetaData.loadJson(metapath)
    outpath = outdir.resolve()
    outpath.mkdir(parents=True, exist_ok=True)

    plt_yticks = np.arange(-0.1, 1.2, 0.1)


    # string with all params in exp format
    pfname = getPlotFname(model, outpath)

    transform = []
    if TF_ROTATE:
        transform.append( TFs.RandomRotate(metadata=metadata) )
    if TF_POS_NOISE:
        transform.append( TFs.AddNoise(target='pos', std=POS_NOISE_STD, metadata=metadata) )
    
    transform = T.Compose(transform)

    # dataset and transforms
    ds = MapGraph(gpath, device=DEVICE, transform=transform, normalizeZScore=True, metadata=metadata)
    
    print(f" - Using device: {DEVICE}")
    print(f" - Dataset length: {len(ds)}")

    match model:
        case 'grusage':
            runner = runGruSage
        case 'sagegru':
            runner = runSageGru
        case 'grugat':
            runner = runGruGat
        case 'grufc':
            runner = runGruFc
        case _:
            raise ValueError(f"Unknown model type: {model}")
    
    (tot_tracc, tot_vacc) = runner(ds, metadata=metadata)
    fig, (ax_plot, ax_text) = plt.subplots(
        1, 2,
        figsize=(10,4),
        gridspec_kw={'width_ratios': [3, 2]}
    )
    ax_plot.plot(tot_vacc[0,:], label='Val. Acc.')
    ax_plot.plot(tot_tracc[0,:], linestyle='--', label='Tr. Acc.')
    ax_plot.set_ylim(bottom=0,top=1)
    ax_plot.set_yticks(plt_yticks)
    ax_plot.grid(True)
    ax_plot.legend()
    ax_plot.set_title(f'Validation Accuracy for label #{lbnum}')
    
    # text box with final results
    params_text = getParams(model)
    ax_text.axis('off')
    ax_text.text(0,0.95, params_text, va='top')

    fig.tight_layout()
    plt.savefig(outpath / pfname )
    plt.close(fig)

def runGruGat(ds:MapGraph,metadata:MetaData):
    model = GRUGAT(
        dynamic_features_num=metadata.n_node_temporal_features,
        has_dims=metadata.has_dims,
        has_aggregated_edges=metadata.aggregate_edges,
        frames_num=metadata.frames_num,
        emb_dim=EMB_DIM,
        gru_hidden_size=GG_GRU_HIDDEN_SIZE,
        gru_num_layers=GG_GRU_NUM_LAYERS,
        fc1dims=GG_FCDIMS1,
        gat_edge_fnum=None,
        gat_inner_dims=GG_GAT_HIDDEN_DIMS,
        gat_nheads = GG_GAT_NHEADS,
        fc2dims=GG_FCDIMS2,
        out_dim=len(metadata.active_labels),
        num_st_types=NUM_POSSIBLE_STATION_TYPES,
        dropout=GG_DROPOUT,
        negative_slope=GG_NEGSLOPE,
        gat_concat=GG_GAT_HEADS_CONCAT,
        global_pooling=GG_GPOOLING
    )
    d_train,d_eval = psplit(ds)
    # create data loaders
    dl_train = GDL(d_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_eval = GDL(d_eval, batch_size=BATCH_SIZE, shuffle=True)

    (_, tot_tracc),(_, tot_vacc) = train_model(
        model,
        dl_train,
        dl_eval,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
        verbose=VERBOSE,
        progress_logging=PROGRESS_LOGGING,
        active_labels=metadata.active_labels,
        neg_over_pos_ratio=metadata.getNegOverPosRatio()
    )
    return (tot_tracc, tot_vacc)

def runGruSage(ds:MapGraph,metadata:MetaData):
    model = GruSage(
        dynamic_features_num=metadata.n_node_temporal_features,
        has_dims=metadata.has_dims,
        has_aggregated_edges=metadata.aggregate_edges,
        frames_num=metadata.frames_num,
        gru_hidden_size=GS_GRU_HIDDEN_SIZE,
        gru_num_layers=GS_GRU_NUM_LAYERS,
        fc1dims=GS_FC1_DIMS,
        sage_hidden_dims=GS_SAGE_HIDDEN_DIMS,
        fc2dims=GS_FC2_DIMS,
        out_dim=len(metadata.active_labels),
        num_st_types=NUM_POSSIBLE_STATION_TYPES,
        emb_dim=EMB_DIM,
        dropout=GS_DROPOUT,
        negative_slope=GS_NEGSLOPE,
        global_pooling=GS_GPOOLING
    )
    d_train,d_eval = psplit(ds)
    # create data loaders
    dl_train = GDL(d_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_eval = GDL(d_eval, batch_size=BATCH_SIZE, shuffle=True)

    (_, tot_tracc),(_, tot_vacc) = train_model(
        model,
        dl_train,
        dl_eval,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
        verbose=VERBOSE,
        progress_logging=PROGRESS_LOGGING,
        active_labels=metadata.active_labels,
        neg_over_pos_ratio=metadata.getNegOverPosRatio()
    )
    return (tot_tracc, tot_vacc)

def runSageGru(ds:MapGraph,metadata:MetaData):
    model = SageGru(
        batch_size=BATCH_SIZE,
        dynamic_features_num=metadata.n_node_temporal_features,
        has_dims=metadata.has_dims,
        frames_num=metadata.frames_num,
        sage_hidden_dims=SG_SAGE_HIDDEN_DIMS,
        fc1dims=SG_FC1_DIMS,
        gru_hidden_size=SG_GRU_HIDDEN_SIZE,
        gru_num_layers=SG_GRU_NUM_LAYERS,
        fc2dims=SG_FC2_DIMS,
        out_dim=len(metadata.active_labels),
        num_st_types=NUM_POSSIBLE_STATION_TYPES,
        emb_dim=EMB_DIM,
        dropout=SG_DROPOUT,
        negative_slope=SG_NEGSLOPE,
        global_pooling='double'
    )
    d_train,d_eval = psplit(ds)
    # create data loaders
    dl_train = GDL(d_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_eval = GDL(d_eval, batch_size=BATCH_SIZE, shuffle=True)

    (_, tot_tracc),(_, tot_vacc) = train_model(
        model,
        dl_train,
        dl_eval,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
        verbose=VERBOSE,
        progress_logging=PROGRESS_LOGGING,
        active_labels=metadata.active_labels,
        neg_over_pos_ratio=metadata.getNegOverPosRatio()
    )
    return (tot_tracc, tot_vacc)

def runGruFc(ds:MapGraph,metadata:MetaData):
    model = GruFC(
        dynamic_features_num=metadata.n_node_temporal_features,
        has_dims=metadata.has_dims,
        frames_num=metadata.frames_num,
        gru_hidden_size=GF_GRU_HIDDEN_SIZE,
        gru_num_layers=GF_GRU_NUM_LAYERS,
        fc_dims=GF_FCDIMS,
        out_dim=len(metadata.active_labels),
        num_st_types=NUM_POSSIBLE_STATION_TYPES,
        emb_dim=EMB_DIM,
        dropout=GF_DROPOUT,
        negative_slope=GF_NEGSLOPE,
        global_pooling=GF_GPOOLING
    )
    d_train,d_eval = psplit(ds)
    # create data loaders
    dl_train = GDL(d_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_eval = GDL(d_eval, batch_size=BATCH_SIZE, shuffle=True)

    (_, tot_tracc),(_, tot_vacc) = train_model(
        model,
        dl_train,
        dl_eval,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
        verbose=VERBOSE,
        progress_logging=PROGRESS_LOGGING,
        active_labels=metadata.active_labels,
        neg_over_pos_ratio=metadata.getNegOverPosRatio()
    )
    return (tot_tracc, tot_vacc)

if __name__ == '__main__':
    main()
