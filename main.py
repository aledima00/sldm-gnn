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
from src.sage import GraphSAGEGraphLevel
from src.gat import GRUGAT
import src.transforms as TFs
from src.utils import train_model, split_tr_ev_3to1, MetaData

colorama.init(autoreset=True)


# general params
EMB_DIM = 12
NUM_POSSIBLE_STATION_TYPES = 256
VERBOSE = False
PROGRESS_LOGGING = 'clilog'  # options: 'clilog', 'tqdm', 'none'
DF_ACTIVE_LABELS = [0,1,2,3,4,5,6,7,8]
POS_NOISE_STD = 0.1

# ------------------- SAGE parameters -------------------
SAGE_NUM_TEMPORAL_FEATURES = 8
SAGE_HIDDEN_DIMS = [128, 64]
SAGE_FC_DIMS = [50, 50]
SAGE_DROPOUT = 0.2
# training params
SAGE_EPOCHS = 10
SAGE_BATCH_SIZE = 32
SAGE_LR = 0.005
SAGE_WEIGHT_DECAY = 5e-4

# ------------------- GRUGAT parameters -------------------
GG_NUM_TEMPORAL_FEATURES = 6 # NO SIN-COS TIME ENCODING
GG_GRU_HIDDEN_SIZE = 128
GG_GRU_NUM_LAYERS = 1
GG_GAT_EDGE_FNUM = 1
GG_GAT_HIDDEN_DIMS = [96,96]
GG_GAT_NHEADS = 8
GG_FCDIMS = []
GG_DROPOUT = 0.05
GG_NEGSLOPE = 0.2
# training params
GG_EPOCHS = 10
GG_BATCH_SIZE = 32
GG_LR = 1e-3
GG_WEIGHT_DECAY = 5e-4

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

Model_opts_type = Lit['sage','gat']

    

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

def getPlotFname(model:Lit['sage','gat'], lbnum:int)->str:
    if model == 'sage':
        mstr = f"RUN_LB{lbnum}_SAGE_HDIMS{ 'x'.join([str(h) for h in SAGE_HIDDEN_DIMS])}_FCDIMS{ 'x'.join([str(f) for f in SAGE_FC_DIMS])}_EMB{EMB_DIM}_DRP{SAGE_DROPOUT:.1e}"
        pstr = f"EP{SAGE_EPOCHS}_BS{SAGE_BATCH_SIZE}_LR{SAGE_LR:.1e}_WD{SAGE_WEIGHT_DECAY:.1e}"
    elif model == 'gat':
        mstr = f"RUN_LB{lbnum}_GRUHS{GG_GRU_HIDDEN_SIZE}_GRUNL{GG_GRU_NUM_LAYERS}_GEFN{GG_GAT_EDGE_FNUM}_GTHDIMS{'x'.join([str(h) for h in GG_GAT_HIDDEN_DIMS])}_GTNHDS{GG_GAT_NHEADS}_FCDIMS{'x'.join([str(f) for f in GG_FCDIMS])}_EMB{EMB_DIM}_DRP{GG_DROPOUT:.1e}"
        pstr = f"EP{GG_EPOCHS}_BS{GG_BATCH_SIZE}_LR{GG_LR:.1e}_WD{GG_WEIGHT_DECAY:.1e}"
    else:
        raise ValueError(f"Unknown model type: {model}")
    tot_str = f"{mstr}__{pstr}"
    tot_str = re.sub(r'e([+-]?)(\d+)',stripnum,tot_str)
    return tot_str.replace('+','').replace('.','p').replace('-','n') + '.png'
    
@click.command()
@click.argument('inputdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.argument('outdir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.option('-l', '--label-num', 'lbnum', type=int, required=True, prompt='Label number to train the model on')
@click.option('-m', '--model', 'model', type=click.Choice(['sage','gat'], case_sensitive=False), required=True, prompt='Choose GNN model', help='GNN model to use')
def main(inputdir,outdir,lbnum:int, model:str):

    path = inputdir.resolve()
    gpath = path / '.graphs'
    metapath = gpath / 'metadata.json'
    metadata = MetaData.loadJson(metapath)
    outpath = outdir.resolve()
    outpath.mkdir(parents=True, exist_ok=True)

    plt_yticks = np.arange(-0.1, 1.2, 0.1)


    # string with all params in exp format
    pfname = getPlotFname(model,lbnum)
    
    transform = T.Compose([
        TFs.RandomRotate(metadata=metadata),
        TFs.AddNoise(target='pos', std=POS_NOISE_STD, metadata=metadata),
    ])

    # dataset and transforms
    ds = MapGraph(gpath, device=DEVICE, transform=transform, normalizeZScore=True, metadata=metadata)
    
    print(f" - Using device: {DEVICE}")
    print(f" - Dataset length: {len(ds)}")

    match model:
        case 'sage':
            runner = runSage
        case 'gat':
            runner = runGruGat
        case _:
            raise ValueError(f"Unknown model type: {model}")
    
    (tot_tracc, tot_vacc) = runner(ds, metadata=metadata)
    fig = plt.figure()
    plt.plot(tot_vacc[0,:], label='Val. Acc.')
    plt.plot(tot_tracc[0,:], linestyle='--', label='Tr. Acc.')
    plt.ylim(bottom=0,top=1)
    plt.yticks(plt_yticks)
    plt.grid(True)
    plt.legend()
    plt.title(f'Validation Accuracy for label #{lbnum}')
    plt.savefig(outpath / pfname )
    plt.close(fig)

def runGruGat(ds:MapGraph,metadata:MetaData):
    model = GRUGAT(
        dynamic_features_num=GG_NUM_TEMPORAL_FEATURES,
        has_dims=metadata.has_dims,
        gru_hidden_size=GG_GRU_HIDDEN_SIZE,
        gru_num_layers=GG_GRU_NUM_LAYERS,
        gat_edge_fnum=GG_GAT_EDGE_FNUM,
        gat_inner_dims=GG_GAT_HIDDEN_DIMS,
        gat_nheads = GG_GAT_NHEADS,
        fc_dims=GG_FCDIMS,
        out_dim=len(metadata.active_labels),
        num_st_types=NUM_POSSIBLE_STATION_TYPES,
        emb_dim=EMB_DIM,
        negative_slope=GG_NEGSLOPE,
        dropout=GG_DROPOUT
    )
    d_train,d_eval = psplit(ds)
    # create data loaders
    dl_train = GDL(d_train, batch_size=GG_BATCH_SIZE, shuffle=True)
    dl_eval = GDL(d_eval, batch_size=GG_BATCH_SIZE, shuffle=True)

    (_, tot_tracc),(_, tot_vacc) = train_model(
        model,
        dl_train,
        dl_eval,
        epochs=GG_EPOCHS,
        lr=GG_LR,
        weight_decay=GG_WEIGHT_DECAY,
        device=DEVICE,
        verbose=VERBOSE,
        progress_logging=PROGRESS_LOGGING,
        active_labels=metadata.active_labels
    )
    return (tot_tracc, tot_vacc)

def runSage(ds:MapGraph,metadata:MetaData):
    model = GraphSAGEGraphLevel(
        dynamic_features_num=SAGE_NUM_TEMPORAL_FEATURES,
        has_dims=metadata.has_dims,
        frames_num=metadata.frames_num,
        hidden_dims=SAGE_HIDDEN_DIMS,
        fcdims=SAGE_FC_DIMS,
        out_dim=len(metadata.active_labels),
        num_st_types=NUM_POSSIBLE_STATION_TYPES,
        emb_dim=EMB_DIM,
        dropout=SAGE_DROPOUT
    )
    d_train,d_eval = psplit(ds)
    # create data loaders
    dl_train = GDL(d_train, batch_size=SAGE_BATCH_SIZE, shuffle=True)
    dl_eval = GDL(d_eval, batch_size=SAGE_BATCH_SIZE, shuffle=True)

    (_, tot_tracc),(_, tot_vacc) = train_model(
        model,
        dl_train,
        dl_eval,
        epochs=SAGE_EPOCHS,
        lr=SAGE_LR,
        weight_decay=SAGE_WEIGHT_DECAY,
        device=DEVICE,
        verbose=VERBOSE,
        progress_logging=PROGRESS_LOGGING,
        active_labels=metadata.active_labels
    )
    return (tot_tracc, tot_vacc)

if __name__ == '__main__':
    main()
