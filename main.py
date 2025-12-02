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

from src.dataset import MapGraph
from src.sage import GraphSAGEGraphLevel
from src.gat import GATGraphLevel
import src.transforms as TFs
from src.labels import LabelsEnum
from src.utils import train_model, Progress_logging_options, split_tr_ev_3to1, MetaData

colorama.init(autoreset=True)



# data description
EMB_DIM = 12
NUM_POSSIBLE_STATION_TYPES = 256
NUM_TEMPORAL_FEATURES = 8
NUM_STATIC_FEATURES = 2
FRAMES_PER_PACK = 20

# learning parameters
DF_EPOCHS = 100
DF_BATCH_SIZE = 32
DF_LR = 0.005
DF_WEIGHT_DECAY = 5e-4
DF_ACTIVE_LABELS = [0,1,2,3,4,5,6,7,8]
DF_DROPOUT = 0.2

# gnn parameters
SAGE_HIDDEN_DIMS = [128, 64]
GAT_HIDDEN_DIMS = [128, 128]
# only for GAT
GAT_ATTENTION_HEADS = 8

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

Model_opts_type = Lit['sage','gat']

def rungnn(ds:MapGraph, verbose:bool=False, model:Model_opts_type='sage', progress_logging:Progress_logging_options='clilog',*,active_labels, wd:float,lr:float,bs:int, epochs:int,dropout:float,hidden_dims:list[int],emb_dim:int):
    in_dim = FRAMES_PER_PACK * NUM_TEMPORAL_FEATURES  + NUM_STATIC_FEATURES
    print(f" - Using device: {DEVICE}")
    print(f" - Dataset length: {len(ds)}")

    # split train and eval
    d_train,d_eval = split_tr_ev_3to1(ds)
    print(f"{Style.DIM}Train set length: {len(d_train)}{Style.RESET_ALL}")
    print(f"{Style.DIM}Validation set length: {len(d_eval)}{Style.RESET_ALL}")

    # create data loaders
    dl_train = GDL(d_train, batch_size=bs, shuffle=True)
    dl_eval = GDL(d_eval, batch_size=bs, shuffle=False)

    match model:
        case 'sage':
            print(f"{Fore.CYAN}Using GraphSAGE model.{Style.RESET_ALL}")
            model = GraphSAGEGraphLevel(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=len(active_labels), num_st_types=NUM_POSSIBLE_STATION_TYPES, emb_dim=emb_dim, dropout=dropout)
        case 'gat':
            print(f"{Fore.CYAN}Using GAT model.{Style.RESET_ALL}")
            model = GATGraphLevel(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=len(active_labels), num_st_types=NUM_POSSIBLE_STATION_TYPES, emb_dim=emb_dim, heads=8, dropout=dropout)
        case _:
            raise ValueError(f"Unknown model type: {model}")
    return train_model(model,dl_train,dl_eval,epochs=epochs,lr=lr,weight_decay=wd,device=DEVICE,verbose=verbose, progress_logging=progress_logging, active_labels=active_labels)


def autorun():
    path = Path(__file__).resolve().parent / 'input' / 'random_random_rsc'
    outpath = Path(__file__).resolve().parent / 'out' / '2'
    plt_yticks = np.arange(-0.1, 1.2, 0.1)

    lr = DF_LR
    bs = DF_BATCH_SIZE
    wd = DF_WEIGHT_DECAY
    epochs = DF_EPOCHS

    # ----------------------------- compare results -----------------------------
    # 1) SLT: Single-Label Trained models
    # 2) ALT: All-Labels Trained model

    (alt_pl_tracc, alt_tot_tracc),(alt_pl_vacc, alt_tot_vacc) = rungnn(
        path,
        active_labels=[lb.value for lb in LabelsEnum], 
        progress_logging='tqdm',
        save=False,
        epochs=epochs,
        lr=lr,
        wd=wd,
        bs=bs
    )
    slt_avg_tot_tracc = np.zeros_like(alt_tot_tracc)
    slt_avg_tot_vacc = np.zeros_like(alt_tot_vacc)

    for lb in LabelsEnum:
        print(f" ------------------------ LABEL #{lb.value} ({lb.name}) ------------------------ ")
        lbval = lb.value
        ((_, tot_tracc), (_, tot_vacc)) = rungnn(
            path,
            active_labels=[lbval],
            progress_logging='tqdm',
            save=False,
            epochs=epochs,
            lr=lr,
            wd=wd,
            bs=bs
        )
        fig = plt.figure()
        plt.plot(tot_vacc[0,:], label=f'SLT Model - Val. Acc.')
        plt.plot(alt_pl_vacc[lbval,:], label='ALT Model - Val. Acc.')
        plt.plot(tot_tracc[0,:], linestyle='--', label=f'SLT Model - Tr. Acc.')
        plt.plot(alt_pl_tracc[lbval,:], linestyle='--', label='ALT Model - Tr. Acc.')
        plt.ylim(bottom=0,top=1)
        plt.yticks(plt_yticks)
        plt.grid(True)
        plt.legend()
        plt.title(f'Validation Accuracy for label #{lb.value} - "{lb.name}"')
        plt.savefig(outpath / f'label_{lb.value}_cmp.png')
        plt.close(fig)

        slt_avg_tot_tracc += tot_tracc
        slt_avg_tot_vacc += tot_vacc

    
    slt_avg_tot_tracc /= len(LabelsEnum)
    slt_avg_tot_vacc /= len(LabelsEnum)
    fig = plt.figure()
    plt.plot(alt_tot_vacc[0,:], label='ALT Model - Val. Acc.')
    plt.plot(alt_tot_tracc[0,:], label='ALT Model - Tr. Acc.', linestyle='--')
    plt.plot(slt_avg_tot_vacc[0,:], label='SLT Avg. - Val. Acc.')
    plt.plot(slt_avg_tot_tracc[0,:], label='SLT Avg. - Tr. Acc.', linestyle='--')
    plt.title('Overall Accuracy (all labels)')
    plt.legend()
    plt.ylim(bottom=0,top=1)
    plt.yticks(plt_yticks)
    plt.grid(True)
    plt.savefig(outpath / f'overall_tot_acc_bs{bs}_lr{lr}_wd{wd}.png')
    plt.close(fig)

@click.command()
@click.argument('inputdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.argument('outdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.option('-l', '--label-num', 'lbnum', type=int, required=True, prompt='Label number to train the model on')
def runModel(inputdir,outdir,lbnum:int):
    path = inputdir.resolve()
    gpath = path / '.graphs'
    metapath = gpath / 'metadata.json'
    metadata = MetaData.loadJson(metapath)
    outpath = outdir.resolve()
    outpath.mkdir(parents=True, exist_ok=True)

    plt_yticks = np.arange(-0.1, 1.2, 0.1)

    lr = 4e-3
    bs = 32
    wd = 1e-5
    epochs = 10
    dropout = 0.3
    hidden_dims = [128, 64]
    emb_dim = 10
    pos_noise = 0.15

    # string with all params in exp format
    params_str = f"LR{lr:.2e}_BS{bs:02d}_WD{wd:.2e}_EP{epochs:02d}_DRP{dropout:.2f}_HID{'x'.join([str(h) for h in hidden_dims])}_EMB{emb_dim:02d}_PNS{pos_noise:.2e}"\
        .replace('.','p')\
        .replace('-','m')
    
    transform = T.Compose([
        TFs.RandomRotate(metadata=metadata),
        TFs.AddNoise(target='pos', std=pos_noise, metadata=metadata),
    ])

    # dataset and transforms
    ds = MapGraph(gpath, device=DEVICE, transform=transform, normalizeZScore=True, metadata=metadata)
    

    (_, tot_tracc),(_, tot_vacc) = rungnn(
        ds,
        verbose=False,
        model='sage',
        progress_logging='clilog',
        active_labels=[lbnum],
        wd=wd,
        lr=lr,
        bs=bs,
        epochs=epochs,
        dropout=dropout,
        hidden_dims=hidden_dims,
        emb_dim=emb_dim
    )
    fig = plt.figure()
    plt.plot(tot_vacc[0,:], label='Val. Acc.')
    plt.plot(tot_tracc[0,:], linestyle='--', label='Tr. Acc.')
    plt.ylim(bottom=0,top=1)
    plt.yticks(plt_yticks)
    plt.grid(True)
    plt.legend()
    plt.title(f'Validation Accuracy for label #{lbnum}')
    plt.savefig(outpath / f'RUN_{params_str}.png')
    plt.close(fig)


if __name__ == '__main__':
    #autorun()
    runModel()
