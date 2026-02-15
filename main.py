import torch
from torch_geometric.loader import DataLoader as GDL
import torch_geometric.transforms as T
from pathlib import Path
import colorama
from colorama import Fore, Style, Back
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
from src.utils import train_model, split_tr_ev_3to1, MetaData, ParamSweepContext
colorama.init(autoreset=True)

PROGRESS_LOGGING = 'clilog'  # options: 'clilog', 'tqdm', 'none'


GRUSAGE_PARAMS_DICT = {
    "epochs":[200],
    "batch_size":[32],
    "lr":[3e-4],
    "weight_decay":[5e-5],

    "tf_rotate":[False], #FIXME: bad implementation of rotation for map data
    "tf_pos_noise":[True],
    "pos_noise_std":[0.2],
    "pos_noise_std_max":[0.2],
    "pos_noise_prop_to_speed":[True],

    "emb_dim":[8],
    "num_possible_station_types":[256],

    "gs_dropout":[0.1],
    "gs_neg_slope":[0.05],

    "gs_hidden_size":[64],
    "gs_gru_hidden_size":(lambda hs: hs, "gs_hidden_size"),
    "gs_gru_num_layers":[1],
    "gs_fc1_dims":(lambda hs: [],"gs_hidden_size") , #+[] #TODO:REMOVE
    "gs_sage_hidden_dims":(lambda hs: [hs, hs],"gs_hidden_size"),
    "gs_pooling":['double'],
    "gs_fc2_dims":(lambda hs: [hs//3],"gs_hidden_size"),

    "gs_map_hidden_size":[32],
    "gs_mapenc_lane_embdim":(lambda mhs: mhs//4,"gs_map_hidden_size"),
    "gs_mapenc_sage_hdims":(lambda mhs: [mhs, mhs],"gs_map_hidden_size"),
    "gs_map_attention_topk":[5]
}


# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def stripnum(match:re.Match)->str:
    sign = match.group(1).replace('+','')
    num = match.group(2)
    if int(num) == 0:
        return ''
    else:
        return f"E{sign}{num}"

def getPlotFname(outdir:Path,mapIncluded:bool)->str:
    #TODO: launch checks at the beginning of the main script
    fnamebase = f"GRUSAGE_{'MAP_' if mapIncluded else ''}RUN_"
    for i in range(1,1001):
        fname = f"{fnamebase}{i:03d}.png"
        if not (outdir / fname).exists():
            return fname
        
def getParams(bin_stats:tuple|None,  tot_vacc:np.ndarray, cut:int|None=None,*,combDict:dict) -> str:
    """ Parameters as string for plot text box """

    EMB_DIM = combDict.get('emb_dim')
    EPOCHS = combDict.get('epochs')
    BATCH_SIZE = combDict.get('batch_size')
    LR = combDict.get('lr')
    WEIGHT_DECAY = combDict.get('weight_decay')
    GS_GRU_HS = combDict.get('gs_gru_hidden_size')
    GS_GRU_NL = combDict.get('gs_gru_num_layers')
    GS_FC1_DIMS = combDict.get('gs_fc1_dims')
    GS_SAGE_HIDDEN_DIMS = combDict.get('gs_sage_hidden_dims')
    GS_FC2_DIMS = combDict.get('gs_fc2_dims')
    GS_DROPOUT = combDict.get('gs_dropout')
    GS_NEGSLOPE = combDict.get('gs_neg_slope')
    GS_MELD = combDict.get('gs_mapenc_lane_embdim')
    GS_MESD = combDict.get('gs_mapenc_sage_hdims')
    GS_MAPATTENTION_TOPK = combDict.get('gs_map_attention_topk')
    params = f"GRUSAGE model parameters:\n - Embedding size for station types: {EMB_DIM}\n - GRU: hidden size = {GS_GRU_HS}, num layers = {GS_GRU_NL}\n - FC1 dims: {GS_FC1_DIMS}\n - SAGE hidden dims: {GS_SAGE_HIDDEN_DIMS}\n - FC2 dims: {GS_FC2_DIMS}\n - Regularization: Dropout = {GS_DROPOUT}, ReLU Neg. slope = {GS_NEGSLOPE}\nMap Input processing:\n - Map Encoder: Lane emb.dim = {GS_MELD}, Sage HDims = {GS_MESD}\n - Map Spatial Attn: topk = {GS_MAPATTENTION_TOPK}\n"

    params += "\n"
    params += f"Tr. Params: EP: {EPOCHS}, BS: {BATCH_SIZE}, LR: {LR}, WD: {WEIGHT_DECAY}\n"
    params += "Data Augmentation:\n"
    if combDict.get('tf_rotate'):
        params += " - Random Rotate\n"
    if combDict.get('tf_pos_noise'):
        if combDict.get('pos_noise_prop_to_speed'):
            POS_NOISE_STD_MAX = combDict.get('pos_noise_std_max')
            params += f" - Add Noise on Positions (X,Y) prop to speed, with max std: {POS_NOISE_STD_MAX}\n"
        else:
            POS_NOISE_STD = combDict.get('pos_noise_std')
            params += f" - Add Noise on Positions (X,Y) with std: {POS_NOISE_STD}\n"
    if cut is not None:
        params += f" - Cutting after: {cut} frames\n"

    best_vacc_idx = tot_vacc[0,:].argmax()
    best_vacc = tot_vacc[0,best_vacc_idx]
    params += f"\nBest Val. Acc.: {best_vacc:.4f}, @ep.{best_vacc_idx}\n"

    if bin_stats is not None:
        (bin_cm_flat_values, bin_rocauc_values) = bin_stats

        best_rocauc_idx = bin_rocauc_values[0,:].argmax()
        best_rocauc = bin_rocauc_values[0,best_rocauc_idx]
        params += f"Best Val. ROC AUC: {best_rocauc:.4f}, @ep.{best_rocauc_idx}\n"
        
    return params
    
@click.command()
@click.argument('inputdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.argument('outdir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.option('-l', '--label-num', 'lbnum', type=int, required=True, prompt='Label number to train the model on')
@click.option('--cut', type=int, default=None, help='If set, cuts frames after the given number, allowing prediction at earlier timesteps')
@click.option('--include-map', is_flag=True, default=False, help='If set, includes map information as node features (if available in dataset)')
@click.option('-v', '--verbose','verbosity_level', count=True, help='Verbosity level: -v for verbose, -vv for more verbose, -vvv for debug.')
def main(inputdir:Path,outdir:Path,lbnum:int, cut:int|None, include_map:bool, verbosity_level:int):
    psc=ParamSweepContext(GRUSAGE_PARAMS_DICT)
    tot_cmb = len(psc)
    print(f"TOT_COMBINATIONS={tot_cmb}")
    if not click.confirm("Do you want to proceed to train with all of the combinations?",default=True):
        return

    for i,combDict in enumerate(psc.combinations()):
        
        print(f"{Fore.BLACK}{Back.MAGENTA}{Style.BRIGHT}Starting training @ combination {i+1}/{tot_cmb}{Style.RESET_ALL}")

        inpath = inputdir.resolve()
        outpath = outdir.resolve()
        outpath.mkdir(parents=True, exist_ok=True)

        # string with all params in exp format
        pfname = getPlotFname(outpath,mapIncluded=include_map)

        tr_gpath = inpath / 'train' / '.graphs'
        ev_gpath = inpath / 'eval' / '.graphs'
        tr_metadata = MetaData.loadJson(tr_gpath / 'metadata.json')
        ev_metadata = MetaData.loadJson(ev_gpath / 'metadata.json')

        transform = []
        if combDict.get('tf_rotate'):
            transform.append( TFs.RandomRotate(metadata=tr_metadata) )
        if combDict.get('tf_pos_noise'):
            posnoisestd = combDict.get('pos_noise_std')
            posnoisestdmax = combDict.get('pos_noise_std_max')
            posnoiseproptospeed = combDict.get('pos_noise_prop_to_speed')
            transform.append( TFs.AddNoise(target='pos', std=posnoisestdmax if posnoiseproptospeed else posnoisestd, prop_to_speed=posnoiseproptospeed, metadata=tr_metadata) )
        if cut is not None:
            transform.append( TFs.CutFrames(cut) )
        
        transform = T.Compose(transform)
        
        
        print(f" - Using device: {DEVICE}")

        d_train = MapGraph(tr_gpath, device=DEVICE, transform=transform, normalizeZScore=True, metadata=tr_metadata)
        mu_sigma = d_train.getMuSigma()
        d_eval = MapGraph(ev_gpath, device=DEVICE, transform=transform, normalizeZScore=True, metadata=ev_metadata, zscore_mu_sigma=mu_sigma)

        print(f"{Style.DIM}Train set length: {len(d_train)}{Style.RESET_ALL}")
        print(f"{Style.DIM}Validation set length: {len(d_eval)}{Style.RESET_ALL}")
        # create data loaders
        dl_train = GDL(d_train, batch_size=combDict.get('batch_size'), shuffle=True)
        dl_eval = GDL(d_eval, batch_size=combDict.get('batch_size'), shuffle=True)

        # load map data if required
        if include_map:
            map_path = inpath / '.map' / 'vmap.pth'
            map_tensors = torch.load(map_path, map_location=DEVICE)
            print(f"{Style.DIM}Loaded map tensors from {map_path}{Style.RESET_ALL}")
        else:
            map_tensors = None

        model = GruSage(
            dynamic_features_num=tr_metadata.n_node_temporal_features,
            frames_num=tr_metadata.frames_num,
            gru_hidden_size=combDict.get('gs_gru_hidden_size'),
            gru_num_layers=combDict.get('gs_gru_num_layers'),
            fc1dims=combDict.get('gs_fc1_dims'),
            sage_hidden_dims=combDict.get('gs_sage_hidden_dims'),
            fc2dims=combDict.get('gs_fc2_dims'),
            out_dim=len(tr_metadata.active_labels),
            num_st_types=combDict.get('num_possible_station_types'),
            emb_dim=combDict.get('emb_dim'),
            dropout=combDict.get('gs_dropout'),
            negative_slope=combDict.get('gs_neg_slope'),
            global_pooling=combDict.get('gs_pooling'),
            map_tensors=map_tensors,
            mapenc_lane_embdim=combDict.get('gs_mapenc_lane_embdim'),
            mapenc_sage_hdims=combDict.get('gs_mapenc_sage_hdims'),
            map_attention_topk=combDict.get('gs_map_attention_topk')
        )
        
        (tot_tracc, tot_vacc, bin_stats) = runModel(model, tr_metadata, dl_train, dl_eval, verbosity_level=verbosity_level, combDict=combDict)
        plotAccuracies(tot_tracc,tot_vacc,bin_stats, outpath / pfname, lbnum, cut=cut, combDict=combDict)

def plotAccuracies(tot_tracc:np.ndarray, tot_vacc:np.ndarray, bin_stats:tuple|None, outfile:Path,lbnum:int,*,cut,combDict:dict):
    fig, (ax_plot, ax_text) = plt.subplots(
        1, 2,
        figsize=(10,4),
        gridspec_kw={'width_ratios': [3, 2]}
    )
    plt_yticks = np.arange(-0.1, 1.2, 0.1)
    ax_plot.plot(tot_vacc[0,:], color='blue', label='Val. Acc.')
    ax_plot.plot(tot_tracc[0,:], color='orange', linestyle='--', label='Tr. Acc.')

    if bin_stats is not None:
        (bin_cm_flat_values, bin_rocauc_values) = bin_stats
        tn=bin_cm_flat_values[0,:]
        fp=bin_cm_flat_values[1,:]
        fn=bin_cm_flat_values[2,:]
        tp=bin_cm_flat_values[3,:]
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        ax_plot.plot(bin_rocauc_values[0,:], color='purple', label='Val. ROC AUC')
        ax_plot.plot(precision, color='green', alpha=0.2, label='Val. Precision')
        ax_plot.plot(recall, color='red', alpha=0.2, label='Val. Recall')

    ax_plot.set_ylim(bottom=0,top=1)
    ax_plot.set_yticks(plt_yticks)
    ax_plot.grid(True)
    ax_plot.legend()
    ax_plot.set_title(f'Validation Accuracy for label #{lbnum}')
    
    # text box with final results
    params_text = getParams(bin_stats, tot_vacc, cut=cut, combDict=combDict)
    ax_text.axis('off')
    ax_text.text(0,0.95, params_text, va='top')

    fig.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)

def runModel(model,train_metadata:MetaData, dl_train, dl_eval, verbosity_level:int,*,combDict:dict):
    (_, tot_tracc),(_, tot_vacc), bin_stats = train_model(
        model,
        dl_train,
        dl_eval,
        epochs=combDict.get('epochs'),
        lr=combDict.get('lr'),
        weight_decay=combDict.get('weight_decay'),
        device=DEVICE,
        verbose=verbosity_level>=1,#TODO: implement fine-grained verbosity levels
        progress_logging=PROGRESS_LOGGING,
        active_labels=train_metadata.active_labels,
        neg_over_pos_ratio=train_metadata.getNegOverPosRatio()
    )
    return (tot_tracc, tot_vacc, bin_stats)

if __name__ == '__main__':
    main()
