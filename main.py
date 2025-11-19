import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GDL
from pathlib import Path
from src.dataset import MapGraph
from src.sage import GraphSAGEGraphLevel
from src.gat import GATGraphLevel
import colorama
from colorama import Fore, Back, Style
import numpy as np
from src.labels import LabelsEnum
from src.tprint import TabPrint
from typing import Literal as Lit
from tqdm.auto import tqdm
import click
colorama.init(autoreset=True)

# graph building parameters
RADIUS_EDGE_CONN = 20

# data description
EMB_DIM = 12
NUM_POSSIBLE_STATION_TYPES = 256
NUM_TEMPORAL_FEATURES = 5
NUM_STATIC_FEATURES = 2
FRAMES_PER_PACK = 20

# learning parameters
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-5
WEIGHT_DECAY = 2e-4
ACTIVE_LABELS = [0,1,2,3,4,5,6,7,8]

# gnn parameters
SAGE_HIDDEN_DIMS = [256, 128, 64, 32, 16]
GAT_HIDDEN_DIMS = [128, 128]
# only for GAT
GAT_ATTENTION_HEADS = 8

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def getLbName(label_idx:int,active_labels)->str:
    try:
        return LabelsEnum(active_labels[label_idx]).name
    except ValueError:
        return "UNKNOWN_LABEL"

def split_tr_ev_3to1(dataset:MapGraph)->tuple[MapGraph,MapGraph]:
    total_len = len(dataset)
    train_len = (total_len * 3) // 4
    val_len = total_len - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    return train_ds, val_ds

Progress_logging_options = Lit['clilog', 'tqdm', 'none']

def train_model(model:torch.nn.Module, train_loader:GDL, eval_loader:GDL, epochs:int=10, lr:float=1e-3, weight_decay:float=1e-5, device:str='cpu', verbose:bool=False, *, progress_logging:Progress_logging_options='clilog', active_labels):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    tprint = TabPrint(tab="   ", enabled=(progress_logging=='clilog'))

    act_labels_num = len(ACTIVE_LABELS)

    for epoch in tqdm(range(epochs), desc="Training Epochs", disable=(progress_logging!='tqdm')):
        tprint(f"\n{Back.CYAN}{Fore.YELLOW} ---------- Epoch {epoch+1}/{epochs} ---------- {Style.RESET_ALL}")
        with tprint.tab:
            tprint(f"{Fore.YELLOW}{Style.BRIGHT}--> Training...   {Style.RESET_ALL}")
            with tprint.tab:
                model.train()
                train_total_loss = 0
                tot_mlb = 0
                tot_correct = torch.zeros((1,act_labels_num), device=device, dtype=torch.long)
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    logits = model(batch)
                    scores = torch.sigmoid(logits)
                    y = batch.y.float().view(batch.num_graphs, act_labels_num)
                    train_loss = criterion(logits, y)
                    train_loss.backward()
                    train_total_loss += train_loss.item() * batch.num_graphs
                    optimizer.step()

                    # Accuracy con threshold 0.5
                    preds = (scores >= 0.5).float()
                    corr = (preds == y).long().sum(dim=0)
                    tot_correct += corr
                    tot_mlb += batch.num_graphs
                    acc = corr.sum().item() / (batch.num_graphs * act_labels_num)
                    if verbose:
                        tprint(f"{Style.DIM}Training Batch Loss: {train_loss.item():.4f}, Training Batch Accuracy: {acc:.4f}{Style.RESET_ALL}")
            avg_train_loss = train_total_loss / len(train_loader)
            if verbose:
                tprint(f"Training Loss: {avg_train_loss:.4f}")
            tot_train_accuracy = tot_correct.sum().item() / (tot_mlb * act_labels_num)
            per_label_train_acc = (tot_correct.sum(dim=0).cpu().float().numpy() / tot_mlb).tolist()
            tprint(f"{Fore.GREEN}{Style.BRIGHT}Training Accuracy: {tot_train_accuracy:.4f}{Style.RESET_ALL}")
            if verbose:
                tprint(f"Per-Label Training Accuracy:")
                with tprint.tab:
                    for i, acc in enumerate(per_label_train_acc):
                        tprint(f'label "{getLbName(i, active_labels)}" -> {acc:.4f}')

            # Evaluation
            tprint(f"{Fore.YELLOW}{Style.BRIGHT}--> Validating ...   {Style.RESET_ALL}")
            with tprint.tab:
                model.eval()
                val_total_loss = 0
                tot_mlb = 0
                tot_correct = torch.zeros((1,act_labels_num), device=device, dtype=torch.long)

                with torch.no_grad():
                    for batch in eval_loader:
                        batch = batch.to(device)
                        logits = model(batch)
                        scores = torch.sigmoid(logits)
                        y = batch.y.float().view(batch.num_graphs, act_labels_num)
                        val_loss = criterion(logits, y)
                        val_total_loss += val_loss.item() * batch.num_graphs

                        # Accuracy con threshold 0.5
                        preds = (scores >= 0.5).float()
                        tot_correct += (preds == y).long().sum(dim=0)
                        tot_mlb += batch.num_graphs
            avg_val_loss = val_total_loss / len(eval_loader)
            if verbose:
                tprint(f"Validation Loss: {avg_val_loss:.4f}")
            tot_accuracy = tot_correct.sum().item() / (tot_mlb * act_labels_num)
            per_label_val_acc = (tot_correct.sum(dim=0).cpu().float().numpy() / tot_mlb).tolist()
            tprint(f"{Fore.GREEN}{Style.BRIGHT}Validation Accuracy: {tot_accuracy:.4f}{Style.RESET_ALL}")
            tprint(f"Per-Label Eval Accuracy:")
            with tprint.tab:
                for i, acc in enumerate(per_label_val_acc):
                    tprint(f'label "{getLbName(i, active_labels)}" -> {acc:.4f}')

@click.command()
@click.option('-D', '--dirpath', 'dirpath', type=click.Path(exists=True,file_okay=False,dir_okay=True), default=None, help='Path to the dataset directory. The directory must contain 3 files, namely "packs.parquet", "labels.parquet" and "vinfo.parquet"', required=True)
@click.option('--no-dims-features', is_flag=True, default=False, help='Do not include vehicle dimensions (width and length) in the node features.')
@click.option('-P','--pos-rescaling', type=click.Choice(MapGraph.pos_rescaling_opt_type.__args__), default='center', help='Position rescaling option to apply to node features. Default is "center".')
@click.option('-b', '--build-only', is_flag=True, default=False, help='Only build and save the graphs from the raw dataset without training the model.')
@click.option('-r', '--rebuild', is_flag=True, default=False, help='Rebuild the dataset graphs even if they already exist on disk.')
@click.option('-m', '--model', type=click.Choice( ['sage','gat']), default='sage', help='Type of GNN model to use: GraphSAGE or GAT.')
@click.option('-v', '--verbose', is_flag=True, default=False, help='Enable verbose output.')
@click.option('--progress-logging', type=click.Choice(Progress_logging_options.__args__), default='clilog', help='Choice for visualization of progress in training.')
def main(dirpath, verbose, build_only, rebuild, pos_rescaling, model, no_dims_features, progress_logging, active_labels=ACTIVE_LABELS):
    in_dim = FRAMES_PER_PACK * NUM_TEMPORAL_FEATURES  + (0 if no_dims_features else NUM_STATIC_FEATURES)
    # load data
    dpath = Path(dirpath).resolve()
    ds = MapGraph(dpath, active_labels=active_labels, m_radius=RADIUS_EDGE_CONN, rebuild=rebuild, pos_rescaling=pos_rescaling, use_dims_features=not no_dims_features)
    print(f" - Using device: {DEVICE}")
    print(f" - Dataset length: {len(ds)}")
    ds.save(tqdm=True)
    if build_only:
        print(f"{Fore.GREEN}✔ Graphs built and saved. Exiting as per --build-only/-b flag.{Style.RESET_ALL}")
        return

    # split train and eval
    d_train,d_eval = split_tr_ev_3to1(ds)
    print(f"{Style.DIM}Train set length: {len(d_train)}{Style.RESET_ALL}")
    print(f"{Style.DIM}Validation set length: {len(d_eval)}{Style.RESET_ALL}")

    # create data loaders
    dl_train = GDL(d_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_eval = GDL(d_eval, batch_size=BATCH_SIZE, shuffle=False)

    match model:
        case 'sage':
            print(f"{Fore.CYAN}Using GraphSAGE model.{Style.RESET_ALL}")
            model = GraphSAGEGraphLevel(in_dim=in_dim, hidden_dims=SAGE_HIDDEN_DIMS, out_dim=len(active_labels), num_st_types=NUM_POSSIBLE_STATION_TYPES, emb_dim=EMB_DIM)
        case 'gat':
            print(f"{Fore.CYAN}Using GAT model.{Style.RESET_ALL}")
            model = GATGraphLevel(in_dim=in_dim, hidden_dims=GAT_HIDDEN_DIMS, out_dim=len(active_labels), num_st_types=NUM_POSSIBLE_STATION_TYPES, emb_dim=EMB_DIM, heads=8)
        case _:
            raise ValueError(f"Unknown model type: {model}")
    train_model(model,dl_train,dl_eval,epochs=EPOCHS,lr=LR,weight_decay=WEIGHT_DECAY,device=DEVICE,verbose=verbose, progress_logging=progress_logging, active_labels=active_labels)


if __name__ == "__main__":
    main()