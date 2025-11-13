import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GDL
from pathlib import Path
from src.dataset import MapGraph
from src.sage import GraphSAGEGraphLevel
import colorama
from colorama import Fore, Back, Style
import numpy as np
from src.labels import getLbName
from src.tprint import TabPrint
colorama.init(autoreset=True)

MLB_LABELS_NUM = 9
RADIUS_EDGE_CONN = 5.0

EMB_DIM = 12
NUM_POSSIBLE_STATION_TYPES = 256
NUM_STATIC_FEATURES = 5
NUM_EMBEDDED_FEATURES = 1
FRAMES_PER_PACK = 20
BATCH_SIZE = 2

# gnn parameters
IN_DIM = FRAMES_PER_PACK * NUM_STATIC_FEATURES  # 20 frames, each
HIDDEN_DIMS = [128, 128]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

def split_tr_ev_3to1(dataset:MapGraph)->tuple[MapGraph,MapGraph]:
    total_len = len(dataset)
    train_len = (total_len * 3) // 4
    val_len = total_len - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    return train_ds, val_ds

def train_model(model:torch.nn.Module, train_loader:GDL, eval_loader:GDL, epochs:int=10, lr:float=1e-3, weight_decay:float=1e-5, device:str='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    tprint = TabPrint(tab="   ")

    for epoch in range(epochs):
        tprint(f"\n{Back.CYAN}{Fore.YELLOW} ---------- Epoch {epoch+1}/{epochs} ---------- {Style.RESET_ALL}")
        with tprint.tab:
            tprint(f"{Fore.YELLOW}{Style.BRIGHT}--> Training...   {Style.RESET_ALL}")
            with tprint.tab:
                model.train()
                train_total_loss = 0
                tot_mlb = 0
                tot_correct = torch.zeros((1,MLB_LABELS_NUM), device=device, dtype=torch.long)
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    logits = model(batch)
                    scores = torch.sigmoid(logits)
                    y = batch.y.float().view(batch.num_graphs, MLB_LABELS_NUM)
                    train_loss = criterion(logits, y)
                    train_loss.backward()
                    train_total_loss += train_loss.item() * batch.num_graphs
                    optimizer.step()

                    # Accuracy con threshold 0.5
                    preds = (scores >= 0.5).float()
                    corr = (preds == y).long().sum(dim=0)
                    tot_correct += corr
                    tot_mlb += batch.num_graphs
                    acc = corr.sum().item() / (batch.num_graphs * MLB_LABELS_NUM)
                    tprint(f"{Style.DIM}Training Batch Loss: {train_loss.item():.4f}, Training Batch Accuracy: {acc:.4f}{Style.RESET_ALL}")
            avg_train_loss = train_total_loss / len(train_loader)
            tprint(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
            tot_train_accuracy = tot_correct.sum().item() / (tot_mlb * MLB_LABELS_NUM)
            per_label_train_acc = (tot_correct.sum(dim=0).cpu().float().numpy() / tot_mlb).tolist()
            tprint(f"{Fore.GREEN}{Style.BRIGHT}Training Accuracy: {tot_train_accuracy:.4f}{Style.RESET_ALL}")
            tprint(f"Per-Label Training Accuracy:")
            with tprint.tab:
                for i, acc in enumerate(per_label_train_acc):
                    tprint(f'label "{getLbName(i)}" -> {acc:.4f}')

            # Evaluation
            
            tprint(f"{Fore.YELLOW}{Style.BRIGHT}--> Validating ...   {Style.RESET_ALL}")
            with tprint.tab:
                model.eval()
                val_total_loss = 0
                tot_mlb = 0
                tot_correct = torch.zeros((1,MLB_LABELS_NUM), device=device, dtype=torch.long)

                with torch.no_grad():
                    for batch in eval_loader:
                        batch = batch.to(device)
                        logits = model(batch)
                        scores = torch.sigmoid(logits)
                        y = batch.y.float().view(batch.num_graphs, MLB_LABELS_NUM)
                        val_loss = criterion(logits, y)
                        val_total_loss += val_loss.item() * batch.num_graphs

                        # Accuracy con threshold 0.5
                        preds = (scores >= 0.5).float()
                        tot_correct += (preds == y).long().sum(dim=0)
                        tot_mlb += batch.num_graphs
            avg_val_loss = val_total_loss / len(eval_loader)
            tprint(f"Validation Loss: {avg_val_loss:.4f}")
            tot_accuracy = tot_correct.sum().item() / (tot_mlb * MLB_LABELS_NUM)
            per_label_val_acc = (tot_correct.sum(dim=0).cpu().float().numpy() / tot_mlb).tolist()
            tprint(f"{Fore.GREEN}{Style.BRIGHT}Validation Accuracy: {tot_accuracy:.4f}{Style.RESET_ALL}")
            tprint(f"Per-Label Eval Accuracy:")
            with tprint.tab:
                for i, acc in enumerate(per_label_val_acc):
                    tprint(f'label "{getLbName(i)}" -> {acc:.4f}')

if __name__ == "__main__":
    # load data
    parentpath = Path(__file__).parent.resolve()
    xpath = parentpath / 'input' / 'pdata.parquet'
    lpath = parentpath / 'input' / 'plabels_encoded.csv'
    ds = MapGraph(xpath, labelspath=lpath, n_labels=MLB_LABELS_NUM, m_radius=RADIUS_EDGE_CONN)
    print(f"Dataset length: {len(ds)}")
    ds.save(tqdm_pos=0)

    # split train and eval
    d_train,d_eval = split_tr_ev_3to1(ds)
    print(f"Train set length: {len(d_train)}")
    print(f"Validation set length: {len(d_eval)}")

    # create data loaders
    dl_train = GDL(d_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_eval = GDL(d_eval, batch_size=BATCH_SIZE, shuffle=False)

    model = GraphSAGEGraphLevel(in_dim=IN_DIM, hidden_dims=HIDDEN_DIMS, out_dim=MLB_LABELS_NUM, num_st_types=NUM_POSSIBLE_STATION_TYPES, emb_dim=EMB_DIM)
    train_model(model,dl_train,dl_eval,epochs=5,lr=1e-3,weight_decay=1e-5,device=DEVICE)