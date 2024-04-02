import config
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm, trange

base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks.criterion import loss_fun, acc_fun, miou_fun
    from Networks.dl_v3 import Seg, SegMultiscale
    from Networks.model import SegSAM
    from Data.Gaofen import train_loader, val_loader, len_train, len_val
    from Log.Logger import getLogger

save_folder = os.path.abspath(os.path.join(base_folder, './save'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SegSAM().to(device)
for fun in (loss_fun, acc_fun, miou_fun):
    fun = fun.to(device)
logger = getLogger("ISPRS Water-body Segmentation")


def train_epochs(model, start, end, lr=0.0001, transfer=False):
    # --- validate the model ---
    lossT, accT, miouT, lossV, accV, miouV = (0, 0, 0, 0, 0, 0,)
    epoch_message = "\ntrain_loss: {:.3f},\ttrain_acc: {:.3f}\ttrain_miou: {:.3f}\nval_loss  : {:.3f}\tval_acc  : {:.3f}\tval_miou  : {:.3f}\n"
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for x, y, z in tqdm(val_loader, desc="Validating Batch"):
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            
            z = torch.logical_or(z, y).to(torch.float32)
            y_pre = model(x).to(torch.float32)
            y_pre = y_pre * z
            
            loss = loss_fun(y_pre, y)
            acc = acc_fun(y_pre, y)
            miou = miou_fun(y_pre, y)
            
            lossV += loss * len(y)
            accV += acc * len(y)
            miouV += miou * len(y)
    lossV = float((lossV / len_val).cpu())
    accV = float((accV / len_val).cpu())
    miouV = float((miouV / len_val).cpu())
    print(epoch_message.format(
        lossT, accT, miouT, lossV, accV, miouV
    ))


# training
if __name__ == "__main__":
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    train_epochs(
        model,
        0,
        20,
        0.0002,
        transfer=True,
    )