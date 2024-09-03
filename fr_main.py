import torch.nn as nn
from fr_utils import save_chkpt, train
from fr_ds import Fr_Ds
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from fr_model import Fr


def get_trf():
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
    ])

    return trf

    

LEARNING_RATE = 7e-5
BATCH_SIZE = 8
LOAD_CHEKPT = True
SAVE_CHEKPT = True
EPOCHS = 10

train_path = '/mnt/c/Users/Krish/Downloads/fr/train_ds/'

if __name__ == '__main__':

    wandb.login(key='8e7aa7fe4ddfe9a81bfda79a06aef00e7729417d')
    wandb.init(project='Fr', name='fr2')

    wandb.config.update({
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE
    })

    trf = get_trf()


    loss = nn.TripletMarginLoss(margin=3.0)
    model = Fr()

    model.to('cuda')

    if LOAD_CHEKPT:
        chkpt=torch.load('models/fr_iter11.pth.tar')
        model.load_state_dict(chkpt['state_dict'])


    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_CHEKPT:
        optim.load_state_dict(chkpt['optimizer'])

    wandb.watch(model, log='all', log_freq=100)


    
    for i in range(EPOCHS):

        if i % 4 == 0:  loader = DataLoader(Fr_Ds(trf=trf, path=train_path), batch_size=BATCH_SIZE, shuffle=True)
        run_loss = train(optim=optim, loader=loader, loss=loss, model=model)

        wandb.log({
        "train_loss": run_loss,
    }, step=i + 1)
        
        chkpt = {
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict()
        }
        print(f'epoch" {i + 1}, train loss: {run_loss}')

        if SAVE_CHEKPT: save_chkpt(chkpt)