import torch

from tqdm import tqdm

### Cosine similarity for embed and test func

def save_chkpt(state, path='models/fr_iter11.pth.tar'):
    torch.save(state, path)
    print('chkpt savd')
    return

def train(optim, loader, loss, model, device='cuda'):
    loop = tqdm(loader)
    run_loss = 0.0
    for idx, (x, p, n, y)  in enumerate(loop):
        x = x.to(device)
        p = p.to(device)
        n = n.to(device)

        x_emb = model(x) 
        p_emb = model(p) 
        n_emb = model(n) 

        losss = loss(x_emb, p_emb, n_emb)

        optim.zero_grad()
        losss.backward()
        optim.step()

        loop.set_postfix(loss=losss.item())
        run_loss += losss.item()

    return run_loss / len(loader)
        

#no classes
def test(optim, loader, loss, model, device='cuda'):

    pass