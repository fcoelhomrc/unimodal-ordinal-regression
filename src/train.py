import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('loss')
parser.add_argument('rep', type=int)
parser.add_argument('output')
parser.add_argument('--datadir', default='/data/ordinal')
parser.add_argument('--lamda', type=float)
parser.add_argument('--batchsize', type=int, default=32)
args = parser.parse_args()

from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from time import time
import torch
import losses, data
from models import MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################## DATASET ##############################

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((int(256*1.05), int(256*1.05)), antialias=True),
    transforms.RandomCrop((256, 256)),
    transforms.ColorJitter(0.1, 0.1),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ds = getattr(data, args.dataset)
ds = ds(args.datadir, transform, 'train', args.rep)
K = ds.K
tr = DataLoader(ds, args.batchsize, True, num_workers=8, pin_memory=True)

############################## LOSS ##############################

if args.lamda is None:
    loss_fn = getattr(losses, args.loss)(K)
else:
    loss_fn = getattr(losses, args.loss)(K, args.lamda)

############################## MODEL ##############################

if ds.modality == 'tabular':
    model = MLP(ds[0][0].shape[0], 128, loss_fn.how_many_outputs())
    epochs = 1000
else:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(512, loss_fn.how_many_outputs())
    epochs = 10 # 25 # 100 TODO: do not commit this change
loss_fn.to(device)
model = model.to(device)
model.loss_fn = loss_fn

############################## TRAIN OPTS ##############################

opt = torch.optim.Adam(model.parameters(), 1e-4)

############################## TRAIN ##############################

train_time = 0
model.train()
for epoch in range(epochs):
    print(f'* Epoch {epoch+1}/{epochs}')
    tic = time()
    avg_loss = 0
    for X, Y in tr:
        X = X.to(device)
        Y = Y.to(device)
        Yhat = model(X)
        loss_value = loss_fn(Yhat, Y).mean()
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        avg_loss += float(loss_value) / len(tr)
    toc = time()
    print(f'- {toc-tic:.0f}s - Loss: {avg_loss}')
    train_time += toc-tic

model.train_time = train_time
torch.save(model.cpu(), os.path.join(os.getcwd(), f"model-{args.dataset}-{args.loss}-{args.rep}.pth"))  # TODO: do not commit this change
