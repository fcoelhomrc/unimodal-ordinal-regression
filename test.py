# We are following the same recipe as https://arxiv.org/abs/1911.10720

# silence warning when using this without GPU
import warnings

warnings.filterwarnings('ignore', message="Can't initialize NVML")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('loss')
parser.add_argument('--datadir', default='/data/ordinal')
parser.add_argument('--lamda')
parser.add_argument('--reps', nargs='+', type=int)
parser.add_argument('--print-lambda', action='store_true')
parser.add_argument('--only-metric', type=int)
parser.add_argument('--store-predictions', action='store_true')

args = parser.parse_args()

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import metrics, losses, data
from models import MLP
import os

save_dir = "predictions"  # for saving predictions (predicted probas + ground truth label)
if os.path.exists(save_dir):
    os.mkdir(save_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################## DATASET ##############################

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((256, 256), antialias=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

############################## LOSS ##############################

ds = getattr(data, args.dataset)
K = ds(args.datadir, transform, 'test', 0).K
if args.loss != 'DummyModel':
    loss_fn = getattr(losses, args.loss)(K)

############################## EVAL ##############################

def compute_metrics(rep, metrics_list):
    global loss_fn
    ts = ds(args.datadir, transform, 'test', rep)
    ts = DataLoader(ts, 64, num_workers=4, pin_memory=True)
    if args.lamda is None:
        model_name = f'model-{args.dataset}-{args.loss}-{rep}.pth'
    else:
        model_name = f'model-{args.dataset}-{args.loss}-{rep}-lambda-{args.lamda}.pth'
    if args.loss == 'DummyModel':
        tr = ds(args.datadir, transform, 'train', rep)
        tr = DataLoader(tr, 64, num_workers=4, pin_memory=True)
        YY_true_tr = torch.cat([Y for _, Y in tr]).to(device)
        dummyK = torch.mode(YY_true_tr)[0]  # majority class
        YY_true = torch.cat([Y for _, Y in ts]).to(device)
        YY_pred = torch.tensor([dummyK]*len(YY_true), dtype=torch.int64, device=device)
        PP_pred = torch.tensor([[0]*dummyK + [1] + [0]*(K-dummyK-1)]*len(YY_true), dtype=torch.float32, device=device)
        return torch.tensor([metric(PP_pred, YY_pred, YY_true) for metric in metrics_list], device=device)
    else:
        model = torch.load(model_name, map_location=device)
        model.eval()
    if hasattr(model, 'loss_fn'):
        loss_fn = model.loss_fn
    YY_pred = []
    YY_true = []
    for X, Y in ts:
        X = X.to(device)
        Y = Y.to(device)
        with torch.no_grad():
            Yhat = model(X)
        YY_pred.append(Yhat)
        YY_true.append(Y)
    YY_pred = torch.cat(YY_pred)
    PP_pred = loss_fn.to_proba(YY_pred)  # class probabilities
    YY_pred = loss_fn.to_classes(YY_pred)  # class labels
    YY_true = torch.cat(YY_true)

    if args.store_predictions:
        assert PP_pred.shape == YY_true.shape  # sanity check
        torch.save(
            PP_pred, os.path.join(save_dir, f"pred-{args.dataset}-{args.loss}-{rep}-lambda-{args.lamda}.pt")
        )
        torch.save(
            YY_true, os.path.join(save_dir, f"true-{args.dataset}-{args.loss}-{rep}-lambda-{args.lamda}.pt")
        )

    return torch.tensor([metric(PP_pred, YY_pred, YY_true) for metric in metrics_list], device=device)

metrics_list = [
    metrics.Percentage(metrics.acc),
    metrics.mae,
    metrics.Percentage(metrics.quadratic_weighted_kappa),
    metrics.Percentage(metrics.kendall_tau),
    metrics.Percentage(metrics.times_unimodal_wasserstein),
    metrics.zero_mean_error,
    metrics.negative_log_likelihood,
]
precisions = [1, 2, 1, 1, 1, 2, 2]

if args.only_metric is not None:
    metrics_list = [metrics_list[args.only_metric]]
    precisions = [precisions[args.only_metric]]

results = torch.stack([compute_metrics(rep, metrics_list) for rep in args.reps])
#print(args.loss.replace('_', r'\_'), end='')
#print(args.dataset, args.loss, sep=' & ', end='')
if args.print_lambda:
    print(f' & {args.lamda}', end='')
if len(args.reps) == 1:
    print(' & ' + ' & '.join([f'{result:.{precision}f}' for precision, result in zip(precisions, results[0])]), end='') # + r' \\')
else:
    stds = [torch.std(r[~r.isnan()]) for r in results.T]
    print(' & ' + ' & '.join([f'${mean:.{precision}f}\\color{{gray}}\\pm{std:.{precision}f}$' for precision, mean, std in zip(precisions, results.nanmean(0), stds)]), end='') # + r' \\')
