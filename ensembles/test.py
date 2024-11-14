# silence warning when using this without GPU
import warnings
warnings.filterwarnings('ignore', message="Can't initialize NVML")
warnings.filterwarnings('ignore', category=FutureWarning)

from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('dataset')
# parser.add_argument('loss')
# parser.add_argument('--datadir', default='/data/ordinal')
# parser.add_argument('--lamda')
# parser.add_argument('--reps', nargs='+', type=int)
# parser.add_argument('--print-lambda', action='store_true')
# parser.add_argument('--only-metric', type=int)
# parser.add_argument('--store-predictions', action='store_true')

# args = parser.parse_args()

from torch.utils.data import DataLoader
import torch
import src.metrics, src.data
import os

from ensembles.utils import EnsembleDataset
from ensembles.models import MedianEnsemble, WassersteinEnsemble


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Selected device: {device}")

base_models = [
    "CrossEntropy", "POM", "OrdinalEncoding", "CDW_CE", "UnimodalNet",
    "PoissonUnimodal", "VS_SL", "ORD_ACL", "CrossEntropy_UR", "BinomialUnimodal_CE",
]
print(f"Selected base models: {base_models}")

datasets = [
    "FOCUSPATH"
]
reps = [1, 2, 3, 4]

print(f"Selected datasets: {datasets}")
print(f"Selected reps: {reps}")

ensembles = [
    MedianEnsemble(), WassersteinEnsemble()
]
print(f"Selected ensembles: {ensembles}")

############################## EVAL ##############################

def compute_metrics(dataset, rep, ensemble, base_models, metrics_list):

    ds = EnsembleDataset(dataset=dataset, loss=base_models, rep=rep)
    ts = DataLoader(ds, 64)

    YY_pred = []
    YY_true = []
    for X, Y in ts:
        X = X.to(device)
        Y = Y.to(device)
        with torch.no_grad():
            Yhat = ensemble(X)  # class labels
        YY_pred.append(Yhat)
        YY_true.append(Y)
    YY_pred = torch.cat(YY_pred)
    # PP_pred = loss_fn.to_proba(YY_pred)  # class probabilities
    # YY_pred = loss_fn.to_classes(YY_pred)  # class labels
    YY_true = torch.cat(YY_true)
    PP_pred = None  # TODO: some ensembles don't aggregate probabilities...
    return torch.tensor([metric(PP_pred, YY_pred, YY_true) for metric in metrics_list], device=device)


metrics_names = [
    "Acc",
    "MAE",
    "QWK",
    "tau",
    # "%Unimodal",
    "ZME",
    # "NLL",
]

metrics_list = [
    src.metrics.Percentage(src.metrics.acc),
    src.metrics.mae,
    src.metrics.Percentage(src.metrics.quadratic_weighted_kappa),
    src.metrics.Percentage(src.metrics.kendall_tau),
    # src.metrics.Percentage(src.metrics.times_unimodal_wasserstein),  # needs probas
    src.metrics.zero_mean_error,
    # src.metrics.negative_log_likelihood,  # needs probas
]
precisions = [1, 2, 1, 1, 1, 2, 2]

# if args.only_metric is not None:
#    metrics_list = [metrics_list[args.only_metric]]
#    precisions = [precisions[args.only_metric]]

# TODO: implement this with script arguments
# results = torch.stack([compute_metrics(rep, metrics_list) for rep in args.reps])

for dataset in datasets:
    for ensemble in ensembles:
        results = torch.stack([compute_metrics(dataset, rep, ensemble, base_models, metrics_list) for rep in tqdm(reps)])

        #print(args.loss.replace('_', r'\_'), end='')
        #print(args.dataset, args.loss, sep=' & ', end='')
        # if args.print_lambda:
        #     print(f' & {args.lamda}', end='')
        # if len(args.reps) == 1:
        #     print(' & ' + ' & '.join([f'{result:.{precision}f}' for precision, result in zip(precisions, results[0])]), end='') # + r' \\')
        # else:
        #     stds = [torch.std(r[~r.isnan()]) for r in results.T]
        #     print(' & ' + ' & '.join([f'${mean:.{precision}f}\\color{{gray}}\\pm{std:.{precision}f}$' for precision, mean, std in zip(precisions, results.nanmean(0), stds)]), end='') # + r' \\')

        stds = [torch.std(r[~r.isnan()]) for r in results.T]
        print(f'{dataset} {ensemble.__class__.__name__}')
        print(''.join([f'{name}: {mean:.{precision}f} +- {std:.{precision}f}\n' for name, precision, mean, std in zip(metrics_names, precisions, results.nanmean(0), stds)]), end='')
        # print(' & ' + ' & '.join([f'${mean:.{precision}f}\\color{{gray}}\\pm{std:.{precision}f}$' for precision, mean, std in zip(precisions, results.nanmean(0), stds)]), end='') # + r' \\')
