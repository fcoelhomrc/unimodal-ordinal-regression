import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ensembles.utils import EnsembleDataset
from ensembles.models import WassersteinEnsemble, MedianEnsemble, DummyEnsemble
import src.metrics
import src.losses

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Selected device: {device}")

models = [
    "PoissonUnimodal", "BinomialUnimodal_CE",
    "PolyaUnimodal", "NegativeBinomialUnimodal"
]

print(f"Selected models: {models}")

datasets = [
    "FOCUSPATH",
    "FGNET",
    "SMEAR2005"
]
reps = [1, 2, 3, 4]

print(f"Selected datasets: {datasets}")
print(f"Selected reps: {reps}")

ensembles = [
    MedianEnsemble(), WassersteinEnsemble()
]
print(f"Selected ensembles: {ensembles}")


def plot_probas(dataset, rep, model_name):
    # model = torch.load(model_name, map_location=device)
    model = DummyEnsemble()
    ds = EnsembleDataset(dataset=dataset, loss=[model_name], rep=rep)
    ts = DataLoader(ds, 64, shuffle=True)

    # Plot probas to see what is going on
    N_plot = 16
    count = 0
    fig, axs = plt.subplots(4, N_plot//4, figsize=(16, 16))

    color_map = {
        "FOCUSPATH": "tab:blue",
        "SMEAR2005": "tab:orange",
        "FGNET": "tab:green",
    }


    for X, Y in ts:
        N_classes = X.shape[-1]
        cls = np.arange(1, N_classes + 1)
        for i in range(N_plot):
            axs.flatten()[count].bar(cls, X[i].cpu().detach().numpy().squeeze(),
                                     label="Probas", color=color_map.get(dataset, "gray"),
                                     edgecolor="k", alpha=0.9)
            axs.flatten()[count].axvline(Y[i].cpu().detach().numpy(), color="k", ls="--", label="Ground truth")
            axs.flatten()[count].set_xlabel("Classes")
            axs.flatten()[count].set_ylabel("Probabilities")
            axs.flatten()[count].legend()
            axs.flatten()[count].grid()
            count += 1
        break
    fig.suptitle(f"{dataset} {rep} {model_name}")
    fig.tight_layout()
    fig.show()

for dataset in datasets:
    for model_name in models:
        rep = 1
        plot_probas(dataset, rep, model_name)



exit()
#############

def compute_metrics(dataset, rep, model_name, metrics_list):

    # model = torch.load(model_name, map_location=device)
    model = DummyEnsemble()
    ds = EnsembleDataset(dataset=dataset, loss=[model_name], rep=rep)
    ts = DataLoader(ds, 64)

    YY_pred = []
    YY_true = []
    for X, Y in ts:
        X = X.to(device)
        Y = Y.to(device)
        with torch.no_grad():
            Yhat = model(X)  # class labels
        YY_pred.append(Yhat)
        YY_true.append(Y)
    YY_pred = torch.cat(YY_pred)
    # PP_pred = loss_fn.to_proba(YY_pred)  # class probabilities
    # YY_pred = loss_fn.to_classes(YY_pred)  # class labels
    YY_true = torch.cat(YY_true)
    PP_pred = None
    return torch.tensor([metric(PP_pred, YY_pred, YY_true) for metric in metrics_list], device=device)


metrics_names = [
    "Acc",
    "MAE",
    "QWK",
    "tau",
    "%Unimodal",
    "ZME",
    "NLL",
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
    for model_name in models:
        results = torch.stack([compute_metrics(dataset, rep, model_name, metrics_list) for rep in tqdm(reps)])

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
        print(f'{dataset} {model_name}')
        print(''.join([f'{name}: {mean:.{precision}f} +- {std:.{precision}f}\n' for name, precision, mean, std in zip(metrics_names, precisions, results.nanmean(0), stds)]), end='')
        # print(' & ' + ' & '.join([f'${mean:.{precision}f}\\color{{gray}}\\pm{std:.{precision}f}$' for precision, mean, std in zip(precisions, results.nanmean(0), stds)]), end='') # + r' \\')
