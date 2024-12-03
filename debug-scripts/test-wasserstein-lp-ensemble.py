import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from ensembles.models import WassersteinEnsemble_LP, MajorityVotingEnsemble
from ensembles.utils import EnsembleDataset

from tqdm import tqdm

plt.figure()
losses = [
    "BinomialUnimodal_CE", "PoissonUnimodal", "CrossEntropy"
]

results = {loss: [] for loss in losses}
results["Ensemble"] = []
results["Baseline"] = []

revs = [1, 2, 3, 4]
# revs = [1]
baseline = MajorityVotingEnsemble()
model = WassersteinEnsemble_LP(verbose=False)

for rev in revs:
    ds = EnsembleDataset(dataset="FOCUSPATH", loss=losses, rep=rev)
    batch_size = 32
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    do_single_plot = True
    total_acc = 0
    baseline_total_acc = 0

    for x, y in tqdm(dl, desc="ensemble"):
        probas, ypred = model(x)
        acc = (ypred == y).sum()
        total_acc += acc.item()

        baseline_probas, baseline_ypred = baseline(x)
        baseline_acc = (baseline_ypred == y).sum()
        baseline_total_acc += baseline_acc.item()

        if do_single_plot:
            _, num_models, classes = x.shape
            cls = np.arange(1, classes + 1)
            for i in range(num_models):
                plt.plot(cls, x[0, i].cpu(), label=losses[i], ls="-", lw=1.0, alpha=0.9)
            plt.plot(cls, probas[0].cpu(), label="Ensemble", ls="-", c="k", lw=2.5)

            plt.legend()
            plt.grid()
            plt.xlabel("Classes")
            plt.ylabel("Probability")
            plt.ylim(0, 1.0)
            plt.title("Wasserstein LP Ensemble")
            plt.show()
            do_single_plot = False

    total_acc /= len(ds)
    baseline_total_acc /= len(ds)
    results["Ensemble"].append(total_acc)
    results["Baseline"].append(baseline_total_acc)

for key, value in results.items():
    if len(value) > 0:
        value = np.array(value)
        print(f"{key} -> {value.mean():.2f} ± {2 * value.std():.2f}")