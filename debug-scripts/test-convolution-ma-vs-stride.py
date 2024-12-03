import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from ensembles.models import ConvolutionEnsemble_MovingAverage, ConvolutionEnsemble_Stride
from ensembles.utils import EnsembleDataset


plt.figure()
losses = [
    "BinomialUnimodal_CE", "PoissonUnimodal", "CrossEntropy"
]

results = {loss: [] for loss in losses}
results["Ensemble"] = []
# revs = [1, 2, 3, 4]
revs = [1]
# model = ConvolutionEnsemble_MovingAverage()
model = ConvolutionEnsemble_Stride()

compute_single = True
for rev in revs:
    # Compute for one at a time
    for loss in losses:
        if not compute_single:
            break
        ds = EnsembleDataset(dataset="FOCUSPATH", loss=[loss], rep=rev)
        batch_size = 32
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        total_acc = 0
        for x, y in dl:
            probas, ypred = model(x)
            acc = (ypred == y).sum()
            total_acc += acc.item()
        total_acc /= len(ds)
        print(f"{loss} -> {total_acc:3f}")
        results[loss].append(total_acc)

    # Compute with all at same time
    ds = EnsembleDataset(dataset="FOCUSPATH", loss=losses, rep=rev)
    batch_size = 32
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    do_single_plot = True
    total_acc = 0
    for x, y in dl:
        probas, ypred = model(x)
        acc = (ypred == y).sum()
        total_acc += acc.item()

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
            plt.title("Ensemble v.s. Base Models")
            plt.show()
            do_single_plot = False

    total_acc /= len(ds)
    # print(f"Ensemble -> {total_acc:3f}")
    results["Ensemble"].append(total_acc)

for key, value in results.items():
    if len(value) > 0:
        value = np.array(value)
        print(f"{key} -> {value.mean():.2f} ± {2 * value.std():.2f}")