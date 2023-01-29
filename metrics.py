import torch

def acc(ppred, ypred, ytrue):
    return (ypred == ytrue).float().mean()

def mae(ppred, ypred, ytrue):
    return (ypred - ytrue).abs().float().mean()

def f1(ppred, ypred, ytrue):
    tp = torch.logical_and(ytrue, ypred).float().mean()
    fp = torch.logical_and(ytrue == 0, ypred).float().mean()
    fn = torch.logical_and(ytrue, ypred == 0).float().mean()
    return tp / (tp + 0.5*(fp+fn))

def Balanced(fn):
    def f(ppred, ypred, ytrue):
        KK = torch.unique(ytrue)
        return sum(fn(ppred[ytrue == k], ypred[ytrue == k], ytrue[ytrue == k]) for k in KK) / len(KK)
    f.__name__ = f'balanced_{fn.__name__}'
    return f

def MacroAvg(fn):
    def f(ppred, ypred, ytrue):
        KK = torch.unique(ytrue)
        return sum(fn(ppred, (ypred == k).long(), (ytrue == k).long()) for k in KK) / len(KK)
    f.__name__ = f'macro_{fn.__name__}'
    return f

def Percentage(fn):
    def f(ppred, ypred, ytrue):
        return 100*fn(ppred, ypred, ytrue)
    return f

def is_unimodal(p):
    zero = torch.zeros(1, device=p.device)
    p = torch.sign(torch.round(torch.diff(p, prepend=zero, append=zero), decimals=2))
    p = torch.diff(p[p != 0])
    p = p[p != 0]
    return len(p) <= 1

def times_unimodal_wasserstein(ppred, ypred, ytrue):
    return sum(is_unimodal(p) for p in ppred) / len(ppred)
