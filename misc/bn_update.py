import itertools
import torch

def bn_update(model, loader):
    """Computes precise BN stats on training data."""
    model.train()
    for inputs, _, _labels in itertools.islice(loader, len(loader)):
    # for inputs, _labels in loader:
        model(inputs.cuda())

    model.eval()
    with torch.no_grad():
        # Retrieve the BN layers
        bns = [m for m in model.modules() if (isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d))]
        # bns = [m for name, m in model.named_modules() if "bn" in name] 
        # Initialize stats storage
        mus = [torch.zeros_like(bn.running_mean) for bn in bns]
        sqs = [torch.zeros_like(bn.running_var) for bn in bns]

        # Remember momentum values
        moms = [bn.momentum for bn in bns]
        # Disable momentum
        for bn in bns:
            bn.momentum = 1.0

        # Accumulate the stats across the data samples
        for inputs, _, _labels in itertools.islice(loader, len(loader)):
        # for inputs, _labels in loader:
            model(inputs.cuda())
            # Accumulate the stats for each BN layer
            for i, bn in enumerate(bns):
                m, v = bn.running_mean, bn.running_var
                sqs[i] += (v + m * m) / len(loader)
                mus[i] += m / len(loader)

        # Set the stats and restore momentum values
        for i, bn in enumerate(bns):
            bn.running_var = sqs[i] - mus[i] * mus[i]
            bn.running_mean = mus[i]
            bn.momentum = moms[i]