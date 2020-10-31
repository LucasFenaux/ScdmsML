import time
import torch
from torch.utils.data import DataLoader
from progress.bar import Bar
from ScdmsML.src.utils.misc import AverageMeter


def train_nn(batch_loader: DataLoader, model: torch.nn.Module, criterion, optimizer, testing: bool,
             device: torch.device):
    if testing:
        model.eval()
        bar = Bar('Testing', max=len(batch_loader))
    else:
        model.train()
        bar = Bar('Training', max=len(batch_loader))

    # Progress bar stuff
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    for batch_idx, (inputs, target) in enumerate(batch_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        target = target.to(device)
        output = model(inputs)

        # cast to double otherwise BCE/MSE is not happy
        total_loss = criterion(output, target)

        # Record loss
        losses.update(total_loss.item(), inputs.size(0))

        if not testing:
            # Compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | Gen Loss: {gen_loss: .4f} | Action Loss: {action_loss: .4f}'.format(
            batch=batch_idx + 1,
            size=len(batch_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            loss=losses.avg,
            gen_loss=float(0),
            action_loss=float(total_loss)
        )
        bar.next()

    bar.finish()
    return losses.avg
