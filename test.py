import torch
from utils import accuracy, AverageMeter

def test(model, dataloader, epoch=9999):
    acc_meter = AverageMeter(name='accuracy')
    n_iters = len(dataloader)
    model.eval()
    with torch.no_grad():
        for iter_idx, (images, labels) in enumerate(dataloader):

            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)

            acc, acc2 = accuracy(outputs, labels, topk=(1, 2))
            acc_meter.update(acc[0], images.shape[0])

            print(f"[Epoch {epoch}] iter {iter_idx} / {n_iters}: Test accuracy {acc_meter.val:.2f}({acc_meter.avg:.2f})", end='\r')
    print("")
    print(f"Epoch {epoch} Test: Accuracy {acc_meter.avg}")
    return acc_meter.avg
