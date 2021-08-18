from utils import accuracy, AverageMeter
import wandb

def train(model, dataloader, criterion, optimizer, epoch=9999):
    acc_meter = AverageMeter(name='accuracy')
    loss_meter = AverageMeter(name='loss')
    n_iters = len(dataloader)
    model.train()
    for iter_idx, (images, labels) in enumerate(dataloader):

        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc, acc2= accuracy(outputs, labels, topk=(1, 2))
        loss_meter.update(loss.item(), images.shape[0])
        acc_meter.update(acc[0], images.shape[0])

        print(f"[Epoch {epoch}] iter {iter_idx} / {n_iters}: Loss {loss_meter.val:.4f}({loss_meter.avg:.4f}) Train accuracy {acc_meter.val:.2f}({acc_meter.avg:.2f})", end='\r')

    print("")
    print(f"Epoch {epoch} training finished")
    wandb.log({
        "Loss": loss,
        "train_acc": acc
    })
