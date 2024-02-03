import time
import os
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from timm.models.vision_transformer import Attention, Block, VisionTransformer

import timm
import tor
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import json
import tor.jetson_utils
# import tlt.models

cuda_id = 0
import csv
def main():
#    valdir = os.path.join('/home/datasets/imagenet/imagenet/val/')
#    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                        std=[0.229, 0.224, 0.225])
#
#    val_dataset = datasets.ImageFolder(valdir, 
#    transforms.Compose([
#            transforms.Resize(256),
#            transforms.CenterCrop(224),
#            transforms.ToTensor(),
#            normalize,
#        ]))
#
#    val_loader = torch.utils.data.DataLoader(val_dataset,
#        batch_size=256, shuffle=False,
#        num_workers=16, pin_memory=False)

    #initialize model
    # Use any ViT model here (see timm.models.vision_transformer)
    # LV ViT S: lvvit_s
    # DEiT B 16: deit_base_patch16_224
    # DEiT S 16: deit_small_patch16_224
    # deit_small_patch16_shrink_base
    model_name = "deit_small_patch16_224"

    # Load a pretrained model
    # model = timm.create_model(model_name, pretrained=True).cuda()
    # ckpt = torch.load('lvvit_s-26M-224-83.3.pth.tar')
    # model.load_state_dict(ckpt)

    # # print(ckpt.state_dict())
    # #evalute model accuracy
    # top1, top5 = validate(model, val_loader)
    # torch.set_printoptions(precision=2)
    # n_digits = 2
    # top1 = torch.round(top1 * 10**n_digits) / (10**n_digits)
    # # top1 = torch.round((top1), decimals=2)
    # top5 = torch.round((top5),decimals=2)

    # print(f"Model Performance before ToMe:Acc@1 {torch.round(top1 * 10**n_digits) / (10**n_digits)} Acc@5 {top5}")

    # ToMe with r=16
    # model.r = 16
    # model.keep_rate = 0.95
    # model.drop_loc=[3, 6, 9]

    device = "cuda:0"
    runs = 50
    batch_size = 64 #256  # Lower this if you don't have that much memory
    # input_size = model.default_cfg["input_size"]
    # baseline_throughput = tome.utils.benchmark(
    #     model,
    #     device=device,
    #     verbose=True,
    #     runs=runs,
    #     batch_size=batch_size,
    #     input_size=input_size
    # )
    # print(baseline_throughput)

    path = 'results_exp/' + model_name
    if not os.path.exists(path):
        os.makedirs(path)
    
    csv_file = open(os.path.join(path, 'results.csv'), 'a')
    writer = csv.writer(csv_file)
    device = "cuda:0"
    runs = 50
    batch_size = 64 #256  # Lower this if you don't have that much memory

    for r_in in range(20, 80, 20):  
        for kr in range(9, 0, -1):
            # Load a pretrained model
            model = timm.create_model(model_name, pretrained=True).cuda()  
            # ckpt = torch.load('lvvit_s-26M-224-83.3.pth.tar')
            # model.load_state_dict(ckpt)

            # Patch the model with ToR
            input_size = model.default_cfg["input_size"]

            # Patch the model with ToR
            tor.patch.timm(model)

            model.r = r_in
            model.keep_rate = 0 #0.1 * kr
            model.drop_loc=[3, 6, 9]
            model.token_fusion = True
            model.merge_rate = 0 #0.1
            print("r value:", r_in) #, "keep rate:", 0.1*kr)

            # evalute model accuracy
            #top1, top5 = validate(model, val_loader)
            #top1 = torch.round(top1, decimals=3)
            #top5 = torch.round(top5, decimals=3)

            latency, tome_throughput, power, energy = tor.jetson_utils.benchmark(
                model,
                device=device,
                verbose=True,
                runs=50,
                batch_size=batch_size,
                input_size=input_size
            )
            results = [
                r_in,
                round(latency, 2),
                round(tome_throughput, 2),
                round(power, 2),
                round(energy, 2)]
            
            print(results)
            writer.writerow(results)
            break
    
    # print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")

def validate(model, val_loader, num_classes=1000):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(cuda_id, non_blocking=False)
            target = target.cuda(cuda_id, non_blocking=False)
            # compute output
            output = model(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, num_classes)))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        return top1.avg, top5.avg


# compute the accuracy for a given result
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# statistic averaging
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    main()
