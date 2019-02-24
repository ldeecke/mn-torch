import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision

from collections import defaultdict
import functools; print = functools.partial(print, flush=True)
import itertools
import numpy as np
import os
import sys
import time

from nn.resnet import resnet20, resnet56, resnet110
from util.helpers.nn import accuracy
from util.helpers.setup import checkpoint, get_data_loader, make_dirs, newline, print_legend, save_model_info, save_log
from util.parser import get_default_parser

def main():
    parser = get_default_parser()
    config = parser.parse_args()

    if config.seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    make_dirs(config.ckpt_path, config.data_path)
    out = open(os.path.join(config.ckpt_path, 'console.out'), "w")

    train_loader, img_size, n_train, config.num_classes = get_data_loader(config)
    test_loader, _, n_test, _ = get_data_loader(config, train=False)

    save_model_info(config, file=out)
    print_legend(file=out)

    if config.model == "resnet20":
        f = resnet20(config)
    elif config.model == "resnet44":
        f = resnet44(config)
    elif config.model == "resnet56":
        f = resnet56(config)
    elif config.model == "resnet110":
        f = resnet110(config)
    f.cuda()

    log = {"a_train": defaultdict(float), "a_train_5": defaultdict(float), "a_test": defaultdict(float), "a_test_5": defaultdict(float)}
    loss = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(f.parameters(),
        lr=config.lr_sgd,
        momentum=config.momentum_sgd,
        weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=list(map(int, config.milestones.split(","))), gamma=config.gamma)

    for epoch in range(config.num_epochs):
        for i, (x, labels) in enumerate(train_loader):
            t = time.time()

            f.train()
            f.zero_grad()

            x = Variable(x).type(torch.cuda.FloatTensor)
            labels = Variable(labels).type(torch.cuda.LongTensor)

            y = f(x)
            l = loss(y, labels)

            l.backward()
            optim.step()

            a_train_top_1, a_train_top_5 = accuracy(y, labels, topk=(1, 5))
            log["a_train"][epoch] = (i * log["a_train"][epoch] + a_train_top_1.item()) / (i+1)
            log["a_train_5"][epoch] = (i * log["a_train_5"][epoch] + a_train_top_5.item()) / (i+1)

            print("\033[47m[%d]\033[0m [%d] [%2.3f] \033[41m[%2.3f]\033[0m [%2.3f]" %
                (epoch, i, time.time() - t, log["a_train"][epoch], l.item()),
                end="\r",
                file=out)

        sched.step()
        newline(f=out)

        with torch.no_grad():
            for i, (x, labels) in enumerate(test_loader):
                t = time.time()

                f.eval()

                x = Variable(x).type(torch.cuda.FloatTensor)
                labels = Variable(labels).type(torch.cuda.LongTensor)

                y = f(x)

                a_test_top_1, a_test_top_5 = accuracy(y, labels, topk=(1, 5))
                log["a_test"][epoch] = (i * log["a_test"][epoch] + a_test_top_1.item()) / (i+1)
                log["a_test_5"][epoch] = (i * log["a_test_5"][epoch] + a_test_top_5.item()) / (i+1)

                print("\033[47m[%d]\033[0m [%d] [%2.3f] \033[43m[%2.3f]\033[0m" %
                    (epoch, i, time.time() - t, log["a_test"][epoch]),
                    end="\r",
                    file=out)

        checkpoint(config, f)
        save_log(config, epoch, log)
        newline(f=out)

if __name__ == "__main__":
    main()
