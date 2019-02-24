import numpy as np
import os
import sys
import time
import torch

from util.data import CIFAR10, CIFAR100

newline = lambda f: print("", file=f, flush=True)
ul = lambda s: "\x1b[4m%s\x1b[0m" % s
hl = lambda s: "\x1b[4;33m%s\x1b[0m" % s


def checkpoint(config, f):
    torch.save(f.state_dict(), os.path.join(config.ckpt_path, 'f.pth'))


def get_data_loader(config, **kwargs):
    if config.dataset == "cifar10":
        return CIFAR10(config.data_path, config.batch_size, **kwargs)
    elif config.dataset == "cifar100":
        return CIFAR100(config.data_path, config.batch_size, **kwargs)


def make_dirs(*args):
    for dir in args:
        os.makedirs(dir, exist_ok=True)


def print_legend(file=None):
    print("\033[47m[epoch]\033[0m \033[41m[train accuracy]\033[0m \033[43m[test accuracy]\033[0m", file=file)


def save_model_info(config, file=None):
    passed_cmds = [k.replace("--", "") for k in sys.argv[1:] if '--' in k]
    config_dict = vars(config)

    for j, (k, i) in enumerate(config_dict.items()):
        print("%s: %s" % (k, hl(i) if k in passed_cmds else ul(i)), file=file)
    newline(file)

    with open(config.ckpt_path + "/preprocessor.dat", 'w') as f:
        f.write(str(config_dict))


def save_log(config, epoch, log):
    extracted = np.vstack([np.arange(epoch+1), [list(l.values()) for l in log.values()]]).T
    fmt = ["%i"] + ["%2.8e"] * len(log)
    header = "epoch," + ",".join([k for k in log.keys()])

    np.savetxt(
        os.path.join(config.ckpt_path, 'log.dat'),
        extracted,
        delimiter=",",
        header=header,
        fmt=fmt,
        comments="")
