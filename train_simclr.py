import os
import numpy as np
import hydra
from omegaconf import OmegaConf
from sslearning.data.data_loader import check_file_list
from torchvision import transforms
from torchsummary import summary

# Model utils
from sslearning.models.accNet import SSLNET, Resnet
from sslearning.data.datautils import (
    RandomSwitchAxisTimeSeries,
    RotationAxisTimeSeries,
)

# Data utils
from sslearning.data.data_loader import (
    simclr_subject_collate,
    worker_init_fn,
    SIMCLR_dataset,
)

# Torch
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import torch.optim as optim

# Torch DDP
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Plotting
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import signal
import time
import sys
from sslearning.pytorchtools import EarlyStopping

import warnings

cuda = torch.cuda.is_available()
now = datetime.now()

""""
Muti-tasking learning for self-supervised wearable models

Our input data will be unlabelled. This script can assign pre-text
task labels to all the data. All the task labels
will be generated all the time but by specifying which tasks to use,
we can train on only a subset of these tasks.

Whenever we introduce a new task, there are several things to change.
1. Dataloader and dataset classes to handle the data generation
2. In the train step, update the `compute_loss` and `get_task_loss` functions.
3. Update the inference step

Example usage:
    python mtl.py data=day_sec_test task=time_reversal augmentation=all

    # multi-processed distributed parallel (DPP)
    python mtl.py data=day_sec_10k task=time_reversal
    augmentation=all model=resnet
    dataloader.num_sample_per_subject=1500 data.batch_subject_num=14
    dataloader=ten_sec model.lr_scale=True
    runtime.distributed=True

"""


################################
#
#
#       DDP functions
#
#
################################
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_program():
    while True:
        time.sleep(1)
        print("a")


def signal_handler(signal, frame):
    # your code here
    cleanup()
    sys.exit(0)


################################
#
#
#       helper functions
#
#
################################
def set_seed(my_seed=0):
    random_seed = my_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed_all(random_seed)


def set_up_data4train(X1, X2, cfg, my_device, rank):
    X1, X2 = (
        Variable(X1),
        Variable(X2),
    )

    if cfg.runtime.distributed:
        X1 = X1.to(rank, dtype=torch.float)
        X2 = X2.to(rank, dtype=torch.float)
    else:
        X1 = X1.to(my_device, dtype=torch.float)
        X2 = X2.to(my_device, dtype=torch.float)
    return X1, X2


def evaluate_model(model, data_loader, cfg, my_device, rank, my_criterion):
    model.eval()
    losses = []

    for i, (X1, X2) in enumerate(data_loader):
        with torch.no_grad():
            (
                X1,
                X2,
            ) = set_up_data4train(X1, X2, cfg, my_device, rank)

            # obtain two views of the same data
            h1 = model(X1)
            h2 = model(X2)
            loss = my_criterion(h1, h2)
            losses.append(loss.item())

    losses = np.array(losses)

    return (losses,)


def log_performance(current_loss, writer, mode, epoch, task_name):
    # We want to have individual task performance
    # and an average loss performance
    # train_loss: numpy array
    # mode (str): train or test
    # overall = np.mean(np.mean(train_loss))
    # rotataion_loss = np.mean(train_loss[:, ROTATION_IDX])
    # task_loss: is only true for all task config
    loss = np.mean(current_loss)

    writer.add_scalar(mode + "/" + task_name + "_loss", loss, epoch)

    return loss


def set_linear_scale_lr(model, cfg):
    """Allow for large minibatch
    https://arxiv.org/abs/1706.02677
    1. Linear scale learning rate in proportion to minibatch size
    2. Linear learning scheduler to allow for warm up for the first 5 epoches
    """
    if cfg.model.lr_scale:
        # reference batch size and learning rate
        # lr: 0.0001 batch_size: 512
        reference_lr = 0.0001
        ref_batch_size = 512.0
        optimizer = optim.Adam(
            model.parameters(), lr=reference_lr, amsgrad=True
        )
        k = (
            1.0
            * cfg.dataloader.num_sample_per_subject
            * cfg.data.batch_subject_num
        ) / ref_batch_size
        scale_ratio = k ** (1.0 / 5.0)
        # linear warm up to account for large batch size
        lambda1 = lambda epoch: scale_ratio**epoch
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.model.learning_rate, amsgrad=True
        )
        lambda1 = lambda epoch: 1.0**epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    return optimizer, scheduler


# contrastive loss
# Taken from https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7
class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = (
            self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))
            / self.temperature
        )

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        # SIMCLR
        labels = (
            torch.from_numpy(np.array([0] * N))
            .reshape(-1)
            .to(positive_samples.device)
            .long()
        )  # .float()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


def compute_acc(logits, true_y):
    pred_y = torch.argmax(logits, dim=1)
    acc = torch.sum(pred_y == true_y)
    acc = 1.0 * acc / (pred_y.size()[0])
    return acc


def compute_loss(
    cfg,
    aot_y,
    scale_y,
    permute_y,
    time_w_y,
    aot_y_pred,
    scale_y_pred,
    permute_y_pred,
    time_w_h_pred,
):
    entropy_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    total_task = 0
    total_acc = 0
    aot_loss = 0
    permute_loss = 0
    scale_loss = 0
    time_w_loss = 0

    if cfg.runtime.distributed:
        aot_loss = entropy_loss_fn(aot_y_pred, aot_y)
        permute_loss = entropy_loss_fn(permute_y_pred, permute_y)
        scale_loss = entropy_loss_fn(scale_y_pred, scale_y)
        time_w_loss = entropy_loss_fn(time_w_h_pred, time_w_y)

        dummy_loss = 0.0 * (
            aot_loss + permute_loss + scale_loss + time_w_loss
        )  # all the output needs to be in loss
        total_loss += dummy_loss

        aot_loss = aot_loss.item()
        permute_loss = permute_loss.item()
        scale_loss = scale_loss.item()
        time_w_loss = time_w_loss.item()

    if cfg.task.time_reversal:
        total_loss += entropy_loss_fn(aot_y_pred, aot_y)
        total_acc += compute_acc(aot_y_pred, aot_y)
        total_task += 1

    if cfg.task.permutation:
        total_loss += entropy_loss_fn(permute_y_pred, permute_y)
        total_acc += compute_acc(permute_y_pred, permute_y)
        total_task += 1

    if cfg.task.scale:
        total_loss += entropy_loss_fn(scale_y_pred, scale_y)
        total_acc += compute_acc(scale_y_pred, scale_y)
        total_task += 1

    if cfg.task.time_warped:
        total_loss += entropy_loss_fn(time_w_h_pred, time_w_y)
        total_acc += compute_acc(time_w_h_pred, time_w_y)
        total_task += 1

    return (
        total_loss / total_task,
        total_acc / total_task,
        [aot_loss, permute_loss, scale_loss, time_w_loss],
    )


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    n_gpus = torch.cuda.device_count()
    signal.signal(signal.SIGINT, signal_handler)

    if cfg.runtime.distributed:
        if n_gpus < 4:
            print(f"Requires at least 4 GPUs to run, but got {n_gpus}.")
        else:
            cfg.runtime.multi_gpu = True
            mp.spawn(main_worker, nprocs=n_gpus, args=(cfg,), join=True)
    else:
        main_worker(-1, cfg)


def main_worker(rank, cfg):
    if cfg.runtime.distributed:
        setup(rank, 4)
    set_seed()
    print(OmegaConf.to_yaml(cfg))

    ####################
    #   Setting macros
    ###################
    num_epochs = cfg.runtime.num_epoch
    lr = cfg.model.learning_rate  # learning rate in SGD
    batch_subject_num = cfg.data.batch_subject_num
    GPU = cfg.runtime.gpu
    multi_gpu = cfg.runtime.multi_gpu
    gpu_ids = cfg.runtime.gpu_ids
    is_epoch_data = cfg.runtime.is_epoch_data
    # mixed_precision = cfg.model.mixed_precision
    # useAugment = cfg.runtime.augment

    # data config
    train_data_root = cfg.data.train_data_root
    test_data_root = cfg.data.test_data_root
    train_file_list_path = cfg.data.train_file_list
    test_file_list_path = cfg.data.test_file_list
    log_interval = cfg.data.log_interval
    gpu_id2save = 0
    if cfg.runtime.distributed is False or (
        cfg.runtime.distributed and rank == gpu_id2save
    ):
        main_log_dir = cfg.data.log_path
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        log_dir = os.path.join(
            main_log_dir,
            cfg.model.name + "_" + cfg.task.task_name + "_" + dt_string,
        )
        writer = SummaryWriter(log_dir)

    check_file_list(train_file_list_path, train_data_root, cfg)
    check_file_list(test_file_list_path, test_data_root, cfg)

    # y_path = cfg.data.y_path
    main_log_dir = cfg.data.log_path
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    log_dir = os.path.join(main_log_dir, cfg.model.name + "_" + dt_string)
    general_model_path = os.path.join(
        main_log_dir,
        "models",
        cfg.model.name
        + "_len_"
        + str(cfg.dataloader.epoch_len)
        + "_sR_"
        + str(cfg.data.ratio2keep)
        + "_"
        + dt_string
        + "_",
    )
    model_path = general_model_path + ".mdl"
    num_workers = 8
    true_batch_size = batch_subject_num * cfg.dataloader.num_sample_per_subject
    if true_batch_size > 2000 and cfg.model.lr_scale is False:
        warnings.warn(
            "Batch size > 2000 but learning rate not using linear scale. \n "
            + "Model performance is going to be worse. Fix: run with  "
            + "cfg.model.lr_scale=True"
        )

    print("Model name: %s" % cfg.model.name)
    print("Learning rate: %f" % lr)
    print("Number of epoches: %d" % num_epochs)
    print("GPU usage: %d" % GPU)
    print("Subjects per batch: %d" % batch_subject_num)
    print("True batch size : %d" % true_batch_size)
    print("Tensor log dir: %s" % log_dir)

    ####################
    #   Model construction
    ###################
    if GPU >= -1:
        my_device = "cuda:" + str(GPU)
    elif multi_gpu is True and cfg.runtime.distributed is False:
        my_device = "cuda:0"  # use the first GPU as master
    else:
        my_device = "cpu"

    if cfg.task.task_name == "simclr":
        z_size = 100
        model = Resnet(
            output_size=z_size,
            resnet_version=cfg.model.resnet_version,
            epoch_len=cfg.dataloader.epoch_len,
            is_simclr=True,
        )
        criterion = SimCLR_Loss(batch_size=true_batch_size, temperature=0.5)
    else:
        model = SSLNET(output_size=2, flatten_size=1024)  # VGG
    model = model.float()
    print(model)

    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Num of paras %d " % pytorch_total_params)
    # check if each process is having the same input
    if cfg.runtime.distributed:
        print("Training using DDP")
        torch.cuda.set_device(rank)
        model.cuda(rank)
        ngpus_per_node = 4
        cfg.data.batch_subject_num = int(
            cfg.data.batch_subject_num / ngpus_per_node
        )
        num_workers = int(num_workers / ngpus_per_node)
        model = DDP(model, device_ids=[rank], output_device=rank)
    elif multi_gpu:
        print("Training using multiple GPUS")
        model = nn.DataParallel(model, device_ids=gpu_ids)
        model.to(my_device)
    else:
        print("Training using device %s" % my_device)
        model.to(my_device, dtype=torch.float)
        model.to(my_device, dtype=torch.float)

    if GPU == -1 and multi_gpu is False:
        summary(
            model,
            (3, cfg.dataloader.sample_rate * cfg.dataloader.epoch_len),
            device="cpu",
        )
    elif GPU == 0:
        summary(
            model,
            (3, cfg.dataloader.sample_rate * cfg.dataloader.epoch_len),
            device="cuda",
        )

    ####################
    #   Set up data
    ###################
    my_transform = transforms.Compose(
        [RandomSwitchAxisTimeSeries(), RotationAxisTimeSeries()]
    )

    train_dataset = SIMCLR_dataset(
        train_data_root,
        train_file_list_path,
        cfg,
        is_epoch_data=is_epoch_data,
        transform=my_transform,
    )
    if cfg.runtime.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_subject_num,
        collate_fn=simclr_subject_collate,
        shuffle=train_shuffle,
        sampler=train_sampler,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
    )

    test_dataset = SIMCLR_dataset(
        test_data_root, test_file_list_path, cfg, is_epoch_data=is_epoch_data
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_subject_num,
        collate_fn=simclr_subject_collate,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
    )

    ####################
    #   Set up Training
    ###################
    optimizer, scheduler = set_linear_scale_lr(model, cfg)
    total_step = len(train_loader)

    print("Start training")
    # scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(
        patience=cfg.model.patience, path=model_path, verbose=True
    )

    for epoch in range(num_epochs):
        if cfg.runtime.distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        train_losses = []

        for i, (X1, X2) in enumerate(train_loader):
            # the labels for all tasks are always generated

            (
                X1,
                X2,
            ) = set_up_data4train(X1, X2, cfg, my_device, rank)

            # obtain two views of the same data
            h1 = model(X1)
            h2 = model(X2)
            loss = criterion(h1, h2)

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            if i % log_interval == 0:
                msg = (
                    "Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},".format(
                        epoch + 1,
                        num_epochs,
                        i,
                        total_step,
                        loss.item(),
                    )
                )
                print(msg)
            train_losses.append(loss.cpu().detach().numpy())

        if epoch < cfg.model.warm_up_step:
            scheduler.step()

        train_losses = np.array(train_losses)

        test_losses = evaluate_model(
            model, test_loader, cfg, my_device, rank, criterion
        )

        # logging
        if cfg.runtime.distributed is False or (
            cfg.runtime.distributed and rank == gpu_id2save
        ):
            log_performance(
                train_losses,
                writer,
                "train",
                epoch,
                cfg.task.task_name,
            )
            test_loss = log_performance(
                test_losses,
                writer,
                "test",
                epoch,
                cfg.task.task_name,
            )

            # save regularly
            if cfg.runtime.distributed is False or (
                cfg.runtime.distributed and rank == gpu_id2save
            ):
                if epoch % 5 == 0 and cfg.data.data_name == "100k":
                    epoch_model_path = general_model_path + str(epoch) + ".mdl"
                    torch.save(model.state_dict(), epoch_model_path)

            early_stopping(test_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    if cfg.runtime.distributed:
        cleanup()


if __name__ == "__main__":
    main()
