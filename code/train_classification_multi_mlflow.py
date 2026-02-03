"""
This script was primarily written by Seowung Leem, and revised by Yunchao Yang.
The multi-mlflow version of the code is developed for multi-gpu application for the HiperGator Development Environment.
Therefore, it might not work on some on different PC.

For any questions, please email Seowung Leem.

institute: leem.s@ufl.edu
personal: dlatjdnd@gmail.com
"""


# Basic library
import os
import random
import copy
import numpy as np
import pandas as pd
import argparse
from datetime import timedelta

# Pytorch Related Library
import torch
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torchmetrics

# Monai Library
from monai.data import DataLoader, DistributedSampler
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    Resize,
    ScaleIntensity)

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from time import time


# defining the input arguments for the script.
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--random_state', default=0, type=int, help='random seed for reproducibility')
parser.add_argument('--local-rank', type=int, help='multigpu param')
parser.add_argument('--left_image_dir', default='/red/ruogu.fang/share/UKB/data/Eye/21015_fundus_left_1_good/',
                    type=str,
                    help='There are 2 image directories. You need to put left fundus image dir here.')

parser.add_argument('--right_image_dir',
                    default='/red/ruogu.fang/share/UKB/data/Eye/21016_fundus_right_1_horizontal_good/',
                    type=str,
                    help='There are 2 image directories. You need to put right fundus image dir here. '
                         'Keep in mind either left or right fundus images should be flipped horizontally.')

parser.add_argument('--csv_dir', default='/red/ruogu.fang/leem.s/NSF-SCH/data/classification_data2.csv', type=str,
                    help='Where the csv file for the dataset is stored.')

parser.add_argument('--left_eye_code', default='_21015_0_0.png', type=str,
                    help='The code added to eid of the UKB subjects')

parser.add_argument('--right_eye_code', default='_21016_0_0.png', type=str,
                    help='The code added to eid of the UKB subjects')

parser.add_argument('--label', type=str, nargs='+', help='Column names of the csv file. It should be the label.')
parser.add_argument('--exclude', type=float, nargs='+',
                    help='The exclude code. example, -1, -2, -3 negative integers are not labels, but exclusion code for reason')

parser.add_argument('--working_dir', default='Swin_classification', type=str, help='where you would save the result')
parser.add_argument('--model_name', default='Swin_classification', type=str, help='name of saved model.')
parser.add_argument('--base_model', default='microsoft/swin-large-patch4-window12-384-in22k', type=str,
                    help='the string of model from hugging-face library')

parser.add_argument('--input_size', default=576, type=int, help='input size for the model')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate of the training')
parser.add_argument('--epoch', default=100, type=int, help='maximum number of epoch')
parser.add_argument('--batch_size', default=32, type=int, help='batch size of the training')

parser.add_argument('--logdir', default="./log", type=str, help='directory where you save the log files.')

# set argument as input variables
args = parser.parse_args()

# initialize a process group, every GPU runs in a process
dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=10))

# set random seed
random_state = args.random_state

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ["PYTHONHASHSEED"] = str(random_state)

# setting the directories for training
left_image_dir = args.left_image_dir
right_image_dir = args.right_image_dir
csv_dir = args.csv_dir
left_eye_code = args.left_eye_code
right_eye_code = args.right_eye_code
label = args.label
ex = args.exclude

# extract the patient eid with both images and label csv files
left_file_list = os.listdir(left_image_dir)
right_file_list = os.listdir(right_image_dir)
left_eid_list = [s.replace(left_eye_code, '') for s in left_file_list]
right_eid_list = [s.replace(right_eye_code, '') for s in right_file_list]

# read the labels from the csv file & process it.
csv_df = pd.read_csv(csv_dir)
convert_dict = {'eid': str}
csv_df = csv_df.astype(convert_dict)

# Dropping the n/a samples for each column
left_label_df = csv_df[csv_df['eid'].isin(left_eid_list)]
left_label_df = left_label_df.dropna(subset=label)
right_label_df = csv_df[csv_df['eid'].isin(right_eid_list)]
right_label_df = right_label_df.dropna(subset=label)

# Adding a path column in the dataframe to align eid and image files.
left_label_df['image'] = left_label_df['eid'] + left_eye_code
left_label_df['path'] = left_image_dir + left_label_df['image']
right_label_df['image'] = right_label_df['eid'] + right_eye_code
right_label_df['path'] = right_image_dir + right_label_df['image']

# concatenate the left & right dataframe.
label_df = pd.concat([right_label_df, left_label_df], ignore_index=True, axis=0)

# excluding the labels of excluding criteria in each column
if ex:
    for lbl in label:
        for i in ex:
            label_df = label_df[label_df[lbl] != i]

# preparing the data split
from sklearn.model_selection import train_test_split

# The split should be based on eid. Therefore, we should first acquire the eid list from the dataframe.
eid_list = np.unique(label_df['eid'].values).tolist()

# splitting the samples for classification
remain, test = train_test_split(eid_list, train_size=0.8, random_state=random_state)
train, val = train_test_split(remain, train_size=0.9, random_state=random_state)

# based on eid, split the dataset
train_df = label_df[label_df['eid'].isin(train)]
val_df = label_df[label_df['eid'].isin(val)]
test_df = label_df[label_df['eid'].isin(test)]

# get raw values
X_train = train_df['path'].values.tolist()
y_train = train_df[label].values.tolist()

X_val = val_df['path'].values.tolist()
y_val = val_df[label].values.tolist()

X_test = test_df['path'].values.tolist()
y_test = test_df[label].values.tolist()


# printing the dataset size.
if dist.get_rank() == 0:
    print('The size of training samples {}'.format(len(y_train)))
    print('The size of validation samples {}'.format(len(y_val)))
    print('The size of test samples {}'.format(len(y_test)))


# Classification dataset definition
class ClassificationDataset_All(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = np.array(labels)  # YY
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize((args.input_size, args.input_size)),
        ScaleIntensity(),
    ]
)

val_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize((args.input_size, args.input_size)),
        ScaleIntensity()
    ]
)

# Dataset & Dataloader
train_ds = ClassificationDataset_All(X_train, y_train, train_transforms)
train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)

val_ds = ClassificationDataset_All(X_val, y_val, val_transforms)
val_sampler = DistributedSampler(dataset=val_ds, even_divisible=True, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=val_sampler)

test_ds = ClassificationDataset_All(X_test, y_test, val_transforms)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

# Defining the model
from transformers import ViTForImageClassification, SwinForImageClassification

model_name_or_path = args.base_model
device = torch.device(f"cuda:{args.local_rank}")
torch.cuda.set_device(device)


# Specialized model with multiple outputs.
class SwinforClassification(torch.nn.Module):
    def __init__(self, model_name_or_path='microsoft/swin-large-patch4-window12-384-in22k', n_label=7):
        super().__init__()
        model = SwinForImageClassification.from_pretrained(model_name_or_path, num_labels=n_label,
                                                           ignore_mismatched_sizes=True)
        self.swin = model.swin
        self.regressor_sex = torch.nn.Linear(1536, 2)
        self.regressor_smoking = torch.nn.Linear(1536, 2)
        self.regressor_sleeplessness = torch.nn.Linear(1536, 3)
        self.regressor_alcohol = torch.nn.Linear(1536, 6)
        self.regressor_depression = torch.nn.Linear(1536, 2)
        self.regressor_economic_status = torch.nn.Linear(1536, 5)


    def forward(self, x):
        y_swin = self.swin(x).pooler_output
        y_sex = self.regressor_sex(y_swin)
        y_smoking = self.regressor_smoking(y_swin)
        y_sleeplessness = self.regressor_sleeplessness(y_swin)
        y_alcohol = self.regressor_alcohol(y_swin)
        y_depression = self.regressor_depression(y_swin)
        y_economic_status = self.regressor_economic_status(y_swin)

        return y_sex, y_smoking, y_sleeplessness, y_alcohol, y_depression, y_economic_status


if 'vit' in args.base_model:
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(args.label))

elif 'swin' in args.base_model:
    model = SwinforClassification(model_name_or_path=args.base_model, n_label=6)

# put model to device
model.to(device)

# evaluation metric.
train_roc_sex = torchmetrics.AUROC(task="multiclass", num_classes=2).to(device)
valid_roc_sex = torchmetrics.AUROC(task="multiclass", num_classes=2).to(device)

train_roc_smoking = torchmetrics.AUROC(task="multiclass", num_classes=2).to(device)
valid_roc_smoking = torchmetrics.AUROC(task="multiclass", num_classes=2).to(device)

train_roc_sleeplessness = torchmetrics.AUROC(task="multiclass", num_classes=3).to(device)
valid_roc_sleeplessness = torchmetrics.AUROC(task="multiclass", num_classes=3).to(device)

train_roc_alcohol = torchmetrics.AUROC(task="multiclass", num_classes=6).to(device)
valid_roc_alcohol = torchmetrics.AUROC(task="multiclass", num_classes=6).to(device)

train_roc_depression = torchmetrics.AUROC(task="multiclass", num_classes=2).to(device)
valid_roc_depression = torchmetrics.AUROC(task="multiclass", num_classes=2).to(device)

train_roc_economic_status = torchmetrics.AUROC(task="multiclass", num_classes=5).to(device)
valid_roc_economic_status = torchmetrics.AUROC(task="multiclass", num_classes=5).to(device)

# Defining the variables for training
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


y_train_np = np.array(y_train, dtype=int)

def compute_class_weights(labels_1d):
    """
    labels_1d: 1D numpy array with integer classes, assumed 0..C-1
    returns: 1D torch.FloatTensor of length C with inverse-frequency weights,
             normalized so mean weight ~ 1.
    """
    labels_1d = np.asarray(labels_1d, dtype=int)
    counts = np.bincount(labels_1d)
    # Avoid division by zero for missing classes
    counts[counts == 0] = 1
    freq = counts.astype(np.float32) / counts.sum()
    weights = 1.0 / freq
    # Normalize so that average weight is 1.0
    weights = weights * (len(weights) / weights.sum())
    return torch.tensor(weights, dtype=torch.float32, device=device)

# y_train_np has shape [N, 6] with columns in the same order as args.label
# Here we assume that the columns are ordered as:
# [sex, smoking, sleeplessness, alcohol, depression, economic_status]

sex_weights             = compute_class_weights(y_train_np[:, 0])
smoking_weights         = compute_class_weights(y_train_np[:, 1])
sleeplessness_weights   = compute_class_weights(y_train_np[:, 2])
alcohol_weights         = compute_class_weights(y_train_np[:, 3])
depression_weights      = compute_class_weights(y_train_np[:, 4])
economic_status_weights = compute_class_weights(y_train_np[:, 5])

if dist.get_rank() == 0:
    print("Class weights (sex):", sex_weights.detach().cpu().numpy())
    print("Class weights (smoking):", smoking_weights.detach().cpu().numpy())
    print("Class weights (sleeplessness):", sleeplessness_weights.detach().cpu().numpy())
    print("Class weights (alcohol):", alcohol_weights.detach().cpu().numpy())
    print("Class weights (depression):", depression_weights.detach().cpu().numpy())
    print("Class weights (economic_status):", economic_status_weights.detach().cpu().numpy())

# different loss functions for each risk factors.
'''
loss_function_sex = torch.nn.CrossEntropyLoss(weight=sex_weights)
loss_function_smoking = torch.nn.CrossEntropyLoss(weight=smoking_weights)
loss_function_sleeplessness = torch.nn.CrossEntropyLoss(weight=sleeplessness_weights)
loss_function_alcohol = torch.nn.CrossEntropyLoss(weight=alcohol_weights)
loss_function_depression = torch.nn.CrossEntropyLoss(weight=depression_weights)
loss_function_economic_status = torch.nn.CrossEntropyLoss(weight=economic_status_weights)
'''
loss_function_sex = torch.nn.CrossEntropyLoss()
loss_function_smoking = torch.nn.CrossEntropyLoss()
loss_function_sleeplessness = torch.nn.CrossEntropyLoss()
loss_function_alcohol = torch.nn.CrossEntropyLoss()
loss_function_depression = torch.nn.CrossEntropyLoss()
loss_function_economic_status = torch.nn.CrossEntropyLoss()

max_epochs = args.epoch
model = DistributedDataParallel(model, device_ids=[device])

######################################################################################################################
args.rank = dist.get_rank()
args.logdir = os.path.join(args.logdir,
                           f"run_{os.environ['SLURM_JOB_NAME']}" + datetime.now().strftime("%m-%d-%Y-%H:%M:%S"))  # YY
writer = SummaryWriter(log_dir=args.logdir)
if args.rank == 0:
    print(f"[{args.rank}] " + f"Writing Tensorboard logs to {args.logdir}")

#####################################################################################################################


# Training the model
def train_model(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=max_epochs,
                optimizer=optimizer,
                loss_functions=None,
                model_name=args.model_name,
                working_dir='Swin_Classification'):

    """
    :param model: The model you want to train (it should be in the device)
    :param train_loader: Defined train loader (it should be in the device)
    :param val_loader: Defined val loader (it should be in the device)
    :param max_epochs: maximum number of epochs.
    :param optimizer: pytorch optimizer
    :param loss_functions: the list of loss functions for each label. It should be in the format of lists.
    :param model_name: The model name. Usually, the default is Swin, because we used Swin Transformer.
    :param working_dir: Where you would save the model.
    :return: Trained model weights
    """

    # defining the path for saving the model.
    if dist.get_rank() == 0 and not os.path.exists(os.path.join('./savedmodel', working_dir)):
        os.makedirs(os.path.join('./savedmodel', working_dir))
    else:
        pass

    # initialization of the variable for analysis
    best_loss = 0
    best_metric = -1
    best_metric_epoch = 0
    epoch_loss_values = []
    val_interval = 1
    val_loss = 0

    global_step = 0  #####YY####

    for epoch in range(max_epochs):
        if dist.get_rank() == 0:
            print("-" * 10, flush=True)
            print(f"[{dist.get_rank()}] " + "-" * 10 + f" epoch {epoch + 1}/{max_epochs}")

        # Turn on the training mode
        model.train()
        epoch_loss = 0
        step = 0
        train_sampler.set_epoch(epoch)

        for step, (inputs, labels) in enumerate(train_loader):  # YY # for inputs, labels in train_loader:
            tik = time()
            step += 1
            global_step += 1

            # transfer the data to gpu
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Feed-Forward
            outputs = model(inputs)

            # compute the separate loss
            loss_sex = loss_functions[0](outputs[0], labels[:, 0].long())
            loss_smoking = loss_functions[1](outputs[1], labels[:, 1].long())
            loss_sleeplessness = loss_functions[2](outputs[2], labels[:, 2].long())
            loss_alcohol = loss_functions[3](outputs[3], labels[:, 3].long())
            loss_depression = loss_functions[4](outputs[4], labels[:, 4].long())
            loss_economic_status = loss_functions[5](outputs[5], labels[:, 5].long())

            # compute the loss and back-propagate the gradient.
            loss = loss_sex + loss_smoking + loss_sleeplessness + loss_alcohol + loss_depression + loss_economic_status
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # calculate the accuracy for trainset
            batch_roc_sex = train_roc_sex(torch.tensor(outputs[0]), labels[:, 0].long())
            batch_roc_smoking = train_roc_smoking(torch.tensor(outputs[1]), labels[:, 1].long())
            batch_roc_sleeplessness = train_roc_sleeplessness(torch.tensor(outputs[2]), labels[:, 2].long())
            batch_roc_alcohol = train_roc_alcohol(torch.tensor(outputs[3]), labels[:, 3].long())
            batch_roc_depression = train_roc_depression(torch.tensor(outputs[4]), labels[:, 4].long())
            batch_roc_economic_status = train_roc_economic_status(torch.tensor(outputs[5]), labels[:, 5].long())

            print(f"[{args.rank}] " + f"train: " +
                  f"epoch {epoch}/{max_epochs - 1}, " +
                  f"step_within_epoch {step}/{len(train_loader) - 1}, " +
                  f"loss: {loss.item():.2f}, " +
                  f"batch_roc for sex: {batch_roc_sex.item():.2f}, " +
                  f"batch_roc for smoking: {batch_roc_smoking.item():.2f}, " +
                  f"batch_roc for sleeplessness: {batch_roc_sleeplessness.item():.2f}, " +
                  f"batch_roc for alcohol: {batch_roc_alcohol.item():.2f}, " +
                  f"batch_roc for depression: {batch_roc_depression.item():.2f}, " +
                  f"batch_roc for economic_status: {batch_roc_economic_status.item():.2f}, " +
                  f"time: {(time() - tik):.2f}s")

            writer.add_scalar("train/batch_loss", scalar_value=loss.item(), global_step=global_step)
            writer.add_scalar("train/batch_roc_sex", scalar_value=batch_roc_sex.item(), global_step=global_step)
            writer.add_scalar("train/batch_roc_smoking", scalar_value=batch_roc_smoking.item(), global_step=global_step)
            writer.add_scalar("train/batch_roc_sleeplessness", scalar_value=batch_roc_sleeplessness.item(),
                              global_step=global_step)
            writer.add_scalar("train/batch_roc_alcohol", scalar_value=batch_roc_alcohol.item(), global_step=global_step)
            writer.add_scalar("train/batch_roc_depression", scalar_value=batch_roc_depression.item(),
                              global_step=global_step)
            writer.add_scalar("train/batch_roc_economic_status", scalar_value=batch_roc_economic_status.item(),
                              global_step=global_step)

        # https://devblog.pytorchlightning.ai/torchmetrics-pytorch-metrics-built-to-scale-7091b1bec919
        total_train_roc_sex = train_roc_sex.compute()
        total_train_roc_smoking = train_roc_smoking.compute()
        total_train_roc_sleeplessness = train_roc_sleeplessness.compute()
        total_train_roc_alcohol = train_roc_alcohol.compute()
        total_train_roc_depression = train_roc_depression.compute()
        total_train_roc_economic_status = train_roc_economic_status.compute()

        # learning rate decay update
        scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        if dist.get_rank() == 0:
            print(
                f"[{dist.get_rank()}] " +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg roc for sex = {total_train_roc_sex:.2f}" +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg roc for smoking = {total_train_roc_smoking:.2f}" +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg roc for sleeplessness = {total_train_roc_sleeplessness:.2f}" +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg roc for alcohol = {total_train_roc_alcohol:.2f}" +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg roc for depression = {total_train_roc_depression:.2f}" +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg roc for economic status = {total_train_roc_economic_status:.2f}"
            )
            writer.add_scalar("train/epoch_loss", scalar_value=epoch_loss, global_step=epoch)  # YY
            writer.add_scalar("train/epoch_roc_sex", scalar_value=total_train_roc_sex, global_step=epoch)  # YY
            writer.add_scalar("train/epoch_roc_smoking", scalar_value=total_train_roc_smoking,
                              global_step=epoch)  # YY
            writer.add_scalar("train/epoch_roc_sleeplessness", scalar_value=total_train_roc_sleeplessness,
                              global_step=epoch)  # YY
            writer.add_scalar("train/epoch_roc_alcohol", scalar_value=total_train_roc_alcohol, global_step=epoch)  # YY
            writer.add_scalar("train/epoch_roc_depression", scalar_value=total_train_roc_depression,
                              global_step=epoch)  # YY
            writer.add_scalar("train/epoch_roc_economic_status", scalar_value=total_train_roc_economic_status,
                              global_step=epoch)  # YY

        # validation mode
        if (epoch + 1) % val_interval == 0:
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for val_step, val_data in enumerate(val_loader):
                    val_images, val_labels = (
                        val_data[0].to(device, non_blocking=True),
                        val_data[1].to(device, non_blocking=True),
                    )
                    outputs = model(val_images)
                    val_loss_sex = loss_functions[0](outputs[0], val_labels[:, 0].long())
                    val_loss_smoking = loss_functions[1](outputs[1], val_labels[:, 1].long())
                    val_loss_sleeplessness = loss_functions[2](outputs[2], val_labels[:, 2].long())
                    val_loss_alcohol = loss_functions[3](outputs[3], val_labels[:, 3].long())
                    val_loss_depression = loss_functions[4](outputs[4], val_labels[:, 4].long())
                    val_loss_economic_status = loss_functions[5](outputs[5], val_labels[:, 5].long())

                    val_loss += val_loss_sex + val_loss_smoking + val_loss_sleeplessness + val_loss_alcohol + val_loss_depression + val_loss_economic_status

                    batch_roc_sex_valid = valid_roc_sex(torch.tensor(outputs[0]), val_labels[:, 0].long())
                    batch_roc_smoking_valid = valid_roc_smoking(torch.tensor(outputs[1]), val_labels[:, 1].long())
                    batch_roc_sleeplessness_valid = valid_roc_sleeplessness(torch.tensor(outputs[2]),
                                                                            val_labels[:, 2].long())
                    batch_roc_alcohol_valid = valid_roc_alcohol(torch.tensor(outputs[3]), val_labels[:, 3].long())
                    batch_roc_depression_valid = valid_roc_depression(torch.tensor(outputs[4]), val_labels[:, 4].long())
                    batch_roc_economic_status_valid = valid_roc_economic_status(torch.tensor(outputs[5]),
                                                                                val_labels[:, 5].long())

                    if dist.get_rank() == 0:  # print only for rank 0
                        print(f"Batch {val_step} AUROC for sex is {batch_roc_sex_valid}")
                        print(f"Batch {val_step} AUROC for smoking is {batch_roc_smoking_valid}")
                        print(f"Batch {val_step} AUROC for sleeplessness is {batch_roc_sleeplessness_valid}")
                        print(f"Batch {val_step} AUROC for alcohol is {batch_roc_alcohol_valid}")
                        print(f"Batch {val_step} AUROC for depression is {batch_roc_depression_valid}")
                        print(f"Batch {val_step} AUROC for economic status is {batch_roc_economic_status_valid}")

                        writer.add_scalar("validation/batch_loss", scalar_value=val_loss, global_step=global_step)
                        writer.add_scalar("validation/batch_roc_age_valid", scalar_value=batch_roc_sex_valid.item(),
                                          global_step=global_step)
                        writer.add_scalar("validation/batch_roc_education_valid",
                                          scalar_value=batch_roc_smoking_valid.item(),
                                          global_step=global_step)
                        writer.add_scalar("validation/batch_roc_sleep_valid",
                                          scalar_value=batch_roc_sleeplessness_valid.item(),
                                          global_step=global_step)
                        writer.add_scalar("validation/batch_roc_BMI_valid", scalar_value=batch_roc_alcohol_valid.item(),
                                          global_step=global_step)
                        writer.add_scalar("validation/batch_roc_dbp_valid",
                                          scalar_value=batch_roc_depression_valid.item(),
                                          global_step=global_step)
                        writer.add_scalar("validation/batch_roc_sbp_valid",
                                          scalar_value=batch_roc_economic_status_valid.item(),
                                          global_step=global_step)

                    epoch_val_loss += val_loss

                epoch_val_loss /= val_step
                total_val_roc_sex = valid_roc_sex.compute()
                total_val_roc_smoking = valid_roc_smoking.compute()
                total_val_roc_sleeplessness = valid_roc_sleeplessness.compute()
                total_val_roc_alcohol = valid_roc_alcohol.compute()
                total_val_roc_depression = valid_roc_depression.compute()
                total_val_roc_economic_status = valid_roc_economic_status.compute()

                result_sex = total_val_roc_sex.cpu().detach().numpy()
                result_smoking = total_val_roc_smoking.cpu().detach().numpy()
                result_sleeplessness = total_val_roc_sleeplessness.cpu().detach().numpy()
                result_alcohol = total_val_roc_alcohol.cpu().detach().numpy()
                result_depression = total_val_roc_depression.cpu().detach().numpy()
                result_economic_status = total_val_roc_economic_status.cpu().detach().numpy()

                val_list = [result_sex, result_smoking, result_sleeplessness, result_alcohol, result_depression,
                            result_economic_status]
                val_result = sum(val_list) / len(val_list)

                if dist.get_rank() == 0:  # print only for rank 0
                    print(f"AUROC sex on all data: {result_sex}")
                    print(f"AUROC smoking on all data: {result_smoking}")
                    print(f"AUROC sleeplessness on all data: {result_sleeplessness}")
                    print(f"AUROC alcohol on all data: {result_alcohol}")
                    print(f"AUROC depression on all data: {result_depression}")
                    print(f"AUROC economic status on all data: {result_economic_status}")

                    writer.add_scalar("validation/AUROC_sex", scalar_value=result_sex, global_step=epoch)
                    writer.add_scalar("validation/AUROC_smoking", scalar_value=result_smoking, global_step=epoch)
                    writer.add_scalar("validation/AUROC_sleeplessness", scalar_value=result_sleeplessness,
                                      global_step=epoch)
                    writer.add_scalar("validation/AUROC_alcohol", scalar_value=result_alcohol, global_step=epoch)
                    writer.add_scalar("validation/AUROC_depression", scalar_value=result_depression, global_step=epoch)
                    writer.add_scalar("validation/AUROC_economic status", scalar_value=result_economic_status,
                                      global_step=epoch)

                if epoch == 0:
                    best_loss = val_loss
                    best_model = model
                    best_metric_epoch = epoch
                    best_metric = val_result

                elif epoch != 0:
                    if val_result > best_metric:
                        best_loss = val_loss
                        best_model = model
                        best_metric = val_result
                        best_metric_epoch = epoch
                        if dist.get_rank() == 0:
                            print(f"best_result={best_metric}, best model has been updated")

                            torch.save(best_model.module.state_dict(),
                                       os.path.join(args.logdir,
                                                    model_name
                                                    + str(best_metric_epoch + 1)
                                                    + f"loss{epoch_loss:.2f}"
                                                    + f"r2{best_metric:.2f}"
                                                    + '.pth'))

                if dist.get_rank() == 0:
                    print(
                        f"current epoch: {epoch + 1}, current MSE: {val_loss}",
                        f" best MSE: {best_loss}",
                        f" at epoch: {best_metric_epoch + 1}"
                    )

        train_roc_sex.reset()
        train_roc_smoking.reset()
        train_roc_sleeplessness.reset()
        train_roc_alcohol.reset()
        train_roc_depression.reset()
        train_roc_economic_status.reset()

        valid_roc_sex.reset()
        valid_roc_smoking.reset()
        valid_roc_sleeplessness.reset()
        valid_roc_alcohol.reset()
        valid_roc_depression.reset()
        valid_roc_economic_status.reset()

    print(f"[{dist.get_rank()}] " + f"train completed, epoch losses: {epoch_loss_values}")
    if dist.get_rank() == 0:
        torch.save(best_model.module.state_dict(),
                   os.path.join('./savedmodel', working_dir,
                                model_name
                                + str(best_metric_epoch + 1)
                                + f"loss{epoch_loss:.2f}"
                                + f"acc{best_metric:.2f}"
                                + '.pth'))
    best_model_wts = copy.deepcopy(best_model.module.state_dict())
    return best_model_wts


best_model_wts = train_model(model=model, train_loader=train_loader, val_loader=val_loader, max_epochs=max_epochs,
                             optimizer=optimizer,
                             loss_functions=[loss_function_sex, loss_function_smoking, loss_function_sleeplessness,
                                             loss_function_alcohol, loss_function_depression,
                                             loss_function_economic_status],
                             model_name=args.model_name,
                             working_dir=args.working_dir)

dist.destroy_process_group()
