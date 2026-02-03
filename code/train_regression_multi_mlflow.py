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
parser.add_argument('--random_state', default=0, type=int, help='random seed')
parser.add_argument('--local-rank', type=int)
parser.add_argument('--left_image_dir', default='/red/ruogu.fang/UKB/data/Eye/21015_fundus_left_1/', type=str,
                    help='There are 2 image directories. You need to put left fundus image dir here.')
parser.add_argument('--right_image_dir', default='/red/ruogu.fang/UKB/data/Eye/21016_fundus_right_1_horizontal/',
                    type=str,
                    help='There are 2 image directories. You need to put right fundus image dir here. Keep in mind either left or right fundus images should be flipped horizontally.')

parser.add_argument('--csv_dir', default='/red/ruogu.fang/leem.s/NSF-SCH/data/regression_data.csv', type=str,
                    help='Where the csv file for the dataset is stored.')

parser.add_argument('--left_eye_code', default='_21015_0_0.png', type=str,
                    help='The code added to eid of the UKB subjects')
parser.add_argument('--right_eye_code', default='_21016_0_0.png', type=str,
                    help='The code added to eid of the UKB subjects')

parser.add_argument('--label', type=str, nargs='+', help='Column names of the csv file. It should be the label.')
parser.add_argument('--exclude', type=float, nargs='+',
                    help='The exclude code. example, -1, -2, -3 negative integers are not labels, but exclusion code for reason')

parser.add_argument('--working_dir', default='Swin_regression', type=str, help='where you would save the result')
parser.add_argument('--model_name', default='Swin_regression', type=str, help='name of saved model.')
parser.add_argument('--base_model', default='microsoft/swin-large-patch4-window12-384-in22k', type=str,
                    help='the string of model from hugging-face library')

parser.add_argument('--input_size', default=224, type=int, help='input size for the model')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate of the training')
parser.add_argument('--epoch', default=100, type=int, help='maximum number of epoch')
parser.add_argument('--batch_size', default=32, type=int, help='batch size of the training')

parser.add_argument('--logdir', default="./log", type=str, help='log')

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

# excluding the exclude labels in each column
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

# get mean and std values for training set for normalization of the data.
mu = np.mean(y_train, axis=0)
std = np.std(y_train, axis=0)

# simple normalization
y_train_norm = (y_train - mu) / std
y_val_norm = (y_val - mu) / std
y_test_norm = (y_test - mu) / std

# printing the dataset size.
if dist.get_rank() == 0:
    print('The size of training samples {}'.format(len(y_train)))
    print('The size of validation samples {}'.format(len(y_val)))
    print('The size of test samples {}'.format(len(y_test)))


# Regression dataset definition
class RegressionDataset_All(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
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
train_ds = RegressionDataset_All(X_train, y_train_norm, train_transforms)
train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)

val_ds = RegressionDataset_All(X_val, y_val_norm, val_transforms)
val_sampler = DistributedSampler(dataset=val_ds, even_divisible=True, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=val_sampler)

test_ds = RegressionDataset_All(X_test, y_test_norm, val_transforms)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

# Defining the model
from transformers import ViTForImageClassification, SwinForImageClassification

model_name_or_path = args.base_model
device = torch.device(f"cuda:{args.local_rank}")
torch.cuda.set_device(device)


# Specialized model with multiple outputs.
class SwinforRegression(torch.nn.Module):
    def __init__(self, model_name_or_path=args.base_model, n_label=7):
        super().__init__()
        model = SwinForImageClassification.from_pretrained(model_name_or_path, num_labels=n_label,
                                                           ignore_mismatched_sizes=True)
        self.swin = model.swin
        self.regressor_age = torch.nn.Linear(1536, 1)
        self.regressor_education = torch.nn.Linear(1536, 1)
        self.regressor_sleep = torch.nn.Linear(1536, 1)
        self.regressor_BMI = torch.nn.Linear(1536, 1)
        self.regressor_dbp = torch.nn.Linear(1536, 1)
        self.regressor_sbp = torch.nn.Linear(1536, 1)
        self.regressor_HbA1C = torch.nn.Linear(1536, 1)

    def forward(self, x):
        y_swin = self.swin(x).pooler_output
        y_age = self.regressor_age(y_swin)
        y_education = self.regressor_education(y_swin)
        y_sleep = self.regressor_sleep(y_swin)
        y_BMI = self.regressor_BMI(y_swin)
        y_dbp = self.regressor_dbp(y_swin)
        y_sbp = self.regressor_sbp(y_swin)
        y_HbA1C = self.regressor_HbA1C(y_swin)

        return y_age, y_education, y_sleep, y_BMI, y_dbp, y_sbp, y_HbA1C


if 'vit' in args.base_model:
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(args.label))

elif 'swin' in args.base_model:
    model = SwinforRegression(model_name_or_path=args.base_model, n_label=7)

metric = torchmetrics.MeanSquaredError().to(device)
model.metric = metric
model.to(device)

# evaluation metric.
train_r2_age = torchmetrics.R2Score().to(device)
valid_r2_age = torchmetrics.R2Score().to(device)

train_r2_education = torchmetrics.R2Score().to(device)
valid_r2_education = torchmetrics.R2Score().to(device)

train_r2_sleep = torchmetrics.R2Score().to(device)
valid_r2_sleep = torchmetrics.R2Score().to(device)

train_r2_BMI = torchmetrics.R2Score().to(device)
valid_r2_BMI = torchmetrics.R2Score().to(device)

train_r2_dbp = torchmetrics.R2Score().to(device)
valid_r2_dbp = torchmetrics.R2Score().to(device)

train_r2_sbp = torchmetrics.R2Score().to(device)
valid_r2_sbp = torchmetrics.R2Score().to(device)

train_r2_HbA1C = torchmetrics.R2Score().to(device)
valid_r2_HbA1C = torchmetrics.R2Score().to(device)

# Defining the variables for training
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# different loss functions for each risk factors.
loss_function_age = torch.nn.MSELoss()
loss_function_education = torch.nn.MSELoss()
loss_function_sleep = torch.nn.MSELoss()
loss_function_BMI = torch.nn.MSELoss()
loss_function_dbp = torch.nn.MSELoss()
loss_function_sbp = torch.nn.MSELoss()
loss_function_HbA1C = torch.nn.MSELoss()
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
                working_dir='./red/ruogu.fang/leem.s/NSF-SCH/code/savedmodel'):

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

    global_step = 0

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
            labels = labels.type(torch.Tensor)

            # transfer the data to gpu
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Feed-Forward
            outputs = model(inputs)

            # compute the separate loss
            loss_age = loss_functions[0](outputs[0].squeeze(), labels[:, 0])
            loss_education = loss_functions[1](outputs[1].squeeze(), labels[:, 1])
            loss_sleep = loss_functions[2](outputs[2].squeeze(), labels[:, 2])
            loss_BMI = loss_functions[3](outputs[3].squeeze(), labels[:, 3])
            loss_dbp = loss_functions[4](outputs[4].squeeze(), labels[:, 4])
            loss_sbp = loss_functions[5](outputs[5].squeeze(), labels[:, 5])
            loss_HbA1C = loss_functions[6](outputs[6].squeeze(), labels[:, 6])

            # compute the loss and back-propagate the gradient.
            loss = loss_age + loss_education + loss_sleep + loss_BMI + loss_dbp + loss_sbp + loss_HbA1C
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # calculate the accuracy for trainset
            batch_r2_age = train_r2_age(torch.tensor(outputs[0]), labels[:, 0].unsqueeze(1))
            batch_r2_education = train_r2_education(torch.tensor(outputs[1]), labels[:, 1].unsqueeze(1))
            batch_r2_sleep = train_r2_sleep(torch.tensor(outputs[2]), labels[:, 2].unsqueeze(1))
            batch_r2_BMI = train_r2_BMI(torch.tensor(outputs[3]), labels[:, 3].unsqueeze(1))
            batch_r2_dbp = train_r2_dbp(torch.tensor(outputs[4]), labels[:, 4].unsqueeze(1))
            batch_r2_sbp = train_r2_sbp(torch.tensor(outputs[5]), labels[:, 5].unsqueeze(1))
            batch_r2_HbA1C = train_r2_HbA1C(torch.tensor(outputs[6]), labels[:, 6].unsqueeze(1))

            print(f"[{args.rank}] " + f"train: " +
                  f"epoch {epoch}/{max_epochs - 1}, " +
                  f"step_within_epoch {step}/{len(train_loader) - 1}, " +
                  f"loss: {loss.item():.2f}, " +
                  f"batch_r2 for age: {batch_r2_age.item():.2f}, " +
                  f"batch_r2 for education: {batch_r2_education.item():.2f}, " +
                  f"batch_r2 for sleep duration: {batch_r2_sleep.item():.2f}, " +
                  f"batch_r2 for BMI: {batch_r2_BMI.item():.2f}, " +
                  f"batch_r2 for diastolic bp: {batch_r2_dbp.item():.2f}, " +
                  f"batch_r2 for systolic bp: {batch_r2_sbp.item():.2f}, " +
                  f"batch_r2 for HbA1C: {batch_r2_HbA1C.item():.2f}, " +
                  f"time: {(time() - tik):.2f}s"
                  )
            writer.add_scalar("train/batch_loss", scalar_value=loss.item(), global_step=global_step)
            writer.add_scalar("train/batch_r2_age", scalar_value=batch_r2_age.item(), global_step=global_step)
            writer.add_scalar("train/batch_r2_education", scalar_value=batch_r2_age.item(), global_step=global_step)
            writer.add_scalar("train/batch_r2_sleep", scalar_value=batch_r2_age.item(), global_step=global_step)
            writer.add_scalar("train/batch_r2_BMI", scalar_value=batch_r2_age.item(), global_step=global_step)
            writer.add_scalar("train/batch_r2_dbp", scalar_value=batch_r2_age.item(), global_step=global_step)
            writer.add_scalar("train/batch_r2_sbp", scalar_value=batch_r2_age.item(), global_step=global_step)
            writer.add_scalar("train/batch_r2_HbA1C", scalar_value=batch_r2_age.item(), global_step=global_step)

        # https://devblog.pytorchlightning.ai/torchmetrics-pytorch-metrics-built-to-scale-7091b1bec919
        total_train_r2_age = train_r2_age.compute()
        total_train_r2_education = train_r2_education.compute()
        total_train_r2_sleep = train_r2_sleep.compute()
        total_train_r2_BMI = train_r2_BMI.compute()
        total_train_r2_dbp = train_r2_dbp.compute()
        total_train_r2_sbp = train_r2_sbp.compute()
        total_train_r2_HbA1C = train_r2_HbA1C.compute()

        # learning rate decay update
        scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        if dist.get_rank() == 0:
            print(
                f"[{dist.get_rank()}] " +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg r2 for age = {total_train_r2_age:.2f}" +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg r2 for education = {total_train_r2_education:.2f}" +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg r2 for sleep duration = {total_train_r2_sleep:.2f}" +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg r2 for BMI = {total_train_r2_BMI:.2f}" +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg r2 for diastolic bp = {total_train_r2_dbp:.2f}" +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg r2 for systolic bp = {total_train_r2_sbp:.2f}" +
                f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}, avg r2 for HbA1C = {total_train_r2_HbA1C:.2f}"
            )
            writer.add_scalar("train/epoch_loss", scalar_value=epoch_loss, global_step=epoch)  # YY
            writer.add_scalar("train/epoch_r2_age", scalar_value=total_train_r2_age, global_step=epoch)  # YY
            writer.add_scalar("train/epoch_r2_education", scalar_value=total_train_r2_education,
                              global_step=epoch)  # YY
            writer.add_scalar("train/epoch_r2_sleep", scalar_value=total_train_r2_sleep, global_step=epoch)  # YY
            writer.add_scalar("train/epoch_r2_BMI", scalar_value=total_train_r2_BMI, global_step=epoch)  # YY
            writer.add_scalar("train/epoch_r2_dbp", scalar_value=total_train_r2_dbp, global_step=epoch)  # YY
            writer.add_scalar("train/epoch_r2_sbp", scalar_value=total_train_r2_sbp, global_step=epoch)  # YY
            writer.add_scalar("train/epoch_r2_HbA1C", scalar_value=total_train_r2_HbA1C, global_step=epoch)  # YY

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
                    val_loss_age = loss_functions[0](outputs[0].squeeze(), val_labels[:, 0])
                    val_loss_education = loss_functions[1](outputs[1].squeeze(), val_labels[:, 1])
                    val_loss_sleep = loss_functions[2](outputs[2].squeeze(), val_labels[:, 2])
                    val_loss_BMI = loss_functions[3](outputs[3].squeeze(), val_labels[:, 3])
                    val_loss_dbp = loss_functions[4](outputs[4].squeeze(), val_labels[:, 4])
                    val_loss_sbp = loss_functions[5](outputs[5].squeeze(), val_labels[:, 5])
                    val_loss_HbA1C = loss_functions[6](outputs[6].squeeze(), val_labels[:, 6])
                    val_loss = val_loss_age + val_loss_education + val_loss_sleep + val_loss_BMI + val_loss_dbp + val_loss_sbp + val_loss_HbA1C


                    batch_r2_age_valid = valid_r2_age(torch.tensor(outputs[0]), val_labels[:, 0].unsqueeze(1))
                    batch_r2_education_valid = valid_r2_education(torch.tensor(outputs[1]),
                                                                  val_labels[:, 1].unsqueeze(1))
                    batch_r2_sleep_valid = valid_r2_sleep(torch.tensor(outputs[2]), val_labels[:, 2].unsqueeze(1))
                    batch_r2_BMI_valid = valid_r2_BMI(torch.tensor(outputs[3]), val_labels[:, 3].unsqueeze(1))
                    batch_r2_dbp_valid = valid_r2_dbp(torch.tensor(outputs[4]), val_labels[:, 4].unsqueeze(1))
                    batch_r2_sbp_valid = valid_r2_sbp(torch.tensor(outputs[5]), val_labels[:, 5].unsqueeze(1))
                    batch_r2_HbA1C_valid = valid_r2_HbA1C(torch.tensor(outputs[6]), val_labels[:, 6].unsqueeze(1))

                    if dist.get_rank() == 0:  # print only for rank 0
                        print(f"Batch {val_step} R2 for age is {batch_r2_age_valid}")
                        print(f"Batch {val_step} R2 for education is {batch_r2_education_valid}")
                        print(f"Batch {val_step} R2 for sleep is {batch_r2_sleep_valid}")
                        print(f"Batch {val_step} R2 for BMI is {batch_r2_BMI_valid}")
                        print(f"Batch {val_step} R2 for dbp is {batch_r2_dbp_valid}")
                        print(f"Batch {val_step} R2 for sbp is {batch_r2_sbp_valid}")
                        print(f"Batch {val_step} R2 for HbA1C is {batch_r2_HbA1C_valid}")

                        writer.add_scalar("validation/batch_loss", scalar_value=val_loss, global_step=global_step)
                        writer.add_scalar("validation/batch_r2_age_valid", scalar_value=batch_r2_age_valid.item(),
                                          global_step=global_step)
                        writer.add_scalar("validation/batch_r2_education_valid",
                                          scalar_value=batch_r2_education_valid.item(),
                                          global_step=global_step)
                        writer.add_scalar("validation/batch_r2_sleep_valid", scalar_value=batch_r2_sleep_valid.item(),
                                          global_step=global_step)
                        writer.add_scalar("validation/batch_r2_BMI_valid", scalar_value=batch_r2_BMI_valid.item(),
                                          global_step=global_step)
                        writer.add_scalar("validation/batch_r2_dbp_valid", scalar_value=batch_r2_dbp_valid.item(),
                                          global_step=global_step)
                        writer.add_scalar("validation/batch_r2_sbp_valid", scalar_value=batch_r2_sbp_valid.item(),
                                          global_step=global_step)
                        writer.add_scalar("validation/batch_r2_HbA1C_valid", scalar_value=batch_r2_HbA1C_valid.item(),
                                          global_step=global_step)

                    epoch_val_loss += val_loss

                epoch_val_loss /= val_step
                total_val_r2_age = valid_r2_age.compute()
                total_val_r2_education = valid_r2_education.compute()
                total_val_r2_sleep = valid_r2_sleep.compute()
                total_val_r2_BMI = valid_r2_BMI.compute()
                total_val_r2_dbp = valid_r2_dbp.compute()
                total_val_r2_sbp = valid_r2_sbp.compute()
                total_val_r2_HbA1C = valid_r2_HbA1C.compute()

                result_age = total_val_r2_age.cpu().detach().numpy()
                result_education = total_val_r2_education.cpu().detach().numpy()
                result_sleep = total_val_r2_sleep.cpu().detach().numpy()
                result_BMI = total_val_r2_BMI.cpu().detach().numpy()
                result_dbp = total_val_r2_dbp.cpu().detach().numpy()
                result_sbp = total_val_r2_sbp.cpu().detach().numpy()
                result_HbA1C = total_val_r2_HbA1C.cpu().detach().numpy()

                val_list = [result_age, result_education, result_sleep, result_BMI, result_dbp, result_sbp,
                            result_HbA1C]
                val_result = sum(val_list) / len(val_list)

                if dist.get_rank() == 0:  # print only for rank 0
                    print(f"R2 age on all data: {result_age}")
                    print(f"R2 education on all data: {result_education}")
                    print(f"R2 sleep on all data: {result_sleep}")
                    print(f"R2 BMI on all data: {result_BMI}")
                    print(f"R2 dbp on all data: {result_dbp}")
                    print(f"R2 sbp on all data: {result_sbp}")
                    print(f"R2 HbA1C on all data: {result_HbA1C}")

                    writer.add_scalar("validation/R2_age", scalar_value=result_age, global_step=epoch)
                    writer.add_scalar("validation/R2_education", scalar_value=result_education, global_step=epoch)
                    writer.add_scalar("validation/R2_sleep", scalar_value=result_sleep, global_step=epoch)
                    writer.add_scalar("validation/R2_BMI", scalar_value=result_BMI, global_step=epoch)
                    writer.add_scalar("validation/R2_dbp", scalar_value=result_dbp, global_step=epoch)
                    writer.add_scalar("validation/R2_sbp", scalar_value=result_sbp, global_step=epoch)
                    writer.add_scalar("validation/R2_HbA1C", scalar_value=result_HbA1C, global_step=epoch)

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

        train_r2_age.reset()
        train_r2_education.reset()
        train_r2_sleep.reset()
        train_r2_BMI.reset()
        train_r2_dbp.reset()
        train_r2_sbp.reset()
        train_r2_HbA1C.reset()

        valid_r2_age.reset()
        valid_r2_education.reset()
        valid_r2_sleep.reset()
        valid_r2_BMI.reset()
        valid_r2_dbp.reset()
        valid_r2_sbp.reset()
        valid_r2_HbA1C.reset()

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
                             loss_functions=[loss_function_age, loss_function_education, loss_function_sleep,
                                             loss_function_BMI, loss_function_dbp, loss_function_sbp,
                                             loss_function_HbA1C],
                             model_name=args.model_name,
                             working_dir=args.working_dir)

dist.destroy_process_group()
