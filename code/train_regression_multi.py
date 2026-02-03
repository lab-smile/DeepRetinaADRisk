# Basic library
import os
import random
import copy
import numpy as np
import pandas as pd
import tqdm
import argparse
from datetime import timedelta

# Pytorch Related Library
import torch
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torchmetrics

# Scikit-Learn Library
from sklearn.model_selection import train_test_split

# Monai Library
from monai.config import print_config
from monai.data import DataLoader, DistributedSampler
from monai.metrics import ROCAUCMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    Resize,
    RandZoom,
    ScaleIntensity)

# defining the input arguments for the script.
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--random_state', default=0, type=int, help='random seed')
parser.add_argument('--local_rank', type=int)
parser.add_argument('--image_dir', default='/red/ruogu.fang/UKB/data/Eye/21015_fundus_left_1/', type=str, help='random seed')
parser.add_argument('--csv_dir', default='/red/ruogu.fang/leem.s/NSF-SCH/data/age.csv', type=str, help='random seed')

parser.add_argument('--eye_code', default='_21015_0_0.png', type=str, help='random seed')
parser.add_argument('--label_code', default='21003-0.0', type=str, help='random seed')
parser.add_argument('--exclude', type=float, nargs='+')

parser.add_argument('--working_dir', default='ViT_age', type=str, help='random seed')
parser.add_argument('--model_name', default='ViT_age', type=str, help='random seed')
parser.add_argument('--base_model', default='google/vit-base-patch16-224-in21k', type=str, help='the string of model from hugging-face library')

parser.add_argument('--input_size', default=224, type=int, help='input size for the model')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate of the training')
parser.add_argument('--epoch', default=100, type=int, help='maximum number of epoch')
parser.add_argument('--batch_size', default=32, type=int, help='batch size of the training')

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
image_dir = args.image_dir
csv_dir = args.csv_dir
eye_code = args.eye_code
label_code = args.label_code
ex = args.exclude

# extract the patient eid with both images and label csv files
file_list = os.listdir(image_dir)
eid_list = [s.replace(eye_code, '') for s in file_list]

csv_df = pd.read_csv(csv_dir)
convert_dict = {'eid': str}
csv_df = csv_df.astype(convert_dict)

label_df = csv_df[csv_df['eid'].isin(eid_list)]
label_df = label_df.dropna(subset=[label_code])

if ex:
    for i in ex:
        label_df = label_df[label_df[label_code] != i]

label_df['image'] = label_df['eid'] + eye_code
label_df['path'] = image_dir + label_df['image']

# Defining the input & output for the data
X = label_df['path'].values.tolist()
y = label_df[label_code].values.tolist()

# Defining the train, val, test split
X_train, X_remain, y_train, y_remain = train_test_split(X, y, train_size=0.8, random_state=random_state)
X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, train_size=0.5, random_state=random_state)

print('The size of training samples {}'.format(len(y_train)) )
print('The size of validation samples {}'.format(len(y_val)))
print('The size of test samples {}'.format(len(y_test)))

# Regression dataset definition
class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]
    
class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


image_size = args.input_size

train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize((image_size, image_size)),
        ScaleIntensity(),
        # RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
    ]
)

val_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize((image_size, image_size)),
        ScaleIntensity()
    ]
)

# Dataset & Dataloader
train_ds = RegressionDataset(X_train, y_train, train_transforms)
train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)

val_ds = RegressionDataset(X_val, y_val, val_transforms)
val_sampler = DistributedSampler(dataset=val_ds, even_divisible=True, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=val_sampler)

test_ds = RegressionDataset(X_test, y_test, val_transforms)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

# Defining the model
from transformers import ViTFeatureExtractor, ViTForImageClassification, SwinForImageClassification
model_name_or_path = args.base_model

device = torch.device(f"cuda:{args.local_rank}")
torch.cuda.set_device(device)

if 'vit' in args.base_model:
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(label_df[label_code].unique()))

elif 'swin' in args.base_model:
    model = SwinForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(label_df[label_code].unique()),
        ignore_mismatched_sizes=True)

metric = torchmetrics.MeanAbsoluteError().to(device)
model.metric = metric
model.to(device)

# Defining the variables for training
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
loss_function = torch.nn.MSELoss()
max_epochs = args.epoch
model = DistributedDataParallel(model, device_ids=[device])

# Training the model
def train_model(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=max_epochs,
                optimizer=optimizer,
                loss_function=loss_function,
                model_name='ViT_age',
                working_dir='./red/ruogu.fang/leem.s/NSF-SCH/code/savedmodel'):

    if not os.path.exists(os.path.join('./savedmodel', working_dir)):
        os.makedirs(os.path.join('./savedmodel', working_dir))
    else:
        pass

    best_loss = 0
    best_metric = -1
    best_metric_epoch = 0
    epoch_loss_values = []
    metric_values = []
    val_interval = 1

    for epoch in range(max_epochs):
        print("-" * 10, flush=True)
        print(f"[{dist.get_rank()}] " + "-" * 10 + f" epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        train_sampler.set_epoch(epoch)
        for inputs, labels in train_loader:
            step += 1
            labels = labels.type(torch.Tensor)
            labels = labels.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs[0], labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            epoch_len = len(train_ds) // train_loader.batch_size
            #print(f"[{dist.get_rank()}] " + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        
        scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"[{dist.get_rank()}] " + f"epoch {epoch + 1}, average loss: {epoch_loss:.4f}")


        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device, non_blocking=True),
                        val_data[1].to(device, non_blocking=True),
                    )
                    #y_pred = torch.cat([y_pred, model(val_images)[0]], dim=0)
                    #y = torch.cat([y, val_labels], dim=0)
                    
                    outputs = model(val_images)[0]
                    val_loss = loss_function(outputs.squeeze(), val_labels)
                    result=metric(outputs.squeeze().clone().detach(), val_labels)
                    
                    if dist.get_rank() == 0:  # print only for rank 0
                        print(f"Batch MAE: {result}")
                
                result = metric.compute()
                result = result.cpu().detach().numpy()
                print(f"Mean Absolute Error on all data: {result}, accelerator rank: {dist.get_rank()}")

                if epoch == 0:
                    best_loss = val_loss
                    best_model = model
                    best_metric_epoch = epoch
                    best_metric = result
                    
                elif epoch != 0:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model = model
                        best_metric = result
                        best_metric_epoch = epoch
                        if dist.get_rank()==0:
                            print("the best model has been updated")

                if dist.get_rank()==0:
                    print(
                        (f"current epoch: {epoch + 1}, current MAE: {result}"),
                        (f" best MAE: {best_metric}"),
                        (f" at epoch: {best_metric_epoch+1}")
                    )
            
        metric.reset()
    
    print(f"[{dist.get_rank()}] " + f"train completed, epoch losses: {epoch_loss_values}")
    torch.save(best_model.module.state_dict(), os.path.join('./savedmodel', working_dir, model_name+str(best_metric_epoch+1)+'.pth'))
    best_model_wts = copy.deepcopy(best_model.module.state_dict())
    return best_model_wts

best_model_wts = train_model(model=model, train_loader=train_loader, val_loader=val_loader, max_epochs=max_epochs, optimizer=optimizer,
            loss_function=loss_function, model_name=args.model_name, working_dir=args.working_dir)

dist.destroy_process_group()