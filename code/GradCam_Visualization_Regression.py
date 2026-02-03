import os
import random
import copy
import cv2
import numpy as np
import pandas as pd
import tqdm
import argparse
from PIL import Image
from datetime import timedelta

# Pytorch Related Library
import torch
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.distributed as dist

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

from transformers import ViTForImageClassification, SwinForImageClassification, AutoFeatureExtractor
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image

parser = argparse.ArgumentParser(description='Parameters')

parser.add_argument('--feat', type=int, default=0)

random_state = 0

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ["PYTHONHASHSEED"] = str(random_state)
args = parser.parse_args()


class SwinforRegression(torch.nn.Module):
    def __init__(self, model_name_or_path='microsoft/swin-large-patch4-window12-384-in22k', n_label=7):
        super().__init__()
        model = SwinForImageClassification.from_pretrained(model_name_or_path, num_labels=n_label,
                                                           ignore_mismatched_sizes=True)
        self.swin = model.swin
        self.regressor_age = torch.nn.Linear(1536, 1)
        self.regressor_education = torch.nn.Linear(1536, 1)
        self.regressor_BMI = torch.nn.Linear(1536, 1)
        self.regressor_dbp = torch.nn.Linear(1536, 1)
        self.regressor_sbp = torch.nn.Linear(1536, 1)
        self.regressor_HbA1C = torch.nn.Linear(1536, 1)

    def forward(self, x):
        y_swin = self.swin(x).pooler_output
        y_age = self.regressor_age(y_swin)
        y_education = self.regressor_education(y_swin)
        y_BMI = self.regressor_BMI(y_swin)
        y_dbp = self.regressor_dbp(y_swin)
        y_sbp = self.regressor_sbp(y_swin)
        y_HbA1C = self.regressor_HbA1C(y_swin)

        return y_age, y_education, y_BMI, y_dbp, y_sbp, y_HbA1C


class SingleOutputModel(torch.nn.Module):
    def __init__(self, model, output_index=0):
        super().__init__()
        self.model = model
        self.output_index = output_index

    def forward(self, x):
        return self.model(x)[self.output_index]


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        return model_output[self.category]


def reshape_transform(tensor, height=18, width=18):
    result = tensor[:, :, :].reshape(tensor.size(0),
                                     height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device to use')

    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    args = parser.parse_args()

    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


def get_saliency_map_regression(img_path='../AutoMorph/Regression/',
                                model_path='./log/regression/Swin_regression6loss3.96r20.26.pth',
                                feat_idx=0):
    label_df = pd.read_csv('./data/test_base_regression.csv', index_col=0) 
    img_list = label_df['image'].tolist()
    img_paths = label_df['path'].tolist()
    
    result_dir = os.path.join(img_path, 'ScoreCAM2')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    from transformers import ViTForImageClassification, SwinForImageClassification, AutoFeatureExtractor

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = SwinforRegression(model_name_or_path='microsoft/swin-large-patch4-window12-384-in22k', n_label=7)
    model.eval()
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    
    for img_name, img_path in zip(img_list, img_paths):
        print('Processing the image: ', img_name)      
        if not os.path.exists(os.path.join(result_dir,str(feat_idx))):
            os.makedirs(os.path.join(result_dir,str(feat_idx)))
                
        class_index = feat_idx
        model_for_gradcam = SingleOutputModel(model, class_index)
        model.to(device)
        model.eval()
                
        img = Image.open(img_path)
        img = img.resize((576,576))
        img = np.array(img)
        img = img.astype(np.float32)
        img = img/255
        convert_tensor = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((576,576)),
                                             ScaleIntensity()
                                            ])

        input_tensor = convert_tensor(img).to(device)
        pred_ori = model(input_tensor.unsqueeze(0))
        prob = torch.nn.functional.softmax(torch.tensor(pred_ori[class_index].detach().cpu()))[0]
        output_index = int(torch.argmax(pred_ori[class_index]))
        
        #gt = label_df[label_df['image'] == img_name].iloc[:,class_index+1]
        targets = [ClassifierOutputTarget(0)]
        target_layers = [model_for_gradcam.model.swin.encoder.layers[-1].blocks[-1].layernorm_before]
        cam = ScoreCAM(model=model_for_gradcam, target_layers=target_layers, reshape_transform=reshape_transform)
        pred = model_for_gradcam(input_tensor.unsqueeze(0))
        output = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets, aug_smooth=False, eigen_smooth=True)
        cam = output[0, :]
        cam_image = show_cam_on_image(img, cam)
        output_name = img_name[:-4] + '_' + str(class_index) + '_' + str(output_index) + '.jpg'
        cv2.imwrite(os.path.join(result_dir, str(feat_idx), output_name), cam_image)
        torch.cuda.empty_cache()
        #np.save(os.path.join(result_dir, str(feat_idx), img_name[:-4] + '.npy'), cam)

get_saliency_map_regression(feat_idx = args.feat)