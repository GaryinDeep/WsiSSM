import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
import PIL
import argparse
from einops import rearrange
import sklearn
from sklearn.metrics import accuracy_score # 添加此句以解决sklearn隐藏错误

import torch
import torch.nn.functional as F

from lib.dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_pkl, load_pkl
np.random.seed(0)
os.environ['CUDA_VISIBLE_DEVICES'] = "1" 

def evaluation(preds, targets, mode="micro", smooth = 1e-7):

    def metric(m1,m2):
        intersection = (m1 * m2).sum() 
        union = m1.sum() + m2.sum()            
        # conventional
        accuracy = sklearn.metrics.accuracy_score(m2, m1) # 准确率 
        P = sklearn.metrics.precision_score(m2, m1)
        R = sklearn.metrics.recall_score(m2, m1)            
        sensitivity = sklearn.metrics.recall_score(m2, m1, pos_label=1) # recall
        specificity = sklearn.metrics.recall_score(m2, m1, pos_label=0)            
        # dice and IOU
        dice = (2. * intersection + smooth) / (union + smooth)
        iou = (intersection + smooth) / (union - intersection + smooth)
        return accuracy, P, R, sensitivity, specificity, dice, iou

    if mode=="micro":
        for index, (p, t) in enumerate(zip(preds, targets)):   
            if index ==0:
                m1 = p.reshape(-1)  # Flatten
                m2 = t.reshape(-1) # Flatten
            else:
                m1 = np.concatenate([m1, p.reshape(-1)])
                m2 = np.concatenate([m2, t.reshape(-1)])
        accuracy, P, R, sensitivity, specificity, dice, iou = metric(m1,m2)
 
    elif mode=="macro":
        metric_list = []
        for index, (p, t) in enumerate(zip(preds, targets)): 
            m1 = p.reshape(-1)  # Flatten
            m2 = t.reshape(-1) # Flatten
            metric_list.append(metric(m1,m2))
        accuracy = np.mean([i[0] for i in metric_list])
        P = np.mean([i[1] for i in metric_list])
        R = np.mean([i[2] for i in metric_list])
        sensitivity = np.mean([i[3] for i in metric_list])
        specificity = np.mean([i[4] for i in metric_list])
        dice = np.mean([i[5] for i in metric_list])
        iou = np.mean([i[6] for i in metric_list])

    return accuracy, P, R, sensitivity, specificity, dice, iou


def get_numpy_from_tensor(tensor):
    return tensor.cpu().detach().numpy()


def bulit_model(model_name):
    # load args
    with open(args_path)  as f :
        for i in f:
            args = i
    # load model
    from lib.models.WsiSSM.classification_head_puzzle import get_model
    model = get_model(**eval(args))

    print("################# load  finetuned ###################")
    if os.path.exists(finetuned_path):
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(finetuned_path, weights_only=False))
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                print("Missing or Unexpected module")
        else:
            print("\033[92m Successfully Loaded Finetuned WsiSSM model from {} \033[00m".format(finetuned_path))
    else:
        raise NotImplementedError
    
    return model



dataset_name = "camelyon"
# dataset_name = "PANDA"

split = "splits_0"         
# split = "splits_external"  

dice_model = "micro"
# dice_model = "macro" # for splits_micro

model_name = "our"

th= int(255*0.5)


data_dir = f"./data/{dataset_name}/FEATURES_dino"
args_path = f"./data/{dataset_name}/results/WsiSSM_puzzle_stage2/experiment_task_1_tumor_vs_normal_WsiSSM_puzzle_stage2_back1.txt"
finetuned_path = f"./data/{dataset_name}/results/WsiSSM_puzzle_stage2/fold_0/checkpoint_best.pt"


if dataset_name == "camelyon":
    csv_path = "./data/camelyon/dataset_csv/camelyon_16_17.csv"
    split_path = f"./data/camelyon/dataset_csv/splits/camelyon_16_17_seg/{split}.csv"
    till_level = 1    # till_level == patch_level 

elif dataset_name == "PANDA":
    csv_path = "./data/PANDA/dataset_csv/PANDA.csv"
    split_path = f"./data/PANDA/dataset_csv/splits/task_1_tumor_vs_normal_100_seg/{split}.csv"
    till_level = 0


pixel_max_value = 255
seed = 1
basic_patch_size  = 256
scale = 2**(till_level)
patch_size  = basic_patch_size * scale

print("################# Building model ###################")
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
model = bulit_model(model_name)
model = model.to(device) # 存储到GPU
model.eval()

print("################# Load Dataset ###################")
label_dict = {'normal_tissue':0, 'tumor_tissue':1}
# label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2}  

n_classes=len(label_dict)
dataset = Generic_MIL_Dataset(csv_path = csv_path,
                        data_dir= data_dir,
                        shuffle = False, 
                        seed = seed, 
                        print_info = True,
                        label_dict = label_dict,
                        patient_strat=False,
                        ignore=[])
dataset.load_from_h5(True)  # load from .h5 (include coords)
dataset_csv = pd.read_csv(csv_path)


split_csv = pd.read_csv(split_path)
name_list = split_csv["test"].dropna(axis=0, how='any').tolist()

pred_mask_list, mask_list = [], []
for image_name in name_list:
    print("################# Load Data ###################")
    item_NO = dataset_csv[dataset_csv['slide_id'] == image_name].index.tolist()[0]  
    image, label, coords = dataset.__getitem__(item_NO)
    image = image.unsqueeze(0) # (1,L,D)
    coords = torch.LongTensor(coords).unsqueeze(0) # (1,L,2)
    print(image.shape) 

    # out of memmory

    print("################# Predict ###################")
    with torch.no_grad():
        from lib.models.WsiSSM.classification_head_puzzle import make_cam
        from lib.trainer.trainer_WsiSSM_puzzle import tile_features, merge_features
        images = image.to(device)
        img_coords= coords.to(device)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _, features = model(images, img_coords) # (b,2)
            cams =  make_cam(features)  # （B,2,L）  

            num_pieces = 4
            tiled_images, tiled_coords, levels  = tile_features(images, img_coords, num_pieces)
            tiled_features = []
            for index,(tiled_image, tiled_coord) in enumerate(zip(tiled_images, tiled_coords)): # (B,L/4,2), (L/4,2)
                if tiled_image.shape[1] != 0:
                    tiled_logit, tiled_feature = model(tiled_image, tiled_coord) # (b,2), (B,2,L/4)
                    tiled_features.append(tiled_feature) # [4, (B,2,L/4) ]
            re_features = merge_features(tiled_features, levels) # (B,2,L)
            re_cams =  make_cam(re_features)

        cam = get_numpy_from_tensor(cams[0][label]) # (L)
        re_cams = get_numpy_from_tensor(re_cams[0][label]) # (L)
        img_coords = get_numpy_from_tensor(img_coords[0]) # (L, 2)


    print("################# genearating cam pic ###################")
    x_min, y_min = min(img_coords[:,1]), min(img_coords[:,0])
    img_coords[:,1] = img_coords[:,1] - [x_min]*len(img_coords)
    img_coords[:,0] = img_coords[:,0] - [y_min]*len(img_coords)
    w = max(img_coords[:,1]) +1
    h = max(img_coords[:,0]) +1
    print("w ", w, "h ", h)

    pic = np.zeros((w,h,1), dtype=np.uint8) # (w,h,1) 
    for value, value_puzzle, coord in zip(cam,re_cams, img_coords):
        pic[coord[1]:coord[1]+patch_size, coord[0]:coord[0]+patch_size] = (value*pixel_max_value).astype(np.uint8)   
    print("pic shape ", pic.shape)


    print("################# Load mask ###################")
    mask_path = os.path.join(data_dir, "mask_files", image_name+".png")
    if not os.path.exists(mask_path):
        print(mask_path)
        continue
    mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # (w,h)
    mask_list.append(mask_gt)
    print(image_name, mask_gt.shape)
    # 缩放
    pic_resize = cv2.resize(pic, (mask_gt.shape[1], mask_gt.shape[0]), interpolation=cv2.INTER_LINEAR)  # (w,h,1) 
    print(pic_resize.shape)
    # 阈值
    thresh, pred_mask = cv2.threshold(src=pic_resize, thresh=th, maxval=1, type=cv2.THRESH_BINARY)
    print(pred_mask.shape)
    pred_mask_list.append(pred_mask)

    # image = PIL.Image.fromarray(mask_gt)
    # image.save("mask_gt.png")
    # image = PIL.Image.fromarray(pred_mask)
    # image.save("pred_mask.png")

    accuracy, P, R, sensitivity, specificity, dice, iou = evaluation(pred_mask, mask_gt)
    print(image_name, accuracy, P, R, sensitivity, specificity, dice, iou)


# 计算指标
accuracy, P, R, sensitivity, specificity, dice, iou = evaluation(pred_mask_list, mask_list, mode=dice_model)
print(accuracy, P, R, sensitivity, specificity, dice, iou)

