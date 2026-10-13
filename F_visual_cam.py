import os
import math 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
import argparse
from einops import rearrange

import torch
import torch.nn.functional as F

from lib.dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_pkl, load_pkl

os.environ['CUDA_VISIBLE_DEVICES'] = "0" 



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



model_name = "our"

# with_bg = False
with_bg = True

dataset_name = "camelyon"
# dataset_name = "PANDA"

# camelyon
image_name = "tumor_110"
# image_name = "test_048"

# PANDA
# image_name = "65529d6dd8a2d9596cae939b6aef2c65"
# image_name = "e891d0d99672757464ee838a0794f424"

slide_ext = ".tif"
# slide_ext = ".svs"

th= 0.5

num_pieces = 4
coord_downsample = 1  


data_dir = f"./data/{dataset_name}/FEATURES_dino"
if dataset_name == "camelyon":
    args_path = f"./data/{dataset_name}/results/WsiSSM_puzzle_stage2/experiment_task_1_tumor_vs_normal_WsiSSM_puzzle_stage2_back1.txt"
    finetuned_path = f"./data/{dataset_name}/results/WsiSSM_puzzle_stage2/fold_0_back1/checkpoint_best.pt"
elif dataset_name == "PANDA":
    args_path = f"./data/{dataset_name}/results/WsiSSM_puzzle_stage2/experiment_task_1_tumor_vs_normal_WsiSSM_puzzle_stage2.txt"
    finetuned_path = f"./data/{dataset_name}/results/WsiSSM_puzzle_stage2/fold_0/checkpoint_best.pt"


if dataset_name == "camelyon":
    csv_path = "./data/camelyon/dataset_csv/camelyon_16_17.csv"
    dataset_csv_path = "./data/camelyon/dataset_csv/camelyon_16_17.csv"
    slice_path = "./data/camelyon/camelyon_16_17/images/" + image_name + slide_ext
    mask_slice_path = "./data/camelyon/camelyon_16_17/mask/"+image_name+"_mask.tif"
    till_level = 1

elif dataset_name == "PANDA":
    csv_path = "./data/PANDA/dataset_csv/PANDA.csv"
    dataset_csv_path = "./data/PANDA/dataset_csv/PANDA.csv"
    slice_path = "./data/PANDA/PANDA/images/" + image_name + slide_ext
    mask_slice_path = "./data/PANDA/PANDA/mask/"+image_name+"_mask.tif"
    till_level = 0


pixel_max_value = 255
seed = 1
basic_patch_size  = 256
scale = 2**(till_level)
patch_size  = basic_patch_size * scale


output_image_path = f"./data/{dataset_name}/pic/ori/" + image_name +"_patho.png"
output_stage_path = f"./data/{dataset_name}/pic/heatmap/" + model_name+"_"+image_name +"_stage.png"
output_puzzle_stage_path = f"./data/{dataset_name}/pic/heatmap/" + model_name+"_"+image_name +"_puzzle_stage.png"
output_pred_mask_path = f"./data/{dataset_name}/pic/heatmap/" + model_name+"_"+image_name +"_pred_mask.png"

# 蒙版
mask_path =f"./data/{dataset_name}/pic/mask/"+image_name+"_patho_mask.png"
mask_recode_path =f"./data/{dataset_name}/pic/mask/"+image_name+"_patho_mask_recode.png"
output_path = f"./data/{dataset_name}/pic/merge/"+image_name+"_anno.png"

for file_path in [output_image_path, output_stage_path, mask_path, output_path]:
    folder_path = "/".join(file_path.split("/")[:-1])
    if not os.path.exists(folder_path):
        print(f"creating: {folder_path}")
        os.makedirs(folder_path)


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
dataset_csv = pd.read_csv(dataset_csv_path)

print("################# Load Data ###################")
item_NO = dataset_csv[dataset_csv['slide_id'] == image_name].index.tolist()[0]  
print(image_name, "item_NO in dataset_csv", item_NO)

image, label, coords = dataset.__getitem__(item_NO)
image = image.unsqueeze(0) # (1,L,D)
coords = torch.LongTensor(coords).unsqueeze(0) # (1,L,2)
print(image.shape)   
  


print("################# Predict ###################")
with torch.no_grad():
    from lib.models.WsiSSM.classification_head_puzzle import make_cam
    from lib.trainer.trainer_WsiSSM_puzzle import tile_features, merge_features
    images = image.to(device)
    img_coords= coords.to(device)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        _, features = model(images, img_coords) # (b,2)
        cams =  make_cam(features)  # （B,2,L）  
            
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

print("################# Show ###################")
print("################# genearating cam pic ###################")
img_coords =  img_coords*coord_downsample
x_min, y_min = min(img_coords[:,1]), min(img_coords[:,0])
img_coords[:,1] = img_coords[:,1] - [x_min]*len(img_coords)
img_coords[:,0] = img_coords[:,0] - [y_min]*len(img_coords)
w = max(img_coords[:,1]) +1
h = max(img_coords[:,0]) +1
print("w ", w, "h ", h)

pic = np.zeros((w,h,1), dtype=np.uint8) # (w,h,1) 
pic_puzzle = np.zeros((w,h,1), dtype=np.uint8)
for value, value_puzzle, coord in zip(cam,re_cams, img_coords):
    pic[coord[1]:coord[1]+patch_size, coord[0]:coord[0]+patch_size] = (value*pixel_max_value).astype(np.uint8)
    pic_puzzle[coord[1]:coord[1]+patch_size, coord[0]:coord[0]+patch_size] = (value_puzzle*pixel_max_value).astype(np.uint8)
print("pic shape ", pic.shape)
print("pic_puzzle shape ", pic_puzzle.shape)



print("################# retriving roi ###################")
if not os.path.exists(output_image_path):
    import openslide
    slide = openslide.open_slide(slice_path)
    print("ori size: ",slide.dimensions)
    region = slide.read_region((y_min, x_min), till_level, (h//scale, w//scale))
    print("load successfully")
    print("region size: ",region.size)

    visual_level = 0.02
    region_ = region.resize((int(h//scale*visual_level), int(w//scale*visual_level)), PIL.Image.LANCZOS)
    print("region resize: ",region_.size)
    region_.save(output_image_path)

print("################# output whole cam ###################")
image = cv2.imread(output_image_path)
height, width, _ = image.shape
pic_resize = cv2.resize(pic, (width, height))
pic_puzzle_resize = cv2.resize(pic_puzzle, (width, height))
print("pic_resize shape ",pic_resize.shape)
print("pic_puzzle_resize shape ",pic_puzzle_resize.shape)

heatmap = cv2.applyColorMap(pic_resize, cv2.COLORMAP_JET)  #CAM resize match input image size
alpha = 0.4
if not with_bg:
    cv2.imwrite(output_stage_path, heatmap)
else:
    result = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    cv2.imwrite(output_stage_path, result)

print("################# output puzzle cam ###################")
heatmap = cv2.applyColorMap(pic_puzzle_resize, cv2.COLORMAP_JET)  #CAM resize match input image size
if not with_bg:
    cv2.imwrite(output_puzzle_stage_path, heatmap)
else:    
    result = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    cv2.imwrite(output_puzzle_stage_path, result)

print("################# output pred mask ###################")
# thresholds = list(np.arange(0.10, 0.50, 0.05))
th= int(pixel_max_value*th)
thresh, pred_mask = cv2.threshold(src=pic_resize, thresh=th, maxval=pixel_max_value, type=cv2.THRESH_BINARY)
cv2.imwrite(output_pred_mask_path, pred_mask)



# print("################# retriving mask roi ###################")
# mask_slide = openslide.open_slide(mask_slice_path)
# print("mask.tif size: ", mask_slide.dimensions)
# region = mask_slide.read_region((y_min//coord_downsample, x_min//coord_downsample), till_level, (h//scale//coord_downsample, w//scale//coord_downsample))
# print("load successfully")
# print("roi shape:", region.size)

# visual_level = 0.02
# mask = region.resize((int(h//scale*visual_level), int(w//scale*visual_level)), PIL.Image.LANCZOS)
# print(mask.size)
# mask.save(mask_path)
# # visual
# mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# mask = mask*255
# print("mask shape: ", mask.shape)
# cv2.imwrite(mask_recode_path, mask)

# print("################# Merging ###################") 
# # read ori image
# image = cv2.imread(output_image_path)
# print("image shape: ", image.shape)
# height, width, _ = image.shape
# # b,g,r = cv2.split(image)
# # image = cv2.merge([r,g,b])


# mask = cv2.imread(mask_path)
# print("mask shape: ", mask.shape)
# mask = cv2.resize(mask, (width, height))

# # merging
# color = (0,255,0)
# color = np.asarray(color).reshape(1, 1, 3)
# # colored_mask = np.expand_dims(mask, -1).repeat(3, axis=-1)
# colored_mask = np.asarray(mask)
# masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
# heatmap = masked.filled()
# print("heatmap shape: ", heatmap.shape)
# alpha = 0.4  # Changing the alpha will change the opacity of the overlay.
# image_combined = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
# cv2.imwrite(output_path, image_combined)