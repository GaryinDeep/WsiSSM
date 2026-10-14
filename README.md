WsiSSM: Weakly Supervised Learning with Subset-matching Strategy for Histopathology Whole Slide Images Classification and Segmentation

Official PyTorch implementation of WsiSSM (Sparse-CAM backbone + Subset-Matching).

A weakly supervised framework that unifies classification and segmentation on gigapixel whole-slide images (WSIs) using only slide-level labels. It achieves state-of-the-art segmentation among MIL methods while keeping strong diagnostic performance.

# Environment
`conda env create -f environment.yaml`

If pytorch is not work:
`pip uninstall torch torchvision -y`
`pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118`

Other dependencies (download from ):
`cd ./data`
`pip install  flash_attn-2.6.3+cu118torch2.3cxx11abiFALSE-cp39-cp39-linux_x86_64.whl`
`pip install  torch_cluster-1.6.3+pt23cu118-cp39-cp39-linux_x86_64.whl`
`pip install  torch_scatter-2.1.2+pt23cu118-cp39-cp39-linux_x86_64.whl`
`pip install  torch_sparse-0.6.18+pt23cu118-cp39-cp39-linux_x86_64.whl`
`pip install  torch_spline_conv-1.2.2+pt23cu118-cp39-cp39-linux_x86_64.whl`

# Preprocessing

Download and Place the images (from camelyon dataset) in the xxx\images folder,then:
```
python 01_create_patches_fp.py \
--source ./data/camelyon/camelyon_16_17/images \
--save_dir ./data/camelyon/SEG_dino \
--patch_size 256 \
--seg \
--patch \
--stitch \
--auto_level
```

```
CUDA_VISIBLE_DEVICES=0,1,2 python 02_extract_features_fp.py \
--data_h5_dir ./data/camelyon/SEG_dino \
--data_slide_dir ./data/camelyon/camelyon_16_17/images \
--slide_ext .tif \
--csv_path ./data/camelyon/SEG_dino/process_list_autogen.csv \
--feat_dir ./data/camelyon/FEATURES_dino \
--batch_size 512 \
--model_name dino_v2 \
--json_dir ./data/model/config.json \
--checkpoint_path ./data/model/pytorch_model.bin
```

Split dataset:
```
python 03_create_splits_seq.py \
--label_frac 1.0 \
--seed 1 \
--k 10 \
--task task_1_tumor_vs_normal \
--val_frac 0.1 \
--test_frac 0.1 \
--csv_path ./data/camelyon/dataset_csv/camelyon_16_17.csv \
--split_path ./data/PANDA/dataset_csv/splits \
--patient_strat
```


# Train and evaluate 

Stage1:
```
CUDA_VISIBLE_DEVICES=2 python 04_train_gigapath_puzzle.py \
--data_dir ./data/camelyon/FEATURES_dino \
--csv_path ./data/camelyon/dataset_csv/camelyon_16_17.csv \
--split_dir ./data/camelyon/dataset_csv/splits/camelyon_16_17  \
--results_dir ./data/camelyon/results/gigapath_puzzle_stage1 \
--k 1 \
--task task_1_tumor_vs_normal \
--exp_code task_1_tumor_vs_normal_gigapath_puzzle_stage1 \
--model_arch gigapath_slide_enc12l768d \
--pretrained ./data/model/slide_encoder.pth \
--input_dim 1536 \
--latent_dim 768 \
--max_epochs 20 \
--patience 5 \
--stop_epoch 5 \
--with_cuda \
--save_best_only \
--gc 32 \
--weight_decay 0.05 \
--layer_decay 0.95 \
--bag_loss ce \
--blr 0.002 \
--feat_layer 11 \
--dropout 0.1 \
--global_pool
```

stage2:
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python 04_train_gigapath_puzzle.py \
--data_dir ./data/camelyon/FEATURES_dino \
--csv_path ./data/camelyon/dataset_csv/camelyon_16_17.csv \
--split_dir ./data/camelyon/dataset_csv/splits/camelyon_16_17  \
--results_dir ./data/camelyon/results/gigapath_puzzle_stage2 \
--k 1 \
--task task_1_tumor_vs_normal \
--exp_code task_1_tumor_vs_normal_gigapath_puzzle_stage2 \
--model_arch gigapath_slide_enc12l768d \
--pretrained ./data/model/slide_encoder.pth \
--finetuned ./data/camelyon/results/gigapath_puzzle_stage1/fold_0/checkpoint_best.pt \
--input_dim 1536 \
--latent_dim 768 \
--max_epochs 20 \
--patience 5 \
--stop_epoch 5 \
--with_cuda \
--save_best_only \
--gc 32 \
--weight_decay 0.05 \
--layer_decay 0.95 \
--bag_loss ce \
--blr 0.002 \
--feat_layer 11 \
--dropout 0.1 \
--global_pool \
--stage 2 \
--re_loss_option masking \
--save_best_eval loss
```

# Seg
`python F_visual_cam.py`
`python F_dice.py`
