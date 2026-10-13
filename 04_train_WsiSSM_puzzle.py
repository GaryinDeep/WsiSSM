from lib.dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_pkl, load_pkl
from lib.dataset_modules.data_loader_WsiSSM import get_simple_loader, get_split_loader
from lib.models.WsiSSM.classification_head_puzzle import get_model
from lib.trainer.trainer_WsiSSM_puzzle import WsiSSMPuzzleTrainer, WsiSSMPuzzleTrainer_stage2

import argparse
import os
import pandas as pd
import numpy as np
import re

import torch
from torch.utils.tensorboard import SummaryWriter

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":

    # Generic training settings
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--data_dir', type=str, default=None, help='data directory')
    parser.add_argument('--csv_path', type=str, default= "dataset_csv/data.csv", help='data description csv')
    parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use, ' 
                            +'instead of infering from the task and label_frac argument (default: None)')    
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    # dataset 
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=None, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=None, help='end fold (default: -1, first fold)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)') 
    parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
    parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
    # trainer
    parser.add_argument('--distributed_mode', type=str, choices = ['DDP', 'DataParallel', 'SingleKernal', 'FSDP', "Pipeline","tensor_parallel"], default='SingleKernal')
    parser.add_argument('--with_cuda', action='store_true', default=False, help='with GPU?')
    parser.add_argument("--local-rank", type=int, default= -1, help="number of cpu threads to use during batch generation")
    parser.add_argument('--max_epochs', type=int, default=200,help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--T_0', type=int, default=1000, help='T_0 of CosineAnnealing schedule') 
    parser.add_argument('--T_mult', type=int, default=1, help='T_mult of CosineAnnealing schedule') 
    parser.add_argument('--eta_min', type=float, default=1e-5, help='eta_min of CosineAnnealing schedule') 
    parser.add_argument("--verbose-steps", type=int, default=50 , help="每个epoch隔几个step打印一次")  
    parser.add_argument("--logging_filename", type=str, default="train_log.txt", help="log file")  
    parser.add_argument("--save_best_only", action='store_true', default=False, help="")  
    parser.add_argument("--save_best_eval", type=str, choices = ['acc', 'loss'], default='acc')  
    parser.add_argument("--label_mode", type=str, choices = ['muiticlass', 'muititask'], default='muiticlass')  
    parser.add_argument('--batch_size',     type=int, default=1, help='Current version only supports batch size of 1')
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                        help='slide-level classification loss function (default: ce)')   
    parser.add_argument('--opt', type=str, choices = ['adam', 'sgd', 'adamw'], default='adamw')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='betas of optimizer') 
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay of optimizer')
    parser.add_argument('--layer_decay', type=float, default=0.95, help='Layer-wise learning rate decay')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--blr', type=float, default=4e-3, help='Base learning rate, will caculate the learning rate based on batch size')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')                        
    parser.add_argument('--fp16', action='store_true', default=True, help='Fp16 training')
    parser.add_argument('--gc', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stop')
    parser.add_argument('--stop_epoch', type=int, default=50, help='min srop epoch for early stop')
    # model  
    parser.add_argument('--model_arch', type=str, default='longnet_enc12l768d')
    parser.add_argument('--input_dim', type=int, default=1536, help='Dimension of input tile embeddings')
    parser.add_argument('--latent_dim', type=int, default=768, help='Hidden dimension of the slide encoder')
    parser.add_argument('--feat_layer', type=str, default='11', 
                        help='The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers')
    parser.add_argument('--pretrained', type=str, default=None, help='Pretrained WsiSSM slide encoder')
    parser.add_argument('--freeze', action='store_true', default=False, help='Freeze pretrained model')
    parser.add_argument('--global_pool', action='store_true', default=False, help='Use global pooling, will use [CLS] token if False')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')                       
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='Drop path rate')
    # puzzle sepcial
    parser.add_argument('--bag_weight', type=float, default=0.7,help='weight coefficient for bag-level loss (default: 0.7)')
    parser.add_argument('--finetuned', type=str, default='', help='finetuned WsiSSM slide encoder')
    parser.add_argument('--stage', type=int, default=1,help='training stage')
    parser.add_argument('--num_pieces', type=int, default=4, help='puzzle pieces number')
    parser.add_argument("--feature_level", type=str, choices = ['ori', 'cam'], default='ori', help='feature level of tiled_loss') 
    parser.add_argument("--re_loss", type=str, choices = ['L1_Loss', 'L2_Loss'], default='L1_Loss', help='re_loss function') 
    parser.add_argument("--re_loss_option", type=str, choices = ['masking', 'selection',"whole", "None"], default='whole', help='mode of re_loss') 
    
    args = parser.parse_args()
    
    print("################# Settings ###################")
    # set the learning rate
    eff_batch_size = args.batch_size * args.gc
    if args.lr is None or args.lr < 0:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    settings = {'data_dir' : args.data_dir,
        'csv_path':args.csv_path,
        'split_dir' : args.split_dir,
        'results_dir': args.results_dir,
        'testing':args.testing ,
        'weighted_sample': args.weighted_sample,
        'num_splits': args.k, 
        'k_start': args.k_start,
        'k_end': args.k_end,
        'seed': args.seed,
        'task': args.task,
        'experiment': args.exp_code,
        'max_epochs': args.max_epochs, 
        'opt': args.opt,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'model_arch': args.model_arch ,
        'input_dim': args.input_dim ,
        'latent_dim': args.latent_dim ,
        'feat_layer': args.feat_layer ,
        'pretrained': args.pretrained ,
        'freeze': args.freeze ,
        'global_pool': args.global_pool ,
        'dropout': args.dropout ,
        'drop_path_rate': args.drop_path_rate ,
        'num_pieces': args.num_pieces
        }
    for key, val in settings.items():
        print("{}:  {}".format(key, val))  
    if not os.path.isdir(args.results_dir): # save setting
        os.mkdir(args.results_dir)
    with open(os.path.join(args.results_dir, 'experiment_{}.txt'.format(args.exp_code)), 'w') as f: 
        print(settings, file=f) 
    f.close()

    print("################# Load Dataset ###################")
    if args.task == 'task_1_tumor_vs_normal':
        label_dict = {'normal_tissue':0, 'tumor_tissue':1}
    elif args.task == 'task_2_tumor_subtyping':
        label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2}  
    else:
        raise NotImplementedError
    args.n_classes=len(label_dict)
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = label_dict,
                            patient_strat=False,
                            ignore=[])
    dataset.load_from_h5(True)  # load from .h5 (include coords)

    print("################# Training ###################")
    # reset result output
    result_path = os.path.join(args.results_dir, "result.xlsx")
    result_xlsx = pd.DataFrame(columns=["task", "k", "accuracy", "P", "R", "F1score", "AUROC", "AUPRC", "sensitivity", "specificity"])
    result_xlsx.to_excel(result_path, index = False)
    # DDP设置
    if args.distributed_mode == "DDP": 
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group('nccl', init_method='env://') # initialize the process group
        os.environ['RANK'] = str(0) # 设置环境 必须放在这三条语句最后，否则死锁
    if args.distributed_mode == "Pipeline ": 
        1

      
    # running k fold
    start,end = 0, args.k
    if args.k_start:
        start = args.k_start  
    if args.k_end:
        end = args.k_end
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    for k in np.arange(start, end):     
        print("\nLoading train、val、test Dataset")
        assert os.path.isdir(args.split_dir) # check split dir
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                                csv_path='{}/splits_{}.csv'.format(args.split_dir, k)) # load k-fold dataset
        print("\nCreating Dataloader") 
        seed_torch(args.seed)  # set random seed
        train_loader = get_split_loader(train_dataset, training=True,testing = args.testing,
                                     weighted = args.weighted_sample, distributed_mode=args.distributed_mode) # batch size=1
        val_loader = get_split_loader(val_dataset,  testing = args.testing)
        test_loader = get_split_loader(test_dataset, testing = args.testing)

        print("\nBuilding model")
        model = get_model(**vars(args))
        if args.stage != 1:   # load stage1 model
            args.finetuned = re.sub(r"fold_\d+", f"fold_{k}",args.finetuned)
            if os.path.exists(args.finetuned):
                missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.finetuned, weights_only=False))
                if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                        print("Missing or Unexpected module")
                else:
                    print("\033[92m Successfully Loaded Finetuned WsiSSM model from {} \033[00m".format(args.finetuned))
            elif "ab" not in args.exp_code:
                raise NotImplementedError
     
        print("\nCreating Trainer") 
        if args.stage == 1:   # stage1
            trainer = WsiSSMPuzzleTrainer(args,
                            model,
                            train_dataloader=train_loader,
                            valid_dataloader=val_loader, 
                            test_dataloader = test_loader,
                            save_directory = os.path.join(args.results_dir, "fold_{}".format(k)),
                            )
        else:
            trainer = WsiSSMPuzzleTrainer_stage2(args,
                            model,
                            train_dataloader=train_loader,
                            valid_dataloader=val_loader, 
                            test_dataloader = test_loader,
                            save_directory = os.path.join(args.results_dir, "fold_{}".format(k)),
                            )
      
        trainer.train_model(args.max_epochs, result_path, k)
        trainer.eval_model(result_path, k)

        # # another eval set
        # print("\nLoading external test Dataset") 
        # _, _, test_dataset = dataset.return_splits(from_id=False, 
        #                         csv_path='{}/splits_external.csv'.format(args.split_dir)) # load k-fold dataset
        # print("\nCreating Dataloader") 
        # test_loader = get_split_loader(test_dataset, testing = args.testing)  
        # trainer.test_data = test_loader
        # trainer.eval_model(result_path, "external")        



        




