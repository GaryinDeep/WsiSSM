import random
import scipy
import numpy as np 
import pandas as pd
import os
import json
import pickle
import tqdm
import sklearn
from sklearn.metrics import accuracy_score # 添加此句以解决sklearn隐藏错误
import functools
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)



def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def get_layer_id(name, num_layers):
    # ------------------------------------------------------------------------------------------
    # References:
    # BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    # ------------------------------------------------------------------------------------------
    if 'cls_token' in name or 'pos_embed' in name:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('slide_encoder.encoder.layers'):
        return int(name.split('.')[3]) + 1
    else:
        return num_layers

        
def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    # ------------------------------------------------------------------------------------------
    # References:
    # BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    # ------------------------------------------------------------------------------------------
    param_group_names = {}
    param_groups = {}

    # ---- NEW: robust num_layers detection / fallback ----
    if not hasattr(model, "slide_encoder"):
        return [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "weight_decay": weight_decay,
            "lr_scale": 1.0,
        }]

    num_layers = model.slide_encoder.encoder.num_layers + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        if 'mask_token' in n or 'slide_encoder.decoder' in n:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
        
        layer_id = get_layer_id(n, num_layers)

        group_name = n + "_%d_%s" % (layer_id + 1, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


class EarlyStopping:

    def __init__(self, patience:int=20, stop_epoch:int=50, verbose:bool=False, 
                 save_best_eval="acc", distributed_mode="SingleKernal", save_model_flag=True,
                 save_directory=None, ckpt_name='checkpoint.pt', ):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        :param patience: How long to wait after last time validation loss improved.
        :param stop_epoch: Earliest epoch possible for stopping
        :param verbose: If True, prints a message for each validation loss improvement. 
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.save_best_eval = save_best_eval
        self.distributed_mode = distributed_mode
        self.save_model_flag = save_model_flag
        self.model_path = os.path.join(save_directory, ckpt_name)
        os.makedirs(save_directory) if not os.path.exists(save_directory) else None

        # Ensure save_directory exists
        self.max_acc = float('-inf')
        self.min_loss = float('inf')
        self.early_stop = False
        self.counter = 0
            
    def __call__(self, epoch, score, model):
        if self.save_best_eval=="acc" and  score >= self.max_acc:
            if self.verbose:
                print(f'Validation acc increased ({self.max_acc:.6f} --> {score:.6f}).  Saving model ...')
            if self.save_model_flag:
                self.save_checkpoint(model)
            self.max_acc = score
            self.counter = 0  
        elif self.save_best_eval=="loss" and  score <= self.min_loss:    
            if self.verbose:
                print(f'Validation loss decreased ({self.min_loss:.6f} --> {score:.6f}).  Saving model ...')
            if self.save_model_flag:    
                self.save_checkpoint(model)
            self.min_loss = score
            self.counter = 0      
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True

    def save_checkpoint(self, model):# Save the model's state_dict
        if self.distributed_mode == "DDP" or self.distributed_mode == "DataParallel":
            print("Model Saved on:", self.model_path)
            torch.save(model.module.state_dict(), self.model_path)  # ddp后需要保存model.module
        elif self.distributed_mode == "SingleKernal":
            print("Model Saved on:", self.model_path)        
            torch.save(model.state_dict(), self.model_path)
            
        

class Trainer:

    def __init__(self,
                args,
                model,
                train_dataloader: DataLoader,
                valid_dataloader: DataLoader = None,
                test_dataloader: DataLoader = None,
                save_directory: str = None,
                train_writer=None, 
                val_writer=None,
                ):

        distributed_mode_ParamList = ["DDP", "DataParallel", "SingleKernal", "FSDP", "Pipeline","tensor_parallel"]
        save_best_eval_ParamList = ["acc", "loss"]
        label_mode_ParamList = ["muiticlass", "muititask"]
        opt_ParamList = ["adam","adamw", "sgd", ]

        assert args.distributed_mode in distributed_mode_ParamList, 'error value of param distributed_mode:{}'.format(args.distributed_mode)
        assert args.save_best_eval in save_best_eval_ParamList, 'error value of param save_best_eval:{}'.format(args.save_best_eval)
        assert args.label_mode in label_mode_ParamList, 'error value of param label_mode:{}'.format(args.label_mode)
        assert args.opt in  opt_ParamList, 'error value of param opt:{}'.format(args.opt)

        self.args = args
        self.model = model
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader

        self.verbose_steps = args.verbose_steps
        self.logging_filename = args.logging_filename
        self.save_directory = save_directory
        self.train_writer = train_writer
        self.val_writer = val_writer
        self.global_step = 0

        self.bag_weight = args.bag_weight
        self.seed = args.seed
        self.exp_code = args.exp_code
        self.n_classes = args.n_classes

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        # Setup cuda device for training : Distributed GPU training 
        self.distributed_mode = args.distributed_mode
        self.local_rank = args.local_rank
        self.with_cuda = args.with_cuda
        # DDP 
        if self.distributed_mode == "DDP": 
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.model = self.model.to(self.device) # 存储到GPU
            # self.model = DDP(self.model, device_ids=[self.local_rank]) 
            self.model = DDP(self.model, device_ids=[self.local_rank],output_device=[self.local_rank], find_unused_parameters=True) # 存在冗余参数时
        # DataParallel
        elif self.distributed_mode == "DataParallel": 
            self.device = torch.device("cuda:0") # 此处设置主卡
            self.model = self.model.to(self.device) # 存储到GPU
            num_of_gpus = torch.cuda.device_count()
            if self.with_cuda and num_of_gpus > 1:
                print("Using %d GPUS for mamba" % num_of_gpus)
                self.model = nn.DataParallel(self.model, device_ids=list(range(num_of_gpus))) # 指定哪些卡用于训练 必须包含主卡(self.device)
        # single GPU
        elif self.distributed_mode == "SingleKernal":
            print("Using GPUS?",  torch.cuda.is_available())
            self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.with_cuda) else "cpu")
            self.model = self.model.to(self.device) # 存储到GPU
        # FSDP
        elif self.distributed_mode == "FSDP":
            torch.cuda.set_device(args.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.model = self.model.to(self.device) # 存储到GPU
            my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e5)
            self.model = FSDP(self.model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True), sharding_strategy=ShardingStrategy.FULL_SHARD)
        # pipeline
        elif self.distributed_mode == "Pipeline":
            1
        # tensor_parallel
        elif self.distributed_mode == "tensor_parallel":
            import tensor_parallel as tp
            self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.with_cuda) else "cpu")
            self.model = tp.tensor_parallel(self.model, [f"cuda:{i}" for i in range(torch.cuda.device_count())]) # 存储到GPU       

        # DDP存储模型时通过local_rank限制单进程写入,只需保存设备0中的模型:
        self.save_model_flag = True 
        if self.distributed_mode == "DDP" and self.local_rank != 0:    
            self.save_model_flag = False

        # Using cross entropy Loss function
        print('\nInit loss function...', end=' ')
        self.label_mode = args.label_mode
        if self.label_mode == "muiticlass":
            if args.bag_loss == 'svm':
                from topk.svm import SmoothTop1SVM
                self.criterion = SmoothTop1SVM(n_classes = args.n_classes)
                if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == 'cuda':
                    self.criterion = self.criterion.cuda()  
            else:              
                self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        elif self.label_mode == "muititask":
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.KL = nn.KLDivLoss(reduction="batchmean")
        print('Done!')

        # Optimizer and Scheduler
        print('\nInit optimizer ...', end=' ')
        if args.opt == "adam":
            self.optim = optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            self.optim = optim.SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        elif args.opt == 'adamw':
            param_groups = param_groups_lrd(self.model, args.weight_decay,layer_decay=args.layer_decay)
            self.optim = optim.AdamW(param_groups, lr=args.lr)
            # self.optim = optim.AdamW(params=self.model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
        print('\nInit Scheduler ...', end=' ')
        self.optim_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
        print('Done!')

        # training mode (save each rpoch / save best)
        self.save_best_only = args.save_best_only
        self.save_best_eval = args.save_best_eval
        self.early_stop = EarlyStopping(patience=args.patience, stop_epoch=args.stop_epoch, verbose=True, 
                                    save_best_eval=self.save_best_eval, distributed_mode=self.distributed_mode, save_model_flag=self.save_model_flag,
                                    save_directory=self.save_directory, ckpt_name='checkpoint_best.pt')
        if self.save_best_only:
            print("Using early stop for training")

        print("trainer initialization completed")

        self.gc = args.gc
        # set up the fp16 scaler
        self.fp16 = args.fp16
        self.fp16_scaler = torch.cuda.amp.GradScaler()
        if args.fp16:
            print('Using fp16 training')        
    
        if args.num_pieces:
            self.num_pieces = args.num_pieces 
        if args.feature_level:
            self.feature_level = args.feature_level
        if args.re_loss:     
            from lib.models.WsiSSM.classification_head_puzzle import L1_Loss, L2_Loss
            if args.re_loss == 'L1_Loss':
                self.re_criterion = L1_Loss
            else:
                self.re_criterion = L2_Loss   
        if args.re_loss_option:
            self.re_loss_option = args.re_loss_option


    def train_model(self, epochs, result_path=None, k=0):
        print_network(self.model) # print model
        print("Training Start")  # 删除多余的model文件夹
        self.flag = k
        for epoch in range(epochs):
            # train
            self.model.train()
            self.train_iteration(epoch, self.train_data)  # train loop
            if not self.save_best_only:# save each epoch  
                self.early_stop.ckpt_name = 'checkpoint_epoch_{}.pt'.format(epoch)
                self.early_stop.save_checkpoint(self.model)   
            # valid
            if self.valid_data is not None: 
                self.model.eval()
                results = self.eval_iteration(epoch, self.valid_data, str_code="valid") # valid loop
                if self.save_best_only: # save best only 
                    if self.save_best_eval=="acc":
                        self.early_stop(epoch, results["accuracy"], self.model)
                    elif self.save_best_eval=="loss":
                        self.early_stop(epoch, results["loss"], self.model)
            # early stop
            if self.early_stop.early_stop:
                print("Early stopping")
                break
        # test     
        if self.test_data is not None: 
            # load model   
            if self.save_best_only:
                self.model.load_state_dict(torch.load(self.early_stop.model_path, weights_only=False))
                self.model = self.model.to(self.device)   
            # predict
            self.model.eval()
            results = self.eval_iteration(epoch, self.test_data,str_code="test")
            # save result
            accuracy, P, R, F1score, AUROC, AUPRC = results["accuracy"], results["P"], results["R"], results["F1score"], results["AUROC"], results["AUPRC"]
            print("\n accuracy:{} | P:{} | R:{} | F1score:{} | AUROC:{} | AUPRC:{} |".format(accuracy, P, R, F1score, AUROC, AUPRC))
            sensitivity, specificity = results["sensitivity"], results["specificity"]
            print("\n sensitivity:{} | specificity:{} ".format(sensitivity, specificity))
            if result_path: # save test result
                result_xlsx = pd.read_excel(result_path)
                result_xlsx.loc[len(result_xlsx)] = [self.exp_code, k, accuracy, P, R, F1score, AUROC, AUPRC, sensitivity, specificity]     
                result_xlsx.to_excel(result_path, index = False)


    def eval_model(self, result_path=None, k=0):
        if self.test_data is not None: 
            self.flag = k
            self.model.load_state_dict(torch.load(self.early_stop.model_path, weights_only=False))
            print("load finetuned model")
            
            self.model = self.model.to(self.device) 
            # predict
            self.model.eval()
            results = self.eval_iteration(0, self.test_data,str_code="test")   
            accuracy, P, R, F1score, AUROC, AUPRC = results["accuracy"], results["P"], results["R"], results["F1score"], results["AUROC"], results["AUPRC"]
            sensitivity, specificity = results["sensitivity"], results["specificity"]
            # print("\n accuracy:{} | P:{} | R:{} | F1score:{} | AUROC:{} | AUPRC:{} |".format(accuracy, P, R, F1score, AUROC, AUPRC))
            # print("\n sensitivity:{} | specificity:{} ".format(sensitivity, specificity))
            if result_path: # save test result
                result_xlsx = pd.read_excel(result_path)
                result_xlsx.loc[len(result_xlsx)] = [self.exp_code, k, accuracy, P, R, F1score, AUROC, AUPRC, sensitivity, specificity]     
                result_xlsx.to_excel(result_path, index = False)                     
        else:
            raise NotImplementedError    


    def train_iteration(self, epoch, data_loader):
        """
        loop over the data_loader for training,backward operation is activated 
        
        :param epoch: current epoch index
        :para += corrtskm data_loader: torch.utils.data.DataLoader for iteration
        :return: None
        """  
        # Setting the tqdm progress bar
        str_code = "train"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc="EP_%s:%d" % (str_code, epoch),
                            total=len(data_loader),
                            bar_format="{l_bar}{r_bar}")        

        avg_loss = 0.0
        avg_inst_loss = 0.0
        total_correct = 0
        total_element = 0
        for batch_idx, (data, label) in data_iter:

            data, label = data.to(self.device), label.to(self.device)
            # 1. forward the prediction
            logits, Y_prob, Y_hat, _, instance_dict = self.model(data, label=label, instance_eval=True)
            # 2. Adding loss1
            if self.label_mode == "muiticlass":
                loss = self.criterion(logits.contiguous().view(-1, logits.size(-1)), label.contiguous().view(-1)) 
            elif self.label_mode == "muititask":
                loss = self.criterion(logits.contiguous().view(-1), label.view(-1))            
            # loss2
            instance_loss = instance_dict['instance_loss']
            # loss
            total_loss = self.bag_weight * loss + (1-self.bag_weight) * instance_loss 

            # 3. backward and optimization only in train
            self.optim.zero_grad()     # 梯度归零  
            total_loss.backward()            # 后传
            self.optim.step()          # 更新权重
            self.optim_schedule.step() # 按schedule更新lr

            # 3.statistics
            avg_loss += loss.item()
            avg_inst_loss += instance_loss.item() 
            correct = Y_hat.float().eq(label.float()).float().sum().item() # prediction accuracy  计算token的预测准确率
            element = label.nelement()  # tensor (张量)中 元素的个数
            total_correct += correct
            total_element += element    
            if batch_idx % self.verbose_steps == 0: # 每几个batch打印一次
                post_fix = {"epoch": epoch, "iter": batch_idx,
                            "avg_loss": "{:.4f}".format(avg_loss / (batch_idx + 1)),
                            "avg_inst_loss": "{:.4f}".format(avg_inst_loss / (batch_idx + 1)),
                            "avg_acc": "{:.4f}%".format(total_correct / total_element * 100),
                            "loss": "{:.4f}".format(loss.item()),
                            "instance_loss": "{:.4f}".format(instance_loss.item()),
                            "acc": "{:.4f}%".format(correct/element*100), }                
                data_iter.write(str(post_fix))   # 打印

            # 清理中间变量, otherwise out of memory while test
            del data, label
            del logits, Y_prob, total_loss
            torch.cuda.empty_cache() if self.with_cuda else None # 清理显存
            
        # 打印日志
        print("EP{}_{} | avg_loss={:.4f} | avg_acc={:.4f}".format(epoch,
                    str_code,avg_loss / len(data_iter),total_correct * 100.0 / total_element) )
        # # 保存记录日志
        if not os.path.exists(self.save_directory): 
            os.makedirs(self.save_directory) # Ensure save_directory exists
        if self.save_model_flag:  
            log_txt_formatter = "Train [Epoch] {epoch_str:03d} [Loss] {loss_str} [Acc] {Acc_str}%\n"
            with open(os.path.join(self.save_directory, self.logging_filename), "a") as f: # 追加模式
                to_write = log_txt_formatter.format(epoch_str = epoch,
                                                    loss_str=avg_loss / len(data_iter),
                                                    Acc_str = total_correct * 100.0 / total_element )
                f.write(to_write) # 自动关闭              
    

    def eval_iteration(self, epoch, data_loader, str_code=None):
        """
        loop over the data_loader for testing,backward operation is not activated 
        
        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :return: None
        """  
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc="EP_%s:%d" % (str_code, epoch),
                            total=len(data_loader),
                            bar_format="{l_bar}{r_bar}")        

        avg_loss = 0.0
        avg_inst_loss = 0.0
        total_correct = 0
        total_element = 0
        all_outputs = []    
        all_labels= [] 
        with torch.no_grad():
            for batch_idx, (data, label) in data_iter:

                data, label = data.to(self.device), label.to(self.device)
                # 1. forward the prediction
                logits, Y_prob, Y_hat, _, instance_dict = self.model(data, label=label, instance_eval=True)
                all_outputs += logits.detach().cpu().tolist() # (n, 2)
                all_labels +=  label.detach().cpu().tolist() # (n)

                # 2. Adding loss1
                if self.label_mode == "muiticlass":
                    loss = self.criterion(logits.contiguous().view(-1, logits.size(-1)), label.contiguous().view(-1)) 
                elif self.label_mode == "muititask":
                    loss = self.criterion(logits.contiguous().view(-1), label.view(-1))            
                # loss2
                instance_loss = instance_dict['instance_loss']
                # loss
                total_loss = self.bag_weight * loss + (1-self.bag_weight) * instance_loss 

                # 3.statistics
                avg_loss += loss.item()
                avg_inst_loss += instance_loss.item() 
                correct = Y_hat.float().eq(label.float()).float().sum().item() # prediction accuracy  计算token的预测准确率
                element = label.nelement()  # tensor (张量)中 元素的个数
                total_correct += correct
                total_element += element               
                if batch_idx % self.verbose_steps == 0: # 每几个batch打印一次
                    post_fix = {"epoch": epoch, "iter": batch_idx,
                                "avg_loss": "{:.4f}".format(avg_loss / (batch_idx + 1)),
                                "avg_inst_loss": "{:.4f}".format(avg_inst_loss / (batch_idx + 1)),
                                "avg_acc": "{:.4f}%".format(total_correct / total_element * 100),
                                "loss": "{:.4f}".format(loss.item()),
                                "instance_loss": "{:.4f}".format(instance_loss.item()),
                                "acc": "{:.4f}%".format(correct/element*100), }                
                    data_iter.write(str(post_fix))   # 打印

                # 清理中间变量, otherwise out of memory while test
                del data, label
                del logits, Y_prob, total_loss
                torch.cuda.empty_cache() if self.with_cuda else None # 清理显存

        # 打印日志
        print("EP{}_{} | avg_loss={:.4f} | avg_acc={:.4f}".format(epoch,
                    str_code,avg_loss / len(data_iter),total_correct * 100.0 / total_element) )
        # 保存记录日志
        if self.save_model_flag and str_code=="valid":  
            log_txt_formatter = "Valid [Epoch] {epoch_str:03d} [Loss] {loss_str} [Acc] {Acc_str}%\n"
            with open(os.path.join(self.save_directory, self.logging_filename), "a") as f: # 追加模式
                to_write = log_txt_formatter.format(epoch_str = epoch,
                                                    loss_str=avg_loss / len(data_iter),
                                                    Acc_str = total_correct * 100.0 / total_element )
                f.write(to_write) # 自动关闭  
       
        results = self.calculate_scores(all_outputs, all_labels, save_output = True, save_path = os.path.join(self.save_directory.replace("results", "pred_results"), f"{self.flag}.npz"))
        results["loss"] = avg_loss / len(data_iter)  

        # self.confident_inerval(all_outputs, all_labels, save_output = True, save_path = os.path.join(self.save_directory.replace("results", "pred_results"), f"{self.flag}_interval.npz"))

        return results      


    def calculate_scores(self, all_outputs, all_labels, autothresh = False, save_output = False, save_path = './data/data.npz', print_flag= True, act = "softmax"):
        """
        :param label_mode: 选择标签的模式 
        :param pos_label: 选择画出哪个类
        """
        # save_output = True
        
        if self.label_mode == "muiticlass":

            all_labels = np.array(all_labels) 
            all_outputs = np.array(all_outputs) 
            all_pred_int = np.argmax(all_outputs, axis=-1)
            if save_output:
                if not os.path.exists(save_path):
                    os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)
                np.savez(save_path, array1=all_outputs, array2=all_labels)
                
            if all_outputs.shape[1] <= 2:  #二分类
                pos_label=1
                if autothresh:
                    all_pred_float = F.softmax(torch.tensor(all_outputs), dim=-1)[:,pos_label].numpy()
                    False_Positive_Rates, True_Positive_Rates, thresholds = sklearn.metrics.roc_curve(all_labels, all_pred_float, pos_label=pos_label) 
                    maxindex = (True_Positive_Rates-False_Positive_Rates).tolist().index(max(True_Positive_Rates-False_Positive_Rates))
                    threshold = thresholds[maxindex]
                    all_pred_int = np.zeros(all_pred_float.shape)
                    all_pred_int[all_pred_float>=threshold] =1

                accuracy = sklearn.metrics.accuracy_score(all_labels, all_pred_int)    # 准确率 
                P = sklearn.metrics.precision_score(all_labels, all_pred_int)
                R = sklearn.metrics.recall_score(all_labels, all_pred_int)
                F1score = sklearn.metrics.f1_score(all_labels, all_pred_int)
                AUROC = sklearn.metrics.roc_auc_score(all_labels, all_outputs[:,pos_label]) #logit也行
                AUPRC = (sklearn.metrics.average_precision_score(all_labels, all_outputs[:,pos_label])) 
                sensitivity = sklearn.metrics.recall_score(all_labels, all_pred_int, pos_label=1) # recall
                specificity = sklearn.metrics.recall_score(all_labels, all_pred_int, pos_label=0)
                # False_Positive_Rates, True_Positive_Rates, thresholds = sklearn.metrics.roc_curve(all_labels, all_outputs[:,pos_label], pos_label=pos_label) 
                # Precision_Rate, Recall_Rate, thresholds = sklearn.metrics.precision_recall_curve(all_labels, all_outputs[:,pos_label], pos_label=pos_label)

            else:  # 多分类
                            
                def specificity_macro(y_true, y_pred, labels=None):
                    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels)
                    n_classes = cm.shape[0]
                    specs = []
                    per_class_specs = []
                    for i in range(n_classes):
                        TP = cm[i, i]
                        FN = cm[i, :].sum() - TP
                        FP = cm[:, i].sum() - TP
                        TN = cm.sum() - (TP + FN + FP)
                        spec_i = TN / (TN + FP) if (TN + FP) > 0 else 0.0
                        per_class_specs.append(spec_i)
                        specs.append(spec_i)
                    return np.mean(specs), np.array(per_class_specs)    
                
                accuracy = sklearn.metrics.accuracy_score(all_labels, all_pred_int)    # 准确率 
                P = sklearn.metrics.precision_score(all_labels, all_pred_int,  average='macro')
                R = sklearn.metrics.recall_score(all_labels, all_pred_int,  average='macro')
                F1score = sklearn.metrics.f1_score(all_labels, all_pred_int,  average='macro')
                AUROC = sklearn.metrics.roc_auc_score(all_labels, F.softmax(torch.tensor(all_outputs), dim=-1), multi_class='ovr')
                AUPRC = sklearn.metrics.average_precision_score(all_labels, F.softmax(torch.tensor(all_outputs), dim=-1), average="macro")
                sensitivity = sklearn.metrics.recall_score(all_labels, all_pred_int, average="macro")
                specificity, per_class_spec = specificity_macro(all_labels, all_pred_int)
                # for i in range(self.n_classes):
                #     False_Positive_Rates, True_Positive_Rates, thresholds = sklearn.metrics.roc_curve(all_labels, F.softmax(torch.tensor(all_outputs), dim=-1)[:, i], pos_label=i) 
                #     Precision_Rate, Recall_Rate, thresholds = sklearn.metrics.precision_recall_curve(all_labels, F.softmax(torch.tensor(all_outputs), dim=-1)[:, i], pos_label=i)
            
            if print_flag:
                print(f"\n accuracy:{accuracy} | P:{P} | R:{R} | F1score:{F1score} | AUROC:{AUROC} | AUPRC:{AUPRC} |")
                print(f"\n sensitivity:{sensitivity} | specificity:{specificity} \n")
            results = {"accuracy":accuracy, "P":P, "R":R, "F1score":F1score, "AUROC":AUROC,"AUPRC":AUPRC,"sensitivity":sensitivity, "specificity":specificity}
            return results

        elif self.label_mode == "muititask": # 多任务
            all_pred_int = np.where(all_outputs>0,1,0)


    def confident_inerval(self, all_outputs, all_labels, autothresh = True, save_output = False, save_path = './data/data.npz'):
        
        # np.random.seed(6) # camelyon
        np.random.seed(self.seed) # PANDA

        all_outputs = np.array(all_outputs) 
        all_labels = np.array(all_labels)

        a = 0.05 # 置信度
        n = 10 # 样本量
        sample_acc_list,sample_P_list,sample_R_list, sample_F1_list  = [],[],[],[]
        sample_AUROC_list, sample_AUPRC_list  = [], []
        sample_sensitivity_list, sample_specificity_list = [],[]
        all_sample_outputs, all_sample_labels= [], []
        while( len(sample_acc_list)<n ):
            sample_index = list(np.random.choice(range(len(all_outputs)),size=len(all_outputs), replace=True)) # bootstrap放回采样
            sample_outputs = [all_outputs[i] for i in sample_index]
            sample_labels = [all_labels[i] for i in sample_index]

            if np.unique(sample_labels).size != np.unique(all_labels).size:   # 只有一个类别
                continue

            results = self.calculate_scores(sample_outputs, sample_labels, autothresh = autothresh, print_flag= False)
            accuracy, P, R, F1score = results["accuracy"], results["P"], results["R"], results["F1score"]
            AUROC, AUPRC = results["AUROC"], results["AUPRC"]
            sensitivity, specificity = results["sensitivity"], results["specificity"]

            sample_acc_list.append(accuracy)
            sample_P_list.append(P)
            sample_R_list.append(R)
            sample_F1_list.append(F1score)
            sample_AUROC_list.append(AUROC)
            sample_AUPRC_list.append(AUPRC)
            sample_sensitivity_list.append(sensitivity)
            sample_specificity_list.append(specificity)

            all_sample_outputs.append(sample_outputs)
            all_sample_labels.append(sample_labels)
                    
        if save_output:
            if not os.path.exists(save_path):
                os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)
            np.savez(save_path, array1=np.array(all_sample_outputs), array2=np.array(all_sample_labels))

        # 均值 
        accuracy_mean= np.mean(sample_acc_list)
        P_mean= np.mean(sample_P_list)
        R_mean= np.mean(sample_R_list)
        F1_mean =  np.mean(sample_F1_list)
        AUROC_mean= np.mean(sample_AUROC_list)
        AUPRC_mean= np.mean(sample_AUPRC_list)
        sensitivity_mean= np.mean(sample_sensitivity_list)
        specificity_mean= np.mean(sample_specificity_list)

        # 标准误差
        accuracy_se= scipy.stats.sem(sample_acc_list)
        P_se=scipy.stats.sem(sample_P_list)
        R_se= scipy.stats.sem(sample_R_list)
        F1_se = scipy.stats.sem(sample_F1_list)
        AUROC_se = scipy.stats.sem(sample_AUROC_list)
        AUPRC_se = scipy.stats.sem(sample_AUPRC_list)
        sensitivity_se= scipy.stats.sem(sample_sensitivity_list)
        specificity_se=scipy.stats.sem(sample_specificity_list)

        # t分布临界值
        t_ = scipy.stats.t.ppf(q=a/2, df=n-1) 
        # 上下限
        accuracy_interval = t_*accuracy_se
        P_interval = t_*P_se
        R_interval = t_*R_se
        F1_interval = t_*F1_se
        AUROC_interval = t_*AUROC_se
        AUPRC_interval = t_*AUPRC_se
        sensitivity_interval = t_*sensitivity_se
        specificity_interval = t_*specificity_se

        # 打印结果
        print(f'accuracy_mean: {round(accuracy_mean, 4)}+{round(accuracy_interval, 4)}\nP: {round(P_mean, 4)}+{round(P_interval, 4)}\nR:{round(R_mean, 4)}+{round(R_interval, 4)}\nF1:{round(F1_mean, 4)}+{round(F1_interval, 4)}')
        print(f'AUROC: {round(AUROC_mean, 4)}+{round(AUROC_interval, 4)}\nAUPRC:{round(AUPRC_mean, 4)}+{round(AUPRC_interval, 4)}')
        print(f'sensitivity: {round(sensitivity_mean, 4)}+{round(sensitivity_interval, 4)}\nspecificity: {round(specificity_mean, 4)}+{round(specificity_interval, 4)}')

