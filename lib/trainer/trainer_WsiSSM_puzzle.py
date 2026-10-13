import math 
import torch.distributed as dist
from torch import nn

from lib.models.WsiSSM.classification_head_puzzle import global_average_pooling_1d, make_cam
from lib.trainer.trainer import *

class WsiSSMPuzzleTrainer(Trainer):

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
        total_correct = 0
        total_element = 0
        for batch_idx, (data, label, img_coords) in data_iter:
            images, label, img_coords = data.to(self.device), label.to(self.device), img_coords.to(self.device)

            if images.shape[1] >=50000:
                continue

            with torch.cuda.amp.autocast(dtype=torch.float16 if self.fp16 else torch.float32):
                # 1. forward the prediction
                logits, features = self.model(images, img_coords) # (b,2)
                # 2. Adding loss1
                if self.label_mode == "muiticlass":
                    loss = self.criterion(logits.contiguous().view(-1, logits.size(-1)), label.contiguous().view(-1)) 
                elif self.label_mode == "muititask":
                    loss = self.criterion(logits.contiguous().view(-1), label.view(-1))            
                loss /= self.gc

                # 3. backward and optimization only in train
                if self.fp16_scaler is None:
                    loss.backward()            # 后传
                    if (batch_idx + 1) % self.gc == 0:
                        self.optim.step()          # 更新权重
                        self.optim_schedule.step() # 按schedule更新lr 
                        self.optim.zero_grad()     # 梯度归零  
                else:
                    self.fp16_scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.gc == 0:
                        self.fp16_scaler.step(self.optim)
                        self.fp16_scaler.update()
                        self.optim_schedule.step() # 按schedule更新lr 
                        self.optim.zero_grad()                    

            # 3.statistics
            avg_loss += loss.item() * self.gc
            correct = logits.argmax(dim=-1).eq(label).sum().item() # prediction accuracy  计算token的预测准确率
            element = label.nelement()  # tensor (张量)中 元素的个数
            total_correct += correct
            total_element += element    
            if batch_idx % self.verbose_steps == 0: # 每几个batch打印一次
                post_fix = {"epoch": epoch, "iter": batch_idx,
                            "avg_loss": "{:.4f}".format(avg_loss / (batch_idx + 1)),
                            "avg_acc": "{:.4f}%".format(total_correct / total_element * 100),
                            "loss": "{:.4f}".format(loss.item()),
                            "acc": "{:.4f}%".format(correct/element*100), }                
                data_iter.write(str(post_fix))   # 打印

            # 清理中间变量, otherwise out of memory while test
            del data, label, logits
            torch.cuda.empty_cache() if self.with_cuda else None # 清理显存
        
        if self.distributed_mode == "FSDP":
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)

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
            for batch_idx, (data, label, img_coords) in data_iter:
                images, label, img_coords = data.to(self.device), label.to(self.device), img_coords.to(self.device)
                
                with torch.cuda.amp.autocast(dtype=torch.float16 if self.fp16 else torch.float32):
                    # 1. forward the prediction
                    logits, features = self.model(images, img_coords) # (b,2)
                    all_outputs += logits.detach().cpu().tolist() # (n, 2)
                    all_labels +=  label.detach().cpu().tolist() # (n)                      
                    # 2. Adding loss1
                    if self.label_mode == "muiticlass":
                        loss = self.criterion(logits.contiguous().view(-1, logits.size(-1)), label.contiguous().view(-1)) 
                    elif self.label_mode == "muititask":
                        loss = self.criterion(logits.contiguous().view(-1), label.view(-1))            
                    
                # 3.statistics
                avg_loss += loss.item()
                correct = logits.argmax(dim=-1).eq(label).sum().item() # prediction accuracy  计算token的预测准确率
                element = label.nelement()  # tensor (张量)中 元素的个数
                total_correct += correct
                total_element += element    
                if batch_idx % self.verbose_steps == 0: # 每几个batch打印一次
                    post_fix = {"epoch": epoch, "iter": batch_idx,
                                "avg_loss": "{:.4f}".format(avg_loss / (batch_idx + 1)),
                                "avg_acc": "{:.4f}%".format(total_correct / total_element * 100),
                                "loss": "{:.4f}".format(loss.item()),
                                "acc": "{:.4f}%".format(correct/element*100), }                
                    data_iter.write(str(post_fix))   # 打印

                # 清理中间变量, otherwise out of memory while test
                del data, label, logits, features
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

        self.confident_inerval(all_outputs, all_labels, save_output = True, save_path = os.path.join(self.save_directory.replace("results", "pred_results"), f"{self.flag}_interval.npz"))

        return results   



class WsiSSMPuzzleTrainer_stage2(Trainer):

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
        total_correct = 0
        total_element = 0
        for batch_idx, (data, label, img_coords) in data_iter:
            images, label, img_coords = data.to(self.device), label.to(self.device), img_coords.to(self.device)

            if images.shape[1] >=50000:
                continue
            
            with torch.cuda.amp.autocast(dtype=torch.float16 if self.fp16 else torch.float32):
                ###############################################################################
                # Normal
                ###############################################################################
                # 1. forward the prediction
                logits, features = self.model(images, img_coords) # (b,2), (B,2,L)
                # 2. Adding loss1
                if self.label_mode == "muiticlass":
                    loss = self.criterion(logits.contiguous().view(-1, logits.size(-1)), label.contiguous().view(-1)) 
                elif self.label_mode == "muititask":
                    loss = self.criterion(logits.contiguous().view(-1), label.view(-1))            
                loss /= self.gc

                # 3. backward and optimization only in train
                if self.fp16_scaler is None:
                    loss.backward()   # 后传
                else:
                    self.fp16_scaler.scale(loss).backward()

                ###############################################################################
                # Puzzle Module
                ###############################################################################
                # 1. forward the prediction
                tiled_images, tiled_coords, levels  = tile_features(images, img_coords, self.num_pieces)

                tiled_features = []
                for index,(tiled_image, tiled_coord) in enumerate(zip(tiled_images, tiled_coords)): # (B,L/4,2), (L/4,2)
                    if tiled_image.shape[1] != 0:
                        tiled_logit, tiled_feature = self.model(tiled_image, tiled_coord) # (b,2), (B,2,L/4)
                        tiled_features.append(tiled_feature) # [4, (B,2,L/4) ]
                re_features = merge_features(tiled_features, levels) # (B,2,L)
                
                if self.feature_level == 'cam':
                    features =  make_cam(features.detach())  # （B,2,L）reset
                    re_features = make_cam(re_features) # （B,2,L）  
                else:
                    features = features.detach()

                re_logits = global_average_pooling_1d(re_features) # (b,2)

                # 2. Adding loss2
                if self.label_mode == "muiticlass":
                    tiled_loss = self.criterion(re_logits.contiguous().view(-1, re_logits.size(-1)), label.contiguous().view(-1)) 
                elif self.label_mode == "muititask":
                    tiled_loss = self.criterion(re_logits.contiguous().view(-1), label.view(-1))            

                # 2. Adding loss3
                if self.re_loss_option == 'whole':
                    re_loss = self.re_criterion(features, re_features).mean()
                elif self.re_loss_option == 'masking':
                    class_mask = label.unsqueeze(1).unsqueeze(2) # (B,1,1)
                    re_loss = self.re_criterion(features, re_features) * class_mask # # (B,2,L)
                    re_loss = re_loss.mean()
                elif self.re_loss_option == 'selection':
                    for b_index in range(label.shape[0]):
                        class_indices = label[b_index]  # (1)
                        selected_features = features[:,class_indices]  # （B,L）  
                        selected_re_features = re_features[:,class_indices] # （B,L）  
                        re_loss = self.re_criterion(selected_features, selected_re_features).mean()                     
                else:
                    re_loss = torch.zeros(1).to(self.device)

                # puzzle_loss = tiled_loss # ab3
                puzzle_loss = tiled_loss + re_loss
                puzzle_loss /= self.gc

                # 3. backward and optimization only in train
                if self.fp16_scaler is None:
                    puzzle_loss.backward()            # 后传
                    if (batch_idx + 1) % self.gc == 0:
                        self.optim.step()          # 更新权重
                        self.optim_schedule.step() # 按schedule更新lr 
                        self.optim.zero_grad()     # 梯度归零  
                else:
                    self.fp16_scaler.scale(puzzle_loss).backward()
                    if (batch_idx + 1) % self.gc == 0:
                        self.fp16_scaler.step(self.optim)
                        self.fp16_scaler.update()
                        self.optim_schedule.step() # 按schedule更新lr 
                        self.optim.zero_grad()                    

            # 4.statistics
            avg_loss += (loss.item() + puzzle_loss.item()) * self.gc
            correct = logits.argmax(dim=-1).eq(label).sum().item() # prediction accuracy  计算token的预测准确率
            element = label.nelement()  # tensor (张量)中 元素的个数
            total_correct += correct
            total_element += element    
            if batch_idx % self.verbose_steps == 0: # 每几个batch打印一次
                post_fix = {"epoch": epoch, "iter": batch_idx,
                            "avg_loss": "{:.4f}".format(avg_loss / (batch_idx + 1)),
                            "avg_acc": "{:.4f}%".format(total_correct / total_element * 100),
                            "loss": "{:.4f}".format(loss.item()),
                            "puzzle_loss": "{:.4f}".format(puzzle_loss.item()),
                            "acc": "{:.4f}%".format(correct/element*100), }                
                data_iter.write(str(post_fix))   # 打印
            
            # print(f"memory allocated:{torch.cuda.memory_allocated()/1024/1024/1024}GB, {torch.cuda.memory_allocated()/1024/1024}MB")
            # print(f"memory reserved:{torch.cuda.max_memory_reserved()/1024/1024/1024}GB, {torch.cuda.max_memory_reserved()/1024/1024}MB")
            # 清理中间变量, otherwise out of memory while test
            del data, label, logits, features
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
            for batch_idx, (data, label, img_coords) in data_iter:
                images, label, img_coords = data.to(self.device), label.to(self.device), img_coords.to(self.device)

                with torch.cuda.amp.autocast(dtype=torch.float16 if self.fp16 else torch.float32):
                    ###############################################################################
                    # Normal
                    ###############################################################################
                    # 1. forward the prediction
                    logits, features = self.model(images, img_coords) # (b,2)
                    all_outputs += logits.detach().cpu().tolist() # (n, 2)
                    all_labels +=  label.detach().cpu().tolist() # (n)                      
                    # 2. Adding loss1
                    if self.label_mode == "muiticlass":
                        loss = self.criterion(logits.contiguous().view(-1, logits.size(-1)), label.contiguous().view(-1)) 
                    elif self.label_mode == "muititask":
                        loss = self.criterion(logits.contiguous().view(-1), label.view(-1))            
                    
                # 3.statistics
                avg_loss += loss.item()
                correct = logits.argmax(dim=-1).eq(label).sum().item() # prediction accuracy  计算token的预测准确率
                element = label.nelement()  # tensor (张量)中 元素的个数
                total_correct += correct
                total_element += element    
                if batch_idx % self.verbose_steps == 0: # 每几个batch打印一次
                    post_fix = {"epoch": epoch, "iter": batch_idx,
                                "avg_loss": "{:.4f}".format(avg_loss / (batch_idx + 1)),
                                "avg_acc": "{:.4f}%".format(total_correct / total_element * 100),
                                "loss": "{:.4f}".format(loss.item()),
                                "acc": "{:.4f}%".format(correct/element*100), }                
                    data_iter.write(str(post_fix))   # 打印

                # 清理中间变量, otherwise out of memory while test
                del data, label, logits, features
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
        
        self.confident_inerval(all_outputs, all_labels, save_output = True, save_path = os.path.join(self.save_directory.replace("results", "pred_results"), f"{self.flag}_interval.npz"))
        
        return results   




def tile_features(features, coords, num_pieces):
    """
    Current version only support batch_size == 1
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+
    
    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    """
    B,L,D = features.shape 
    features = features.reshape((B*L,D)) 
    coords = coords.reshape((B*L,2)) 
    
    h_min, h_max = min(coords[:,0]), max(coords[:,0])
    w_min, w_max = min(coords[:,1]), max(coords[:,1])

    num_pieces_per_line = int(math.sqrt(num_pieces))
    h_per_patch = math.ceil((h_max-h_min+1) / num_pieces_per_line)
    w_per_patch = math.ceil((w_max-w_min+1) / num_pieces_per_line)

    levels = []
    for i in range(num_pieces):
        levels.append([]) # init
    for index, coord in enumerate(coords):
        h_level = (coord[0]-h_min) // h_per_patch
        w_level = (coord[1]-w_min) // w_per_patch
        level = (h_level) * num_pieces_per_line + (w_level)
        levels[level].append(index)

    # patches = [torch.index_select(features, 0, torch.LongTensor(i)).unsqueeze(0) for i in levels] # [4,(1,L/4,D)]
    # patches_coords = [torch.index_select(coords, 0, torch.LongTensor(i)) for i in levels] # [4,(L/4,2)]
    patches = [features[i].unsqueeze(0) for i in levels]  # [4,(1,L/4,D)]
    patches_coords = [coords[i] for i in levels]  # [4,(L/4,2)]
    return patches, patches_coords, levels



def tile_features_average(features, coords, num_pieces):
    """
    Current version only support batch_size == 1
    +-----+
    |  1  | 
    +-----+
    |  2  | 
    +-----+
    |  3  | 
    +-----+
    |  4  | 
    +-----+
    """
    B,L,D = features.shape 
    features = features.reshape((B*L,D)) 
    coords = coords.reshape((B*L,2)) 

    L_per_patch = math.ceil(L/num_pieces)

    patches = []
    patches_coords = []
    for patch in torch.split(features, L_per_patch, dim=0):
        patches.append(patch)    
    for patches_coord in torch.split(coords, L_per_patch, dim=0):
        patches_coords.append(patches_coord)        

    patches = nn.utils.rnn.pad_sequence(patches, batch_first=True, padding_value=0) # pad
    patches_coords=-nn.utils.rnn.pad_sequence(patches, batch_first=True, padding_value=0) # pad
    return patches, patches_coords



def merge_features(features, levels):
    """
    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+
    """
    features = torch.cat(features, dim=-1) # (B,D,L)
    
    loca=[]  # (L)
    for i in levels:
        loca += i
    reverse_index = [] # (L)
    for i in range(len(loca)):
        reverse_index.append(loca.index(i))

    if len(reverse_index) != features.shape[-1]:
        raise ValueError("shape do not match")
    re_features = features[:,:,reverse_index]
    return re_features





# import torch
# import math

# features = torch.randn(1,20,768)
# coords= torch.tensor([
# [1,1],[1,2],[1,3],[1,4],[1,5],
# [2,1],[2,2],[2,3],[2,4],[2,5],
# [3,1],[3,2],[3,3],[3,4],[3,5],
# [4,1],[4,2],[4,3],[4,4],[4,5],
# ])
# num_pieces = 4

# B,L,D = features.shape 
# features = features.reshape((B*L,D)) 
# coords = coords.reshape((B*L,2)) 

# h_min, h_max = min(coords[:,0]), max(coords[:,0])
# w_min, w_max = min(coords[:,1]), max(coords[:,1])

# num_pieces_per_line = int(math.sqrt(num_pieces))
# h_per_patch = math.ceil((h_max-h_min+1) / num_pieces_per_line)
# w_per_patch = math.ceil((w_max-w_min+1) / num_pieces_per_line)

# levels = []
# for i in range(num_pieces):
#     levels.append([]) # init
# for index, coord in enumerate(coords):
#     h_level = (coord[0]-h_min) // h_per_patch
#     w_level = (coord[1]-w_min) // w_per_patch
#     level = (h_level) * num_pieces_per_line + (w_level)
#     levels[level].append(index)

# print(features)
# print(levels)
# patches = [torch.index_select(features, 0, torch.LongTensor(i)).unsqueeze(0) for i in levels] # [4,(B,L/4,D)]
# patches_coords = [torch.index_select(coords, 0, torch.LongTensor(i)) for i in levels]

# feature = torch.cat(patches, dim=-2) # (B,L,D)
# print(feature)

# loca=[]
# for i in levels:
#     loca += i

# reverse_index = []
# for i in range(len(loca)):
#     reverse_index.append(loca.index(i))
# re_feature = feature[:,reverse_index]

# print(loca)
