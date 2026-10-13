import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = [item[0] for item in batch]
    # img = torch.cat([item[0] for item in batch], dim = 0)
	img = nn.utils.rnn.pad_sequence(img, batch_first=True, padding_value=0) # pad
	label = torch.LongTensor([item[1] for item in batch])
	coords = torch.LongTensor(np.vstack([item[2] for item in batch]))
	return [img, label, coords]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)
	
def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, distributed_mode=None):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if distributed_mode == "DDP": # DDP设置
				train_sampler = DistributedSampler(split_dataset,shuffle=True) # DataLoader中的shuffle应该设置为False，因为打乱的任务交给了sampler 
				loader = DataLoader(split_dataset, batch_size=1, shuffle=False, sampler = train_sampler, collate_fn = collate_MIL, **kwargs)	
			elif weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset) #  sampler option is mutually exclusive with shuffle
				loader = DataLoader(split_dataset, batch_size=1, shuffle=False, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, shuffle=False, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, shuffle=False, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )

	return loader