import torch
from pathlib import Path
import os
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')
ALPHABET = ["-", "AA", "AE", "AH","AO","AW", "AY", "EH", "ER","EY","IH", "IY","OW","OY", "UH",\
            "UW", "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG",\
            "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH", " "]

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]
ReturnType = Tuple[DataLoader, DataLoader, DataLoader, Dict, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]

#**Maxwell Kounga**

import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import timeit
import itertools
from scipy.io import loadmat
from array import array

class BCIDataset(Dataset):
  def __init__(self, path):
    neural_data, sentence_text, neural_padding, sentence_padding, day = [], [], [], [], []

    # DATALOADING A FOLDER
    file_list = os.listdir(path)
    
    for day_idx, filename in enumerate(file_list):
      print(filename)
      full_name = path + filename
      xy = loadmat(full_name, squeeze_me=True)
      neural_data.append(np.array(xy['neuralData']))
      sentence_text.append(np.array(xy['sentenceText']))
      neural_padding.append(np.array(xy['neuralPadding']))
      sentence_padding.append(np.array(xy['sentencePadding']))
      day = np.concatenate((day, (day_idx * np.ones((xy['neuralData'].shape[0],)))), axis=0)

    neural_data = np.vstack(neural_data)
    sentence_text = np.vstack(sentence_text)
    neural_padding = np.vstack(neural_padding)
    sentence_padding = np.vstack(sentence_padding)

    # Number of Samples
    self.n_samples = sentence_text.shape[0]

    # Create the Neural and Sentence Data and Padding
    self.neural_data = torch.from_numpy(neural_data)
    self.sentence_text = torch.from_numpy(sentence_text)
    self.neural_padding = torch.from_numpy(neural_padding)
    self.sentence_padding = torch.from_numpy(sentence_padding)
    self.day = torch.from_numpy(day.astype(int))

  # Output neural data, sentences, and the padding as auxillary data
  def __getitem__(self, index):
    return self.neural_data[index], self.sentence_text[index], self.neural_padding[index], self.sentence_padding[index], self.day[index]

  def __len__(self):
    return self.n_samples

# BCI Dataloader meant to call the DataLoader function.
def BCIData_loader(cache_dir: str,
				  bsz: int = 64,
				  seed: int = 42,
          shuffle: bool = True):

  # DATALOADING A FOLDER.
  train_str = str(cache_dir) + "train/"
  test_str = str(cache_dir) + "test/"

  trainDataset = BCIDataset(path=train_str)
  testDataset = BCIDataset(path=test_str)

  trainloader = DataLoader(dataset=trainDataset, batch_size=bsz, shuffle=True)
  testloader = DataLoader(dataset=testDataset, batch_size=20, shuffle=False)
  valloader = None

  # Stack the tx1 and spikePow earlier
  neuralData_train, labels_train, _, _, _ = trainDataset[0]
  neuralData_test, labels_test, _, _, _ = testDataset[0]

  N_CLASSES = len(ALPHABET)
  SEQ_LENGTH = [neuralData_train.shape[0],neuralData_test.shape[0]]
  IN_DIM = 256
  TRAIN_SIZE = len(trainDataset)

  aux_loader = {}

  return trainloader, valloader, testloader, aux_loader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE

Datasets = {
	# Data loader.
  	"BCI-classification": BCIData_loader,
}
