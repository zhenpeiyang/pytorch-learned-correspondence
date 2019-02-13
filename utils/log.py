import numpy as np
import torch
import time
import os

class logging():
  """ record msg into log file, and print to screen
  """
  def __init__(self, log_file):
    self.log_file = log_file
  def __call__(self, msg):
    with open(self.log_file,'a') as f:
      f.write(msg + '\n')
      print(msg)

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
