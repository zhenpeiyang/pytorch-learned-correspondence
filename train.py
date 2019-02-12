import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import routines
from utils.routines import v, npy
from utils import log
from tensorboardX import SummaryWriter
import cv2
import helper
import time
import re
import glob
from utils.dotdict import *
from utils.factory import trainer
from archs.cvpr2018 import Net
from utils.callbacks import PeriodicCallback,CallbackLoc
import copy
from tests import test_process
from helper import torch_skew_symmetric
from config import get_config, print_usage
from termcolor import colored, cprint
import parse

def get_dataset(config):
  def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
  
  if 'st_peters' in config.data_tr:
    from datasets.photo_tourism import photo_tourism as Dataset
  else:
    raise Exception("unknown dataset!")

  train_dataset = Dataset('train', config)
  val_dataset   = Dataset('valid', config)
  if config.debug:
    train_loader = DataLoader(train_dataset, batch_size=1, 
      shuffle=True, drop_last=True, collate_fn=routines.collate_fn_cat, 
      worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True,
      drop_last=True, collate_fn=routines.collate_fn_cat, 
      worker_init_fn=worker_init_fn)
  else:
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
      num_workers=config.num_workers, drop_last=True,
      collate_fn=routines.collate_fn_cat, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, 
      num_workers=config.num_workers, drop_last=True,
      collate_fn=routines.collate_fn_cat, worker_init_fn=worker_init_fn)

  return train_loader,val_loader


class learner(object):
  def __init__(self, config,):
    self.config = config
    self.epoch_on_start = 0

    self.tensorboardX = SummaryWriter(
                        log_dir= self.config.EXP_DIR + '/tensorboard')
    self.logger_loss     = log.AverageMeter()
    self.res_dir_va   =  self.config.EXP_DIR + '/valid'
    os.makedirs(self.res_dir_va, exist_ok=True)
    self.va_res_file = os.path.join(self.res_dir_va, "valid", "va_res.txt")
    
    self.global_step = 0
    self.speed_benchmark = True
    if self.speed_benchmark:
      self.time_per_step = log.AverageMeter()

    self.net = Net(self.config).cuda()

    routines.parameters_count(self.net, 'net')

    # setup optimizer
    params = list(self.net.parameters())
    self.optimizer = torch.optim.Adam(params, lr=self.config.train_lr, 
      betas=(0.5, 0.999), weight_decay=0.001)

    # resume if specified
    self.best_va_res = -1
    if self.config.resume: self.load_checkpoint()

  def set_mode(self,mode='train'):
    if mode == 'train':
      self.net.train()
    else:
      return

  def update_lr(self):
    self.lr_scheduler.step()

  def _save_checkpoint(self, net, optimizer, filename, 
        do_cleaning = True, NUM_RETAIN = 3, epoch = None):
    # find previous saved model and only retain  
    # the 3 most recent models
    state = {'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict()}
    
    torch.save(state, filename)
    if do_cleaning:
      dirname = os.path.dirname(filename)

      ckpt = filename.split('/')[-1]
      num  = re.findall(r'\d+', ckpt)[0]
      ckpt = ckpt.replace(num, '*')

      # get all matched checkpoints
      files = glob.glob('%s/%s' % (dirname, ckpt))
      files.sort()

      # delete old checkpoints
      N = len(files) - NUM_RETAIN
      for i in range(N):
        cmd = 'rm %s' % files[i]
        os.system(cmd)

  def save_checkpoint(self, context):
    epoch = context['epoch']
    cprint('save model: %s' % epoch, 'yellow')
    self._save_checkpoint(
        self.net, self.optimizer,
        os.path.join(self.config.EXP_DIR_PARAMS, 
                    'checkpoint_G_%04d.pth.tar' % epoch),
        do_cleaning = True, epoch = epoch)
    

  def load_checkpoint(self):
    try:
      if self.config.ckpt is not None:
        net_path = self.config.ckpt
      else:
        net_path = routines.get_latest_model(
                self.config.EXP_DIR_PARAMS,'checkpoint_G')
      
      checkpoint = torch.load(net_path)
      checkpoint = dotdict(checkpoint)
      state_dict = checkpoint.state_dict
      self.epoch_on_start = checkpoint.epoch + 1

      self.net.load_state_dict(state_dict)
      cprint('resume network weights from %s successfully' \
              % net_path, 'red', attrs=['reverse', 'blink'])

      self.optimizer.load_state_dict(checkpoint.optimizer)
      cprint('resume optimizer from %s successfully' % net_path,
             'red', attrs=['reverse', 'blink'])
      
      if os.path.exists(self.va_res_file):
        with open(self.va_res_file, "r") as ifp:
            dump_res = ifp.read()
        dump_res = parse(
            "{best_va_res:e}\n", dump_res)
        self.best_va_res = dump_res["best_va_res"]
    except Exception as e: 
      print(e)
      print("resume fail")

  def step(self, data, mode = 'train'):
    torch.cuda.empty_cache()
    if self.speed_benchmark:
      step_start=time.time()
    
    with torch.set_grad_enabled(mode == 'train'):
      np.random.seed()
      self.optimizer.zero_grad()
      
      MSEcriterion = torch.nn.MSELoss()
      BCEcriterion = torch.nn.BCELoss()
      CEcriterion  = nn.CrossEntropyLoss(reduce=False)
      
      x_in, y_in  = v(data.xs),v(data.ys)
      R_in, t_inv = (data.Rs),v(data.ts)
      
      x_shp = x_in.shape
      
      logits = self.net(x_in)
      logits = logits.view(x_shp[0], -1)
      weights = F.relu(F.tanh(logits))

      # Make input data (num_img_pair x num_corr x 4)
      
      xx = x_in
      # Create the matrix to be used for the eight-point algorithm
      
      X = torch.transpose(torch.stack([
          xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
          xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
          xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
      ], dim=1), 1, 2)

      print("X shape = {}".format(X.shape))
      wX = weights.view(x_shp[0], x_shp[2], 1) * X
      print("wX shape = {}".format(wX.shape))
      XwX = torch.matmul(torch.transpose(X, 1, 2), wX)
      print("XwX shape = {}".format(XwX.shape))
      
      
      # Recover essential matrix from self-adjoing eigen
      e_hat = []
      for i in range(x_shp[0]):
        # e, vv = torch.eig(XwX[i],eigenvectors=True)
        u, s, vv = torch.svd(XwX[i].cpu())
        e_hat_this = u[:, -1]
        # Make unit norm just in case
        e_hat_this = e_hat_this / torch.norm(e_hat_this)
        e_hat.append(e_hat_this)
      
      e_hat = torch.stack(e_hat).cuda()
      
      gt_geod_d = y_in[:, 0, :]
      # tf.summary.histogram("gt_geod_d", gt_geod_d)

      # Get groundtruth Essential matrix
      e_gt_unnorm = torch.matmul(
          torch_skew_symmetric(t_in).view(x_shp[0], 3, 3),
          R_in.view(x_shp[0], 3, 3)
      ).view(x_shp[0], 9)
      e_gt = e_gt_unnorm / torch.norm(e_gt_unnorm, dim=1, \
        keepdim=True).detach()
      
      # Essential matrix loss
      essential_loss = torch.mean(torch.min(
          torch.sum(torch.pow(e_hat - e_gt, 2), dim=1),
          torch.sum(torch.pow(e_hat + e_gt, 2), dim=1)
      ))

      self.tensorboardX.add_scalars('essential_loss', 
                    {'%s' % (mode):essential_loss}, self.global_step)
      
      # Classification loss
      is_pos = (gt_geod_d < self.config.obj_geod_th).float()
      
      is_neg = (gt_geod_d >= self.config.obj_geod_th).float()

      c = is_pos - is_neg
      classif_losses = -torch.log(F.sigmoid(c * logits))

      # balance
      num_pos = F.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
      num_neg = F.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
      classif_loss_p = torch.sum(
          classif_losses * is_pos, dim=1
      )
      classif_loss_n = torch.sum(
          classif_losses * is_neg, dim=1
      )
      classif_loss = torch.mean(
          classif_loss_p * 0.5 / num_pos +
          classif_loss_n * 0.5 / num_neg
      )
      self.tensorboardX.add_scalars('classif_loss', 
          {'%s' % (mode):classif_loss}, self.global_step)
      self.tensorboardX.add_scalars('classif_loss_p', 
          {'%s' % (mode):torch.mean(classif_loss_p * 0.5 / num_pos)}
          , self.global_step)
      self.tensorboardX.add_scalars('classif_loss_n', 
          {'%s' % (mode):torch.mean(classif_loss_n * 0.5 / num_neg)}
          , self.global_step)
      
      precision = torch.mean(
          torch.sum((logits > 0).float() * is_pos, dim=1) /
          torch.sum((logits > 0).float() *
                        (is_pos + is_neg), dim=1)
      )
      self.tensorboardX.add_scalars('precision', 
          {'%s' % (mode):precision}, self.global_step)
      
      recall = torch.mean(
          torch.sum((logits > 0).float() * is_pos, dim=1) /
          torch.sum(is_pos, dim=1)
      )
      self.tensorboardX.add_scalars('recall', 
          {'%s' % (mode):recall}, self.global_step)
      

      # Check global_step and add essential loss
      loss = 0
      if self.config.loss_essential > 0:
          loss += (
              self.config.loss_essential * essential_loss * 
              (self.global_step >= self.config.loss_essential_init_iter))
      if self.config.loss_classif > 0:
          loss += self.config.loss_classif * classif_loss
      
      self.tensorboardX.add_scalars('loss', 
          {'%s' % (mode):loss}, self.global_step)

      
      if mode == 'train':
        if not torch.isnan(loss).any():
          loss.backward()
        
          torch.nn.utils.clip_grad_norm_(self.net.parameters(), 
                                        self.config.GRAD_CLIP)
          self.optimizer.step()
      
      self.logger_loss.update(loss.data, x_shp[0])
      
      suffix = f"| loss {self.logger_loss.avg:.6f}"
    
      # ----------------------------------------
      # Validation
      if self.global_step % self.config.val_intv == 0: 
      # if 1:
        import ipdb;ipdb.set_trace()
        va_res = 0
        cur_global_step = self.global_step
        va_res, summary_t = test_process(self.net,
            "valid", cur_global_step,
            x_in, y_in, R_in, t_in,
            None, None, None,
            logits, e_hat, loss,
            data["valid"],
            self.res_dir_va, self.config, True)
        for entry in summary_t:
          self.tensorboardX.add_scalars(entry['tag'], 
              entry['val'], self.global_step)
        # Higher the better
        if va_res > best_va_res:
          print(
              "Saving best model with va_res = {}".format(
                  va_res))
          best_va_res = va_res
          # Save best validation result
          with open(self.va_res_file, "w") as ofp:
              ofp.write("{:e}\n".format(best_va_res))
          # Save best model
          self.saver_best.save(
              self.sess, self.save_file_best,
              write_meta_graph=False,
          )

      summary = {'suffix':suffix}
      self.global_step+=1
    
    if self.speed_benchmark:
      self.time_per_step.update(time.time()-step_start,1)
      print(f"time elapse per step: {self.time_per_step.avg}")
    return dotdict(summary)

def main():
  # parse arguments, build exp dir
  config, unparsed = get_config()
  routines.setup_folder(config)

  # build data loader
  train_loader,val_loader = get_dataset(config)

  # build learner
  model = learner(config)
  
  # build trainer and launch training
  mytrainer=trainer(
    model,
    train_loader,
    val_loader,
    max_epoch=200,
    )

  mytrainer.add_callbacks(
    [PeriodicCallback(cb_loc = CallbackLoc.epoch_end,pstep = 5,
                      func = model.save_checkpoint)]
  )

  mytrainer.run()

if __name__ == '__main__':
  main()
