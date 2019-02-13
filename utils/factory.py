from .callbacks import PeriodicCallback, OnceCallback, \
    ScheduledCallback,CallbackLoc
import numpy as np
from progress.bar import Bar
import time
from .dotdict import *


class trainer(object):
  def __init__(self, model,train_loader=None,
              val_loader=None,max_epoch=1000):
    # callbacks types
    self._periodic_callbacks = None
    self._once_callbacks = None
    self._scheduled_callbacks = None
    self.max_epoch=max_epoch
    self.model = model
    self.train_loader=train_loader
    self.val_loader=val_loader
    self.callbacks=[]
    self.context={}
    self.step = 0
    self.epoch = self.model.epoch_on_start
    self.trainTimer     = 0
    self.testTimer      = 0

  def add_callbacks(self, callbacks):
    """Add callbacks.
    Args:
        callbacks: list of callbacks
    """
    if not isinstance(callbacks, list):
        callbacks = [callbacks]
    # keep order
    self.callbacks = self.callbacks + callbacks
    # after add callbacks, update callbacks list.
    self.update_callbacks()

  def update_callbacks(self):
    def _check_type(t, cb):
      return t == cb.__class__ or t in cb.__class__.__bases__

    # clear
    self._periodic_callbacks = []
    self._once_callbacks = []
    self._scheduled_callbacks = []
    # add
    for cb in self.callbacks:
      if _check_type(PeriodicCallback, cb):
        self._periodic_callbacks.append(cb)
      if _check_type(OnceCallback, cb):
        self._once_callbacks.append(cb)
      if _check_type(ScheduledCallback, cb):
        self._scheduled_callbacks.append(cb)
  
    
  def timeElaps(self,start):
    return (time.time() - start)/60

  def run(self):
    """Start training with callbacks.
    """
    
    self.update_callbacks()
    # once_callbacks at train start
    
    for cb in self._once_callbacks:
      if cb.cb_loc == CallbackLoc.train_start:
        cb.run()
    
    try:
      while self.epoch < self.max_epoch:
        # update context
        self.context['epoch']=self.epoch
        self.context['global_step']=self.model.global_step

        # periodic callbacks at epoch start
        for cb in self._periodic_callbacks:
          if (cb.cb_loc == CallbackLoc.epoch_start):
            cb.run(self.context)
        
        # train
        if self.train_loader is not None:
          self.model.train(self.train_loader, self.epoch)

        # val
        if self.val_loader is not None:
          self.model.val(self.val_loader, self.epoch)

          # periodic callbacks at epoch end
          for cb in self._periodic_callbacks:
            if (cb.cb_loc == CallbackLoc.epoch_end and self.epoch % cb.pstep == 0):
              cb.run(self.context)

        self.epoch += 1

    except (KeyboardInterrupt, SystemExit):
      logger.info("Training is stoped.")
    except:
      raise
    finally:
      # once_callbacks at exception
      for cb in self._once_callbacks:
        if cb.cb_loc == CallbackLoc.exception:
          cb.run(self.context)
    # once_callbacks at train end
    for cb in self._once_callbacks:
      if cb.cb_loc == CallbackLoc.train_end:
        cb.run(self.context)
