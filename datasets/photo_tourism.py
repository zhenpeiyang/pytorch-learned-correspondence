import torch.utils.data as data
import numpy as np
import torch
import cv2
import os
import glob
import pickle

class photo_tourism(data.Dataset):
  def __init__(self, split, config, batch_size):
    self.data = self.load_data(config, split)
    self.len = len(self.data['xs'])
    self.batch_size = batch_size
    # self.batch_size = config.batch_size
    
  def load_data(self, config, split):
    print("Loading {} data".format(split))

    # use only the first two characters for shorter abbrv
    split = split[:2]

    # Now load data.
    var_name_list = [
        "xs", "ys", "Rs", "ts",
        "img1s", "cx1s", "cy1s", "f1s",
        "img2s", "cx2s", "cy2s", "f2s",
    ]

    data_folder = config.data_dump_prefix
    if config.use_lift:
        data_folder += "_lift"

    # Let's unpickle and save data
    data = {}
    data_names = getattr(config, "data_" + split)
    data_names = data_names.split(".")
    for data_name in data_names:
        cur_data_folder = "/".join([
            data_folder,
            data_name,
            "numkp-{}".format(config.obj_num_kp),
            "nn-{}".format(config.obj_num_nn),
        ])
        if not config.data_crop_center:
            cur_data_folder = os.path.join(cur_data_folder, "nocrop")
        suffix = "{}-{}".format(
            split,
            getattr(config, "train_max_" + split + "_sample")
        )
        cur_folder = os.path.join(cur_data_folder, suffix)
        ready_file = os.path.join(cur_folder, "ready")
        if not os.path.exists(ready_file):
            # data_gen_lock.unlock()
            raise RuntimeError("Data is not prepared!")

        for var_name in var_name_list:
            cur_var_name = var_name + "_" + split
            in_file_name = os.path.join(cur_folder, cur_var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp)
                else:
                    data[var_name] = pickle.load(ifp)

    return data

  def LoadImage(self, PATH,depth=True):
    if depth:
      img = cv2.imread(PATH,2)/1000.
    else:
      img = cv2.imread(PATH)
    return img
  
  def shuffle(self):
    pass

  def __getitem__(self, index):
    class Record(object):
        pass
    record = Record()
    ind_cur = [index]
    numkps = np.array([self.data['xs'][_i].shape[1] for _i in ind_cur])
    cur_num_kp = numkps.min()
    # Actual construction of the batch
    xs_b = np.array(
        [self.data['xs'][_i][:, :cur_num_kp, :] for _i in ind_cur]
    ).reshape(1, cur_num_kp, 4).transpose(0,2,1)
    ys_b = np.array(
        [self.data['ys'][_i][:cur_num_kp, :] for _i in ind_cur]
    ).reshape(1, cur_num_kp, 2).transpose(0,2,1)
    Rs_b = np.array(
        [self.data['Rs'][_i] for _i in ind_cur]
    ).reshape(1, 9)
    ts_b = np.array(
        [self.data['ts'][_i] for _i in ind_cur]
    ).reshape(1, 3)
    record.xs = xs_b
    record.ys = ys_b
    record.Rs = Rs_b
    record.ts = ts_b
    return record
    
  def __len__(self):
    return self.len

    for i in range(14723):
      m=data_loader.dataset.data['xs'][i].max()
      if m>5: print(m)                                                                            


