import numpy as np 
import os 
import glob
import shutil
import torch
import collections
import sys
if sys.version_info[0] == 2:
  import Queue as queue
  string_classes = basestring
else:
  import queue
  string_classes = (str, bytes)

EXP_BASE_DIR = './experiments'

def v(var, cuda=True, volatile=False):
  # convert torch tensor/ numpy array into torch variable

  if type(var) == torch.Tensor or type(var) == torch.DoubleTensor:
    res = torch.autograd.Variable(var.float(),volatile=volatile)
  elif type(var) == np.ndarray:
    res = torch.autograd.Variable(torch.from_numpy(var).float(),volatile=volatile)
  if cuda:
    res = res.cuda()
  return res

def npy(var):
  # convert torch tensor into numpy variable

  return var.data.cpu().numpy()

def parameters_count(net, name):
  # count parameter of the network

  model_parameters = filter(lambda p: p.requires_grad, net.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print('total parameters for %s: %d' % (name, params))

def get_latest_model(path, identifier):
  # check for latest saved checkpoint

  models = glob.glob('{0}/*{1}*'.format(path, identifier))
  epoch = [int(model.split('_')[-1].split('.')[0]) for model in models]
  ind = np.array(epoch).argsort()
  models = [models[i] for i in ind]
  return models[-1]



def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def collate_fn_cat(batch):
  "Puts each data field into a tensor with outer dimension batch size"
  
  if torch.is_tensor(batch[0]):
    out = None
    return torch.cat(batch, 0, out=out)
    # for rnn variable length input
  elif type(batch[0]).__module__ == 'numpy':
    elem = batch[0]
    if type(elem).__name__ == 'ndarray':
      try:
        torch.cat([torch.from_numpy(b) for b in batch], 0)
      except:
        import ipdb;ipdb.set_trace()
      return torch.cat([torch.from_numpy(b) for b in batch], 0)
    if elem.shape == ():  # scalars
      py_type = float if elem.dtype.name.startswith('float') else int
      return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
  elif isinstance(batch[0], int):
    return torch.LongTensor(batch)
  elif isinstance(batch[0], float):
    return torch.DoubleTensor(batch)
  elif isinstance(batch[0], string_classes):
    return batch
  elif isinstance(batch[0], collections.Mapping):
    return {key: collate_fn_cat([d[key] for d in batch]) for key in batch[0]}
  elif isinstance(batch[0], collections.Sequence):
    transposed = zip(*batch)
    return [collate_fn_cat(samples) for samples in transposed]
  elif isinstance(batch[0], object):
    return {key: collate_fn_cat([getattr(d,key) for d in batch]) for key in batch[0].__dict__.keys()}
    
def setup_folder(args):
  # setup experimental directories

  if not os.path.exists(EXP_BASE_DIR):
    os.mkdir(EXP_BASE_DIR)
  
  args.EXPERIMENT_INDEX = args.exp if args.exp else '404'
  args.EXP_BASE_DIR = './experiments/exp_' + args.EXPERIMENT_INDEX
  args.EXP_DIR = args.EXP_BASE_DIR
  args.EXP_DIR_SAMPLES = args.EXP_DIR + '/samples'
  args.EXP_DIR_PARAMS = args.EXP_DIR + '/params'
  args.EXP_DIR_LOG = os.path.join(args.EXP_DIR, 'exp_{}.csv'.format(args.EXPERIMENT_INDEX))

  if args.d:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.d
  for i in range(torch.cuda.device_count()):
    print("detected gpu:{}\n".format(torch.cuda.get_device_name(i)))

  # cannot set remove and resume both true
  assert(not (args.rm and args.resume))
  # if not specified rm and resume, then the folder must not exists
  if not args.rm and not args.resume:
    assert(not os.path.exists(args.EXP_DIR))
  
  try:
    if args.rm:
      shutil.rmtree(args.EXP_DIR)
  except:
      pass
  
  try:
    if not os.path.exists(args.EXP_DIR):
      os.makedirs(args.EXP_DIR)
  except:
    pass
  
  try:
    if not os.path.exists(args.EXP_DIR_SAMPLES):
      os.makedirs(args.EXP_DIR_SAMPLES)
  except:
      pass
  
  try:
    if not os.path.exists(args.EXP_DIR_PARAMS):
      os.makedirs(args.EXP_DIR_PARAMS)
  except:
      pass
  
  return args