import re, os

import globals as G
import torch.nn.functional as F

from pathlib import Path
from utils.richprint import *

# prefix: model name e.g. "fish", "matte"
# model_engine: "pt"
def load_z_model(prefix, weights_dir, model_engine, verbose=True):
  # (.*) means "match any characters, 0 or more time"
  regex = re.compile(f'^{prefix}(.*)\.{model_engine}$')
  model_files = os.listdir(weights_dir)
  matching_files = [f for f in model_files if regex.match(f)] 
  assert len(matching_files) == 1, f'Exactly one {prefix} model file must exist in {weights_dir}.'
  matched = matching_files[0] # "fish_res=640x400_conf=0.50_split=90%_box=43624_exp=0.pt"
  
  m = regex.match(matched) # extract the matching groups
  path = Path(weights_dir) / m[0] # m[0] is the entire file name => weights/fish_res=640x400_conf=0.50_split=90%_box=43624_exp=0.pt
  
  # key-value pairs to extract model params in a dict
  kv = dict()
  for kv_text in m[1].split('_'): # "m[1]" = "_res=640x400_conf=0.50_split=90%_box=43624_exp=0.pt"
    k, v = G.process_command(kv_text)
    if k is not None and v is not None:
      kv[k] = v
  # treat key="res" and value="640x400"
  if "res" in kv:
    res_x, res_y = [int(val) for val in kv['res'].split('x')]
    kv['res'] = (res_y, res_x)
  # print out
  if verbose:
    rprint(f"[bold cyan]# {prefix}[/bold cyan]\n- Using model file {bold_yellow(str(path))} with confidence {bold_yellow(kv['conf'])}.")
  
  return str(path), {'path': path, **kv}

# frame_chw: [N,3,600,960]
# batch_size: 64 (inference batch size, not video batch size)
def batchify(frame_chw, batch_size):
  n_frames = frame_chw.shape[0] # N
  for i in range(0, n_frames, batch_size):
    yield frame_chw[i : i+batch_size, :, :, :] # [64,3,600,960]
  
# model_engine: "pt"  
def torch_compile(model, model_engine):
  if G.cfg["torch_compile"] and model_engine == "pt":
    model.model = torch.compile(model.model)
  
class YoloPredictionHelper:
  def __init__(self, model, img_shape, print_info=True):
    def stride(val):
      a = (val // 32) * 32 # round it down to multiple of 32 (in this case, (400,600)=>(384,576))
      return a if a==val else a+32 # if no change => return old value; else => round up (416,608) 
    
    self.model = model
    self.img_shape = img_shape # (400, 600)
    self.suitable_shape = (stride(img_shape[0]), stride(img_shape[1])) # (400,600)=>(416,608)
    self.pad_v = (self.suitable_shape[0] - self.img_shape[0]) // 2 # pad 8 upper and 8 lower
    self.pad_h = (self.suitable_shape[1] - self.img_shape[1]) // 2 # pad 4 right and 4 left
  
  # frame_chw: [N,3,600,960]
  # bg_color: color of which to pad input image 
  def batch_predict(self, frame_chw, batch_size, bg_color=114, **kwargs):
    predictions = list() # list[Result]
    # frame_chw_batch: [64,3,600,960]
    for frame_chw_batch in batchify(frame_chw, batch_size):
      # resize from [64,3,600,960] to img_shape=[400,600] (input [64,3,400,600])
      input = F.interpolate(frame_chw_batch, size=self.img_shape, mode="bilinear", align_corners=False)
      # pad to make input shape multiple of 32 => [416,608]
      if self.pad_v > 0 or self.pad_h > 0:
        input = F.pad(input, (self.pad_h, self.pad_h, self.pad_v, self.pad_v), 'constant', bg_color/255)
      predictions += self.model(source=input, imgsz=self.suitable_shape, workspace=4, **kwargs)
    # list[Result] of N (video length) elements of Result type
    return predictions
    