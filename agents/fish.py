import globals as G
import numpy as np
import torch

from kornia.image import tensor_to_image
from pathlib import Path
from segment_anything import SamPredictor
from segment_anything.modeling import Sam
from segment_anything_fast import sam_model_registry, sam_model_fast_registry
from ultralytics import YOLO
from utils import misc_utils

# SamPredictor is init with SamModel
class CustomSamPredictor(SamPredictor):
  def __init__(self, sam_model: Sam, is_quantized=False):
    super().__init__(sam_model)
    self.is_quantized = is_quantized

class FishInferenceAgent:
  def __init__(self, weights_dir: str, model_engine: str, sam_model_type:str, device: str) -> None:
    self.model_engine = model_engine # "pt"
    self.sam_model_type = sam_model_type
    self.device = device
    
    self._init_model(weights_dir)
    self._init_sam_model(weights_dir)
  
  def _init_model(self, weights_dir):
    # 'fish': prefix parameter of models
    # weights_dir: "weights/"
    # model_engine: "pt"
    # weights: path of weights 
    # model_info: {'path': Path(xxx), 'res': (400, 640), 'conf': 0.5, 'split': '90%', 'box': 43624, 'exp': 0}
    weights, model_info = misc_utils.load_z_model('fish', weights_dir, self.model_engine)
    self.conf = model_info['conf']
    img_shape = 400, 600
    if "res" in model_info:
      img_shape = model_info["res"]
    self.model = YOLO(weights, task="detect").to(self.device)
    misc_utils.torch_compile(self.model, self.model_engine) # model_engine: "pt"
    self.prediction_helper = misc_utils.YoloPredictionHelper(self.model, img_shape=img_shape)
      
  def _init_sam_model(self, weights_dir):
    assert self.sam_model_type in ["vit_h", "vit_l", "vit_b", "fast_vit_h", "fast_vit_l", "fast_vit_b"], "Wrong ViT model type."
    model_type = "_".join(self.sam_model_type.split("_")[-2:]) # "vit_h"/"vit_l"/"vit_b" (remove the word "fast")
    self.is_quantized_sam = "fast" in self.sam_model_type
    sam_ckpt_pool = {
      "vit_h": "sam_vit_h_4b8939.pth",
      "vit_l": "sam_vit_l_0b3195.pth",
      "vit_b": "sam_vit_b_01ec64.pth"
    }
    sam_weights = Path(weights_dir) / sam_ckpt_pool[model_type]
    sam_registry = sam_model_fast_registry if self.is_quantized_sam else sam_model_registry
    sam = sam_registry[model_type](checkpoint=sam_weights)
    sam.to(device=self.device)
    self.sam_predictor = CustomSamPredictor(sam, self.is_quantized_sam)
  
  def reset(self):
    self.total_num_frames = 0
    
  def run(self, cam_frame_chw, results=None, mask_results=None, is_inline=False):
    # cam_frame_chw: [4,N,3,600,960]
    self.reset()
    n_cams, n_frames = cam_frame_chw.shape[:2]
    self.total_num_frames = n_frames
    cam_ids = list(range(n_cams)) # [0,1,2,3]
    
    # results: {0: {}, 1: {}, 2: {}, 3: {}}
    # mask_results: similar
    if results is None:
      results = {cam_id: {} for cam_id in cam_ids}
    if mask_results is None:
      mask_results = {cam_id: {} for cam_id in cam_ids}
    frame_hw = cam_frame_chw[0].shape[2:] # [600,960]
    
    for cam_id in cam_ids:
      predictions = self.prediction_helper.batch_predict(
        cam_frame_chw[cam_id],
        batch_size = G.cfg["batch_size"],
        # conf = self.conf # don't apply conf of 0.5 yet (take ALL predictions)
      )
      if is_inline:
        mask_predictions = None 
        if G.active_need_biomass: # alway True (check main.py)
          # predictions: list[Result] from whole video (each frame = 1 Result)
          # cam_frame_chw: [4,N,3,600,960]
          mask_predictions = self.predict_masks(cam_frame_chw[cam_id], predictions)
      else: # in-cage
        pass 
    
  # frame_lst: [N,3,600,960]; N is the number of images
  # predictions: list of N results
  def predict_masks(self, frame_lst, predictions):
    mask_predictions = list()
    # each frame is [3,600,960]
    for frame, prediction in zip(frame_lst, predictions):
      boxes = prediction.boxes.xyxyn # (0,4) if no box
      if len(boxes) == 0:
        # raw_masks: [0,600,960]
        raw_masks = torch.zeros((0, *frame.shape[1:]), dtype=torch.bool, device=boxes.device)
      else:
        print(boxes)
        break
        # tensor_to_image(): kornia function to convert to kornia-image (for computer vision)
        frame_np = (tensor_to_image(frame) * 255).astype(np.uint8) # (600,960,3)
        orig_shape_h, orig_shape_w = frame_np.shape[:2]
      
        raw_masks = None
      mask_predictions.append(raw_masks) # list of N raw_masks
    return mask_predictions
  
  def process_results(self, res, mask_res):
    r = {'fish': dict()}
