import globals as G
import imageio_ffmpeg as ffmpeg
import torch
import torch.nn.functional as F

from utils.richprint import *

def frames_from_video(input_file, batch_size, cam_res = G.work_res_for_cam, device=G.cfg["model_device"]):
  n_cams = G.active_n_cams
  frame_step = G.cfg["frame_step"]
  print(frame_step)
  
  def init_tensor():
    # cam_res: (600,960)
    return torch.empty(
      (batch_size, 3, cam_res[0]*G.active_n_cams, cam_res[1]),
      dtype=torch.uint8,
      device="cpu" # stay here first
    )
  
  def eval_return_batch(video_tensor):
    # video_tensor: [batch,3,600*4,960] => [batch,2400,960,3] => [batch,4,600,960,3] =>[4,batch,3,600,960]
    return (video_tensor.permute(0,2,3,1) \
        .view(-1, n_cams, *cam_res, 3) \
        .permute(1,0,4,2,3) \
        .to(torch.float32) / 255.0 \
      ).to(device)
    
  
  video = ffmpeg.read_frames(input_file, pix_fmt='rgb24')
  meta = video.__next__()
  del meta["ffmpeg_version"]
  # codec=hevc pix_fmt=yuv420p(tv, bt709) fps=40.0 source_size=(960, 2400) size=(960, 2400) rotate=0 duration=6.48
  printBoldYellow(' '.join([f'{k}={v}' for k, v in meta.items()]))
  
  w, h = meta["size"] # 960, 2400
  G.active_hw = (h, w)
  video_tensor = init_tensor() # [batch,3,600*4,960]
  
  current_frame_id = -1
  n_decoded_frames = 0
  for frame_bytes in video: # RGB
    current_frame_id += 1    
    # [2400, 960, 3] => [3,2400,960]
    frame = torch.frombuffer(bytearray(frame_bytes), dtype=torch.uint8) \
      .view(h, w, 3) \
      .permute(2, 0 ,1)
    video_tensor[current_frame_id, :, :, :] = frame
    n_decoded_frames += 1
    # finish running a batch (in this case, whole video)
    if n_decoded_frames % batch_size == 0:
      yield eval_return_batch(video_tensor)
      n_decoded_frames = 0
  # for the leftovers frames
  yield eval_return_batch(video_tensor[:n_decoded_frames, ...])
