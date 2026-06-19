import json, os
import globals as G

from agents import * # import from __init__.py
from utils import db_utils, misc_utils, video_utils
from utils.richprint import *

def run():
  weights_dir, tmp_dir, tmp_video_dir = G.cfg['weights_dir'], G.cfg['tmp_dir'], G.cfg['tmp_video_dir']
  device = G.cfg['model_device']
  verbose = G.cfg['verbose']
  yolo_engine = G.cfg['yolo_engine'] # pt
  sam_model_type = G.cfg['sam_model_type']
  
  agent_dict = dict({
    'FISH': build_fish_agent(weights_dir, yolo_engine, sam_model_type, device),
    'VISUAL_HULL': None
  })
  
  # tmp_video_dir: /run/user/501/tmp
  vid = "FROY-HOVD-0001-20260521-100546!0040-INLN_01_4B23-12370-36100-40259-fwl-0"
  vid_ext = vid + ".mp4"
  
  # set Global variables according to video name
  cmp, loc, fishid, vid_unit, subidx, vid_day, vid_time = db_utils.name_parser(vid)
  video_file_path = os.path.join(tmp_video_dir, vid_ext)
  meshfn = f'_BM_{cmp}-{loc}-{fishid:04}-{vid_day}-{vid_time}!{vid_unit:04}' # unit: 40
  bucket_dict = {'FROY': None}
  result = process(agent_dict, video_file_path, tmp_dir, bucket_dict[cmp], meshfn, verbose=verbose)
  print(result)
  
def process(agent_dict, filepath, output_dir, bucket, meshfn, verbose = False):
  rprint(f"PROCESSING: {bold_red(filepath)}")
  n_cams = G.active_n_cams
  is_inline = '-INLN_' in filepath
  printBoldCyan(
    f"{'INLINE' if is_inline else 'CAGE'} MODE"
    f"  UNIT={G.active_unitid}" # 40
    f"  FPS={G.active_fps}"
    f"  CAMS={G.active_cameramodel}"
    f"  N_FRAMES={G.active_n_frames}"
  )
  
  if is_inline:
    fish_agent = agent_dict['FISH']
    # we need biomass only when the video is SHORT enough, camera model correct, and unit NOT 22
    # unit 22 is dead; unit 30 is Taupiri
    G.active_need_biomass = G.active_cameramodel == "4B23" and G.active_n_frames <= 1200 and G.active_unitid != 22 # original: 100 frames
    # if biomass not considered => skip that video
    if not G.active_need_biomass:
      return -1
    
    # start processing video => "next(video_gen)" will return a batch (tensor [4,batch,3,600,960])
    video_gen = video_utils.frames_from_video(filepath, batch_size=G.active_n_frames) # take all frames in video as batch
    
    # Run fish inference
    # we don't need while loop here, as batch = whole video
    cam_frame_chw = next(video_gen) # [4,batch,3,600,960]
    results = mask_results = None
    results, mask_results = fish_agent.run(cam_frame_chw, results, mask_results, is_inline=True)
  else:
    results = {}
  
  obj = json.dumps(results)
  return json.loads(obj)

if __name__ == '__main__':
  run()
  