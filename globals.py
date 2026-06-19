import os, re

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

cfg = {
  "batch_size": 64, # for YOLO inference
  "frame_step": 1,
  'model_device': 'cpu',
  'sam_model_type': 'fast_vit_h',
  # 'tmp_dir': f'/run/user/{os.getuid()}/tmp', # /run/user/501/tmp
  # 'tmp_video_dir': f'/run/user/{os.getuid()}/tmp',
  'tmp_dir': ROOT_DIR /"tmp",
  'tmp_video_dir': ROOT_DIR / "tmp",
  'torch_compile': False,
  'verbose': False,
  'weights_dir': 'weights/',
  'yolo_engine': 'pt',
}

camera_model_info = {
  "4B23": {"res": (1200, 1920), 'n_cams': 4}
}
video_decoder_lib = 'ffmpeg' # 'gstreamer', 'ffmpeg' or 'opencv'
work_res_for_cam = (600, 960) # will be overwritten from filename when yolo lice model loads

# e.g. conf=0.50 => return "key" and "value" to be added to dict
def process_command(cmd):
  keyvalue = cmd.split('=') # ['conf', '0.50']
  if len(keyvalue) > 1:
    value = process_value(keyvalue[1])
    return keyvalue[0].strip(), value
  # command is key only e.g. "conf"
  return keyvalue[0].strip(), None
  
def process_value(value):
  # ^[-+]? means "begin with - or + sign, 0 or 1 time"
  # \\d* means "digit", 0 or more times
  # \\d+ means "digit", 1 or more times
  # re.S means "consider newline too"
  re_float = re.compile("^[-+]?\\d*\\.\\d+$", re.S)
  re_int = re.compile("^[-+]?\\d+$", re.S)
  if re_float.match(value):
    value = float(value)
  elif re_int.match(value):
    value = int(value)
  # if value is '[2,5]' or 'True' or 'False' => evaluate to actual array and Boolean
  elif value.startswith('[') or value=='True' or value == 'False':
    value = eval(value)
  # if doesn't match e.g. "640x400" then return as is
  return value
