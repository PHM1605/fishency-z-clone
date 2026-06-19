import globals as G

# vid_name: "FROY-HOVD-0001-20260521-100546!0040-INLN_01_4B23-12370-36100-40259-fwl-0.mp4"
def name_parser(vid_name):
  parsed_name = vid_name.split('!')
  cmp, loc, fish_id, day, time = parsed_name[0].split('-')
  
  md = parsed_name[1].split('-')
  unit_id = md.pop(0) # 0040
  G.active_fps = int(md[3][:2]) # md[3]: 40259 => 40
  G.active_n_frames = int(md[3][2:]) # md[3]: 40259 => 259 frames in video
  opmode_subidx_cammodel = md[0].split('_') # ['INLN', '01', '4B23']
  subidx = int(opmode_subidx_cammodel[1]) # 01
  G.active_cameramodel = opmode_subidx_cammodel[-1] # '4B23'
  G.active_unitid = int(unit_id) # 40
  G.active_n_cams = G.camera_model_info[G.active_cameramodel]['n_cams']
  
  return cmp, loc, int(fish_id), int(unit_id), subidx, day, time