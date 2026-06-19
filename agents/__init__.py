from agents.fish import FishInferenceAgent

def build_fish_agent(weights_dir, model_engine, sam_model_type, device):
  return FishInferenceAgent(weights_dir, model_engine, sam_model_type, device)
  