from .train import LoraTraininginComfy, TensorboardAccess
NODE_CLASS_MAPPINGS = {"Lora Training in ComfyUI": LoraTraininginComfy, "Tensorboard Access": TensorboardAccess}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']