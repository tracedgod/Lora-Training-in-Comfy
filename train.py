# Original LoRA train script by @Akegarasu ; rewritten in Python by LJRE.
import subprocess
import os
import sys
import folder_paths
import random
from comfy import model_management
import torch
from packaging import version

os.environ['HF_HOME'] = "huggingface"
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = "1"

#Train data path | 设置训练用模型、图片
#pretrained_model = "E:\AI-Image\ComfyUI_windows_portable_nvidia_cu121_or_cpu\ComfyUI_windows_portable\ComfyUI\models\checkpoints\MyAnimeModel.ckpt"
# is_v2_model = 0 # SD2.0 model | SD2.0模型 2.0模型下 clip_skip 默认无效
# parameterization = 0 # parameterization | 参数化 本参数需要和 V2 参数同步使用 实验性功能
#train_data_dir = "" # train dataset path | 训练数据集路径
reg_data_dir = "" # directory for regularization images | 正则化数据集路径，默认不使用正则化图像。

# Network settings | 网络设置
#network_module = "networks.lora" # 在这里将会设置训练的网络种类，默认为 networks.lora 也就是 LoRA 训练。如果你想训练 LyCORIS（LoCon、LoHa） 等，则修改这个值为 lycoris.kohya
#network_weights = "" # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。
network_dim = 32 # network dim | 常用 4~128，不是越大越好
network_alpha = 32 # network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率。

# Train related params | 训练相关参数
# resolution = "512,512" # image resolution w,h. 图片分辨率，宽,高。支持非正方形，但必须是 64 倍数。
#batch_size = 1 # batch size | batch 大小
#max_train_epoches = 10 # max train epoches | 最大训练 epoch
#save_every_n_epochs = 10 # save every n epochs | 每 N 个 epoch 保存一次

#train_unet_only = 0 # train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启
#train_text_encoder_only = 0 # train Text Encoder only | 仅训练 文本编码器
stop_text_encoder_training = 0 # stop text encoder training | 在第 N 步时停止训练文本编码器

noise_offset = 0 # noise offset | 在训练中添加噪声偏移来改良生成非常暗或者非常亮的图像，如果启用，推荐参数为 0.1
keep_tokens = 0 # keep heading N tokens when shuffling caption tokens | 在随机打乱 tokens 时，保留前 N 个不变。
min_snr_gamma = 0 # minimum signal-to-noise ratio (SNR) value for gamma-ray | 伽马射线事件的最小信噪比（SNR）值  默认为 0

# Learning rate | 学习率
#lr = "1e-4" # learning rate | 学习率，在分别设置下方 U-Net 和 文本编码器 的学习率时，该参数失效
#unet_lr = "1e-4" # U-Net learning rate | U-Net 学习率
#text_encoder_lr = "1e-5" # Text Encoder learning rate | 文本编码器 学习率
#lr_scheduler = "cosine_with_restarts" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
#lr_warmup_steps = 0 # warmup steps | 学习率预热步数，lr_scheduler 为 constant 或 adafactor 时该值需要设为0。
#lr_restart_cycles = 1 # cosine_with_restarts restart cycles | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时起效。

# 优化器设置
#optimizer_type = "AdamW8bit" # Optimizer type | 优化器类型 默认为 AdamW8bit，可选：AdamW AdamW8bit Lion Lion8bit SGDNesterov SGDNesterov8bit DAdaptation AdaFactor prodigy

# Output settings | 输出设置
#output_name = "Pkmn3GTest" # output model name | 模型保存名称
#save_model_as = "safetensors" # model save ext | 模型保存格式 ckpt, pt, safetensors

# Resume training state | 恢复训练设置
save_state = 0 # save training state | 保存训练状态 名称类似于 <output_name>-??????-state ?????? 表示 epoch 数
resume = "" # resume from state | 从某个状态文件夹中恢复训练 需配合上方参数同时使用 由于规范文件限制 epoch 数和全局步数不会保存 即使恢复时它们也从 1 开始 与 network_weights 的具体实现操作并不一致

# 其他设置
min_bucket_reso = 256 # arb min resolution | arb 最小分辨率
max_bucket_reso = 1584 # arb max resolution | arb 最大分辨率
persistent_data_loader_workers = 1 # persistent dataloader workers | 保留加载训练集的worker，减少每个 epoch 之间的停顿
#clip_skip = 2 # clip skip | 玄学 一般用 2
#multi_gpu = 0 # multi gpu | 多显卡训练 该参数仅限在显卡数 >= 2 使用
#lowram = 0 # lowram mode | 低内存模式 该模式下会将 U-net 文本编码器 VAE 转移到 GPU 显存中 启用该模式可能会对显存有一定影响

# LyCORIS 训练设置
algo = "lora" # LyCORIS network algo | LyCORIS 网络算法 可选 lora、loha、lokr、ia3、dylora。lora即为locon
conv_dim = 4 # conv dim | 类似于 network_dim，推荐为 4
conv_alpha = 4 # conv alpha | 类似于 network_alpha，可以采用与 conv_dim 一致或者更小的值
dropout = "0"  # dropout | dropout 概率, 0 为不使用 dropout, 越大则 dropout 越多，推荐 0~0.5， LoHa/LoKr/(IA)^3 暂时不支持

# 远程记录设置
use_wandb = 0 # enable wandb logging | 启用wandb远程记录功能
wandb_api_key = "" # wandb api key | API，通过 https://wandb.ai/authorize 获取
log_tracker_name = "" # wandb log tracker name | wandb项目名称,留空则为"network_train"

#output_dir = ''
logging_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
log_prefix = ''
# mixed_precision = 'fp16'
caption_extension = '.txt'

launch_args = []
ext_args = []

class TrainingUtils:
    @staticmethod
    def free_memory():
        '''Frees up memory by unloading all currently loaded models.'''
        try:
            loaded_models = model_management.current_loaded_models

            if not loaded_models:
                return  # No models to unload

            # Unload all models
            for model in loaded_models:
                model.model_unload()

            loaded_models.clear()

            # Clear cache
            model_management.soft_empty_cache()

        except Exception as e:
            print(f"Warning: Failed to free memory before training: {e}")

    @staticmethod
    def get_train_script(script_name: str):
        """
        Gets the full path to the specified training script.

        Parameters:
            script_name (str): The name of the training script without the `.py` extension.

        Returns:
            tuple: (Full script path, script directory) if found.
            tuple: (None, script directory) if the script file does not exist.
        """
        try:
            # Validate script_name
            if not isinstance(script_name, str) or not script_name.strip():
                raise ValueError("Invalid script_name: Must be a non-empty string.")

            # Get directory paths
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            sd_script_dir = os.path.join(current_file_dir, "sd-scripts")
            train_script_path = os.path.join(sd_script_dir, f"{script_name}.py")

            # Check if the script file exists
            if not os.path.isfile(train_script_path):
                print(f"Warning: Training script '{train_script_path}' not found.")
                return None, sd_script_dir

            return train_script_path, sd_script_dir

        except Exception as e:
            print(f"Error in get_train_script(): {e}")
            return None, None

    @staticmethod
    def get_available_torch_devices():
        '''Gets the amount of available Torch devices'''
        return torch.cuda.device_count()

    @staticmethod
    def is_bf16_supported():
        '''Checks if the GPU is NVIDIA Ampere or newer (Compute Capability >= 8.0)'''
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return major >= 8
        return False
    
    @staticmethod
    def is_torch_bf16_compatible():
        '''Checks if PyTorch version is >=1.10.0 for bf16 support'''
        return version.parse(torch.__version__) >= version.parse("1.10.0")

class LoraTraininginComfy:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "model_type": ("STRING", {"default": "sd1.5", "choices": ["sd1.5", "sd2.0", "sdxl", "sd3", "flux"], "tooltip": "The type of base model to train the LoRA with."},),
            "resolution_width": ("INT", {"default": 1024, "step": 64, "tooltip": "X resolution value"}),
            "resolution_height": ("INT", {"default": 1024, "step": 64, "tooltip": "Y resolution value"}),
            "data_path": ("STRING", {"default": "Insert path of image folders"}),
			"batch_size": ("INT", {"default": 1, "min":1}),
            "max_train_epoches": ("INT", {"default":10, "min":1, "tooltip": "Max epochs to run during training session."}),
            "save_every_n_epochs": ("INT", {"default":10, "min":1, "tooltip": "How many epochs to run before saving a copy of the LoRA."}),
            "output_name": ("STRING", {"default":'Desired name for LoRA.'}),
            "clip_skip": ("INT", {"default":2, "min":1, "tooltip": "Controls how early the processing of prompt by clip network should be stopped (Option is ignored for sdxl)."}),
            "mixed_precision": ("STRING", {"default": "no", "choices": ["no", "fp16", "bf16"], "tooltip": "Use mixed precision training. Choose between fp16 and bf16 training. NOTE: bf16 training is only supported on Nvidia Ampere and up GPUs + PyTorch v1.10 or later!"}),
            "output_dir": ("STRING", {"default":'models/loras'}),
            "dynamo_backend": ("STRING", {"default": "no", "choices": ["no", "eager", "aot_eager", "inductor", "aot_ts_nvfuser", "nvprims_nvfuser", "cudagraphs", "ofi", "fx2trt", "onnxrt", "tensorrt", "aot_torchxla_trace_once", "ipex", "tvm"], "tooltip": "Dynamo backend selection (see https://pytorch.org/docs/stable/torch.compiler.html)."}),
            "multi_gpu": ("BOOLEAN", {"default": False, "tooltip": "Use distributed GPU training."})
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "loratraining"
    OUTPUT_NODE = True
    CATEGORY = "LJRE/LORA"
    
    def loratraining(self, ckpt_name, model_type, resolution_width, resolution_height, data_path, batch_size, max_train_epoches, save_every_n_epochs, output_name, clip_skip, output_dir, mixed_precision, dynamo_backend, multi_gpu):
        #free memory first of all
        TrainingUtils.free_memory()

        #transform backslashes into slashes for user convenience.
        train_data_dir = data_path.replace( "\\", "/")
        
        # Validate inputs
        if data_path == "Insert path of image folders":
            raise ValueError("Please insert the path of the image folders.")
        if output_name == 'Desired name for LoRA.':
            raise ValueError("Please insert the desired name for LoRA.")
        
        # Validate mixed_precision selection
        if mixed_precision == "bf16":
            if not TrainingUtils.is_torch_bf16_compatible():
                raise ValueError(f"bf16 requires PyTorch >= 1.10.0. Installed version: {torch.__version__}")
            if not TrainingUtils.is_bf16_supported():
                raise ValueError("bf16 training is not supported on this GPU, please use fp16 or no mixed precision.")

        #generates a random seed
        theseed = random.randint(0, 2^32-1)

        # Launch args
        if multi_gpu:
            num_gpus = TrainingUtils.get_available_torch_devices()
            if num_gpus > 1:
                launch_args.append("--multi_gpu")
                launch_args.append(f"--num_processes={num_gpus}")
            else:
                print("Warning: multi_gpu is enabled, but only one GPU is available. Ingoring multi_gpu option.")
                launch_args.append("--num_processes=1")
        else:
            launch_args.append("--num_processes=1")

        launch_args.extend([
            "--num_machines=1",
            f"--mixed_precision={mixed_precision}",
            f"--dynamo_backend={dynamo_backend}",
            f"--num_cpu_threads_per_process=8",
        ])

        # Model type script selection
        if model_type in ["sd1.5", "sd2.0"]:
            train_script_name = "train_network"
        elif model_type == "sdxl":
            train_script_name = "sdxl_train_network"
        elif model_type == "sd3":
            train_script_name = "sd3_train_network"
        elif model_type == "flux":
            train_script_name = "flux_train_network"
        else:
            raise ValueError(f"Error: Unsupported model type: {model_type}")

        # Ext args
        ext_args.extend([
            f"--pretrained_model_name_or_path={ckpt_name}",
            f"--train_data_dir={data_path}",
            f"--resolution={resolution_width},{resolution_height}",
            f"--output_dir={output_dir}",
            f"--output_name={output_name}",
            f"--logging_dir={logging_dir}",
            f"--log_prefix={output_name}"
            f"--train_batch_size={batch_size}",
            f"--max_train_epoches={max_train_epoches}",
            f"--save_every_n_epochs={save_every_n_epochs}",
            "--optimizer_type=AdamW8bit",
            "--learning_rate=1e-4"
            "--unet_lr=1e-4",
            "--text_encoder_lr=1e-5",
            "--lr_scheduler=cosine_with_restarts",
            "--lr_warmup_steps=0",
            "--lr_scheduler_num_cycles=1",
            "--network_module=networks.lora",
            "--network_dim=32",
            "--network_alpha=32",
            "--save_precision=fp16",
            f"--seed={theseed}",
            "--cache_latents",
            "--prior_loss_weight=1", 
            "--max_token_length=225",
            "--caption_extension=.txt",
            "--xformers",
            "keep_tokens"
            "--shuffle_caption",
            "--enable_bucket",
            "--no_metadata",
            "--min_bucket_reso=256",
            "--max_bucket_reso=1584",
            "--save_model_as safetensors",
            "--log_with=tensorboard",
        ])

        # Model-specific ext args
        if train_script_name == "train_network":
            if model_type == "sd1.5":
                ext_args.append(f"--clip_skip={clip_skip}")

            if model_type == "sd2.0":
                ext_args.append("--v2")

        elif train_script_name == "sdxl_train_network":
            print("placeholder")

        pretrained_model = folder_paths.get_full_path("checkpoints", ckpt_name)

        # Get the training script path
        nodespath, sd_script_dir = TrainingUtils.get_train_script(train_script_name)
        if not os.path.exists(nodespath):
            raise FileNotFoundError(f"Training script not found at {nodespath}")

        # Base command
        command = (
            f"{sys.executable} -m accelerate.commands.launch "
        )

        # Add launch args
        command += " ".join(launch_args) + " "

        # Add script path
        command += (f"\"{nodespath}\" ")

        # Add script arguments
        if ext_args:
            command += " " + " ".join(ext_args)

        print(f"Executing command: {command}")
        subprocess.run(command, shell=True, cwd=sd_script_dir)
        print(f"Train finished")
        return ()

# class LoraTraininginComfyAdvanced:
#     def __init__(self):
#         pass
#     
#     @classmethod
#     def INPUT_TYPES(s):
#          return {
#             "required": {
#             "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
#             "model_type": ({"default": "sd1.5", "choices": ["sd1.5", "sd2.0", "sdxl", "sd3", "flux"], "tooltip": "The type of model being used to train your LoRA."},),
#             "networkmodule": (["networks.lora", "lycoris.kohya"], ),
#             "networkdimension": ("INT", {"default": 32, "min":0}),
#             "networkalpha": ("INT", {"default":32, "min":0}),
#             "resolution_width": ("INT", {"default": 512, "step": 64}),
#             "resolution_height": ("INT", {"default": 512, "step": 64}),
#             "data_path": ("STRING", {"default": "Insert path of image folders"}),
# 			"batch_size": ("INT", {"default": 1, "min":1}),
#             "max_train_epoches": ("INT", {"default":10, "min":1}),
#             "save_every_n_epochs": ("INT", {"default":10, "min":1}),
#             "keeptokens": ("INT", {"default":0, "min":0}),
#             "minSNRgamma": ("FLOAT", {"default":0, "min":0, "step":0.1}),
#             "learningrateText": ("FLOAT", {"default":0.0001, "min":0, "step":0.00001}),
#             "learningrateUnet": ("FLOAT", {"default":0.0001, "min":0, "step":0.00001}),
#             "learningRateScheduler": (["cosine_with_restarts", "linear", "cosine", "polynomial", "constant", "constant_with_warmup"], ),
#             "lrRestartCycles": ("INT", {"default":1, "min":1}),
#             "optimizerType": (["AdamW8bit", "Lion8bit", "SGDNesterov8bit", "AdaFactor", "prodigy"], ),
#             "output_name": ("STRING", {"default":'Desired name for LoRA.'}),
#             "algorithm": (["lora","loha","lokr","ia3","dylora", "locon"], ),
#             "networkDropout": ("FLOAT", {"default": 0, "step":0.1}),
#             "clip_skip": ("INT", {"default":2, "min":1}),
#             "output_dir": ("STRING", {"default":'models/loras'}),
#             "mixed_precision": ("STRING", {"default": "no", "choices": ["no", "fp16", "bf16"], "tooltip": "Use mixed precision training. Choose between fp16 and bf16 training. NOTE: bf16 training is only supported on Nvidia Ampere GPUs + PyTorch v1.10 or later!"}),
#             "dynamo_backend": ("STRING", {"default": "no", "choices": ["no", "eager", "aot_eager", "inductor", "aot_ts_nvfuser", "nvprims_nvfuser", "cudagraphs", "ofi", "fx2trt", "onnxrt", "tensorrt", "aot_torchxla_trace_once", "ipex", "tvm"], "tooltip": "Represents a dynamo backend (see https://pytorch.org/docs/stable/torch.compiler.html)."}),
#             "train_unet_only": ("BOOLEAN", {"default": False, "tooltip": "Only the LoRA module related to u-net is valid. It may be a good idea to specify it with fine tuning learning."}),
#             "train_text_encoder_only": ("BOOLEAN", {"default": False, "tooltip": "Only the LoRA module related to text encoder is valid. You may be able to expect a textual inversion effect."}),
#             "multi_gpu": ("BOOLEAN", {"default": False, "tooltip": "Use distributed GPU training."}),
#             #"parameterization": ("BOOLEAN", {"default": False, "tooltip": "Enable v-parameterization training (sd1.5 / sd2.0 only)"}),
#             },
#         }
# 
#     RETURN_TYPES = ()
#     RETURN_NAMES = ()
#     FUNCTION = "loratraining"
#     OUTPUT_NODE = True
#     CATEGORY = "LJRE/LORA"
# 
#     def loratraining(self, ckpt_name, model_type, networkmodule, networkdimension, networkalpha, resolution_width, resolution_height, data_path, batch_size, max_train_epoches, save_every_n_epochs, keeptokens, minSNRgamma, learningrateText, learningrateUnet, learningRateScheduler, lrRestartCycles, optimizerType, output_name, algorithm, networkDropout, clip_skip, output_dir, mixed_precision, dynamo_backend, multi_gpu):
#         #free memory first of all
#         TrainingUtils.free_memory()
#         
#         #transform backslashes into slashes for user convenience.
#         train_data_dir = data_path.replace( "\\", "/")
# 
#         # Validate inputs
#         if data_path == "Insert path of image folders":
#             raise ValueError("Please insert the path of the image folders.")
#         if output_name == 'Desired name for LoRA.':
#             raise ValueError("Please insert the desired name for LoRA.")
# 
#         #generates a random seed
#         theseed = random.randint(0, 2^32-1)
# 
#         # ADVANCED parameters
#         network_module=networkmodule
#         resolution = f"{resolution_width},{resolution_height}"        
#         min_snr_gamma = minSNRgamma
#         optimizer_type = optimizerType
#         algo = algorithm
#         dropout = f"{networkDropout}"
# 
#         # Model-specific args
#         if model_type == "sd2.0":
#             ext_args.append("--v2")
#         elif model_type == "sd1.5":
#             ext_args.append(f"--clip_skip={clip_skip}")
#         elif model_type == "sdxl":
#             train_script_name = "sdxl_train_network"
#         elif model_type == "sd3":
#             train_script_name = "sd3_train_network"
#         elif model_type == "flux":
#             train_script_name = "flux_train_network"
#         else:
#             train_script_name = "train_network"
# 
#         if multi_gpu:
#             ext_args.append("--multi_gpu")
# 
#         if lowram:
#             ext_args.append("--lowram")
# 
#         if is_v2_model:
#             ext_args.append("--v2")
#         else:
#             ext_args.append(f"--clip_skip={clip_skip}")
# 
#         if parameterization:
#             ext_args.append("--v_parameterization")
# 
#         if train_unet_only:
#             ext_args.append("--network_train_unet_only")
# 
#         if train_text_encoder_only:
#             ext_args.append("--network_train_text_encoder_only")
# 
#         if network_weights:
#             ext_args.append(f"--network_weights={network_weights}")
# 
#         if reg_data_dir:
#             ext_args.append(f"--reg_data_dir={reg_data_dir}")
# 
#         if optimizer_type:
#             ext_args.append(f"--optimizer_type={optimizer_type}")
# 
#         if optimizer_type == "DAdaptation":
#             ext_args.append("--optimizer_args")
#             ext_args.append("decouple=True")
# 
#         if network_module == "lycoris.kohya":
#             ext_args.extend([
#                 f"--network_args",
#                 f"conv_dim={conv_dim}",
#                 f"conv_alpha={conv_alpha}",
#                 f"algo={algo}",
#                 f"dropout={dropout}"
#             ])
# 
#         if noise_offset != 0:
#             ext_args.append(f"--noise_offset={noise_offset}")
# 
#         if stop_text_encoder_training != 0:
#             ext_args.append(f"--stop_text_encoder_training={stop_text_encoder_training}")
# 
#         if save_state == 1:
#             ext_args.append("--save_state")
# 
#         if resume:
#             ext_args.append(f"--resume={resume}")
# 
#         if min_snr_gamma != 0:
#             ext_args.append(f"--min_snr_gamma={min_snr_gamma}")
# 
#         if persistent_data_loader_workers:
#             ext_args.append("--persistent_data_loader_workers")
# 
#         if use_wandb == 1:
#             ext_args.append("--log_with=all")
#             if wandb_api_key:
#                 ext_args.append(f"--wandb_api_key={wandb_api_key}")
#             if log_tracker_name:
#                 ext_args.append(f"--log_tracker_name={log_tracker_name}")
#         else:
#             ext_args.append("--log_with=tensorboard")
# 
#         pretrained_model = folder_paths.get_full_path("checkpoints", ckpt_name)
# 
#         # Get the training script path
#         nodespath, sd_script_dir = TrainingUtils.get_train_script(train_script_name)
#         if not os.path.exists(nodespath):
#             raise FileNotFoundError(f"Training script not found at {nodespath}")
#         
#         command = (
#             f"{sys.executable} -m accelerate.commands.launch "
#             f"--num_cpu_threads_per_process=8 \"{nodespath}\" "
#             f"--enable_bucket --pretrained_model_name_or_path={pretrained_model} "
#             f"--train_data_dir=\"{train_data_dir}\" --output_dir=\"{output_dir}\" "
#             f"--logging_dir=\"{logging_dir}\" --log_prefix={output_name} "
#             f"--resolution={resolution} --network_module={networkmodule} "
#             f"--max_train_epochs={max_train_epoches} --learning_rate={lr} "
#             f"--unet_lr={learningrateUnet} --text_encoder_lr={learningrateText} "
#             f"--lr_scheduler={learningRateScheduler} --lr_warmup_steps={lr_warmup_steps} "
#             f"--lr_scheduler_num_cycles={lrRestartCycles} --network_dim={networkdimension} "
#             f"--network_alpha={networkalpha} --output_name={output_name} "
#             f"--train_batch_size={batch_size} --save_every_n_epochs={save_every_n_epochs} "
#             f"--mixed_precision=\"fp16\" --save_precision=\"fp16\" --seed={theseed} "
#             f"--cache_latents --prior_loss_weight=1 --max_token_length=225 "
#             f"--caption_extension=\".txt\" --save_model_as={save_model_as} "
#             f"--min_bucket_reso={min_bucket_reso} --max_bucket_reso={max_bucket_reso} "
#             f"--keep_tokens={keeptokens} --xformers --shuffle_caption "
#         )
# 
#         # Add additional arguments
#         if ext_args:
#             command += " " + " ".join(ext_args)
# 
#         print(f"Executing command: {command}")
#         subprocess.run(command, shell=True, cwd=sd_script_dir)
#         print("Train finished")
#         return ()
   
class TensorboardAccess:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
           
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "opentensorboard"
    OUTPUT_NODE = True
    CATEGORY = "LJRE/LORA"

    def opentensorboard(self):
        command = 'tensorboard --logdir="{logging_dir}"'
        subprocess.Popen(command, shell=True)
        return()