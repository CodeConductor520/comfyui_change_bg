from PIL import Image
import base64
from io import BytesIO
import json
import time
from .log import logger_
import numpy as np
import torch

def pil2bs64(image):

    # 将图像转换为字节流
    buffered = BytesIO()
    image.save(buffered, format="PNG")

    # 获取字节流内容
    image_byte = buffered.getvalue()

    # 将字节流进行base64编码
    base64_encoded = base64.b64encode(image_byte).decode('utf-8')
    return base64_encoded

def bs642pil(base64_string):
    # 从base64字符串解码获取字节流
    # base64_string = "your_base64_encoded_string_here"
    image_data = base64.b64decode(base64_string)

    # 将字节流转为BytesIO对象
    image_buffer = BytesIO(image_data)

    # 使用PIL打开图像
    image = Image.open(image_buffer)
    return image

def parse_json(path):
    with open(path, 'r') as json_file:
        params_dic = json.load(json_file)
    return params_dic


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


#PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        # log = logger("befaker_fast", "befaker_fast.log")
        logger_.logger.info(f"{func.__name__} cost time: {execution_time:.4f} s")
        return result
    return wrapper

def video_to_base64(file_path):
    with open(file_path, "rb") as video_file:
        encoded_string = base64.b64encode(video_file.read()).decode("utf-8")

    return encoded_string

def base64_to_video(bs64_video, video_path):
    # video_path = parent_path + str(uuid.uuid1()) + '.mp4'
    with open(video_path, "wb") as f:
        f.write(base64.b64decode(bs64_video))
    return video_path
