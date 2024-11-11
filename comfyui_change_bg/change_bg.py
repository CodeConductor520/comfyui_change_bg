from .briarmbg import BriaRMBG
import os
import torch
from typing import Optional

from torchvision import transforms
from PIL import Image
from .utils import tensor2pil,pil2tensor
from .image_merge import remove_background, merge_bg_fg


current_directory = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

class Load_model:
    def __init__(self):
        pass

    CATEGOPY = "my custom plugin:A_example"

    @classmethod
    def INPUT_TYPES(s):
        return {

        }

    OUTPUT_NODE = True

    RETURN_TYPES = ("MODEL",)

    RETURN_NAMES = ("model",)

    FUNCTION = "load_model"

    def load_model(self,):
        net = BriaRMBG()
        model_path = os.path.join(current_directory, "models/model.pth")
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()
        return ([net][0],)


class Remove_bg:
    def __init__(self):
        pass

    CATEGOPY = "my custom plugin:B_example"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "IMAGE": ("IMAGE",),
                "bg": (["white","green", "red","yellow","自定义背景"],{"default": "white"}),
            },
            "optional": {"bg_image": ("IMAGE", {"default": None}),
            }
        }

    OUTPUT_NODE = True

    RETURN_TYPES = ("IMAGE",)

    RETURN_NAMES = ("image",)

    FUNCTION = "remove_bg"

    def remove_bg(self, model, IMAGE, bg, bg_image:Optional[torch.tensor]=None):
        if IMAGE is not None:
            IMAGE = IMAGE.permute(0, 3, 1, 2)
            transform = transforms.ToPILImage()
            IMAGE = transform(IMAGE.squeeze(0))
            foreground = IMAGE.convert("RGB")
            tensor_fg = pil2tensor(foreground)
            new_ims, mask = remove_background(model, tensor_fg)
            fg = tensor2pil(new_ims)
            if bg is not None:
                if bg == "green":
                    bg_path = os.path.join(current_directory, "bg_tmplate/green.png")
                    background = Image.open(bg_path).convert("RGB")
                elif bg == "white":
                    bg_path = os.path.join(current_directory, "bg_tmplate/white.png")
                    background = Image.open(bg_path).convert("RGB")
                elif bg == "red":
                    bg_path = os.path.join(current_directory, "bg_tmplate/red.png")
                    background = Image.open(bg_path).convert("RGB")
                elif bg == "yellow":
                    bg_path = os.path.join(current_directory, "bg_tmplate/yellow.png")
                    background = Image.open(bg_path).convert("RGB")
                elif bg == "自定义背景":
                    bg_image = bg_image.permute(0, 3, 1, 2)
                    transform = transforms.ToPILImage()
                    bg_image = transform(bg_image.squeeze(0))
                    background = bg_image.convert("RGB")
                else:
                    pass
                # 合并前景背景图像
                pil_res = merge_bg_fg(fg, mask, background)
            else:
                # fg.save("fg_model.png")
                pil_res = fg
            image_tensor = transforms.ToTensor()(pil_res)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.permute(0, 2, 3, 1)
        return (image_tensor,)



