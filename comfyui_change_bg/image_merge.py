import numpy as np
import torch
import os
from PIL import Image

from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from .utils import tensor2pil,pil2tensor,timing_decorator

from .utils import bs642pil

#使用logger_
#logger_.logger.info,logger_.logger.critical等

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


def remove_background(rmbgmodel, image):
    processed_images = []
    processed_masks = []

    for image in image:
        orig_image = tensor2pil(image)
        w, h = orig_image.size
        image = resize_image(orig_image)
        im_np = np.array(image)
        im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = torch.unsqueeze(im_tensor, 0)
        im_tensor = torch.divide(im_tensor, 255.0)
        im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        if torch.cuda.is_available():
            im_tensor = im_tensor.cuda()

        result = rmbgmodel(im_tensor)
        result = torch.squeeze(F.interpolate(result[0][0], size=(h, w), mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_array = (result * 255).cpu().data.numpy().astype(np.uint8)
        pil_im = Image.fromarray(np.squeeze(im_array))
        new_im = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
        new_im.paste(orig_image, mask=pil_im)

        new_im_tensor = pil2tensor(new_im)  # 将PIL图像转换为Tensor
        pil_im_tensor = pil2tensor(pil_im)  # 同上

        processed_images.append(new_im_tensor)
        processed_masks.append(pil_im_tensor)

    new_ims = torch.cat(processed_images, dim=0)
    new_masks = torch.cat(processed_masks, dim=0)

    return new_ims, new_masks

def mask_to_image(masks):
    if masks.ndim == 4:
        # If input has shape [N, C, H, W]
        tensor = masks.permute(0, 2, 3, 1)
        tensor_rgb = torch.cat([tensor] * 3, dim=-1)
        return (tensor_rgb,)
    elif masks.ndim == 3:
        # If Input has shape [N, H, W]
        tensor = masks.unsqueeze(-1)
        tensor_rgb = torch.cat([tensor] * 3, dim=-1)
        return (tensor_rgb,)
    elif masks.ndim == 2:
        # If input has shape [H, W]
        tensor = masks.unsqueeze(0).unsqueeze(-1)
        tensor_rgb = torch.cat([tensor] * 3, dim=-1)
        return (tensor_rgb,)
    else:
        print("Invalid input shape. Expected [N, C, H, W] or [H, W].")
        return masks


#参考链接
#https://blog.csdn.net/hahabeibei123456789/article/details/101512632
# foreground = cv2.imread("fg.png")

def merge_bg_fg(pil_fg, mask, background):
    """
    函数功能：将使用模型推理后得到的前景图像和mask，与背景进行融合
    pil_fg:PIL格式的前景图像
    mask:tensor
    background:PIL格式的背景图像
    """
    mask = mask_to_image(mask)
    alpha = np.array(np.squeeze(mask[0]))
    foreground = np.array(pil_fg.convert("RGB"))
    # background = Image.open(bg_path).convert("RGB")

    # 使用邻近插值进行图像缩放
    resized_background = np.array(background.resize((foreground.shape[1], foreground.shape[0]), Image.NEAREST))

    # Convert uint8 to float
    foreground = foreground.astype(np.float32)
    background = resized_background.astype(np.float32)
    # Multiply the foreground with the alpha matte
    foreground = np.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    resized_background = np.multiply(1.0 - alpha, resized_background)

    # Add the masked foreground and background.
    outImage = np.add(foreground, resized_background)
    outImage = outImage.astype(np.uint8)

    pil_res = Image.fromarray(outImage)
    # res.save("res.png")
    return pil_res

@timing_decorator
def remove_bg(net, pil_fg, bg=None):
    """
    net是模型
    fg_path是前景路径
    bg是数字1,2或者路径
    """
    foreground = pil_fg.convert("RGB")
    tensor_fg = pil2tensor(foreground)
    new_ims, mask = remove_background(net, tensor_fg)
    fg = tensor2pil(new_ims)
    if bg is not None:
        if bg == 1:
            bg_path = "bg_tmplate/blue.png"
            background = Image.open(bg_path).convert("RGB")
        elif bg == 2:
            bg_path = "bg_tmplate/white.png"
            background = Image.open(bg_path).convert("RGB")
        elif type(bg) == str:
            background = bs642pil(bg).convert("RGB")

        else:
            pass
        #合并前景背景图像
        pil_res = merge_bg_fg(fg, mask, background)
    else:
        # fg.save("fg_model.png")
        pil_res = fg
    return pil_res


@timing_decorator
def remove_bg1(net, pil_fg, bg=None):
    """
    net是模型
    fg_path是前景路径
    bg是数字1,2或者路径
    """
    foreground = pil_fg.convert("RGB")
    tensor_fg = pil2tensor(foreground)
    new_ims, mask = remove_background(net, tensor_fg)
    fg = tensor2pil(new_ims)
    if bg is not None:
        if bg == 1:
            bg_path = "bg_tmplate/blue.png"
            background = Image.open(bg_path).convert("RGB")
        elif bg == 2:
            bg_path = "bg_tmplate/white.png"
            background = Image.open(bg_path).convert("RGB")
        elif type(bg) == str:
            background = bs642pil(bg).convert("RGB")

        else:
            pass
        #合并前景背景图像
        pil_res = merge_bg_fg(fg, mask, background)
    else:
        # fg.save("fg_model.png")
        pil_res = fg
        pil_mask = tensor2pil(mask)
    return pil_res, pil_mask


# remove_bg_worker(net,"src.jpg", "bg.png")

if __name__ == '__main__':
    from load_model import remove_net
    dir_path = "/home/meta/comfyui_latest_4/111/frames1"
    # save_png_path = "/home/meta/comfyui_latest_4/111/png_path"
    # save_mask_path = "/home/meta/comfyui_latest_4/111/mask_path"
    # for name in os.listdir(dir_path):
    #     png_path = os.path.join(dir_path,name)
    #     pil_fg = Image.open(png_path)
    #     pil_img, pil_mask = remove_bg1(remove_net, pil_fg)
    #     pil_img.save(os.path.join(save_png_path, name))
    #     pil_mask.save(os.path.join(save_mask_path, name))
        
        
    #resize
    # resize_path = "/home/meta/comfyui_latest_4/111/frames1"
    # save_resize_path = "/home/meta/comfyui_latest_4/111/resize"
    # for name in os.listdir(dir_path):
    #     png_path = os.path.join(dir_path, name)
    #     pil_fg = Image.open(png_path)
    #     pil_fg.resize((1280, 720), Image.NEAREST)
    #     pil_fg.save(os.path.join(save_resize_path, name))



    #裁剪图像
    # crop_path = "/home/meta/comfyui_latest_4/111/birme-540x1280"
    # save_crop_path = "/home/meta/comfyui_latest_4/111/crop_path"
    # for name in os.listdir(crop_path):
    #     png_path = os.path.join(crop_path, name)
    #     pil_fg = Image.open(png_path)
    #     cropped_im = pil_fg.crop((0, 320, 540, 1280))
    #     cropped_im.save(os.path.join(save_crop_path, name))

    # resize
    resize_path = "/home/meta/comfyui_latest_4/111/crop_path"
    save_resize_path = "/home/meta/comfyui_latest_4/111/resize_720_1280"
    for name in os.listdir(resize_path):
        png_path = os.path.join(resize_path, name)
        pil_fg = Image.open(png_path)
        pil_fg.resize((1280, 720), Image.NEAREST)
        pil_fg.save(os.path.join(save_resize_path, name))