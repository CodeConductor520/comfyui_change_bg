项目介绍

1.适配ComfyUI的可实现更换图片背景项目

2.可利用提供的纯色模板背景或上传自定义背景更换图片背景



节点说明



![企业微信截图_17312936702358](https://github.com/user-attachments/assets/5cc60dd4-f0fd-4431-b2fe-92a72199e0a8)
1.Load_model:加载模型

2.change_bg：改变图片背景，可在“bg”的下拉框选择纯色模板背景，或自定义上传背景

安装

1.手动安装

cd custom_nodes

git clone 

2.模型下载地址：https://huggingface.co/dreamer521/change_bg

3.在本文件夹下，新建“models”目录，将下载好的模型放在该文件夹下

工作流

[change_bg.json](https://github.com/user-attachments/files/17694685/change_bg.json)
