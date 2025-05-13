## 背景
作为一种能够处理和理解多种模态数据（如文本、图像、声音等）的人工智能模型，LLAVA系列多模态大模型具有强大的表现力和广泛的应用前景。Megatron官方仓早在060版本便发布llava大模型入口pretrain_vlm.py文件，Mindspeed也需不断做出适配。
## 操作步骤
```
cd Megatron-LM/
ls -n ../MindSpeed/mindSpeed ./mindSpeed
cp ../MindSpeed/tests_extend/llava/pretrain_llava.sh ./pretrain_llava.sh
bash pretrain_llava.sh
```
