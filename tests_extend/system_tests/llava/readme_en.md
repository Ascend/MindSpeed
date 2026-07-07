# Background

As an AI model capable of processing and understanding multimodal data (such as text, images, and audio), the Large Language and Vision Assistant (LLaVA) series of multimodal large models demonstrates powerful expressiveness and broad app prospects. The official Megatron repository released the LLaVA entry file `pretrain_vlm.py` as early as version 060, and MindSpeed also needs to continuously make adaptations.

## Procedure

```bash
cd Megatron-LM/
ls -n ../MindSpeed/mindSpeed ./mindSpeed
cp ../MindSpeed/tests_extend/llava/pretrain_llava.sh ./pretrain_llava.sh
bash pretrain_llava.sh
```
