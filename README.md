<div align="center">

<h2><a href="">[NeurIPS 2025 Spotlight] StreamForest: Efficient Online Video Understanding with Persistent Event Memory</a></h2>

[Xiangyu Zeng](https://scholar.google.com/citations?user=jS13DXkAAAAJ&hl=zh-CN), Kefan Qiu, Qingyu Zhang, [Xinhao Li](https://scholar.google.com/citations?user=evR3uR0AAAAJ&hl=zh-CN), Jing Wang, Jiaxin Li, [Ziang Yan](https://scholar.google.com/citations?user=78lx13MAAAAJ&hl=zh-CN), Kun Tian, Meng Tian, Xinhai Zhao, [Yi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Xm2M8UwAAAAJ), and [Limin Wang](https://scholar.google.com/citations?user=HEuN8PcAAAAJ)

</div>

<p align="center">
        🤗 <a href="https://huggingface.co/collections/MCG-NJU/streamforest-and-odvbench-68da5d8a88f3cf453bca0e09">Model & Data</a> &nbsp&nbsp ｜ &nbsp&nbsp🖥️ <a href="">Demo</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/pdf/2509.24871">Paper</a> &nbsp&nbsp | &nbsp&nbsp 🌐 <a href="">Blog</a>
<br>

</p>


## :fire: Updates

- [x] **2025/09/29**: 🎉🎉🎉Our models, data and code have been released.
- [x] **2025/09/19**: 🎉🎉🎉Our paper has been accepted as Spotlight in the proceedings of NeurIPS 2025.

---
## :parrot: Introduction

![introduction](demo/img/introduction.png)

StreamForest is a novel architecture designed for real-time streaming video understanding with Multimodal Large Language Models (MLLMs). Unlike prior approaches that struggle with memory constraints or coarse spatiotemporal reasoning, StreamForest introduces two complementary innovations:

![Architecture of StreamForest](demo/img/architecture.png)

- Persistent Event Memory Forest (PEMF): A hierarchical, event-level memory system that adaptively organizes and compresses long-term video history using temporal distance, content similarity, and merge frequency. This ensures efficient storage without losing critical contextual information.

- Fine-grained Spatiotemporal Window (FSTW): A short-term perception module that captures detailed local spatiotemporal features around the current moment, enabling precise real-time reasoning.

![ODVBench](demo/img/benchmark.png)

For effective deployment and comprehensive evaluation, we contribute:

- OnlineIT, an instruction-tuning dataset tailored for streaming scenarios, improving both present-moment perception and future prediction.

- ODV-Bench, a benchmark for real-time autonomous driving video understanding.

Extensive experiments demonstrate that StreamForest consistently outperforms previous state-of-the-art streaming video MLLMs and achieves performance comparable to leading offline models. Even under extreme visual token compression, the model preserves nearly all of its accuracy, highlighting its robustness, efficiency, and scalability for real-world streaming video applications.


---


## ⚙️ Installation

- Please execute the following commands to clone the StreamForest source code to your local environment:

```
git clone https://github.com/MCG-NJU/StreamForest.git
cd StreamForest
```



- Create a new environment and install the required dependencies:
```
conda create --name StreamForest
conda activate StreamForest
pip install -r requirements.txt
```


- Search for all instances of `/your_local_path_to` within the codebase and replace them with the absolute path of the StreamForest root directory on your local machine.


---

## 📊 Evaluation

We employ lmms-eval for model evaluation.

#### Preparatory Steps

- Environment Setup: Ensure that all dependencies required by [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) are properly installed.


#### Evaluating StreamForest

- You can run the following command to evaluate StreamForest on eight benchmark datasets, including our proposed ODVBench:

```
bash scripts/eval/run_eval.sh
```

#### Evaluating Other Models

###### Predefined Models

- Execute the script `lmms_eval/scripts/eval_internvl2-8B.sh` to initiate the benchmark evaluation.

###### Custom Models

- To add a custom model, please refer to the implementation of `lmms_eval/models/streamforest.py`.

- Alternatively, you may reuse an existing model integration from [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and adapt it to your needs. Ensure that `lmms_eval/models/__init__.py` is updated accordingly to register the new model.

*Note: Since the video data in ODVBench is associated with query timestamps, it is essential to employ `llava/video_utils.py` for proper video loading. As a reference, you may consult the implementation of the `load_video` function in `lmms_eval/models/streamforest.py`. Incorporate this function into your custom model as needed to ensure full compatibility with the lmms_eval evaluation framework.*

---

## ⚡ Training

#### Data Preparation

- OnlineIT-general: Download the dataset from [StreamForest-Annodata](https://huggingface.co/datasets/Lanxingxuan/StreamForest-Annodata) and place it under the `./anno` directory.

- Video & Image Data: Download annotations from [VideoChat-Flash-Training-Data](https://huggingface.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data/tree/main/annotations) and store them in the `./annotations` directory.

- Inspect all YAML files in `./anno/data_list`. For each entry, download the corresponding image or video data specified by the `data_root` field, and replace the URLs with the paths to your local data storage.

#### Training Procedure

Our training pipeline consists of five stages. The first three stages follow the setup of [VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash), while the fourth and fifth stages involves online video fine-tuning.

- To execute the first three stages of offline video pretraining, run
```
bash scripts/train/stage1-init_connector/s1_siglip_tome64_mlp.sh
bash scripts/train/stage2-visual_pretraining/s2_siglip_tome64_mlp.sh
bash scripts/train/stage3-video_sft/s3_siglip_tome16_mlp.sh
```


- To perform the fourth and fifth stages of online video fine-tuning, run:
```
bash scripts/train/stage4-online_ft/s4_siglip_online_dynamic_tree_memory.sh
bash scripts/train/stage5-drive_ft/s5_siglip_online_tree_memory_drive.sh
```

The table below summarizes the configuration of each stage, including the corresponding scripts, datasets, and released checkpoints.

| Stage | ViT | Connector | LLM | Script | Data | Ckpt |
|:------:|:---:|:----------:|:---:|:--------:|:------:|:------:|
| 1 | :snowflake: | :fire: | :snowflake: | [click me](https://github.com/MCG-NJU/StreamForest/blob/main/scripts/train/stage1-init_connector/s1_siglip_tome64_mlp.sh) |  [click me](https://huggingface.co/datasets/MCG-NJU/StreamForest-Annodata/blob/main/data_list/stage1_init_connector_iv1m.yaml) | - |
| 2 | :fire: | :fire: | :fire: | [click me](https://github.com/MCG-NJU/StreamForest/blob/main/scripts/train/stage2-visual_pretraining/s2_siglip_tome64_mlp.sh) |  [click me](https://huggingface.co/datasets/MCG-NJU/StreamForest-Annodata/blob/main/data_list/stage2_short_pretrain_iv6m.yaml) | - |
| 3 | :fire: | :fire: | :fire: | [click me](https://github.com/MCG-NJU/StreamForest/blob/main/scripts/train/stage3-video_sft/s3_siglip_tome16_mlp.sh) |  [click me](https://huggingface.co/datasets/MCG-NJU/StreamForest-Annodata/blob/main/data_list/stage3_short-long_mix_sft.yaml) | [StreamForest-Pretrain-Qwen2-7B](https://huggingface.co/MCG-NJU/StreamForest-Pretrain-Qwen2-7B) |
| 4 | :snowflake: | :fire: | :fire: | [click me](https://github.com/MCG-NJU/StreamForest/blob/main/scripts/train/stage4-online_ft/s4_siglip_online_dynamic_tree_memory.sh) |  [click me](https://huggingface.co/datasets/MCG-NJU/StreamForest-Annodata/blob/main/data_list/stage4_online_sft.yaml) | [StreamForest-Qwen2-7B](https://huggingface.co/MCG-NJU/StreamForest-Qwen2-7B) |
| 5 | :snowflake: | :fire: | :fire: | [click me](https://github.com/MCG-NJU/StreamForest/blob/main/scripts/train/stage5-drive_ft/s5_siglip_online_tree_memory_drive.sh) |  [click me](https://github.com/MCG-NJU/StreamForest/blob/main/scripts/train/stage5-drive_ft/s5_siglip_online_tree_memory_drive.sh) | [StreamForest-Drive-Qwen2-7B](https://huggingface.co/MCG-NJU/StreamForest-Drive-Qwen2-7B) |

Based on the table above, you can quickly start the 4th and 5th stages of online fine-tuning using the provided checkpoints.


---

## :page_facing_up: Citation

```BibTeX
@misc{zeng2025streamforest,
      title={StreamForest: Efficient Online Video Understanding with Persistent Event Memory}, 
      author={Xiangyu Zeng and Kefan Qiu and Qingyu Zhang and Xinhao Li and Jing Wang and Jiaxin Li and Ziang Yan and Kun Tian and Meng Tian and Xinhai Zhao and Yi Wang and Limin Wang},
      year={2025},
      eprint={2509.24871},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.24871}, 
}
```

---

## :dizzy: Acknowledgement

Thanks to the open source of the following projects:
- [VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash): Efficient architecture, data and training methods for offline video understanding.
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): Original code framework and image and video training data.
- [ToMe](https://github.com/facebookresearch/ToMe): Efficient token merging and compression method.
- [VideoChat-Online](https://github.com/MCG-NJU/VideoChat-Online): Inspiring memory mechanism design and online training data.
- [StreamChat](https://github.com/hmxiong/StreamChat): Design of a training-free tree-structured hierarchical memory mechanism.
- [MovieChat](https://github.com/rese1f/MovieChat): Pioneer of long video memory mechanisms.
- [VideoLLM-Online](https://github.com/showlab/videollm-online): Pioneer of online video understanding methods.
