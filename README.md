<div align="center">

<h2><a href="">[NeurIPS 2025 Spotlight] StreamForest: Efficient Online Video Understanding with Persistent Event Memory</a></h2>

[Xiangyu Zeng](https://scholar.google.com/citations?user=jS13DXkAAAAJ&hl=zh-CN), Kefan Qiu, Qingyu Zhang, [Xinhao Li](https://scholar.google.com/citations?user=evR3uR0AAAAJ&hl=zh-CN), Jing Wang, Jiaxin Li, [Ziang Yan](https://scholar.google.com/citations?user=78lx13MAAAAJ&hl=zh-CN), Kun Tian, Meng Tian, Xinhai Zhao, [Yi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Xm2M8UwAAAAJ), and [Limin Wang](https://scholar.google.com/citations?user=HEuN8PcAAAAJ)

</div>




## :fire: Updates

- [x] **2025/??/??**: 🎉🎉🎉Our models and code are being prepared for open source release.



---


## Installation

- Please execute the following commands to clone the StreamForest source code to your local environment:

```
git clone https://github.com/LanXingXuan/StreamForest_before_opensource.git
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

## Evaluation

We employ lmms-eval for model evaluation.

#### Preparatory Steps

- Environment Setup: Ensure that all dependencies required by [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) are properly installed.


#### Evaluating StreamForest

- You can run the following command to evaluate StreamForest on eight benchmark datasets, including our proposed ODVBench:

```
bash scripts/eval/eval_all.sh
```

#### Evaluating Other Models

###### Predefined Models

- Execute the script `lmms_eval/scripts/eval_internvl2-8B.sh` to initiate the benchmark evaluation.

###### Custom Models

- To add a custom model, please refer to the implementation of `lmms_eval/models/streamforest.py`.

- Alternatively, you may reuse an existing model integration from [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and adapt it to your needs. Ensure that `lmms_eval/models/__init__.py` is updated accordingly to register the new model.

*Note: Since the video data in ODVBench is associated with query timestamps, it is essential to employ `llava/video_utils.py` for proper video loading. As a reference, you may consult the implementation of the `load_video` function in `lmms_eval/models/streamforest.py`. Incorporate this function into your custom model as needed to ensure full compatibility with the lmms_eval evaluation framework.*

---

## Training

#### Data Preparation

- OnlineIT-general: Download the dataset from <OnlineIT_URL_HERE> and place it under the `./anno` directory.

- Video & Image Data: Download annotations from [VideoChat-Flash-Training-Data](https://huggingface.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data/tree/main/annotations) and store them in the `./annotations` directory.

- Inspect all YAML files in `./anno/data_list`. For each entry, download the corresponding image or video data specified by the `data_root` field, and replace the URLs with the paths to your local data storage.

#### Training Procedure

Our training pipeline consists of four stages. The first three stages follow the setup of VideoChat-Flash, while the fourth stage involves online video understanding fine-tuning.

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