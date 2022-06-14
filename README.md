

<div align="center">

# Hybrid Spatio-Temporal Transformer Network for Predicting Ischemic Stroke Lesion Outcomes from 4D CT Perfusion Imaging
  
</div>

Keras implementation of our method for MICCAI2022 paper: "Hybrid Spatio-Temporal Transformer Network for Predicting Ischemic Stroke Lesion Outcomes from 4D CT Perfusion Imaging".

## Abstract
Predicting the follow-up infarct lesion from baseline spatio-temporal (4D) Computed Tomography Perfusion (CTP) imaging is essential for the diagnosis and management of acute ischemic stroke (AIS) patients. However, due to their noisy appearance and high dimensionality, it has been technically challenging to directly use 4D CTP images for this task. Thus, CTP datasets are usually post-processed to generate parameter maps that describe the perfusion situation. Existing deep learning-based methods mainly utilize these maps to make lesion outcome predictions, which may only provide a limited understanding of the spatio-temporal details available in the raw 4D CTP. While a few efforts have been made to incorporate raw 4D CTP data, a more effective spatio-temporal integration strategy is still needed. Inspired by the success of Transformer models in medical image analysis, this paper presents a novel hybrid CNN-Transformer framework that directly maps 4D CTP datasets to stroke lesion outcome predictions. This hybrid prediction strategy enables an efficient modeling of spatio-temporal information, eliminating the need for post-processing steps and hence increasing the robustness of the method. Experiments on a multicenter CTP dataset of 45 AIS patients demonstrate the superiority of the proposed method over the state-of-the-art.

<p align="center">
<img src="https://github.com/kimberly-amador/Spatiotemporal-CNN-Transformer/figures/architecture.png" width="550">
</p>


## Usage

#### Installation

Recommended environment:

- Python 3.8.1
- TensorFlow GPU 2.4.1
- CUDA 11.0.2 
- cuDNN 8.0.4.30

To install the dependencies, run:

```shell
$ git clone https://github.com/kimberly-amador/Spatiotemporal-CNN-Transformer
$ cd Spatiotemporal-CNN-Transformer
$ pip install -r requirements.txt
```

#### Data Preparation
1. Preprocess the dataset. The model takes images of size 384 x 256.
2. Save the preprocessed images as numpy arrays. Use the name convention 'patientID_preprocessed.npz'. Then move all images into a folder. 
3. Create a patient dictionary. This should be a pickle file containing a dict as follows:

```python
partition = {
    'train': {
        's_patientID',
        's_patientID',
        ...
    },
    'val': {
        's_patientID',
        's_patientID',
        ...
    }
    'test': {
        's_patientID',
        's_patientID',
        ...
    }
}
```

#### Train Model

1. Modify the model configuration. The default configuration parameters are in `./model/config_file.py`.
2. Run `python main.py` to start training the model.

## Citation
If you find this code and paper useful for your research, please cite the paper:

```
```

## Acknowledgement
Part of the code is adapted from open-source codebase and original implementations of algorithms, 
we thank these authors for their fantastic and efficient codebase:
* CycleGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
