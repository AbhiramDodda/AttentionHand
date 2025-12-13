# AttentionHand: Text-driven Controllable Hand Image Generation for 3D Hand Reconstruction in the Wild

# This repository is a modified work of https://github.com/redorangeyellowy/AttentionHand
## Now the code supports torch 2.6 version
![introduction](./thumbnail.png)

## Install
```
pip install -r requirements.txt
```


## Inference

1. Download the pre-trained model `attentionhand.ckpt` from [here](https://drive.google.com/drive/folders/1YC-eaTPW5ZtkWQe3y5XXw1-jndmQ-NlO?usp=drive_link).
3. Set your own modalities in `samples`. (But, we provide some samples for fast implementation.)
4. Put samples and downloaded weight as follows.
```
${ROOT}
|-- samples
|   |-- mesh
|   |   |-- ...
|   |-- text
|   |   |-- ...
|   |-- modalities.json
|-- weights
|   |-- attentionhand.ckpt
```
4. Run `inference.py`.

## Train from scratch

1. Download initial model `sd15_ini.ckpt` from [here](https://drive.google.com/drive/folders/1YC-eaTPW5ZtkWQe3y5XXw1-jndmQ-NlO?usp=drive_link).
2. Download pre-processed dataset `dataset.tar.gz` from [here](https://drive.google.com/drive/folders/1YC-eaTPW5ZtkWQe3y5XXw1-jndmQ-NlO?usp=drive_link).
3. Put downloaded weight and dataset as follows.
```
${ROOT}
|-- data
|   |-- mesh
|   |   |-- ...
|   |-- rgb
|   |   |-- ...
|   |-- text
|   |   |-- ...
|   |-- modalities.json
|-- weights
|   |-- sd15_ini.ckpt
```
4. Run `train.py`.

## Fine-tuning

1. Download the pre-trained model `attentionhand.ckpt` from [here](https://drive.google.com/drive/folders/1YC-eaTPW5ZtkWQe3y5XXw1-jndmQ-NlO?usp=drive_link).
2. Set your own modalities in `data` as `datasets.tar.gz` in [here](https://drive.google.com/drive/folders/1YC-eaTPW5ZtkWQe3y5XXw1-jndmQ-NlO?usp=drive_link).
3. Put downloaded weight and dataset as follows.
```
${ROOT}
|-- data
|   |-- mesh
|   |   |-- ...
|   |-- rgb
|   |   |-- ...
|   |-- text
|   |   |-- ...
|   |-- modalities.json
|-- weights
|   |-- attentionhand.ckpt
```
4. Change `resume_path` in `train.py` to `weights/attentionhand.ckpt`.
5. Run `train.py`.


## License and Citation <a name="license-and-citation"></a>

All assets and code are under the [license](./LICENSE) unless specified otherwise.

If this work is helpful for your research, please consider citing the following BibTeX entry.

``` bibtex
@article{park2024attentionhand,
  author  = {Park, Junho and Kong, Kyeongbo and Kang, Suk-Ju},
  title   = {AttentionHand: Text-driven Controllable Hand Image Generation for 3D Hand Reconstruction in the Wild},
  journal = {European Conference on Computer Vision},
  year    = {2024},
}
```
