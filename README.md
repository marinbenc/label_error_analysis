## Impact of Skin Lesion Segmentation Labeling Errors on Convolutional Neural Networks

**TODO** Add link to paper.

## Requirements

- PyTorch  w/ CUDA support
- Conda: `conda env create -f environment.yml; conda activate pt2` (might include a lot of libraries that are not used in this specific project)
- Pip: `pip install -r requirements.txt`

## Data Preparatation

The data links are below. Each dataset can be downloaded and extracted into `data/downloaded_data`, an example of the folder structure:

```
data/downloaded_data
├── isic
│   ├── ISIC2018_Task1-2_Test_Input
│   ├── ISIC2018_Task1-2_Training_Input
│   ├── ISIC2018_Task1-2_Validation_Input
│   ├── ISIC2018_Task1_Test_GroundTruth
│   ├── ISIC2018_Task1_Training_GroundTruth
│   └── ISIC2018_Task1_Validation_GroundTruth
```

To split and prepare the segmentation images, run:

```
cd data
python make_seg_dataset.py
```

### Segmentation Datasets

**ISIC 2018**

https://challenge.isic-archive.com/data/#2018

[1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).

Good starting parameters:

```shell
python train.py --model_type lesion_seg --dataset seg_isic --log_name isic_unet --overwrite --lr 1e-4 --batch-size 32

python train.py --dataset seg_isic --log_name isic_unet_1e-4_8 --overwrite --lr 1e-4 --batch-size 8 --workers 0
```

## Model Training

To train a model, run `python train.py -h` to see the available options.

## Model Testing

Run `python test.py -h` to see options. Test results will be saved in `predictions/LOG_NAME/metrics_TEST_DATASET_NAME.csv`.

Example

```shell
python test.py seg_isic LOG_NAME --dataset_folder valid
```
