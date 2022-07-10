# SIIM-ISIC Melanoma Classification
**Identify Melanoma In Lesion Images**

---

### Competition Overview  
https://www.kaggle.com/c/siim-isic-melanoma-classification  
### Competition Leaderboard
https://www.kaggle.com/c/siim-isic-melanoma-classification/leaderboard
> Secured a **Solo Silver Medal** (103<sup>rd</sup> amongst 3308 teams) on kaggle private leaderboard

<a href="https://www.kaggle.com/c/siim-isic-melanoma-classification/leaderboard"><img src="https://github.com/sajwankit/SIIM-ISIC-Melanoma-Classification/blob/main/images/kaggle_lb.png" align="center" height="200" width="720" ></a> 

---

## Data

* [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification/data)
* [ISIC 2019 TFRecords 256x256](https://www.kaggle.com/cdeotte/isic2019-256x256)
* [ISIC 2019 TFRecords 384x384](https://www.kaggle.com/cdeotte/isic2019-384x384)
* [ISIC 2019 TFRecords 512x512](https://www.kaggle.com/cdeotte/isic2019-512x512)
* [ISIC 2019 TFRecords 768x768](https://www.kaggle.com/cdeotte/isic2019-768x768)
* [Melanoma TFRecords 256x256](https://www.kaggle.com/cdeotte/melanoma-256x256)
* [Melanoma TFRecords 384x384](https://www.kaggle.com/cdeotte/melanoma-384x384)
* [Melanoma TFRecords 512x512](https://www.kaggle.com/cdeotte/melanoma-512x512)
* [Melanoma TFRecords 768x768](https://www.kaggle.com/cdeotte/melanoma-768x768)
* [Melanoma hair](https://www.kaggle.com/nroman/melanoma-hairs)

---

## Solution Overview
 
* Built a deep learning architecture: an ensemble of **convolutional neural network models** with goal to be robust against **overfitting**
* Centered on the **EfficientNet** family of CNNs,
specifically EfficientNet-B6 and B7
* Ensembling models using different sized images, this helps the architecture to learn different patterns from varying sized images, boosting CV and LB score

### Data augmentation and image processing
* Rotation, Sheer, Zoom, Shift Augmentation 
* Horizontal flip, hue, saturation, contrast, brightness
* Dropout
* Advanced hair augmentation

### Cross validation strategy: Triple Stratified Leak-Free KFold CV
> A robust local cross-validation was important in the competition, the dataset was highly imbalanced with only ~78 melanoma examples in the public leaderboard

**Stratify 1 - Isolate Patients**  
A single patient can have multiple images. All images from one patient are fully contained inside a single fold.  

**Stratify 2 - Balance Malignant Images**  
The entire dataset has 1.8% malignant images. Each fold contains 1.8% malignant images.  

**Stratify 3 - Balance Patient Count Distribution**  
Some patients have as many as 115 images and some patients have as few as 2 images. When isolating patients into folds, each record has an equal number of patients.

## Experiment Result (For a selected setting)
<img src="https://github.com/sajwankit/SIIM-ISIC-Melanoma-Classification/blob/main/images/exp_result.png" align="center" height="360" width="720" >
