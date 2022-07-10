# SIIM-ISIC Melanoma Classification
## Identify Melanoma In Lesion Images

---

### Competition Overview  
https://www.kaggle.com/c/siim-isic-melanoma-classification  
### Competition Leaderboard
https://www.kaggle.com/c/siim-isic-melanoma-classification/leaderboard
* Secured a **Solo Silver Medal** (103<sup>rd</sup> amongst 3308 teams) on the private lb  
<a href="https://www.kaggle.com/c/siim-isic-melanoma-classification/leaderboard"><img src="https://github.com/sajwankit/SIIM-ISIC-Melanoma-Classification/blob/siiim/images/kaggle_lb.png" align="center" height="360" width="720" ></a> 

---

### Data

* [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification/data)
* [ISIC 2019 TFRecords 256x256](https://www.kaggle.com/cdeotte/isic2019-256x256)
* [ISIC 2019 TFRecords 384x384](https://www.kaggle.com/cdeotte/isic2019-384x384)
* [ISIC 2019 TFRecords 512x512](https://www.kaggle.com/cdeotte/isic2019-512x512)
* [ISIC 2019 TFRecords 768x768](https://www.kaggle.com/cdeotte/isic2019-768x768)
* [Melanoma TFRecords 256x256](https://www.kaggle.com/cdeotte/melanoma-256x256)
* [Melanoma TFRecords 384x384](https://www.kaggle.com/cdeotte/melanoma-384x384)
* [Melanoma TFRecords 512x512](https://www.kaggle.com/cdeotte/melanoma-512x512)
* [Melanoma TFRecords 768x768](https://www.kaggle.com/cdeotte/melanoma-768x768)
* [Melanoma_hairs](https://www.kaggle.com/nroman/melanoma-hairs)

---

### Solution Overview
 
Built an ensemble of convolutional neural network models with a goal to be more robust against overfitting.  
Ensembling models using different sized images increases CV and LB.
 I also used cosine annealing with warm restarts as the learning rate scheduler. I leveraged the
ISIC 2019 training data, which provided extra melanoma examples to learn from and prevented
overfitting to the 2020 data. 
Trusting CV and diversity was important to achieve a good result on private LB. 

Data Augmentation and Image Processing
We considered a wide range of augmentations in different models:

horizontal/vertical flips
rotation
circular crop (a.k.a microscope augmentation)
dropout
zoom/brightness adjustment
color normalization
On the TTA stage, we used the same augmentations as on the training stage and usually varied the number of augmentations between 10 and 20.

Models
My solution was centered on the EfficientNet family of convolutional neural networks,
specifically EfficientNet-B6 and B7
We were focusing on EfficientNet models and trained a variety of architectures on different image sizes. Most of the final models were using B5 with 512x512 images. We experimented with adding attention layers and meta-data for some models and played with the learning rates and label smoothing parameters. We also explored Densenet and Inception architectures but observed worse performance.

Some models were initialized from the pre-trained weights. To get the pre-trained weights, we fitted CNNs on the complete train + test + external data to predict anatom_site_general_challenge as a surrogate label. Initializing from the pre-trained weights instead of the Imagenet weights improved our CV. I set up a notebook demonstrating the pre-training approach.

The best single model was EN-B5 trained on 384x384 with attention and meta features, which achieved private LB of 0.9380.

**Triple Stratified Leak-Free KFold CV**  
**Stratify 1 - Isolate Patients**  
A single patient can have multiple images. Now all images from one patient are fully contained inside a single TFRecord. This prevents leakage during cross validation

**Stratify 2 - Balance Malignant Images**  
The entire dataset has 1.8% malignant images. Each TFRecord contains 1.8% malignant images. This makes validation score more reliable.

**Stratify 3 - Balance Patient Count Distribution**  
Some patients have as many as 115 images and some patients have as few as 2 images. When isolating patients into TFRecords, each record has an equal number of patients with 115 images, with 100, with 70, with 50, with 20, with 10, with 5, with 2, etc. This makes validation more reliable.

Below are 15 plots showing the histogram of patients and their counts within each TFRecord.
<a href="url"><img src="https://github.com/sajwankit/SIIM-ISIC-Melanoma-Classification/blob/siiim/images/stratified_data.png" align="center" height="360" width="720" ></a>  
  

    
Identifying and Remove Duplicates  
The above 3 stratifications make a more reliable CV and prevent leakage during cross validation. Additionally it has been published by the competition host here that the training data contains 434 duplicate images. If one image is inside your training fold and the duplicate is in your validation fold, this causes a leak which jeopardizes the reliability of your CV. These 434 duplicate images have been removed from my TFRecords to prevent leakage.

Cross Validation
The acronym CV refers to cross validation. We start with the full training dataset picture below to the far left. Next, we divide it into 5 subsets, called Fold 1, Fold 2, Fold 3, Fold 4, Fold 5. Then we train 5 models. We train our first model using data from Folds 2-5 and predict Fold 1. Next we train model 2 using Folds 1, 3, 4, 5 and predict 2. Next 1, 2, 4, 5 and predict 3, etc etc.

Afterward we have predictions for every training image. This compete set of predictions is called OOF, "out of fold" predictions. It is a good practice to save these predictions for every model you build during a competition as oof.csv.

The CV score (or OOF AUC) is then calculated with OOF_AUC = roc_auc_score( train.target, oof.prediction). And this is the best indicator of how your model performs. It is a better indicator than public LB.

During the competition, every model should use the same folds. This is accomplished by using the same seed with sklearn.model_selection.KFold(n_splits = 5, shuffle = True, random_seed = 42)



Submission Files
For each of the 5 fold models above, we predict the test images. Therefore we have 5 predictions for each test image. We take the average of these 5 sets of predictions and this is our submission.csv file that we submit to Kaggle. When you submit this to Kaggle, your LB score should be similar to your CV score.

Model Ensemble
Now say that you build 2 models (that means that you did 5 KFold twice). You now have oof_1.csv, oof_2.csv, sub_1.csv, and sub_2.csv. How do we blend the two models?

We find the weight w such that w * oof_1.predictions + (1-w) * oof_2.predictions has the largest AUC.

 all = []
 for w in [0.00, 0.01, 0.02, ..., 0.98, 0.99, 1.00]:
     ensemble_pred = w * oof_1.predictions + (1-w) * oof_2.predictions
     ensemble_auc = roc_auc_score( oof.target , ensemble_pred )
     all.append( ensemble_auc )
 best_weight = np.argmax( all ) / 100.
Then our submission to kaggle will be

 kaggle_sub = best_weight * sub_1.target + (1-best_weight) * sub_2.target
Starter Notebook
After weeks working on a competition, we will have more than 2 models. So we need a more sophisticated approach than a single for-loop.

The simplest approach is hill climbing (or forward selection). Start with the one model that has highest CV score. Next iterate through all your additional models and find the one model that combines with the first model to generate the highest two model ensemble CV score. Then search for the best third model. Repeat until ensemble CV does not increase anymore.

I posted a starter notebook here which includes all my oof.csv and sub.csv for this Melanoma comp. There are 39 models. Forward selection chooses 8 of them and the resultant ensemble has OOF CV 0.950, Public LB 0.958, and Private LB 0.942
