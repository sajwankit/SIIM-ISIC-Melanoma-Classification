# SIIM-ISIC Melanoma Classification
**Identify Melanoma In Lesion Images**  

**Competition Overview:** https://www.kaggle.com/c/siim-isic-melanoma-classification  

**Solution:  **
Built a large ensemble of diverse CNN models with a goal to be more robust against overfitting. Trusting CV and diversity was important to achieve a good result on private LB.

**Triple Stratified Leak-Free KFold CV**  
**Stratify 1 - Isolate Patients**  
A single patient can have multiple images. Now all images from one patient are fully contained inside a single TFRecord. This prevents leakage during cross validation

**Stratify 2 - Balance Malignant Images**  
The entire dataset has 1.8% malignant images. Each TFRecord contains 1.8% malignant images. This makes validation score more reliable.

**Stratify 3 - Balance Patient Count Distribution**  
Some patients have as many as 115 images and some patients have as few as 2 images. When isolating patients into TFRecords, each record has an equal number of patients with 115 images, with 100, with 70, with 50, with 20, with 10, with 5, with 2, etc. This makes validation more reliable.

Below are 15 plots showing the histogram of patients and their counts within each TFRecord.



Leak Free - Remove Duplicates
The above 3 stratifications make a more reliable CV and prevent leakage during cross validation. Additionally it has been published by the competition host here that the training data contains 434 duplicate images. If one image is inside your training fold and the duplicate is in your validation fold, this causes a leak which jeopardizes the reliability of your CV. These 434 duplicate images have been removed from my TFRecords to prevent leakage.
