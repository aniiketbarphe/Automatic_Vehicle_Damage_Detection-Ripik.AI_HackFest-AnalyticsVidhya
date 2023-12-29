![intro](https://github.com/aniiketbarphe/Automatic_Vehicle_Damage_Detection-Ripik.AI_HackFest-AnalyticsVidhya/assets/84449238/d8516b8d-b8e0-4123-96ad-8dee01fd2f2e)

# Automatic_Vehicle_Damage_Detection-Ripik.AI_HackFest-AnalyticsVidhya

Problem Statement

In the insurance industry, processing claims for vehicle damage is a common and most essential task. With the advancement in AI and Computer Vision, the users can settle the claims online instantly by uploading the images of the damaged car with the insurance company.

Now, insurance companies face the constant challenge of identifying fraudulent claims. It's a common practice for the users to submit the fraudulent images as a part of the claim settlement process. This brings out the threat/challenge to the insurance companies to identify the fraudulent claims which leads to significant financial losses.

Fraudulent claims often involve exaggerating the extent of damage or submitting false claims altogether. In this problem, we will focus on the first type of problem i.e. exaggerating the extent of damage. To mitigate these losses and maintain the integrity of their operations, insurance companies must develop effective methods to flag out these claims most accurately and efficiently. 

The hackathon challenges you to develop a robust and high performance model for classifying an image of a car into different types of damages automatically with the help of computer vision techniques. By accurately identifying the damages, the insurance company can assess the legitimacy of the claim and make informed decisions regarding payouts.


Dataset

You are provided with 3 files: 

Training set (train.zip)
Test set (test.zip)
Sample submission (sample_submission.csv)

Training Dataset

The training set contains a diverse dataset of car images of damaged vehicles from various angles, lighting conditions along with labels indicating the specific type of damage (e.g., dents, scratches, cracks, collision damage, etc)

Dataset Description

train.zip contains 2 files: images folder and train.csv

images folder contains the images to be used for training the model
train.csv contains the 3 columns: image_id, filename and target class of the images present in the training dataset.

![train_image](https://github.com/aniiketbarphe/Automatic_Vehicle_Damage_Detection-Ripik.AI_HackFest-AnalyticsVidhya/assets/84449238/a5fe902d-d7dd-4eb9-9f77-1f5f891095c2)


Test Dataset

In the test set, you are provided with only the images and you need to predict the type of damage for each image present in the test set.

Dataset Description

test.zip contains only images folder and test.csv

images folder contains all the test images for which the prediction is to be done.
test.csv contains 2 columns: image_id and filename and you need to predict the label for each present in the test set.

![test_image](https://github.com/aniiketbarphe/Automatic_Vehicle_Damage_Detection-Ripik.AI_HackFest-AnalyticsVidhya/assets/84449238/974ce8c1-41b9-4fed-bc11-7d96de5054a5)

Sample Submission

The solution file must contain predictions for every image_id in the test set. It must contain only 2 columns - image_id and label.  The solution file format must be similar to that of a sample_submission.csv file.

sample_submission.csv contains 2 variables - image_id and label

![sub](https://github.com/aniiketbarphe/Automatic_Vehicle_Damage_Detection-Ripik.AI_HackFest-AnalyticsVidhya/assets/84449238/da38bf31-210d-4ead-a17d-11386875d4ca)

Public and Private Split

Test data is further divided into Public (40%) and Private (60%) data.

Your initial responses will be checked and scored on the Public data. The final rankings would be based on your private score which will be published once the competition is over.


Evaluation Metric

The model will be evaluated based on the macro f1 score.

Rules and Conditions

The final rankings would be based on private score and model interpretability and will be published once the competition is over.
Setting the final submission is recommended. Without a final submission, the submission corresponding to best public score will be taken as the final submission
Use of external data is not allowed.
No restriction on the no. of submissions per day.
Entries submitted after the contest is closed, will not be considered
The code file pertaining to your final submission is mandatory while setting final submission
The submitted code file must be able to reproduce the similar score to that of the final submission file.
Throughout the hackathon, you are expected to respect fellow hackers and act with high integrity.
Use of multiple Login IDs will lead to immediate disqualification
Analytics Vidhya holds the right to disqualify any participant at any stage of the competition if the participant(s) are deemed to be acting fraudulently.

Link:- 

https://datahack.analyticsvidhya.com/contest/ripikai-hackfest-unleashing-ai-potential/?utm_source=newhomepage#LeaderBoard

# Data manipulation
# Data Visualazation
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
%matplotlib inline
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
 
import warnings
warnings.filterwarnings('ignore')

import pycaret
print('PyCaret: %s' % pycaret.__version__)
from pycaret.classification import *

s = setup(data=train,
          session_id=7010,
          target='ind',
          train_size=0.99,
          fold_strategy='stratifiedkfold',
          fold=5,
          fold_shuffle=True,
          remove_multicollinearity = True,
          normalize = True,
          normalize_method = 'robust')
top4 = compare_models(n_select=5)

print(top4[1])

LDA = tune_model(create_model('lda'), choose_better = True, n_iter = 20)
plot_model(LDA, "feature")

# Additional functions in Pycaret
lgbm  = create_model('lightgbm')      
tuned_lightgbm = tune_model(lgbm)
plot_model(estimator = tuned_lightgbm, plot = 'learning')
plot_model(estimator = tuned_lightgbm, plot = 'auc')
plot_model(estimator = tuned_lightgbm, plot = 'confusion_matrix')
plot_model(estimator = tuned_lightgbm, plot = 'feature')
evaluate_model(tuned_lightgbm)
# plotting number of correctly classified and misclassifed labels
plot_model(tuned_model, plot = 'error')
# plotting classification report
plot_model(tuned_model, plot = 'class_report')

predict_model(tuned_lightgbm, data=test)
predictions = predict_model(tuned_lightgbm, data=test)
predictions.head()

sub['Survived'] = round(predictions['Score']).astype(int)
sub.to_csv('submission.csv',index=False)
sub.head()

# Extra: Blending made easy!
logr  = create_model('lr');          

blend = blend_models(estimator_list=[tuned_lightgbm,logr])
