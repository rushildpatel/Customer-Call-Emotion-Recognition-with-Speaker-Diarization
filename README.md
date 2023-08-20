# Customer-Call-Emotion-Recognition-with-Speaker-Diarization

Emotion is a complex topic, and its study spans multiple disciplines. The importance of human speech emotion recognition has grown in recent years to improve both the naturalness and efficiency of human-machine interactions. Research on speech emotion recognition (SER) has also gained popularity through the use of machine learning techniques.

Among all kind of emotions, six categories are often described as the cardinal ones: **happy, disgust, sad, fear, anger, suprise**.

<p align="center">
    <img src="./emotions_wheel.png" width="400"/>
</p>

Taking audio files labelled as `happy`,`disgust`,`sad`,`fear`,`anger`,`surprise`, and `neutral`, this project aim to build and evaluate few models to classify these sounds.

<br/>

The entire project is made into 2 parts:

Part A: Find the best perfomring model for emotion recognition of single speaker audio clips
The project is structured in nine jupyter notebooks with these parts:
1. **The data Merging Step**: information on sources used and preliminary overview of the samples
2. **Data cleaning + feature extraction**: audio processing for cleaning the samples (resampling, trimming, noise reduction) and description of extracted features
3. **Random Forest**: getting the best hyperparameters (number of decision trees and their maximum depth) with cross validation
4. **Support Vector Machine**: tuning and evaluating the regularization strenght parameter C and the gamma value for an rbf kernel
5. **Dense Networks**: fitting two models, one without hidden layers and the other with two hidden layers
6. **Convolutional Netural Network**: training a Conv Net with MFCC data as two-dimensional arrays instead of using the features
7. **Models performance comparison**: evaluation of model performance with visualizations and final considerations on the project

After comparing all the models we find the best performing model is SVM (Support Vector Machine).

Part B: Split real life audio clip with multiple speakers (usually 2 in a customer call) using speaker diarization method
