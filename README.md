# Comparing Different CNN Architectures to Classify Trading Cards

This paper investigates which Convolutional Neural Networks can be used to classify Yu-Gi-Oh! trading cards based on non-text features of the cards. This problem is made more challenging because the dataset is very imbalanced. To mitigate this oversampling is used. We find that due to overfitting caused by oversampling, accuracy is decreased. Convolutional Neural Networks can still be used to classify trading cards, but not all models are well-suited for this task.

## Running the code

This code depends on the this dataset: https://www.kaggle.com/datasets/archanghosh/yugioh-database/versions/1

The dataset needs to be in the same directory as the code when the code is executed. ``preprocessing.py`` needs to be run first as it transforms the data. Afterwards to models can be trained with or without using oversampling by executing ``train_all_oversample.py`` or ``train_all_normal.py``. 