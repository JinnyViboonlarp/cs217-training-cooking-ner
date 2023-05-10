# cs217-training-cooking-ner

This repo stores the dataset and codes used for training a cooking-domain NER model that is used to suggest labels in the web application hosted on [this repo](https://github.com/JinnyViboonlarp/cs217-project).

## dataset

The dataset used is the ingredient-phrased dataset used in the paper ["A Named Entity Based Approach to Model Recipes"](https://arxiv.org/abs/2004.12184) consisting of 49884 tokens, across 10418 sentences.

The examples of the input sentences are:
- 1/2 large sweet red onion, thinly sliced
- 1/2 cup fresh bean sprouts, or to taste (optional)
- 4 eggs, whites and yolks separated

There are 8 possible labels as follow:

Label | Significance | Example
--- | --- | ---
NAME | name of ingredient | salt, pepper
STATE | processing state of ingredient | ground, thawed
UNIT | measuring unit | gram, cup
QUANTITY | quantity associated with the unit | 1, 1 1/2, 2-4
SIZE | portion size | small, large
TEMP | temperature | hot, frozen
DF | dryness and freshness state | dry, fresh
O | no label presented | and, thinly

There is no marking indicating the start or continuation of an entity span (unlike the BIO format).

## models

The models are trained from scratch in Pytorch. 5 model architectures are experimented with:
- **DAN**: Deep Averaging Network. The sentence embedding (calculated by averaging the embedding of every token in the sentence) is concatenated with the embedding of the token to be predicted, before being sent to a dense layer and then a classification layer.
- **Dense**: Dense/Feedforward Network. The embedding of every token in the sentence is concatenated, before being sent to a dense layer and then a classification layer.
- **CNN**: Convolutional Network. The embedding layer is followed by a CNN layer, which is followed by a classification layer. The CNN layer has the same architecture as described in the paper *Convolutional Neural Networks for Sentence Classification*.
- **LSTM**: LSTM Network. The embedding layer is followed by a LSTM layer, which is followed by a classification layer.
- **Attention**: Attention-Mechanism Network. The embedding layer is followed by a LSTM layer, which is followed by a classification layer.

Regarding the input sentence (to any of the model), the token to be labeled would be appended at the start (index 0) of the sentence, as a way to indicate and "mark out" the interested token to the model.

To train a model, I run `python train.py` followed by a flag indicating the model architecture, such as `python train.py --dan` for running the DAN model. The list of the flag is [--dan, --dense, --cnn, --lstm, -attn] which correspond to the 5 architectures consecutively.

Other flags also can be passed to fine-tune the hyperparameters. The list of all those flags could be found in `train.py`. The default values of those flags are the ones that result in the best models, described in the next section.

## evaluation

As most tokens have non-O labels, and as the labels are relatively evenly distributed, I decide to evaluate the models' performance by accuracy, instead of by F1 like in most NER tasks.

The 49884 tokens/instances are divided into 37214 in training set, 5705 in validation set, and 6965 in testing set.

Model | Accuracy
--- | ---
DAN | **0.887**
Dense | 0.879
CNN | 0.867
LSTM | 0.871
Attention | 0.851

It is a bit surprising that the models with simpler architectures seem to perform better than those with more complex architectures. It seems to me that the reasons for this could be because
- The more complex a model's architecture is, the easier for the weights to overfit the training data.
- The more complex a model's architecture is, the harder and more time-consuming it is to train it to its optimal performance. It might be the case that, despite my intention, the hyperparameters were just not set in a way that would allow the more complex models to reach their peak performance.
- Considering that all out-of-vocab tokens are mapped to the same embedding, it might be the case that an accuracy of 0.90 or less could be the highest that any model could theoretically achieve with this setup.

Since the DAN model is both simplest and best-performing, this model is the one that is used in [our web app](https://github.com/JinnyViboonlarp/cs217-project).
