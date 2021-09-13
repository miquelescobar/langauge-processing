# Word Vectors
Is this homework assignment you will create, analyze and evaluate word vectors, using a database created from the Catalan Wikipedia.

## Task description
You will have to modify the provided baseline notebooks to create a new notebook with an additional contribution to the analysis, optimization or comparative study of the baseline model and word vectors.

## Assignment 1 (word vectors)
* Improve CBOW model
* Position-dependent Weighting. The standard CBOW model (CBOW Training notebook) sums all the context word vectors with the same weight. Implement and evaluate a weighted sum of the context words with:
a) A fixed scalar weight, e.g, (1,2,2,1) to give more weight to the words that are closer to the predicted central word
b) A trained scalar weight for each position
c) A trained vector weight for each position. Each word vector is element-wise multiplied by the corresponding position-dependent weight and then added with the rest of the weighted word vectors.
d) (Optional) Hyperparameter optimization: study the performance of the model as a function of one of its parameters: the embedding size, batch size, optimizer, learning rate/scheduler, number of epochs, sharing input/output embeddings.
e) (Optional) Implement other methods to obtain word embeddings

* Evaluate the performance of the improved Catalan word vectors
a) Implement the WordVector class (Word Vector Analysis notebook) with the most_similar and analogy methods to find closest vectors and analogies using the cosine similarity measure.
b) Intrinsic evaluation: perform an informal evaluation finding good and bad examples of closest words and analogies. You can analyze the behavior of the CBOW word vectors for words with multiple meanings, synonyms and antonyms, word frequency, different types of analogies, bias (gender, race, sexual orientation, etc.).
c) Prediction accuracy: compare the accuracy of the implemented CBOW models in the out-of-domain (el Periódico) test set (prediction of the central word given a context of 2 previous and 2 next words). To obtain your score on the competition test set you have to commit your version of the CBOW Training notebook, wait until the training completes, and then "Submit to competition" the obtained file (submission.csv) in the Output section of the notebook.
d) (Optional) Visualize word analogies or word clustering properties


## Assignment 2 (language modeling)
Improve the prediction accuracy of the baseline TransformerLayer notebook
Suggestions:

* Increase the number of TransformerLayers (2 o more)
* Multilayer perceptron (MLP) over the concatenated input vectors
* TransformerLayer with multihead attention
* Replace the last linear layer and the softmax part of the loss with AdaptiveSoftmax
* Sharing input/output embeddings
* Hyperparameter optimization: embedding size, batch size, pooling layer (mean, max, first, …), optimizer, learning rate/scheduler, number of epochs, etc.
The report should include the modified source code, a simple schematic drawing of the model, your results and conclusions.

Create a comparative table of the studied models respect to the single-layer transformer baseline, including loss, accuracy, training time, number of parameters and hyperparameters differences with at least 4 new models or hyperparameters.
