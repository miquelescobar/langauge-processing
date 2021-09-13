# Language Identification
In this homework assignment you will work on a written Language Identification (LI) task, using a database created from Wikipedia paragraphs.

## Task description
Edit the character-based RNN Baseline to create a new notebook with an additional contribution to the analysis, optimization or comparative study of the proposed model and task. Include the modified code with your comments, results, tables, graphs, and conclusions in the PDF report. Compare the results of at least 6 models

### Assignment tasks
* Hyperparameter optimization: try incrementing embedding size, RNN hidden size, and batch size. You can also compare different optimizers.
* Combination of the output of the max-pool layer with the output of a mean-pool layer (concatenation or addition).
* Add a dropout layer after the pooling layer
* Comparative analysis of the performance with other RNNs or number of layers.

### Other optional tasks
* Comparative analysis with other DNN architectures such as convolutional neural networks.
* Other input features: modify the code to use other features as input such as words, character n-grams, or character counts.
* Description and comparative analysis with other classical language identification systems. You can use an existing implementation, but you will have to describe it with your own words.
