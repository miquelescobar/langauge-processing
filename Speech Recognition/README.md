# Speech Recognition Assignment
In this homework assignment you will work on a simple speech recognition task: digit recognition using dynamic time warping.

## Task description
* Modify the wer function to evaluate the performance of the DTW algorithm for text-dependent speaker identification. What is the probability of detecting the correct speaker of the free-spoken-digit-dataset using the DTW distance to the rest of the recordings?
* Compare the WER with respect to the number of cepstral coefficients. Use the obtained best value for the rest of the tasks.
* Analyze the influence in the recognition accuracy of each of the cepstral normalization steps (mean and variance) with and without littering. Use the default littering parameter (sinus window size) of 22.
* Extend the MFCC parameters with the first-order derivatives, and check that the WER is reduced.
* Complete the DTW algorithm to compute the alignment (backtracking). Plot one example of the alignment with the closest reference when the test signal is different from the reference signal (different digit) or equal (same digit).
* Optional: perform a comparative study of different variants of the DTW algorithm in terms of speed and accuracy.
