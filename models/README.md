# Models
Files of models used in submissions for Kaggle contest.
People responsible: Youssef Ibrahim, Kacper Leszczyński
## How to run
Model files are:

* BERT.py in BERT folder
* time_series.py in time_series folder

To run them, execute them in Python 3.7 with command: "python3 <name of script file>".

You may need to have following packages installed: tensorflow, keras, sklearn, pandas, nltk, numpy

You may also need to copy learning and testing datasets to directory where scripts are ran. They are available in models/test and models/learn directories in the repository.


In case of BERT, you will need to download and unpack the following BERT pre-trained model: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

BERT model requires a lot of computational power, so we encourage use of Python notebook provided in BERT directory. Use it on external platforms (Kaggle Kernels - it allows import of stage test data used in the competition, so it may be better) to use their computational power.

## Custom data
To test the set on a different data set, provide them as an argument to running command, i.e. "python3 <name of script file> <name of data file>".