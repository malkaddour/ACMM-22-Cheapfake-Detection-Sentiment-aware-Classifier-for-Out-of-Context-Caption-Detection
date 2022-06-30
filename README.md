# ACMM-22-Cheapfake-Detection-Sentiment-aware-Classifier-for-Out-of-Context-Caption-Detection

The source code here is used to evaluate our implementation of the sentiment-aware classifier for Task 1 of the ACMM 2022 Grand Challenge on Cheapfakes Detection.

In addition to the packages in the requirements.txt, you will need to:
1. Install cython
2. Install numpy
3. Install setuptools
4. To fix the installation for pycocotools, run:
  "pip3 uninstall -y pycocotools" followed by
    "pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI"
5. Download the required Spacy models using
  "python3 -m spacy download en" && followed by
    "python3 -m spacy download en_core_web_sm"
6. Install Detectron2. The source files for Detectron2 are available in this repo, and the modifications required to compute the Image-text features from COSMOS (https://github.com/shivangi-aneja/COSMOS) and COSMOS on Steroids (https://github.com/acmmmsys/2021-grandchallenge-cheapfakes-cosmosonsteroids) are already done in the source files. Please follow the instructions from https://github.com/facebookresearch/detectron2 to setup Detectron2, and make sure to be in this directory during setup.

# Data and evaluation
The three model weights for the MLPs trained on the validation and test sets are available in "/newresults/modelweights/", and to change the model you wish to use simply modify the corresponding name in line 205 of "run_cheapfake.py". Note that you also need to change the "no_of_layers" in line 171 to 6 if you want to use the "train_full.h5" weights.

The code reads a "test.json" file located in the directory that contains the test data as supplied in COSMOS, and also requires a directory containing the test images. For licensing purposes, these are not provided in this repository. The data was acquired from COSMOS (https://github.com/shivangi-aneja/COSMO) by participating in the 2022 ACMM Grand Challenge for Cheapfake Detection.

The "utils", "models_final", and "model_archs" directories contain the models used for image-text matching, also obtained from COSMOS and COSMOS on Steroids.
To run the code:
1. Inside the "utils/config.py", set the base directory and the data directory. The data directory should contain a "test.json" file and a folder with the test images.
2. NLTK, TextBlob, and VaderSentimentAnalyzer are all in the requirements.txt, and the Vader_lexicon model is downloaded automatically within the code.
3. After setup, running "run_cheapfake.py" will compute the sentiment features and use the COSMOS models and code to compute the image-text features for the "test.json" file. It will then create the MLP and load the weights as described above, and evaluate the performance by using the MLP predictions and comparing with the test.json labels.
4. The output is the "/test_labels.json" file which contains a vector of the test labels predicted by the MLP 
