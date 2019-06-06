# deepDR
## paper "deepDR: A network-based deep learning approach to in silico drug repositioning"

### 'dataset' directory
Contain the gold standard drug-disease set and ten drug-related networks.
### 'preprocessing' directory
Contain the preprocessing code to generate PPMI matrix.
### 'PPMI' directory
Contain the PPMI matrices of ten drug-related networks.
### Tutorial
1. Create two directories "test_models" and "test_results" in the project.
2. To get drug features learned by MDA, run
  - python getFeatures.py example_params.txt
3. To predict drug-disease associations by cVAE, run
  - pretraining with features: python cvae.py --dir dataset -a 6 -b 0.1 -m 300 --save --layer 1000 100
  - refine training with rating: python cvae.py --dir dataset --rating -a 15 -b 3 -m 500 --load 1 --layer 1000 100

### Requirements
deepDR is tested to work under Python 3.6  
The required dependencies for deepDR  are Keras, PyTorch, TensorFlow, numpy, scipy, and scikit-learn.
