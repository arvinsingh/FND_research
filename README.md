# <h2 align='center'>Comparative Evaluation of Machine Learning Algorithms for Fake News Detection</h2>

<div align='center'>
  <a href="https://www.researchgate.net/profile/Arvinder_Bali"><strong>Arvinder Pal Singh Bali</strong></a>
  ·
  <a href="https://www.researchgate.net/profile/Mexson_Fernandes"><strong>Mexson Fernandes</strong></a>
  ·
  <a href="https://www.researchgate.net/profile/Sourabh_Choubey"><strong>Sourabh Choubey</strong></a>
</div>

> [!Note] 
> Published in *[Springer - Advances in Computing and Data Sciences](https://link.springer.com/chapter/10.1007%2F978-981-13-9942-8_40)*

This repository contains the complete data and code used in our research for comparative evaluation of machine learning algorithms for fake news detection.

## Acknowledgements

Special thanks to our beloved [Prof. Pradosh K. Roy](https://www.researchgate.net/profile/PRADOSH_K_Roy) for all his guidance and help he provided us with.

> [!Important]
> Kindly contact us personally if you'd like to read about our research. It makes us happy to know our hardwork is being recognized and read widely.

## Installation

### Library Dependencies

- Python <= 3.5
- [Jupyter Notebook](https://jupyter.org/install)
- Scipy Stack (numpy, scipy and pandas)
- [scikit-learn](http://scikit-learn.org/stable/)
- [XGBoost](http://xgboost.readthedocs.io/en/latest/)
- [gensim (for word2vec)](https://radimrehurek.com/gensim/)
- [NLTK (python NLP library)](http://www.nltk.org)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone git@github.com:arvinsingh/FND_research.git
   cd FND_research
   ```

2. **Download required models:**
   
   Download the GloVe [model](http://nlp.stanford.edu/data/glove.6B.zip) trained on Wikipedia 2014 + Gigaword 5. Convert the file to word2vec.txt using `convert_GloVe2Word2Vec.ipynb`
   
   **OR**
   
   Download datasets and model from [here](https://drive.google.com/drive/folders/1swSy4AKuCf3ykS_vt5Klxfrc0gRzd_JA?usp=sharing) and save under `src/feature_generators/datasets/`

## Procedure to Replicate Results

### Step 1: Feature Generation
**In directory `src/feature_generators/`**

1. Use `prepare_data.ipynb` to prepare the data
2. Then use `gen_features.ipynb` to generate all the required features

All the pickled files are saved under `src/saved_data/`

### Step 2: Model Training
**In directory `src/`**

3. Run `xgb_train.py` to train and make predictions on the test set
   
   Output file: `src/predictions_*.csv` under directory `src/results`

### Step 3: Results Analysis
**In directory `src/`**

4. Use `Result_visualization.ipynb` and `test_xgb_model.ipynb` to study the output and use the model respectively
5. For Cross Validation results check notebook `cross_validation.ipynb`

> [!Note]
> All the output files are also stored under `results/` and all parameters are hard-coded that have been determined using grid and random search methods.

## Citation

If you find this research helpful, please cite the following paper:

```bibtex
@InProceedings{10.1007/978-981-13-9942-8_40,
author="Bali, Arvinder Pal Singh
and Fernandes, Mexson
and Choubey, Sourabh
and Goel, Mahima",
title="Comparative Performance of Machine Learning Algorithms for Fake News Detection",
booktitle="Advances in Computing and Data Sciences",
year="2019",
publisher="Springer Singapore",
address="Singapore",
pages="420--430",,
isbn="978-981-13-9942-8"
}

```