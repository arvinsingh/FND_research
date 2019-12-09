## This repository contains the complete data and code used in the following research article:

**[“Comparative evaluation of Machine Learning algorithms for fake news detection”](https://link.springer.com/chapter/10.1007%2F978-981-13-9942-8_40)**

Authors
	[Arvinder Pal Singh Bali](https://www.researchgate.net/profile/Arvinder_Bali)
	[Mexson Fernandes](https://www.researchgate.net/profile/Mexson_Fernandes)
	[Sourabh Choubey](https://www.researchgate.net/profile/Sourabh_Choubey)

Special thanks to our beloved [Prof. Pradosh K. Roy](https://www.researchgate.net/profile/PRADOSH_K_Roy) for all his guidance and help he provided us with.

Kindy contact us personally if you'd like to read about our research. It makes us happy to know our hardwork is being recognized and read widely.

## Procedure to replicate our results

**1. Install all the dependencies**

`Library Dependencies`
* Python <= 3.5
* [Jupyter Notebook](https://jupyter.org/install)
* Scipy Stack (`numpy`, `scipy` and `pandas`)
* [scikit-learn](http://scikit-learn.org/stable/)
* [XGBoost](http://xgboost.readthedocs.io/en/latest/)
* [gensim (for word2vec)](https://radimrehurek.com/gensim/)
* [NLTK (python NLP library)](http://www.nltk.org)

**2.`clone the repo`**

**3. Download the `GloVe` [model](http://nlp.stanford.edu/data/glove.6B.zip) trained on Wikipedia 2014 + Gigaword 5. Convert the file to word2vec.txt using `convert_GloVe2Word2Vec.ipynb` ``OR`` download datasets and model from [here](https://drive.google.com/drive/folders/1swSy4AKuCf3ykS_vt5Klxfrc0gRzd_JA?usp=sharing) and save under `src/feature_generators/datasets/`.**

**In directory src/feature_generators/**

**4. Use `prepare_data.ipynb` then `gen_features.ipynb` to generate all the required features.**

**All the pickled files are saved under `src/saved_data/`.**

**6. Run `xgb_train.py` to train and make predictions on the test set. Output file is `src/predictions_*.csv` under directory `src/results`.**

**In directory src/**

**7. Use `Result_visualization.ipynb` and `test_xgb_model.ipynb` to study the output and use the model respectively.**

**8. For Cross Validation results check notebooks `cross_validation.ipynb`.**


All the output files are also stored under `results/` and all parameters are hard-coded that have been determined using grid and random search methods. 
