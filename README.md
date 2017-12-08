# Citation Sentiment Classifier

### Dataset:
Code and data for citation sentiment classification reported in <http://www.aclweb.org/anthology/P11-3015>.
The file test.arff contains only the test set with dependency triplets generated with Stanford CoreNLP.
Full corpus available at <http://www.cl.cam.ac.uk/~aa496/citation-sentiment-corpus>.

### Files:
```
face-rating/
├── data/
│   ├── ratings.txt
│   ├── landmarks.txt
│   ├── features_ALL.txt
├── source/
|   ├── machine_learning/    
│       ├── generateFeatures.py
│       ├── trainModel.py
│       ├── cross_validation.py
|   ├── deep_learning/    
│       ├── build_model.py
│       ├── utils.py
│       ├── face_rating.ipynb
|   ├── deeplearning_result.ipynb
|   ├── traditional_result.ipynb
```

### Requirements:

- Python 2.7
- scikit-learn
- pandas
- numpy
