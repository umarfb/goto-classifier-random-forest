# goto-classifier
A machine-learning classifier for classifying GOTO light curves, using Python 3. Currently the aim of the classifier is to classify lightcurves into *flare/variable star* and *supernova/transient* classes.

Links to packages used for simulating light curves:
- Supernova light curves - `sncosmo` http://sncosmo.github.io/
- M-dwarf flares - `AltaiPony` https://github.com/ekaterinailin/AltaiPony

Other Python packages used:
 - `scikit-learn` https://scikit-learn.org/stable/, used for Gaussian Process Regression and Machine learning.
 
 Outline of steps towards making a GOTO classifier
 
 1. Simulating light curves:
  - Supernova light curves were simulated using `sncosmo` and the built-in templates, and then transformed into the GOTO L filter and sampled according to the cadence from real GOTO supernova light curves.
  - M-dwarf flare light curves were simulated using `AltaiPony`. Fractional fluxes were converted to magnitude differences and then scaled to have magnitudes ranging from 12 - 16 mag.
  - Variable star light curves were simulated using Gaussian Processes on real GOTO variable star light curves, and then sampled according to the cadence from real GOTO light curves.
  
 2. Training the model:
  - Split the simulated light curves into a balanced training set and a test set.
  - Use five-fold cross validation to determine best hyperparameters for a `RandomForestClassifier`
  - Use the `RandomForestClassifier` from `scikit-learn` and train on the training set, and then see performance on test set.
 
 3. Testing performance:
  - Use the classifier on real GOTO light curves.
