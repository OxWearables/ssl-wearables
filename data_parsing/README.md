## Baseline pre-processing
All the data are processed to have a unit of G. They will also be clipped to have values between ±3g. 
The range and sampling frequencies of different datasets are 
different. The clipping of values should only be done at the modeling stage. When doing pre-processing, we should 
only extract the raw data. Not sure if this matters:

We have included links to the processed datasets that are publicly available. One would however make a request to access
data from the [UK-Biobank](https://www.ukbiobank.ac.uk), the [Rowlands](https://pubmed.ncbi.nlm.nih.gov/21088628/) dataset and the [MJFF-LR](https://www.synapse.org/#!Synapse:syn20681023/wiki/594678) dataset.

| Dataset  | n |Output_size |  Range  | Sample rate (Hz)  |  Validation method |
|---|---|---|---|---|---|
| Capture-24 | 152 | 6 | - | 100 | Five-fold |
| Rowlands | 55 | 13 | - | 80 | - |
| WISDM | 46 | 18 | - | 30 | Five-fold, subject 1616, 1618, 1637, 1639, 1642 removed |
| MJFF-LR | 28 | 12 | - | 50 | Five-fold, days 2 and 3 and activities finger-to-nose and repeated-arm-movement removed |
| REALWORLD | 14 | 8 | - | 30 | Five-fold, subject 2 removed |
| Opportunity | 4 | 4 | - | 30 | Held-one-subject-out cross-validation. Four locomotion labels. Every subject will have all the activities. Weighted-cross-entropy.|
| PAMAP2 | 8 | 8 | ±16g | 100 | Held-one-subject-out cross-validation, using one in test and one in val. This is because  not all the activities are performed by all subjects.  We will use only activities 1, 2, 3, 4, 12, 13, 16, 17 and remove subject 109. This ensures that every subject will have all eight activities. |
| Clemson | 30 | 1  | ±2g | 15 | Five-fold cross-validation. All subjects have roughly about 300-600 samples. | 
| ADL | 7 | 5 | - | 30 | Held-one-subject out | 


70/10/20 ratio for five-fold cross-validation.

References:

* [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset Data Set 
](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+)
* [Dataset for ADL Recognition with Wrist-worn Accelerometer Data Set](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer?ref=datanews.io)
* [RealWorld](https://sensor.informatik.uni-mannheim.de/#dataset_realworld)
