# QBR
Dataset and source code for QBR


## Overview

### Dataset
There are two domain-specific datasets including legal and medical domain. 
Due to copyright issues, we only provide some illustration examples for the corpus in the document retrieval, the training and the testing set in the scope level fine-tuning.

### Code

### Document-level Retrieval
```bash
cd document_retrieval
```

```bash
# for legal dataset
python retrieval_legal.py 
# for medical dataset
python retrieval_medical.py 
```

### Scope-level Retrieval Training
```bash
cd fine_tuning
```

```bash
# for legal dataset
python main.py -c config/legal.config -g 1
# for medical dataset
python main.py -c config/medical.config -g 1
```

### Testing

[//]: # (> Make sure that the xx exists.)
```bash
cd fine_tuning
```

```bash
# for legal dataset
python test.py -c config/legal.config 
# for medical dataset
python test.py -c config/medical.config 
```