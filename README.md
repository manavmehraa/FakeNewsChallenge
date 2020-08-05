## FNC1 Revisited: Two-Step Multilayer Perceptron based Stance Detection
---
This repository contains submission for MSCI641 Fake News Challenge Default Project. The trained model files an features are stored in **trained.zip**. Unzip this to use pretrained model and predict else make a new **trained** directory and run the main.py file.

#### Predict without training the models

1. ```git clone https://github.com/manavmehra96/fnc_stance_detection.git```
2. ```cd fnc_stance_detection && unzip trained.zip```
3. ```pip install -r requirements.txt```
4. ```python main.py --train_feat n --train_model n```

---
#### Predict with training the models

1. ```git clone https://github.com/manavmehra96/fnc_stance_detection.git```
2. ```cd fnc_stance_detection && mkdir trained```
3. ```pip install -r requirements.txt```
4. ```python main.py --train_feat y --train_model y```

##### The output directory contains the final predicted csv.

---
### Usage

```
main.py [-h] [--train_feat (y/n)] [--train_model (y/n)]

optional arguments:
  -h, --help            show this help message and exit
  --train_feat - Train Features? (y/n)
                        
  --train_model - Train Model? (y/n)
```

---
The file structure of the repository is as follows -

```bash
├── main.py
├── data (dataset)
│   ├─**/*.csv
├── utils
│   ├─*build_model1.py
│   ├─*build_model12.py
│   ├─*features.py
│   ├─*read_data.py
│   ├─*predict.py
│   ├─*prediction.py
│   ├─*score.py
├── output
│   ├─*final_answer.csv
├── trained.zip(all trained models and features)
```
---
## Main Dependencies
    Keras==2.4.3        
    nltk==3.5
    numpy==1.19.0
    pandas==1.0.3
    scikit-learn==0.23.0
    scipy==1.4.1
    tensorflow==2.2.0
---
#### Disclaimer
**The experiments were performed using a Tesla T4 GPU, 30GB memory and 8 core CPU**

#### Credits
Manav Mehra (m3mehra@uwaterloo.ca)
Rajbir Singh (rsrajbir@uwterloo.ca)