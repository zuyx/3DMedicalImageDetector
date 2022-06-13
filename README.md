# 3DMedicalImageDetector
The repoitory is my ongoing research promgram for meidical imaging detection.

## description of the projects componets
- the projects are splitted into 4 parts: criterion,dataset,model,utils,evaluate,trainer,main and infer parts.
- the project include two kind detector: anchor free and anchor based detectors.
- the dataset config include preprocessed data path, train patients id, validattion patients id and so on,which is in the 'dataset/data/data_config.py' script
- the model and experiments config are in  the 'dataset/exp_config.py' script

## guides for running and evaluating models

- you can train the model on the default config, also you can modify the parameters on the parser argument

- commands:

  - you can run the command to train model:
```
python main.py --model {model name} -b {batch size} --save-dir {the path to save trained model} --resume {the .ckpt file you want load}} --anchor 0
```
  - you can run the command to test the model:

```
python infer.py --model {model_name} --save-dir {the path to save the predicted results} --resume {the .ckpt file you want to load}
```

- you can run the script 'evaluate/FROCeval.py' to evaluate the model and get the fpr results:
```
python FROCeval.py
```

