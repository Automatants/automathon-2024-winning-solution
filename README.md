# automathon-2024

## Folder structure
```
├── README.md
├── requirements.txt
├── configs/
│   ├── model1/
│   │   ├── config1.yaml
├── src/
│   ├── model1/
│   │   ├── model1.py
│   │   ├── inference.py
│   ├── model2/
├── checkpoints/
│   ├── model1/
│   │   ├── model1-config1.ckpt
├── data/
│   ├── raw/
│   ├── processed/
├── submissions/
│   ├── model1.csv
```


## Training
```bash
cd src/model1/
python model1.py --config_path ../../configs/model1/config1.yaml
```
It saves the checkpoint file .ckpt in `checkpoints/model1/`

## Inferece
```bash
cd src/model1/
python inference.py --checkpoint_path ../../checkpoints/model1/model1-config1.ckpt --config_path ../../configs/model1/config1.yaml
```



## Config file
```yaml
name: model1-config1
config:
    epochs: 10
    batch_size: 32
    learning_rate: 0.001
    optimizer: adam
    hidden_dim: 128
```


