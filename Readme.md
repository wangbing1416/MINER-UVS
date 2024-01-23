# MINER-UVS
This repo is the released code of our work **Escaping the Neutralization Effect of Modality Features Fusion in Multimodal Fake News Detection**

Our released code follows to "EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection" and "BDANN: BERT-Based Domain Adaptation Neural
Network for Multi-Modal Fake News Detection"

### Requirements

```
torch==1.12.1
cudatoolkit==11.3.1
transformers==4.27.4
```

### Train

- Prepare the datasets Weibo and Gossip. Our datasets are from https://github.com/yaqingwang/EANN-KDD18 and https://github.com/shiivangii/SpotFakePlus,
and you should put them in `./Data`

- Run the python file
```shell
python ./src/run.py
```

- Check log files in `./log`

### Tips
1. When you change the dataset to run, such as changing Weibo to Gossip, you should revise 
```python
import process_weibo as process_data
```
to
```python
import process_gossipcop as process_data
```
in line 10, `run.py`.
2. You should manually split the training set of GossipCop into the divisions of training and validation, then, revise the file road in the function `write_data` in line 89, `process_gossipcop.py`

### Citation
```

```