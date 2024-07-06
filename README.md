# LLMeLog 




This is the basic implementation of: **LLMeLog: An Approach for Anomaly Detection based on LLM-enriched Log Events.**.
- [LLMeLog](#llmelog)
  * [Project Structure](#project-structure)
  * [Datasets](#datasets)
    + [Environment:](#environment)
    + [Preparation](#preparation)
  * [Run](#run)


## Project Structure

```
├─checkpoint      # Saved models
├─bert-base-en    # Pretrained BERT model
├─new_encoder     # Fine-tuned BERT model
├─data            # Log data
├─src             
|  ├─dataset.py   # Load dataset
|  ├─models.py    # Transformer-Based Anomaly Detection model   
|  └─utils.py     # Log Embedding
├─main.py         # entries
└─predata.py      # Data preprocess
```

## Datasets

We used `3` open-source log datasets for evaluation, HDFS, BGL and Thunderbird. 

| Software System | Description                        | Time Span  | # Messages | Data Size | Link                                                      |
|       ---       |           ----                     |    ----    |    ----    |  ----     |                ---                                        |
| HDFS         | Hadoop distributed file system log | 38.7 hours | 11,175,629 | 1.47GB | [Loghub](https://github.com/logpai/loghub/tree/master/HDFS) |
| BGL           | Blue Gene/L supercomputer log | 214.7 days | 4,747,963 | 708.76MB | [Usenix-CFDR Data](https://www.usenix.org/cfdr-data#hpc4) |
| Thunderbird     | Thunderbird supercomputer log      | 244 days   | 211,212,192 | 27.367  GB | [Usenix-CFDR Data](https://www.usenix.org/cfdr-data#hpc4)   |

**Note:** Considering the huge scale of the Thunderbird dataset, we followed the settings of the previous study [LogADEmpirical](https://github.com/LogIntelligence/LogADEmpirical) and selected the earliest 10 million log messages from the Thunderbird dataset for experimentation. 

### Environment

**Key Packages:**

Numpy==1.20.3

Pandas==1.3.5

Pytorch_lightning==1.1.2

torch==1.13.1+cu116

tqdm==4.62.3

transformers==4.15.0

[Drain3](https://github.com/IBM/Drain3)




### Preparation

You need to follow these steps to **completely** run `LLMeLog`.
- **Step 1:** Download [Log Data](#datasets) and put it under `data` folder.
- **Step 2:** Using [Drain](https://github.com/IBM/Drain3) to parse the unstructed logs.
- **Step 3:** Download `bert-base` from [Hugging Face](https://huggingface.co/bert-base-uncased), and put it under `bert-base-en` folder.
- **Step 4:** Enriching the log event with the provided prompt, we recommend using [ChatGPT](https://chat.openai.com/).


## Run
you can run `LLMeLog` on HDFS dataset with this code:

- Preprocessing data for training and evaluation.
```
python predata.py --dataset hdfs
```

- Hierarchical Semantic Fine-tuning.
```
python main.py --mode train --encoder 1 --dataset hdfs --lr 0.0002
```

- Event Embedding within Fine-tuned BERT.
```
python main.py --mode gen --encoder 1 --dataset hdfs --lr 0.0002
```

- Training Transoformer for Log-based Anomaly Detection.
```
python main.py --mode train --dataset hdfs --batch_size 256 --lr 0.0003
```

- Evaluation on HDFS Dataset.
```
python main.py --mode eval --dataset hdfs --batch_size 256 --lr 0.0003 --load_checkpoint True
```