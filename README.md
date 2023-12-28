# LLMeLog

**LLMeLog: An Approach for Anomaly Detection based on LLM-enriched Log Events.**

## Requirements

```
numpy==1.20.3
pandas==1.3.5
pytorch_lightning==1.1.2
torch==1.13.1+cu116
tqdm==4.62.3
transformers==4.15.0
```

## Log data

We used `3` open-source log datasets: HDFS, BGL and Thunderbird. You can find them on the [loghub](https://github.com/logpai/loghub). 



## Preparation

You need to follow these steps to run `LLMeLog`.

- **Step 1:** Download `bert-base` from [Hugging Face](https://huggingface.co/bert-base-uncased), and put it under `bert-base-en` folder.
- **Step 2:** Parsing Log data, we recommend using [Drain](https://github.com/logpai/logparser/tree/main/logparser/Drain#drain).
- **Step 3:** Enriching the log event with the provided prompt, we recommend using [ChatGPT](https://chat.openai.com/).



## Running

you can run `LLMeLog` on HDFS dataset with this code:

```
python predata.py --dataset hdfs
python main.py --mode train --encoder 1 --dataset hdfs --lr 0.0002
python main.py --mode gen --encoder 1 --dataset hdfs --lr 0.0002
python main.py --mode train --dataset hdfs --batch_size 256 --lr 0.0003
python main.py --mode eval --dataset hdfs --batch_size 256 --lr 0.0003 --load_checkpoint True
```

