# simple-gpt

## Intro

This project implements a simple chat program based on the **Transformer** architecture, trained using the [openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext) dataset.

## Run

1. Use the following code to install the required dependencies.

```shell
pip install -r requirements.txt
```

2. Run `data-process.py`, the program will automatically download the dataset and divide it into a training set and a validation set.

3. Run `train.py` to train your own model
4. Run `chat.py` to start the chat using the model you trained.

## References

https://github.com/Infatoshi/fcc-intro-to-llms/tree/main