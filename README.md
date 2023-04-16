## **GPT-2 based PyTorch Text-Generator for ChatGPT Tweets**

## Quick Start

1. Make sure you have PyTorch installed
2. Download the trained models [here](https://drive.google.com/drive/folders/1sO4d16KKXxaJ8ihGjh8EKRAlSxTu9GoE?usp=share_link) and place them in the same folder as ```generate_tweet.py```
```shell
# Install required packages
$ pip install -r requirements.txt
# Run the python file to generate tweets
$ python generate_tweet.py
```

## Like Ratio Classification
We also tried training a classifer to predict the like-to-view ratio of ChatGPT related tweets with limited result. The code and result of our attempt is in the ```Like Classifier```  folder. It is for reference only and ```requirements.txt``` does not contain the packages needed for running that part.


## Author

- Hayden Chiu
- Author Email: [yikhei123@gmail.com](mailto:yikhei123@gmail.com)


## License

- OpenAI/GPT2 follow MIT license, huggingface/pytorch-pretrained-BERT is Apache license. 
- I follow MIT license with original GPT2 repository


## Acknowledgement
[Tae-Hwan Jung (@graykode)](https://github.com/graykode/gpt-2-Pytorch) for referring code.