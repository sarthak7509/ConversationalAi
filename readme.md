# Conversational Ai using Tensorflow
This is the implementation of multi head attention model on the cornell movie dataset. This is an implementation of Transformer models introduced in 2017 and is used in NLP in various task such as Translation (you can visit my other repository Neural Style Translation).

## How to use:-
1) first clone the repository
```shell
$ git clone https://github.com/sarthak7509/ConversationalAi.git
```
2) Download the pretrained weights from [my drive](https://drive.google.com/file/d/1tkWOcmNPeNeRaJY1MhTS9bcWB6gGKkcq/view?usp=sharing) and place it in the model weights section
3) Run conversationAi.py for demo or run app.py to start a local server that return the chatbot value for the POST methon
4) The training files are included under training folder use it to train custom model

## How to use WkiiQna:-
1) As of 1st July 2021 I have used wkiiqna dataset provided by microsoft and used it to train on my existing model architecture with current accuracy of 25% which is not bad for 20 epochs
2) Since the model used is same we just need to download latest weights and provide its path to the model [my drive](https://drive.google.com/drive/folders/1jAuJpbaPXr0Sh6m7MnDCLVvWgVgg0CS-?usp=sharing) as on now the latest weight is model2.h5 and place it in folder model_weight_WkiiQna
3) Training notebook is provided under training folder 
4) Just need to specify the model path in the code where ever it is mentioned
5) Tokenizer is preincluded in the github file
### Note wikki qna bot model weights are preincluded in the git directory name model_weight_WkiiQna. So no need to download current latest weight is model3.h5
# encoder
## Single Encoder Layer
![alt text](/model-graphs/encoder_LAYER.png )

## Encoder
![alt text](/model-graphs/encoder.png )

# Decoder
## Single Decoder Layer
![alt text](/model-graphs/decoder_layer.png )

## Decoder
![alt text](/model-graphs/decoder.png )

# Transformer
![alt text](/model-graphs/Transformer.png )


