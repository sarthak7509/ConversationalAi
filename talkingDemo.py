import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pickle
from colorama import init
from colorama import Fore, Back
from coversationalAi import conversationalApi
import argparse
init()
model = None


ap = argparse.ArgumentParser()

ap.add_argument("-b", "--bot", required=True,
	help="path to the weight of madel to use")
ap.add_argument("-t", "--tokenize", required=True,
	help="path to tokenizer to use")
ap.add_argument("-n", "--name", required=False,
	help="name you want to give bot",default="powerbot")
args = vars(ap.parse_args())



your_name = input('Enter your name: ')
bot_name = args["name"]

print(f"{Back.BLUE}\n{bot_name} almost ready...{Back.RESET}")


def loadModel():
    global model
    #To use bot trained on Cornell movie dataset change its path to respective tokenizer
    with open(args["tokenize"], 'rb') as f:
        tokenizer = pickle.load(f)

    #choosing hyperparameter
    NUM_LAYERS=2
    D_MODEL=256
    NUM_HEADS=8
    UNITS = 512
    DROPOUT=0.1

    #TO use Conrnell bot replace it with corn weight path the weights are available on drive
    model_path = args["bot"]
    model = conversationalApi(tokenizer=tokenizer, model_weight_path=model_path,MAXLENGTH=100,NUM_LAYERS=NUM_LAYERS,D_MODEL=D_MODEL,NUM_HEADS=NUM_HEADS,UNITS=UNITS,DROPOUT=DROPOUT)
print("[+]Loading bot")
loadModel()
print("[+]bot loaded")

print(f"{Back.BLUE}\nPlease start the Asking Qna: {Back.RESET}")
while True:
    print(Fore.LIGHTYELLOW_EX + "")
    prompt = input(f"{your_name}: ")
    print(Fore.RESET + "")
    print(f"{Fore.LIGHTMAGENTA_EX}{bot_name}: {model.predict(prompt)}{Fore.RESET}\n")
