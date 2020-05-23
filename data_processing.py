# data preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
def data_processing():
    with open("data/paracrawl-release1.en-ru.zipporah0-dedup-clean.en") as f:
        en = []
        for step,line in enumerate(f):
            en.append(line)
    print("english length: ", step)
    with open("data/paracrawl-release1.en-ru.zipporah0-dedup-clean.ru") as f:
        ru = []
        for step,line in enumerate(f):
            ru.append(line)
    print("russian length: ", step)
    print("success extract data!")

    df = pd.DataFrame({"text":en, "target": ru})
    df = df.replace('\n','', regex=True)

    train ,valid = train_test_split(df, test_size=0.1, random_state = 42)
    train.to_csv("data/train.csv")
    valid.to_csv("data/val.csv")