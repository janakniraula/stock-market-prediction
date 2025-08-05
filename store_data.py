from utils import Preprocess

symbols = ['NABIL','ADBL','NICA','SHL','EBL','HIDCL']
for symbol in symbols:
    df = Preprocess.getData(symbol)
    df.to_csv(f'./data/{symbol}.csv', index=True)
    print(f"Stored {symbol}")
