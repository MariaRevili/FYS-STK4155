from scipy.io import arff
import pandas as pd

data = arff.loadarff('EEG Eye State.arff')
df = pd.DataFrame(data[0])
df["eyeDetection"]=df["eyeDetection"].astype(int)

df.to_csv(path_or_buf='/path/to/target/destination/eyeData.csv')
