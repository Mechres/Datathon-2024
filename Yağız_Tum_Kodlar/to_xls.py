import pandas as pd

print("Train.csv dönüştür")
data = pd.read_csv('train.csv')
data.to_excel('train.xlsx')


print("Test_x.csv dönüştür.")

test_x = pd.read_csv('test_x.csv')
test_x.to_excel("test_x.xlsx")