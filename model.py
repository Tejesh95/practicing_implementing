import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('house.csv')
x = df[['area','bedrooms','bathrooms','parking','mainroad','guestroom','basement','hotwaterheating','airconditioning']]
y=df['price']
x = pd.get_dummies(x, drop_first=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
# print(model.score(x_test,y_test))
import pickle
pickle.dump(model,open('house_model.pkl','wb'))
