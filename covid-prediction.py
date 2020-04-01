import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")

from plotly.offline import init_notebook_mode, iplot, plot
import plotly.express as px
import plotly.graph_objs as go
import math 
import datetime

#import os
#for dirname, _, filenames in os.walk('./'):#
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


#test=pd.read_csv('covid.csv',parse_dates=True,)
#print (test.tail())
poland_covid_19_summary = pd.read_csv('covid.csv', index_col='Data', parse_dates = True)
poland_covid_19_summary["Zmiana"] = poland_covid_19_summary.Aktywni.diff()
poland_covid_19_summary["Wzrost"] = poland_covid_19_summary.Aktywni.div(other=poland_covid_19_summary.Aktywni.shift(1))
print(poland_covid_19_summary.tail())
poland_covid_19_summary.insert(0, 'Dzien', range(0, len(poland_covid_19_summary)))
sick_model = LinearRegression(fit_intercept=True)
poly = PolynomialFeatures(degree=8)
num_days_poly = poly.fit_transform(poland_covid_19_summary.Dzien.values.reshape(-1,1))
poly_reg = sick_model.fit(num_days_poly, poland_covid_19_summary.Aktywni.values.reshape(-1,1))
predictions_for_given_days = sick_model.predict(num_days_poly)

print("coef_ :",sick_model.coef_,"intercept_:",sick_model.intercept_)
pass
today = poland_covid_19_summary.Dzien.iloc[-1]
print(f'Dzisiaj jest {today} dzień pandemii w Polsce')


tomorrow_value = today

for i in range(0, 2):
    tomorrow_value += 1
    value_prediction = poly.fit_transform(np.array([[tomorrow_value]]))
    prediction = sick_model.predict(value_prediction)
    przyrost = prediction-(poland_covid_19_summary.Suma[poland_covid_19_summary.Dzien.iloc[-1]]) 
    print(f'Prognoza na dzień {tomorrow_value} : {prediction} stwierdzonych przypadków ')
    print (f'Czyli zmiana o {przyrost}')
def model(N, a, alpha, t):
    # we enforce N, a and alpha to be positive numbers using min and max functions
    return max(N, 0) * (1 - math.e ** (min(-a, 0) * t)) ** max(alpha, 0)
def model_loss(params):
    N, a, alpha = params
    model_x = []
    r = 0
    for t in range(len(poland_covid_19_summary)):
        r += (model(N, a, alpha, t) - poland_covid_19_summary.Aktywni.iloc[t]) ** 2
#         print(model(N, a, alpha, t), df.iloc[t, 0])
        return r 
#import numpy as np
from scipy.optimize import minimize
opt = minimize(model_loss, x0=np.array([200000, 0.1, 15]), method='Nelder-Mead', tol=1e-5).x
opt 
model_x = []
for t in range(len(poland_covid_19_summary)):
    model_x.append([poland_covid_19_summary.index[t], model(*opt, t)])
model_sim = pd.DataFrame(model_x, dtype=int)
model_sim.set_index(0, inplace=True)
model_sim.columns = ['Model']
pd.concat([poland_covid_19_summary, model_sim], axis=1)

import datetime
start_date = poland_covid_19_summary.index[0]
n_days = 50
extended_model_x = []
for t in range(n_days):
    extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt, t)])
extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)
extended_model_sim.set_index(0, inplace=True)
extended_model_sim.columns = ['Model']
pd.concat([extended_model_sim, poland_covid_19_summary], axis=1)

print (extended_model_sim)

# @studiobtg  Krzywa logistyczna A/B+EXP((-x+D)/E) A=9200, B=2, D=31,22, E=4,8 .. 4,3
#  7dniowa EXP odcięta * EXP dzienny przyrost * T ) (odcięta 14,41 , dzienny przyrost 13,1%) pesymistyczna (odcięta 3,13 , dzienny przyrost 19,3%) przyrost  (C23/C22)-1)
# siec neuronowa tanh bartosz growiec 
# zwykła exp  2.2604 EXP 0,2897x 
# https://rmostowy.github.io/covid-19/prognoza-polska/
# https://en.wikipedia.org/wiki/Logistic_distribution
# https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions-7784b4d7dd8d
# https://medium.com/analytics-vidhya/predicting-the-spread-of-covid-19-coronavirus-in-us-daily-updates-4de238ad8c26