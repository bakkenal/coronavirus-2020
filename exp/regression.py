from re import T
import sys
sys.path.insert(0, '..')
from utils import data
import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim

plt.style.use('fivethirtyeight')

# bounds
bounds = (0, [100000., 3, 1000000000.])
p0 = np.random.exponential(size=3)

# loading data
BASE_PATH = '../COVID-19/csse_covid_19_data/'
NORMALIZE = True

degree = range(10)
regressions = []

# logistic function
def my_logistic(a, b, c, t):
    return c / (1 + a * np.exp(-b*t))

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
features = []
metaInfo = []
confirmed = data.load_csv_data(confirmed)

for region in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", region)
    cases, labels = data.get_cases_chronologically(df)
    features.append(cases)
    metaInfo.append(labels)

# stripping zeros from beginning
for x in range(len(features)):
    features[x] = [i for i in features[x][0] if i != 0]

print("finished!")
print(features[2])
print("\n\n metaInfo (province, country, lat, long)")
print(metaInfo[2])

fig = plt.figure(figsize=(12, 12))
plt.title("regressions")
plt.xlabel("days since first case")
plt.ylabel("estimated number of cases")

# running regression
for y in range(len(features)):
    country = np.array(features[y])
    x = np.array(range(len(country))) + 1
    (a,b,c),cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)
    plt.plot(x, my_logistic(a, b, c, x), label=metaInfo[y][0][1])
    regressions.append([a, b, c])

print(regressions[10])
print(regressions[20])
print(regressions[30])
print(regressions[40])

plt.ylim([0, 60000])
plt.legend(fontsize='xx-small')

plt.savefig("Regressions.png")
plt.close()