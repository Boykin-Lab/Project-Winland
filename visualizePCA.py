import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make the data
np.random.seed(3)
x = 4 + np.random.normal(0, 2, 24)
y = 4 + np.random.normal(0, 2, len(x))
# size and color:
sizes = np.random.uniform(15, 80, len(x))
colors = np.random.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

plt.show()
"""
Script to visualize a csv file
I want
"""
#column
def read_data(path):

    with open(path) as data_file:
        data = pd.read_csv(data_file)
        buff = []
        for line in data_file:
                line = line.split()
                if line:
                        line = [int(i) for i in line]
                        data 
    headers = list(data.columns)
    
    print(headers)

    data.head()

    return headers

def main():
    csvfile = sys.argv[1]
    print("Main File started")
    read_data(csvfile)


if __name__ == "__main__":

    main()

