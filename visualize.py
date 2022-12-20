import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

"""
Script to visualize a csv file based on the columns we want
"""

def read_data(path):
    buff = pd.read_csv(path, sep='\t')
    df = pd.DataFrame(buff)
    #print(df.head())
    return df

#DONE
def barplot(df):
    print("barplot")
    plt.bar(df.index, df['Proportions'])
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.figure(figsize=(8,5))
    plt.show()
    return

def scatterplot(df,colx,coly):
    print("scatter")
    plt.title("Data from flashPCA2.0 Plotting PC1 -PC2")
    plt.xlabel(colx)
    plt.ylabel(coly)
    plt.scatter(df[colx], df[coly])
    plt.show()


#Create the biplot function
def biplot(df,coeff,labels=None):
    xs = df['PC1']
    ys = df['PC2']
    print(ys)
    tpose = np.ones(coeff)
    n = tpose.transpose(tpose).shape
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, s = 8)
    for i in range(n):
        plt.arrow(0, 0, coeff[0], coeff[1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.show()

def radplot(df):
    print("radviz")
    print(df.head())
    col_names = df.columns.values
    print(df.columns.values)
    pd.plotting.radviz(df[2:],col_names)
    print("radkilled")


def main():
    # path = input("Type p to graph proportion variance, otherwise enter pcs filename:")
    # if path == "p":
    #     pve = input("Type pve file name to graph component explained proportion:")
    #     path = pve
    pca = read_data("UK_overlaps_pcs.txt")
    pca_df = pca[1:]
    print("Data Head")
    print(pca_df.head())
    pve = read_data("pve.txt").to_numpy()
    pve = pd.DataFrame(pve, columns=['Proportions'])
    scatterplot(pca_df,"PC1", "PC2")
    barplot(pve )

    biplot(pca_df, pve["Proportions"], labels=pca_df.columns.values)
    #plot_cat = int(input("Choose plot type (1) barplot, (2) biplot:"))
    #radplot(data[2:])
    print("End")


if __name__ == "__main__":

    main()
