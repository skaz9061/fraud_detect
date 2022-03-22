import pandas as pd
import matplotlib.pyplot as plt


def show_correlation_heatmap(df):
    plt.imshow(df.corr())
    plt.colorbar()
    plt.xticks(range(len(df.columns)), df.columns, rotation='vertical')
    plt.yticks(range(len(df.columns)), df.columns)
    plt.show()


datasource_loc = "creditcard.csv"
df = pd.read_csv(datasource_loc)
show_correlation_heatmap(df)

