import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def read_and_sample(file_path, column_name, max_points=1000000, sep='\t'):
    # Estimate the size of the file
    file_size = os.path.getsize(file_path)

    # Assuming each point in the column takes approximately 8 bytes (for float)
    estimated_points = file_size / 8

    # If the estimated number of points is less than max_points, read the entire column
    if estimated_points <= max_points:
        data = pd.read_csv(file_path, usecols=[column_name], sep=sep)[column_name]
    else:
        # Sample the points
        skip = estimated_points // max_points
        data = pd.read_csv(file_path, usecols=[column_name], skiprows=lambda i: i % skip != 0, sep=sep)[column_name]

    return data

def calculate_percentiles(data, target_percentiles):
    percentiles = data.quantile(target_percentiles)
    return dict(zip(target_percentiles, percentiles))

def plot_distribution(data, column_name, plot_out=None, show=True):
    sns.kdeplot(data, fill=True)
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    if plot_out is not None:
        plt.savefig(plot_out)
    if show:
        plt.show()

def Summarize(file_path, column_name, plot_out=None, target_percentiles=[.1, .25, .5, .75, .9]):
    data = read_and_sample(file_path, column_name)
    percentiles = calculate_percentiles(data, target_percentiles)
    plot_distribution(data, column_name, plot_out=plot_out, show=False)
    return {'percentiles':percentiles, 'plot':plot_out}

