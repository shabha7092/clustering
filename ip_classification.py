import os
import glob
import json
import argparse
import matplotlib
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
from jenks import jenks
from jenkspy import jenks_breaks
import matplotlib.pyplot as plt
plt.ticklabel_format(useOffset=False)
matplotlib.interactive(True)


def read_data_to_dataframe(path):
    '''
    Method to read the data in to DataFrame
    '''
    if path is None:
        raise ValueError('Need Path as argument !')
    files = glob.glob(os.path.join(path, '*'))
    data = pd.concat([pd.read_csv(file, sep=',', index_col=None, header=None) for file in files], ignore_index = True)
    data.columns = ['ip_address', 'bcookies_count']
    return data

def scatter_plot(data=None, output_path=None):
    '''
    Method to draw a Scatter Plot
    '''
    if data is None or output_path is None:
        raise ValueError('Need Data and as argument !')
    data = data.copy()
    data.index = np.arange(1,len(data)+1)
    data = data.reset_index()
    out = output_path + '/Ip-Bcookie-Scatter-Plot'
    fg = sns.FacetGrid(data=data, height=5)
    fg.fig.suptitle('Ip \'s / Bcookie Count Scatter Plot')
    fg.map(plt.scatter, 'index', 'bcookies_count').add_legend()
    fg.savefig(out)
    return data

def scatter_plot_classified(data=None, output_path=None):
    '''
    Method to draw a Scatter Plot
    '''
    if data is None or output_path is None:
        raise ValueError('Need Data and as argument !')
    data = data.copy()
    data.index = np.arange(1,len(data)+1)
    data = data.reset_index()
    labels = data['classification'].unique()
    out = output_path + '/Ip-Bcookie-Scatter-Plot-with-classification'
    fg = sns.FacetGrid(data=data, hue='classification', hue_order=labels, height=5)
    fg.fig.suptitle('Ip \'s / Bcookie Count Scatter Plot with Classification')
    fg.map(plt.scatter, 'index', 'bcookies_count').add_legend()
    fg.savefig(out)
    return None

def pie_chart_classfied(data=None, output_path=None):
    '''
    Method to draw a Pie Plot
    '''
    if data is None or output_path is None:
        raise ValueError('Need Data and as argument !')
    data = data.copy()
    out = output_path + '/Ip-classification-Pie-Plot'
    data = data.groupby('classification').size()
    fg, ax = plt.subplots()
    data.plot(kind='pie', y = 'classification', ax=ax, autopct='%1.1f%%', legend=True)
    ax.set_title('Ip Address Classification Pie Chart')
    ax.set_ylabel('classification')
    fg.savefig(out)
    return None


def read_data_from_json(file=None):
    '''
    Method to read data from json file
    '''
    if file is None:
        raise ValueError('Need File as argument !')
    data = json.load(open(file))
    return data

def run_jenks_algo(data=None, classes=2):
    '''
    Method to execute Jenks natural breaks Algorithm
    '''
    if  data is None:
        raise ValueError('Need Data as argument !')
    jenks_map = {}
    for i in range(2, classes+1):
        iter_map = {}
        gvf, classified, classes = goodness_of_variance_fit(data['bcookies_count'], i)
        iter_map['classified'] = classified
        iter_map['classes'] = classes
        jenks_map[gvf] = iter_map
    best_gvf = max(jenks_map)
    best_iter_map = jenks_map[best_gvf]
    data['classification'] = pd.Series(best_iter_map['classified']).values
    return best_gvf, best_iter_map['classes'], data

def goodness_of_variance_fit(data=None, classes=2):
    '''
    Method to compute goodness of variance fit
    '''
    if data is None:
        raise ValueError('Need Data as argument !')
    #classes = jenks(data, classes)
    classes = jenks_breaks(data.unique(), classes)
    print(classes)
    classified = np.array([classify(i, classes) for i in data])
    maxz = max(classified)
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]
    sdam = np.sum((data - np.mean(data)) ** 2)
    data_sort = [np.array([data[index] for index in zone]) for zone in zone_indices]
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in data_sort])
    gvf = (sdam - sdcm) / sdam
    return gvf, classified, classes


def classify(value, breaks):
    '''
    Method to classify a value
    '''
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1

def write_data(data=None, output_path=None, sep='\t'):
    '''
    Method to write the data to file
    '''
    if data is None or output_path is None:
        raise ValueError('Need Data and Output Path as arguments !')
    data.to_csv(os.path.join(output_path, 'classfications.tsv'), sep = sep)
    return None

def main(input_path, output_path, num_classes):
    output_path = os.path.join(os.getcwd(), "Output/{}".format(dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    os.makedirs(output_path)
    data = read_data_to_dataframe(input_path)
    with open(output_path + '/stats.txt', 'w') as f:
        print('Plotting Scatter Plot', file=f)
        data = scatter_plot(data, output_path)
        print('Executing Jenks natural breaks Algorithm', file=f)
        gvf, classes, data = run_jenks_algo(data, num_classes)
        print('Best Good Variance Fit', file=f)
        print(gvf, file=f)
        print('Classes for Good Variance Fit', file=f)
        print(classes, file=f)
        print('Writing Clasifications to File',file=f)
        write_data(data, output_path)
        print('Plotting Scatter Plot with Classification', file=f)
        scatter_plot_classified(data, output_path)
        print('Plotting Ip Address Classification Pie Plot', file=f)
        pie_chart_classfied(data, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
            default=os.path.join(os.getcwd(), "ipn2"),
            help="path to input data")
    parser.add_argument('--output_path',
            default=os.path.join(os.getcwd(), "Output"),
            help="path to output data")
    parser.add_argument('--num_classes',
            default=3,
            help="number of classes")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.num_classes)