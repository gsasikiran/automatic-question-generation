import os
import wget
import argparse

parser = argparse.ArgumentParser(description='SQuAD Dataset Download')
parser.add_argument('--directory', default='../dataset', type=str, metavar='PATH',
                    help='path to download dataset (default: dataset directory)')

args = parser.parse_args()

def download_squad(directory = '../dataset'):
    '''
    Download SQuAD dataset and place it in dataset folder

    Arguments:
        directory -- location of dataset file, default is in dataset folder
    Returns:
        examples  -- list of example data from dataset
    '''

    print('Downloading SQuAD v.2.0 dataset...')
    urls = ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
    		'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',]

    if not os.path.exists(directory):
    	os.makedirs(directory)

    for url in urls:
    	filename = url.split('/')[-1]
    	if os.path.exists(os.path.join(directory, filename)):
    		print(file, ' already downloaded')
    	else:
    		wget.download(url=url, out=directory)

if __name__ == '__main__':
	download_squad(args.directory)