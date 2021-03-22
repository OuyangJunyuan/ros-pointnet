#!/home/ou/software/anaconda3/envs/dl/bin/python
import sys
import os

dirname, filename = os.path.split(__file__)
sys.path.append(dirname.strip(dirname.split('/')[-1]))

import ROSbridge

if __name__ == '__main__':
    classifier = ROSbridge.CloudClassifier()

