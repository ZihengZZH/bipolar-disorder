import sys, getopt
import numpy as np
import pandas as pd

from src.utility import preproc_baseline_feature
from src.baseline import BaseLine


def run_baseline_system(model_name, feature_name):
    baseline = BaseLine(model_name, feature_name)
    baseline.run()

def main(argv):
    model, feature = '', ''

    try:
        opts, _ = getopt.getopt(argv, "h:m:f:", ["model=", "feature="])
        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print("main.py -m <model_name> -f <feature_name>")
                sys.exit()
            elif opt in ('-m', '--model'):
                model = arg
            elif opt in ('-f', '--feature'):
                feature = arg
        if len(model) != 0 and len(feature) != 0:
            print("Baseline System with model %s and feature %s" % (model, feature))
            run_baseline_system(model, feature)

    except getopt.GetoptError:
        print("main.py -m <model_name> -f <feature_name>")
        sys.exit(2)
    
    

if __name__ == "__main__":
    main(sys.argv[1:])
