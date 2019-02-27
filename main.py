import numpy as np
import pandas as pd

from src.utility import load_label, load_LLD, load_baseline_feature


def main():
    label_train, label_dev = load_label()
    
    load_LLD('MFCC', 'train', 1)
    load_LLD('eGeMAPS', 'dev', 30)
    load_LLD('openFace', 'test', 10)

    load_baseline_feature('BoAW', 'train', 100)
    load_baseline_feature('eGeMAPS', 'dev', 50)
    load_baseline_feature('BoVW', 'test', 30)


if __name__ == "__main__":
    main()
