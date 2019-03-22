import numpy as np
import pandas as pd

from src.utility import preproc_baseline_feature
from src.baseline import BaseLine


def main():
    baseline_mfcc = BaseLine('eGeMAPS')
    baseline_mfcc.run()


if __name__ == "__main__":
    main()
