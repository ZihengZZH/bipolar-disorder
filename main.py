import sys, getopt
from src.model.baseline import BaseLine
from src.utils.vis import visualize_landmarks
from src.experiment import AE_BOW, BAE_XBOW


def display_help():
    print("--" * 10)
    print("Several options to run")
    print("--" * 20)
    print("baseline system")
    print("\tmain.py -m <model_name> -f <feature_name>")
    print("landmark visualization")
    print("\tmain.py -v <partition>")
    print("help list")
    print("\tmain.py -h")
    print("--" * 20)


def run_baseline_system(model_name, feature_name):
    baseline = BaseLine(model_name, feature_name)
    baseline.run()


def main(argv):
    model, feature = '', ''
    vis, exp = '', ''
    try:
        opts, _ = getopt.getopt(argv, "h:m:f:v:x:")
        for opt, arg in opts:
            if opt in ('-h', '--help'):
                display_help()
                sys.exit()
            elif opt in ('-m', '--model'):
                model = arg
            elif opt in ('-f', '--feature'):
                feature = arg
            elif opt in ('-v', '--visualize'):
                vis = arg
            elif opt in ('-x', '--experiment'):
                exp = arg
        if len(model) != 0 and len(feature) != 0:
            print("Baseline System with model %s and feature %s" % (model, feature))
            run_baseline_system(model, feature)
        if len(vis) != 0:
            print("Visualize facial landmarks on videos")
            visualize_landmarks(vis)
        if len(exp) != 0:
            print("Experiment begins")
            AE_BOW()

    except getopt.GetoptError:
        display_help()
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
