import sys, getopt


def display_help():
    print("--" * 20)
    print("Several options to run")
    print("--" * 20)
    print("baseline system")
    print("\tmain.py -b")
    print("experiment system")
    print("\tmain.py -x")
    print("landmark visualization")
    print("\tmain.py -v <partition>")
    print("help list")
    print("\tmain.py -h")
    print("--" * 20)


def run_baseline_system():
    from src.baseline import BaseLine
    models = ['SVM', 'RF', 'RF_cv']
    features = ['ALL', 'MFCC', 'eGeMAPS', 'BoAW', 'AU', 'BoVW']
    print("--" * 20)
    print("Available models:")
    for idx, m in enumerate(models):
        print(idx, m)
    model_id = int(input("choose a model: "))
    print("Available features:")
    for idx, f in enumerate(features):
        print(idx, f)
    feature_id = int(input("choose a feature: "))
    if feature_id != 0:
        baseline = BaseLine(models[model_id], features[feature_id])
        baseline.run()
    else:
        print("\nrunning baseline on all available features", features[1:])
        for i in range(1, 6):
            baseline = BaseLine(models[model_id], features[i])
            baseline.run()
            del baseline

def run_experiment_system():
    from src.experiment import Experiment
    exp = Experiment()
    exp.run()


def main(argv):
    try:
        opts, _ = getopt.getopt(argv, "hbv:x", ["help", "baseline", "visualize", "experiment"])
    except getopt.GetoptError as err:
        print(err)
        display_help()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            display_help()
        elif opt in ('-b', '--baseline'):
            print("Baseline System")
            print("--" * 20)
            run_baseline_system()
        elif opt in ('-x', '--experiment'):
            print("Experiment System")
            print("--" * 20)
            run_experiment_system()
        elif opt in ('-v', '--visualize'):
            print("Visualize facial landmarks on videos")
            print("--" * 20)
            if arg == 'recon':
                from src.utils.vis import visualize_reconstrcution
                visualize_reconstrcution(no_frame=False)
            else:
                from src.utils.vis import visualize_landmarks
                visualize_landmarks(arg, no_frame=False)
    

if __name__ == "__main__":
    main(sys.argv[1:])
