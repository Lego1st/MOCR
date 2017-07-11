from utils import predict
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-n", required = True, type = int, help = "Batch Size")
ap.add_argument("-r", required = True, type = str, help = "Root")
args = ap.parse_args()
predict(args.n, root = args.r)

