#!/usr/bin/python3
import argparse
import knn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run KNN classifier on k within range specified by -l and -r.")
    parser.add_argument("-f", dest="filename", required=True, help="CSV file.")
    parser.add_argument("-l", dest="l", type=int, default=1)
    parser.add_argument("-r", dest="r", type=int, default=10)
    parser.add_argument("-n", "--normalize", dest="normalize", action="store_true", default=False)
    # parser.add_argument("-q", "--quiet", dest="quiet", action="store_true", default=False)
    args = parser.parse_args()

    data = knn.read_csv(args.filename)
    if args.normalize:
        knn.normalize_data(data)

    distances = knn.precalc_data(data)
    for r in range(args.l, args.r+1):
        print(knn.main(["-k", str(r)], data, distances))
