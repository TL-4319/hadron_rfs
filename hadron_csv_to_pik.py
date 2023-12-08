from datetime import datetime
import numpy as np
from timeit import default_timer as timer
import time
import dill as pickle
import numpy as np
import csv
import warnings
warnings.filterwarnings("ignore")
centerXPixel = 320
centerYPixel = 256
degreePerPixel = 0.05

def define_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Convert hadron output csv to pik file")

    parser.add_argument(
        "iname",
        type=str,
        help="Full path to the input csv file.",
    )

    parser.add_argument(
        "oname",
        type=str,
        help="Full path to the output pik file.",
    )

    return parser

if __name__ == "__main__":
    args = define_parser().parse_args()
    with open (args.iname) as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')
        timestamps = []
        meas_delay = []
        meas_list = []
        for row in reader:
            try:
                cur_timestamp = float(row[0])
                cur_meas_delay = float(row[1])
                timestamps.append(cur_timestamp)
                meas_delay.append(cur_meas_delay)
                num_detection = int((len(row) - 2) / 3)
                cur_meas_list = []
                # Handle no measurement cases
                if num_detection == 0:
                    meas_list.append(cur_meas_list)
                    continue
                for i in range(num_detection):
                    bearing = float(row[i*3 + 4])
                    azimuth = float(row[i*3 + 3])
                    xPixel  = (azimuth / degreePerPixel) + centerXPixel
                    yPixel = (bearing / degreePerPixel) + centerYPixel
                    cur_meas_vec = np.array([[xPixel],[yPixel]])
                    cur_meas_list.append(cur_meas_vec)
                meas_list.append(cur_meas_list)
                #print (num_detection)

            except Exception as e:
               #print (e)
                # Handle header rows
                continue

        saved_outputs = {'timestamps' : timestamps,
                         'meas_delays' : meas_delay,
                         'all_meas_list' : meas_list}

        pickle.dump(saved_outputs, open(args.oname, "wb"))
        print ("Conversion completed. Wrote {} entries".format(len(timestamps)))