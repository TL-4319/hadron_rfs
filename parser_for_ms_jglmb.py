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

    parser = argparse.ArgumentParser(
        description="Convert hadron output csv to pik file"
    )

    parser.add_argument(
        "iname",
        type=str,
        help="Full path to the input csv file.",
    )

    parser.add_argument(
        "iname2",
        type=str,
        help="Full path to the second input csv file.",
    )

    parser.add_argument(
        "oname",
        type=str,
        help="Full path to the output pik file.",
    )

    return parser


if __name__ == "__main__":
    args = define_parser().parse_args()

    skip_row_reader1 = 1
    skip_row_reader2 = 16

    j = 0

    with open(args.iname) as csvfile, open(args.iname2) as csvfile2:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        reader2 = csv.reader(csvfile2, delimiter=",", quotechar="|")

        timestamps = []
        meas_delay = []
        meas_list = []

        for _ in range(skip_row_reader1):
            next(reader, None)

        for _ in range(skip_row_reader2):
            next(reader2, None)

        for row1, row2 in zip(reader, reader2):
            try:
                # print(row1[0], row2[0])
                # time.sleep(2)
                cur_timestamp = float(row1[0])
                cur_meas_delay = float(row1[1])
                timestamps.append(cur_timestamp)
                meas_delay.append(cur_meas_delay)
                num_detection_1 = int((len(row1) - 2) / 3)
                num_detection_2 = int((len(row2) - 2) / 3)
                cur_meas_list_1, cur_meas_list_2 = [], []

                # Handle no measurement cases
                if num_detection_1 != 0:
                    for i in range(num_detection_1):
                        bearing = float(row1[i * 3 + 4])
                        azimuth = float(row1[i * 3 + 3])
                        xPixel = (azimuth / degreePerPixel) + centerXPixel
                        yPixel = (bearing / degreePerPixel) + centerYPixel
                        cur_meas_vec = np.array([[azimuth], [bearing]])
                        cur_meas_list_1.append(cur_meas_vec)

                if num_detection_2 != 0:
                    for i in range(num_detection_2):
                        bearing = float(row2[i * 3 + 4])
                        azimuth = float(row2[i * 3 + 3])
                        xPixel = (azimuth / degreePerPixel) + centerXPixel
                        yPixel = (bearing / degreePerPixel) + centerYPixel
                        cur_meas_vec = np.array([[azimuth], [bearing]])
                        cur_meas_list_2.append(cur_meas_vec)

                combined_meas_list = [cur_meas_list_1, cur_meas_list_2]
                meas_list.append(combined_meas_list)

            except Exception as e:
                # print (e)
                # Handle header rows
                continue

        # saved_outputs = {
        #     "timestamps": timestamps,
        #     "meas_delays": meas_delay,
        #     "all_meas_list": meas_list,
        # }

        # print(meas_list)

        pickle.dump(meas_list, open(args.oname, "wb"))
        print("Conversion completed. Wrote {} entries".format(len(timestamps)))
