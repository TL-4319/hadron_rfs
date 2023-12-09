from datetime import datetime
import numpy as np
import carbs.swarm_estimator.tracker as tracker
import serums.models as smodels
import gncpy.filters as gfilts
import gncpy.dynamics as gdyn
import gncpy.plotting as gplot
from timeit import default_timer as timer
import time
import dill as pickle
import matplotlib.pyplot as plt
from serums.models import GaussianScaleMixture
from serums.enums import GSMTypes
import warnings
warnings.filterwarnings("ignore")
from scipy.io import savemat


def _state_mat_fun(t, dt):
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


meas_mat = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)



def set_inner_filter(filt_name, p_noise_pos, p_noise_vel, dt=None):

    if filt_name.lower() == 'kalman_filter':
        filt = gfilts.KalmanFilter()
        # filt.meas_noise = np.diag([0.0118, 0.0084])
        filt.meas_noise = np.diag([1, 1])
    elif filt_name.lower() == 'students_filter':
        filt = gfilts.StudentsTFilter()
        filt.proc_noise_dof = 33
        #filt.meas_noise_dof = 9.0
        #filt.meas_noise = np.diag([0.0033, 0.0076])
        filt.meas_noise_dof = 9
        filt.meas_noise = np.diag([0.0022, 0.0034])
    elif filt_name.lower() == 'gaussian_scale_mixture':
        filt = gfilts.KFGaussianScaleMixtureFilter()
        x_gsm = GaussianScaleMixture(gsm_type=GSMTypes.CAUCHY, scale=np.array([[0.003]]), scale_range=(0, 0.1),df_range=(1, 1))
        y_gsm = GaussianScaleMixture(gsm_type=GSMTypes.CAUCHY, scale=np.array([[0.003]]), scale_range=(0, 0.1),df_range=(1, 1))
        filt.set_meas_noise_model(gsm_lst=[x_gsm, y_gsm],num_parts=20)
    else:
        raise RuntimeError("Invalid inner filter argument.")
    obj = gdyn.DoubleIntegrator()
    filt.set_state_model(dyn_obj=obj)
    filt.set_measurement_model(meas_mat=meas_mat)
    if dt is None:
        dt = 1
        print("HARDCODED dt in inner filt process noise")
    Q00 = p_noise_pos**2*np.eye(2) * dt
    Q11 = p_noise_vel**2 * np.eye(2) * dt
    Q01 = Q11 * dt
    filt.proc_noise = np.vstack((np.hstack((Q00, Q01)), np.hstack((np.zeros((2,2)), Q11))))

    # filt.proc_noise = np.diag([p_noise_pos ** 2, p_noise_pos ** 2, p_noise_pos ** 2,
    #                            p_noise_vel ** 2, p_noise_vel ** 2, p_noise_vel ** 2])

    return filt

def define_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run tracker on input pik file")
    parser.add_argument("in_file", type=str, help="Name of the pickle file containing measurement data")
    parser.add_argument(
        "out_name",
        type=str,
        help="Full path to the file where filter outputs will be saved.",
    )
    parser.add_argument(
        "-t0",
        "--start-time",
        type=float,
        default=0.0,
        help="Starting timestamps (float). The default is 0.",
    )

    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=40,
        help="Time to track objects in seconds (int). The default is 40.",
    )
    parser.add_argument("-f", "--filter", type=str, help="Filter to be used.", default='jointgeneralizedlabeledmultibernoulli')
    parser.add_argument("-i", "--inner-filter", type=str, help="Inner filter to be used.", default='kalman_filter',choices=['kalman_filter', 'students_filter', 'gaussian_scale_mixture'])
    parser.add_argument(
        "--meas-hz",
        type=float,
        default=2.0,
        help="Measurement update in hz. The default is 2.0.",
    )
    parser.add_argument(
        "--pred-hz",
        type=float,
        default=2.0,
        help="Prediction update in hz. The default is 2.0.",
    )
    parser.add_argument(
        "--extract-pred",
        action="store_true",
        help="Flag to extract predicted states from filters. The default is false")

    return parser


def init_rfs_filter(f_type, if_type, dt=None):
    # mu = [np.append(data[0, 1:4].copy(), np.array([0, 0, 0])).reshape((6, 1))]
    # mu = [np.array([0.87, 3.64, 0, 0]).reshape((-1,1)),
    #       np.array([3.6, 0.35, 0, 0]).reshape((-1,1))]
    # mu = [np.array([4.600, 1.290, 0, 0]).reshape((-1,1))]
    # mu = [np.array([4.600, 2.00, 0, 0]).reshape((-1, 1))]
    # cov = [np.diag(np.array([0.75, 0.75, 0.07, 0.07])) ** 2]
    # cov = [np.diag(np.array([0.75, 1.5, 0.07, 0.07])) ** 2]

    # Set birth area at the lef and right edge of the image
    mu = [np.array([640, 200, 0, 0]).reshape((-1, 1)),
          np.array([0, 200, 0, 0]).reshape((-1, 1))]
    cov = [np.diag(np.array([100, 300, 40, 40])) ** 2,
           np.diag(np.array([100, 300, 40, 40])) ** 2]

    if if_type.lower() == 'kalman_filter' or if_type.lower() == 'gaussian_scale_mixture':
        gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[1 / len(mu) for m in mu])
    elif if_type.lower() == 'students_filter':
        gm0 = smodels.StudentsTMixture(means=mu, scalings=cov,dof=8,weights=[1 / len(mu) for m in mu])

    # cov = [np.diag(np.array([0.75, 0.75, 0.5, 0.5])) ** 2,
    #        np.diag(np.array([0.75, 0.75, 0.5, 0.5])) ** 2]
    # if if_type.lower() == 'kalman_filter' or if_type.lower() == 'gaussian_scale_mixture':
    #     gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[0.5, 0.5])
    # elif if_type.lower() == 'students_filter':
    #     gm0 = smodels.StudentsTMixture(means=mu, scalings=cov,dof=8,weights=[0.5, 0.5])

    b_model = [
        (gm0, 0.1),
    ]
    p_noise_pos = 3
    p_noise_vel = 5
    m_noise = 0.2

    filt = set_inner_filter(if_type, p_noise_pos, p_noise_vel, dt=dt)

    rfs_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        #"clutter_den": 1/(6.5*3*np.sqrt(2)/2*3/3),
        #"clutter_rate": 0.01,
        "clutter_den": 1/327680,
        #"clutter_den": 1/1000,
        "clutter_rate": 100,
    }
    glmb_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10 ** -5,
        "max_hyps": 1000,
        "save_covs": True,
        "save_measurements": True,
    }
    if f_type.lower() == "jointgeneralizedlabeledmultibernoulli" or f_type.lower() == 'jglmb':
        if if_type.lower() == 'kalman_filter':
            rfs_cls = tracker.JointGeneralizedLabeledMultiBernoulli
        elif if_type.lower() == 'students_filter':
            rfs_cls = tracker.STMJointGeneralizedLabeledMultiBernoulli
        else:
            rfs_cls = tracker.GSMJointGeneralizedLabeledMultiBernoulli
    elif f_type.lower() == "generalizedlabeledmultibernoulli" or f_type.lower() == 'glmb':
        if if_type.lower() == 'kalman_filter':
            rfs_cls = tracker.GeneralizedLabeledMultiBernoulli
        elif if_type.lower() == 'students_filter':
            rfs_cls = tracker.STMGeneralizedLabeledMultiBernoulli
        else:
            rfs_cls = tracker.GSMGeneralizedLabeledMultiBernoulli
    elif f_type.lower() == "poissonmultibernoullimixture" or f_type.lower() == "pmbm":
        rfs_args["birth_terms"] = [gm0]
        glmb_args = {
            "req_upd": 800,
            "prune_threshold": 10 ** -5,
            "exist_threshold": 10 ** -2,
            "max_hyps": 1000,
            "save_covs": True,
            "save_measurements": True,
        }
        print("initializing pmbm")
        if if_type.lower() == 'kalman_filter':
            rfs_cls = tracker.LabeledPoissonMultiBernoulliMixture
        elif if_type.lower() == 'students_filter':
            rfs_cls = tracker.STMLabeledPoissonMultiBernoulliMixture
        else:
            rfs_cls = tracker.LabeledPoissonMultiBernoulliMixture

    else:
        raise AttributeError("Invalid filter selection: {:s}".format(f_type))

    return rfs_cls(**glmb_args, **rfs_args)


if __name__ == "__main__":

    plt.close("all")

    args = define_parser().parse_args()

    rfs_filt = init_rfs_filter(args.filter.lower(), args.inner_filter.lower())

    timestep_count = 0

    elapsed_time = 0

    #fig = plt.figure()
    #fig.add_subplot(1, 1, 1)

    print("Starting loop... ({} seconds)".format(args.duration))
    # extract_kwargs = {"update": True, "calc_states": True}
    extract_kwargs = {"update": True, "calc_states": True}
    time_counter = 0
    des_meas_rate = 1 / args.meas_hz
    des_pred_rate = 1 / args.pred_hz
    last_pred_time = timer() - des_pred_rate
    last_meas_time = timer() - des_meas_rate
    min_rate = min(des_pred_rate, des_meas_rate)
    print_time = 0

    elapsed_time_vec = []
    real_time_vec = []

    meas_dict = pickle.load(open(args.in_file, 'rb'))
    meas_delays = meas_dict['meas_delays']
    all_meas = meas_dict['all_meas_list']
    timestamps = meas_dict['timestamps']
    time_list = []
    x_list = []
    y_list = []
    time_of_interest = 0
    start_time = timer()
    while elapsed_time < args.duration:
        loop_start_time = timer()
        needs_extract = False
        delta = timer() - last_pred_time
        if delta >= des_pred_rate:
            # print(delta)
            #print("\tElapsed Time: {:.2f}".format(elapsed_time))
            # rfs_filt.predict(
            #     time_counter, filt_args={"state_mat_args": (timer() - last_pred_time,)},
            # )
            rfs_filt.predict(
                timer() - start_time, filt_args={"state_mat_args": (des_pred_rate,)},
            )
            last_pred_time = timer()
            if args.extract_pred:
                needs_extract = True
            #print("\t\tPredict at {:.2f}".format(last_pred_time - start_time),flush=True)
            if last_pred_time - loop_start_time > des_pred_rate:
                print(
                    "Loop {:4d} took longer than desired prediction update. ({:.2f})".format(
                        time_counter, last_pred_time - loop_start_time
                    )
                )
        #print("\t\t\t timer() - last_meas_time: {:.2f}".format(timer() - last_meas_time),flush=True)
        delta = timer() - last_meas_time

        if delta >= des_meas_rate:
            # print(delta)
            # print("\tElapsed Time: {:.2f}".format(elapsed_time))
            cur_ind = np.where(np.abs(np.array(timestamps)-args.start_time-elapsed_time)==np.min(np.abs(np.array(timestamps) - args.start_time - elapsed_time)))[0][0]
            time_of_interest = timestamps[cur_ind]
            meas = all_meas[cur_ind]
            time.sleep(meas_delays[cur_ind])
            time_step = timer() - start_time
            #print (len(meas))
            rfs_filt.correct(time_step, meas)
            #print('correct called with meas: ', meas)
            #print('num hyps: ', len(rfs_filt._hypotheses))
            last_meas_time = timer()
            needs_extract = True
            #time_list.append(elapsed_time)
            if len(meas) == 0:
                continue
            else:
                for arr in meas:
                    time_list.append(elapsed_time)
                    x_list.append(arr[0])
                    y_list.append(arr[1])
            #print("\t\tCorrect at {:.2f}".format(last_meas_time - start_time),flush=True)
            if last_meas_time - loop_start_time > des_meas_rate:
                print(
                    "Loop {:4d} took longer than desired correction update. ({:.2f})".format(
                        time_counter, last_meas_time - loop_start_time
                    )
                )

        if needs_extract:
            extract_kwargs = {"update": True, "calc_states": False}
            rfs_filt.cleanup(extract_kwargs=extract_kwargs)
            elapsed_time_vec.append(timer() - start_time)
            real_time_vec.append(time_of_interest)
            #extract_kwargs = {"update": False, "calc_states": True}
            #rfs_filt.extract_states(**extract_kwargs)
            #rfs_filt.plot_states_labels([0, 1], meas_inds=[0, 1], f_hndl=fig)
            #plt.pause(0.001)

        time_counter = time_counter + 1

        elapsed_time = timer() - start_time
        #elapsed_time += (timer() - loop_start_time)
        if elapsed_time - print_time >= 2:
            print("\tElapsed Time: {:.2f}".format(elapsed_time))
            print_time = elapsed_time


    print("Elapsed Time: {:.2f}".format(elapsed_time))
    extract_kwargs = {"update": False, "calc_states": True}
    rfs_filt.extract_states(**extract_kwargs)


    filt_state = rfs_filt.save_filter_state()
    data = {'rel_time': elapsed_time_vec,
            'states': filt_state['_states'],
            'covs': filt_state['_covs'],
            'labels': filt_state['_labels'],
            'real_time': real_time_vec}
    savemat(
        "/home/lagerprocessor/Projects/hadron_rfs/ir_gm_jglmb_2hz_1701966775_029.mat",
        data)
    rfs_filt.plot_states_labels([0, 1], meas_inds=[0, 1])
    rfs_filt.plot_card_history(time_units='s',time=elapsed_time_vec)

    data = {'rel_time': elapsed_time_vec,
            'states': filt_state['_states'],
            'covs': filt_state['_covs'],
            'labels': filt_state['_labels'],
            'real_time': real_time_vec}
    savemat(
        "ir_gm_jglmb_2hz_170196810_153.mat",
        data)

    #time_list = []
    #x_list = []
    #y_list = []
    #for ii, list in enumerate(all_meas):
    #    if len(list)==0:
    #        continue
    #    else:
   #         for arr in list:
     #           time_list.append(timestamps[ii]-timestamps[0])
     #           x_list.append(arr[0])
     #           y_list.append(arr[1])

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time_list, x_list, 'ko')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Meas x pos")
    plt.ylim(0, 640)
    plt.subplot(2,1,2)
    plt.plot(time_list, y_list, 'ko')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Meas y pos")
    plt.ylim(0, 512)
    plt.show()


