import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from copy import deepcopy
import sys

import scipy.stats as stats

import gncpy.filters as gfilts
import gncpy.dynamics as gdyn
import gncpy.distributions as gdistrib
import gncpy.plotting as pltUtil
import gncpy.data_fusion as gfuse
from carbs.swarm_estimator.tracker import LabeledPoissonMultiBernoulliMixture, MSLabeledPoissonMultiBernoulliMixture, MSJointGeneralizedLabeledMultiBernoulli, JointGeneralizedLabeledMultiBernoulli
import serums.models as smodels
from serums.enums import GSMTypes, SingleObjectDistance

import csv

global_seed = 69
debug_plots = False

class DoubleIntegrator3D(gdyn.LinearDynamicsBase):
    """Implements a double integrator model."""

    state_names = ("x pos", "y pos", "z pos", "x vel", "y vel", "z vel")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_dis_process_noise_mat(self, dt, proc_cov):
        """Discrete process noise matrix.

        Parameters
        ----------
        dt : float
            time difference, unused.
        proc_cov : N x N numpy array
            Covariance matrix of the process noise.

        Returns
        -------
        6 x 6 numpy array
            process noise matrix.

        """
        gamma = np.array([0, 0, 0, 1, 1, 1]).reshape((6, 1))
        return gamma @ proc_cov @ gamma.T

    def get_state_mat(self, timestep, dt):
        """Class method for getting the discrete time state matrix.

        Parameters
        ----------
        timestep : floatfilt.meas_noise = m_noise ** 2 * np.eye(3)
            timestep.
        dt : float
            time difference

        Returns
        -------
        4 x 4 numpy array
            state matrix.

        """
        return np.array(
            [
                [1.0, 0.0, 0.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

def extract_data(filename, skips):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)

        data = []
        time = []
        loop_time = []
        all_meas = []

        for ii, row in enumerate(reader):
            if ii < skips:
                continue
            data.append(row)
            time.append(float(row[0]))
            loop_time.append(float(row[1]))
            meas = []
            # print(ii, skips, filename)
            for jj in range(3, len(row), 3):
                cur_meas_pair = np.array([float(row[jj]), float(row[jj+1])]).reshape([2,1])
                if type(cur_meas_pair[0]) is None:
                    continue
                meas.append(cur_meas_pair)
            # meas.append([])
            all_meas.append(meas)

    return np.array(time), np.array(loop_time), all_meas

def match_data(red_time, blue_time, red_meas, blue_meas):
    if len(blue_time) > len(red_time):
        first = red_time[0]
        last = red_time[-1]
        first_min_diff = np.min(np.abs(blue_time - first))
        last_min_diff = np.min(np.abs(blue_time - last))
        first_2 = np.where(first_min_diff==np.abs((blue_time - first)))[0][0]
        last_2 = np.where(last_min_diff==np.abs((blue_time - last)))[0][0]
        
        if last_2 > len(red_time) - 1:
            case = 0
            loop_range = range(0, len(red_time), 1)
            lower_time = red_time
            lower_meas = red_meas
            higher_time = blue_time
            higher_meas = blue_meas
        else:
            case = 1
            loop_range = range(0, last_2, 1)
            lower_time = blue_time
            lower_meas = blue_meas
            higher_time = red_time
            higher_meas = red_meas

    else:
        first = blue_time[0]
        last = blue_time[-1]
        first_min_diff = np.min(np.abs(red_time - first))
        last_min_diff = np.min(np.abs(red_time - last))
        first_2 = np.where(first_min_diff==np.abs((red_time - first)))[0][0]
        last_2 = np.where(last_min_diff==np.abs((red_time - last)))[0][0]
        
        if last_2 > len(blue_time) - 1:
            case = 1
            loop_range = range(0, len(blue_time), 1)
            lower_time = blue_time
            lower_meas = blue_meas
            higher_time = red_time
            higher_meas = red_meas
        else:
            case = 0
            loop_range = range(0, last_2, 1)
            lower_time = red_time
            lower_meas = red_meas
            higher_time = blue_time
            higher_meas = blue_meas

    matched_lower_time = []
    matched_lower_meas = []
    matched_higher_time = []
    matched_higher_meas = []
    for ii in loop_range:
        cur_lower_time = lower_time[ii]
        higher_time_ind = np.where(np.min(np.abs(higher_time - cur_lower_time))==np.abs(higher_time - cur_lower_time))[0][0]
        cur_higher_time = higher_time[higher_time_ind]

        matched_lower_time.append(cur_lower_time)
        matched_higher_time.append(cur_higher_time)

        for jj, item in enumerate(lower_meas[ii]):
            lower_meas[ii][jj] = item * np.pi / 180
        # matched_lower_meas.append(lower_meas[ii] + [[]])
        matched_lower_meas.append(lower_meas[ii])
        for jj, item in enumerate(higher_meas[higher_time_ind]):
            higher_meas[higher_time_ind][jj] = item * np.pi / 180
        matched_higher_meas.append(higher_meas[higher_time_ind])
        # matched_higher_meas.append(higher_meas[higher_time_ind] + [[]])

    if case == 0:
        new_red_time = matched_lower_time
        new_red_meas = matched_lower_meas
        new_blue_time = matched_higher_time
        new_blue_meas = matched_higher_meas
    else:
        new_red_time = matched_higher_time
        new_red_meas = matched_higher_meas
        new_blue_time = matched_lower_time
        new_blue_meas = matched_lower_meas

    return new_red_time, new_blue_time, new_red_meas, new_blue_meas

def fuse_measurements(red_time, blue_time, red_meas, blue_meas):

    for tt in range(len(red_time)):
        
        est_list = 0
        fused_meas, fused_meas_cov, weight_list = GeneralizedCovarianceIntersection(est_list, cov_list, weight_list, meas_model_list, optimizer=None)

    return 0




def extract_truth(truth_file, meas_time):
    truth_time = []
    global_true = []

    true_obj1 = []
    times_obj1 = []
    true_obj2 = []
    times_obj2 = []

    with open(truth_file) as csvfile:
        reader = csv.reader(csvfile)
        for ii, row in enumerate(reader):
            if ii == 0:
                continue
            numerical_row = []
            for item in row:
                numerical_row.append(float(item))
            cur_gl_true = []
            if ii % 2 == 0:
                temp_time = numerical_row[0] - 1 + 0.89982
                truth_time.append(temp_time)
            else:
                temp_time = numerical_row[0] + 0.39982
                truth_time.append(temp_time)
            temp_pos1 = np.array(numerical_row[1:4])

            temp_pos2 = np.array(numerical_row[4:7])
            if not np.all(np.isnan(temp_pos1)):
                cur_gl_true.append(temp_pos1)
                true_obj1.append(temp_pos1)
                times_obj1.append(temp_time)
            if not np.all(np.isnan(temp_pos2)):
                cur_gl_true.append(temp_pos2)
                true_obj2.append(temp_pos2)
                times_obj2.append(temp_time)
            global_true.append(cur_gl_true)

        new_global_true = []
        interp_times = [times_obj1, times_obj2]
        interp_states = [true_obj1, true_obj2]
        for ii, time in enumerate(meas_time):
            cur_new_true = []
            for jj in range(2):
                if time <= interp_times[jj][-1] and time >= interp_times[jj][0]:
                    closest_ind = np.where(np.abs(interp_times[jj] - time) == np.min(np.abs(interp_times[jj] - time)))[0]
                    if len(closest_ind) == 1:
                        in_times = interp_times[jj][closest_ind[0]-2:closest_ind[0]+2]
                        in_states = interp_states[jj][closest_ind[0]-2:closest_ind[0]+2]
                    else:
                        in_times = interp_times[jj][closest_ind[0]-2:closest_ind[-1]+2]
                        in_states = interp_states[jj][closest_ind[0]-2:closest_ind[-1]+2]
                    cur_new_true.append(interp_truth_points(in_times, in_states, time, len(in_times)))

            new_global_true.append(cur_new_true)

    return meas_time, new_global_true
    # return truth_time, global_true


def interp_truth_points(truth_time, global_true, meas_time, order):
    
    func_poly = 0
    for ii in range(order):
        num = 1
        den = 1
        for jj in range(order):
            # if ii == jj:
            if (truth_time[ii] - truth_time[jj]) == 0:
                continue
            num = num * (meas_time - truth_time[jj])
            den = den * (truth_time[ii] - truth_time[jj])

        func_poly += num/den * global_true[ii]

    return func_poly

def red_azimuth(tt, cur_state, *args):
    x = cur_state[0][0] # East
    y = cur_state[1][0] # Down
    z = cur_state[2][0] # North
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arccos(y/r)
    # azimuth = np.arccos(np.sqrt(x ** 2 + z ** 2)/r)
    return azimuth

def red_bearing(tt, cur_state, *args):
    x = cur_state[0][0] # East
    y = cur_state[1][0] # Down
    z = cur_state[2][0] # North
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # bearing = np.sign(x) * np.arccos(z / np.sqrt(x ** 2 + z ** 2))
    # bearing = np.arctan2(np.sqrt(x ** 2 + z ** 2), z)
    bearing = np.arctan2(x, z)
    return bearing

def blue_azimuth(tt, cur_state, *args):
    x = cur_state[0][0] + 1.5198 #East
    y = cur_state[1][0] # Down
    z = cur_state[2][0] # North
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arccos(y/r)
    # azimuth = np.arccos(np.sqrt(x ** 2 + z ** 2)/r)
    return azimuth

def blue_bearing(tt, cur_state, *args):
    x = cur_state[0][0] + 1.5198 # East
    y = cur_state[1][0] # Down
    z = cur_state[2][0] # North
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # bearing = np.sign(x) * np.arccos(z / np.sqrt(x ** 2 + z ** 2))
    # bearing = np.arctan2(z, np.sqrt(x ** 2 + z ** 2))
    # bearing = np.arctan2(np.sqrt(x ** 2 + z ** 2), z)
    bearing = np.arctan2(x, z)
    return bearing

def _setup_double_int_ekf(dt):
    m_noise = 0.02
    p_noise = 0.2
    # m_model_list = [[m_model_red], [m_model_blue]]
    # m_model_list = [red_azimuth, red_bearing, blue_azimuth, blue_bearing]
    m_model_list = [red_azimuth, red_bearing]
    # m_model_list = [blue_azimuth, blue_bearing]

    meas_noise_list = m_noise * np.eye(4)

    doubleInt = DoubleIntegrator3D()

    filt = gfilts.ExtendedKalmanFilter(cont_cov=False)
    filt.set_state_model(dyn_obj=DoubleIntegrator3D())
    filt.set_measurement_model(meas_fun_lst=m_model_list)
    filt.proc_noise = DoubleIntegrator3D().get_dis_process_noise_mat(
        dt, np.array([[p_noise**2]])
    )
    filt.meas_noise = (60 * np.pi / 180 * m_noise) ** 2 * np.eye(2)

    filt.cov = 0.25 * np.eye(6)
    return filt

def _setup_double_int_gci_ekf(dt):
    m_noise = 0.02
    p_noise = 0.2
    # m_model_list = [[m_model_red], [m_model_blue]]
    m_model_list = [[red_azimuth, red_bearing], [blue_azimuth, blue_bearing]]

    meas_noise_list = [(60 * np.pi / 180 * m_noise) ** 2 * np.eye(2), (60 * np.pi / 180 * m_noise) ** 2 * np.eye(2)]

    doubleInt = DoubleIntegrator3D()

    in_filt = gfilts.ExtendedKalmanFilter(cont_cov=False)
    in_filt.set_state_model(dyn_obj=DoubleIntegrator3D())
    in_filt.proc_noise = DoubleIntegrator3D().get_dis_process_noise_mat(
        dt, np.array([[p_noise**2]])
    )
    in_filt.meas_noise = meas_noise_list[0]
    
    filt = gfilts.GCIFilter(
        base_filter=in_filt,
        meas_model_list=m_model_list,
        meas_noise_list=meas_noise_list,
    )
    filt.cov = 0.25 * np.eye(6)
    return filt

#TODO: Redo  birth model, maybe to something in the 300 yard range? Kinda depends on what truth looks like

def _setup_gm_glmb_double_int_birth():
    mu = [np.array([1.0, 3.0, 15.0, 0.0, 0.0, 0.0]).reshape((6, 1))]
    cov = [np.diag(np.array([1.5, 2., 6., 0.1, 0.1, 0.1])) ** 2]
    gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[1])

    return [
        (gm0, 0.003),
    ]

def _setup_gm_pmbm_double_int_birth():
    # mu = [np.array([15, 1.0, 3.0, 0.0, 0.0, 0.0]).reshape((6, 1))]
    # cov = [np.diag(np.array([6., 1.5, 2, 0.1, 0.1, 0.1])) ** 2]
    mu = [np.array([-1.0, -3.0, 15.0, 0.0, 0.0, 0.0]).reshape((6, 1))]
    cov = [np.diag(np.array([1.5, 2., 6., 0.1, 0.1, 0.1])) ** 2]
    gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[1])

    return [gm0]

def track_drones(red_time, blue_time, red_meas, blue_meas, runnum, truth_time, global_true): 

    dt = 1/8
    rng = rnd.default_rng(global_seed)

    filt = _setup_double_int_gci_ekf(dt)
    # filt = _setup_double_int_ekf(dt)

    state_mat_args = (dt,)
    # state_mat_args = {"dt": dt}
    meas_fun_args = () # ("useless arg",)

    b_model_jglmb = _setup_gm_glmb_double_int_birth()
    b_model_pmbm = _setup_gm_pmbm_double_int_birth()

    JGLMB_RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.97,
        "in_filter": filt,
        "birth_terms": b_model_jglmb,
        "clutter_den": 1**-2,
        "clutter_rate": 1**-2,
    }

    PMBM_RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.97,
        "in_filter": filt,
        "birth_terms": b_model_pmbm,
        "clutter_den": 1**-5,
        "clutter_rate": 1**-5,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-5,
        "max_hyps": 1000,
    }

    JGLMB_args = {
        "req_births": len(b_model_jglmb) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }

    # rfs_tracker1 = JointGeneralizedLabeledMultiBernoulli(**JGLMB_args, **JGLMB_RFS_base_args)
    # rfs_tracker1 = LabeledPoissonMultiBernoulliMixture(
    #     **PMBM_args, **PMBM_RFS_base_args
    # )
    rfs_tracker1 = MSLabeledPoissonMultiBernoulliMixture(
        **PMBM_args, **PMBM_RFS_base_args
    )

    # rfs_tracker1 = LabeledPoissonMultiBernoulliMixture(
    #     **PMBM_args, **PMBM_RFS_base_args
    # )

    # rfs_tracker2 = MSJointGeneralizedLabeledMultiBernoulli(
    #     **JGLMB_args, **JGLMB_RFS_base_args
    # )
    # time = np.arange(t0, t1, dt)
    print("\tStarting filtering")

    #TODO: Restructure data processing loop
    time = np.arange(red_time[0], red_time[-1], dt)
    for kk, tt in enumerate(red_time):
        print(kk)
        # if kk == 319:
            # print("stop here")
        # if np.mod(kk, 10) == 0:
        #     print("\t\t{:.2f}".format(kk))
        #     sys.stdout.flush()
        # if kk == 15:
        #     break
        state_mat_args = (dt,)
        pred_args = {"dyn_fun_params": state_mat_args}
        rfs_tracker1.predict(tt, filt_args=pred_args)
        # rfs_tracker2.predict(tt, filt_args=pred_args)

        meas_in = [red_meas[kk], blue_meas[kk]]
        # meas_in = red_meas[kk]# + blue_meas[kk]
        # meas_in = blue_meas[kk]
        # if meas_in[0] is not None:
        # print(meas_in, type(meas_in[0]))
        cor_args = {"meas_fun_args": meas_fun_args}
        rfs_tracker1.correct(tt, meas_in, filt_args=cor_args)
        # rfs_tracker2.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        rfs_tracker1.cleanup(extract_kwargs=extract_kwargs)
        # rfs_tracker2.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    rfs_tracker1.extract_states(**extract_kwargs)
    # rfs_tracker2.extract_states(**extract_kwargs)

    # rfs_trackers=[rfs_tracker1, rfs_tracker2]
    # tracker_names = ['LPMBM', 'JGLMB']
    rfs_trackers = [rfs_tracker1]
    tracker_names = ['LPMBM']
    for t_ind, rfs_tracker in enumerate(rfs_trackers):
        plot_time = []
        plot_x = []
        plot_y = []
        plot_z = []
        for ii in range(len(rfs_tracker.states)):
            for jj in range(len(rfs_tracker.states[ii])):
                plot_time.append(red_time[ii] - red_time[0])
                plot_x.append(rfs_tracker.states[ii][jj][0])
                plot_y.append(rfs_tracker.states[ii][jj][1])
                plot_z.append(rfs_tracker.states[ii][jj][2])
                
        plot_true_time = []
        plot_true_x = []
        plot_true_y = []
        plot_true_z = []
        for ii in range(len(truth_time)):
            for jj in range(len(global_true[ii])):
                plot_true_time.append(truth_time[ii] - truth_time[0])
                plot_true_x.append(global_true[ii][jj][0])
                plot_true_y.append(global_true[ii][jj][1])
                plot_true_z.append(global_true[ii][jj][2])
        plot_truth_states = [plot_true_x, plot_true_y, plot_true_z]
        plot_states = [plot_x, plot_y, plot_z]
        f_hndl = plt.figure()
        f_hndl.add_subplot(3, 1, 1)
        f_hndl.add_subplot(3, 1, 2)
        f_hndl.add_subplot(3, 1, 3)
        s_lst = deepcopy(rfs_tracker.states)
        l_lst = deepcopy(rfs_tracker.labels)
        for states in s_lst:
                if states is not None and len(states) > 0:
                    x_dim = states[0].size
                    break
        u_lbls = []
        for lbls in l_lst:
            if lbls is None:
                continue
            for lbl in lbls:
                if lbl not in u_lbls:
                    u_lbls.append(lbl)

        # plt.figure()
        cmap = pltUtil.get_cmap(len(u_lbls))
        for c_idx, lbl in enumerate(u_lbls):
            xyz = np.nan * np.ones((x_dim, len(rfs_tracker.states)))
            for tt, lbls in enumerate(l_lst):
                if lbls is None:
                    continue
                if lbl in lbls:
                    lbl_ind = lbls.index(lbl)
                    if s_lst[tt][lbl_ind] is not None:
                        xyz[:, [tt]] = s_lst[tt][lbl_ind].copy()
            
            color = cmap(c_idx)


            settings = {
                        "color": color,
                        "markeredgecolor": "k",
                        "markersize": 20,
                        "marker": '*',
                        "linestyle": (0, (5, 10)),
                        }

            # ylabels = ['North', 'East', 'Down']
            ylabels = ['East', 'Down', 'North']

            for ii in range(3):
                f_hndl.axes[ii].plot(plot_time, plot_states[ii], **settings)
                f_hndl.axes[ii].plot(plot_true_time, plot_truth_states[ii], color="k", marker='.', linestyle='', markeredgecolor='k', markersize=10, label="True Trajectories")
                s = "({:.2f}, {:.1f})".format(lbl[0] - red_time[0], lbl[1])
                # tmp = x.copy()
                # tmp = tmp[:, ~np.any(np.isnan(tmp), axis=0)]
                f_hndl.axes[ii].text(
                    plot_time[0], plot_states[ii][0], s, color=color
                )
                f_hndl.axes[ii].set_ylabel(ylabels[ii])
            f_hndl.axes[ii].set_xlabel('Time (Seconds)')
            f_hndl.suptitle(tracker_names[t_ind] + 'States vs Truth')
            # f_hndl.axes[0].legend()
            # plot truth

        all_errs = []
        first_obj_errs = []
        sec_obj_errs = []
        first_obj_err_times = []
        sec_obj_err_times = []
        for ts, t in enumerate(truth_time):
        # for ts, t in enumerate(truth_time[0:10]):
            cur_err = []
            if len(rfs_tracker.states[ts]) == 1 and len(global_true[ts]) == 1:
                cur_err.append(np.abs(rfs_tracker.states[ts][0:3] - global_true[ts][0].reshape((3,1))))
                first_obj_errs.append(np.abs(rfs_tracker.states[ts][0][0:3] - global_true[ts][0].reshape((3,1))))
            else:
                # for state in plot_states[ts]:
                used = []
                for st_ind, state in enumerate(rfs_tracker.states[ts]):
                    dists = []
                    if st_ind > 1:
                        continue
                    for tr_st, truth_state in enumerate(global_true[ts]):
                        if tr_st in used:
                            dists.append(np.array([np.inf]))
                            continue
                        cur_dist = np.sqrt((state[0] - truth_state[0]) ** 2
                                         + (state[1] - truth_state[1]) ** 2
                                         + (state[2] - truth_state[2]) ** 2)
                        dists.append(cur_dist)
                    # if len(dists) == 1:
                    min_dist = np.where(dists==np.min(dists))[0][0]
                    if st_ind == 0:
                        first_obj_errs.append(np.abs(state[0:3] - global_true[ts][min_dist].reshape((3,1))))
                        first_obj_err_times.append(t)
                    else:
                        sec_obj_errs.append(np.abs(state[0:3] - global_true[ts][min_dist].reshape((3,1))))
                        sec_obj_err_times.append(t)
                    
                    cur_err.append(np.abs(state[0:3] - global_true[ts][min_dist].reshape((3,1))))
                    used.append(min_dist)

            all_errs.append(cur_err)

        plot_err_time1 = []
        plot_err_x1 = []
        plot_err_y1 = []
        plot_err_z1 = []
        plot_err_time2 = []
        plot_err_x2 = []
        plot_err_y2 = []
        plot_err_z2 = []
        for ii in range(len(first_obj_err_times)):
            # for jj in range(len(first_obj_errs[ii])):
            plot_err_time1.append(first_obj_err_times[ii] - first_obj_err_times[0])
            plot_err_x1.append(first_obj_errs[ii][0][0])
            plot_err_y1.append(first_obj_errs[ii][1][0])
            plot_err_z1.append(first_obj_errs[ii][2][0])
        for ii in range(len(sec_obj_err_times)):
            # for jj in range(len(sec_obj_errs[ii])):
            plot_err_time2.append(sec_obj_err_times[ii] - sec_obj_err_times[0])
            plot_err_x2.append(sec_obj_errs[ii][0][0])
            plot_err_y2.append(sec_obj_errs[ii][1][0])
            plot_err_z2.append(sec_obj_errs[ii][2][0])
                    # if min_dist in used:

                    # else:
                    #     used.append(min_dist)

        

        path = "/workspaces/Research Work/hadron_rfs/datasets/"

        plt.figure()
        plt.subplot(3,1,1)
        plt.scatter(plot_err_time1, plot_err_x1, marker='*', color='b')
        plt.scatter(plot_err_time2, plot_err_x2, marker='.', color='k')
        # plt.plot(first_obj_err_times, first_obj_errs[0])
        # plt.plot(sec_obj_err_times, sec_obj_errs[0])
        plt.ylabel('East Errors (m)')
        plt.legend(["Drone 1 errors", "Drone 2 errors"], loc='upper right')

        plt.subplot(3,1,2)
        plt.scatter(plot_err_time1, plot_err_y1, marker='*', color='b')
        plt.scatter(plot_err_time2, plot_err_y2, marker='.', color='k')
        # plt.plot(first_obj_err_times, first_obj_errs[0])
        # plt.plot(sec_obj_err_times, sec_obj_errs[0])
        plt.ylabel('Down Errors (m)')

        plt.subplot(3,1,3)
        plt.scatter(plot_err_time1, plot_err_z1, marker='*', color='b')
        plt.scatter(plot_err_time2, plot_err_z2, marker='.', color='k')
        # plt.plot(first_obj_err_times, first_obj_errs[0])
        # plt.plot(sec_obj_err_times, sec_obj_errs[0])
        plt.ylabel('North Errors (m)')
        plt.xlabel('Time Elapsed (seconds)')
        plt.suptitle('State Errors')


        state_fname = path + tracker_names[t_ind] + '_states_figure_with_labels.png'

        plt.savefig(path + tracker_names[t_ind] + '_state_errors_run_' + str(runnum) + '.png')

        plt.figure()
        plt.subplot(3,1,1)
        plt.scatter(plot_err_time1, plot_err_x1, marker='*', color='b')
        plt.scatter(plot_err_time2, plot_err_x2, marker='.', color='k')
        # plt.plot(first_obj_err_times, first_obj_errs[0])
        # plt.plot(sec_obj_err_times, sec_obj_errs[0])
        plt.xlim([-5, 35])
        plt.ylabel('East Errors (m)')
        plt.legend(["Drone 1 errors", "Drone 2 errors"], loc='upper right')

        plt.subplot(3,1,2)
        plt.scatter(plot_err_time1, plot_err_y1, marker='*', color='b')
        plt.scatter(plot_err_time2, plot_err_y2, marker='.', color='k')
        # plt.plot(first_obj_err_times, first_obj_errs[0])
        # plt.plot(sec_obj_err_times, sec_obj_errs[0])
        plt.xlim([-5, 35])
        plt.ylabel('Down Errors (m)')

        plt.subplot(3,1,3)
        plt.scatter(plot_err_time1, plot_err_z1, marker='*', color='b')
        plt.scatter(plot_err_time2, plot_err_z2, marker='.', color='k')
        # plt.plot(first_obj_err_times, first_obj_errs[0])
        # plt.plot(sec_obj_err_times, sec_obj_errs[0])
        plt.ylabel('North Errors (m)')
        plt.xlabel('Time Elapsed (seconds)')
        plt.xlim([-5, 35])
        plt.suptitle('State Errors')

        plt.savefig(path + tracker_names[t_ind] + '_state_errors_run_first_30s' + str(runnum) + '.png')

        f_hndl.savefig(fname=state_fname)
        # rfs_tracker.plot_card_history(time_units="s")
        rfs_tracker.plot_card_history(time_units="s", time=red_time - red_time[0])
        plt.savefig(path + tracker_names[t_ind] + '_cardinality_run_' + str(runnum) + '.png')

        rfs_tracker.calculate_ospa(global_true, 2, 1, state_inds=[0, 1, 2])
        rfs_tracker.plot_ospa_history()
        plt.savefig(path + tracker_names[t_ind] + '_ospa_run_' + str(runnum) + '.png')

        rfs_tracker.calculate_ospa2(global_true, 2, 1, 5, state_inds=[0, 1, 2], core_method=SingleObjectDistance.EUCLIDEAN)
        rfs_tracker.plot_ospa2_history()
        plt.savefig(path + tracker_names[t_ind] + '_ospa2_run_' + str(runnum) + '.png')
        # rfs_tracker.plot_card_history(time_units="s", time=red_time - red_time[0])

        picklefilename = "gci_pmbm_filter.pik"
        


    print(f"Run finished after {kk:4d} iterations.\n")


if __name__ == "__main__":
    from timeit import default_timer as timer
    import matplotlib
    matplotlib.use("WebAgg")

    path = "/workspaces/Research Work/hadron_rfs/datasets/"

    # data_files = ["blue_data/test1/hadron_data.csv", "blue_data/test2/hadron_data.csv",
    #             "red_data/test1/hadron_data.csv", "red_data/test2/hadron_data.csv"]

    truth_path = path + "truth/test2_truth_camera_frame.csv"

    # data_bonus_lines = [16, 1, 12, 12]

    data_files = ["blue_data/test2/hadron_data.csv", "red_data/test2/hadron_data.csv"]

    data_bonus_lines = [1, 12]
    
    # for ii in range(1, 2):
    blue_time, blue_loop_time, blue_meas = extract_data(path + data_files[0], data_bonus_lines[0])
    red_time, red_loop_time, red_meas = extract_data(path + data_files[1], data_bonus_lines[1])

    r_time, b_time, r_meas, b_meas = match_data(red_time, blue_time, red_meas, blue_meas)

    # all_time, all_meas = fuse_measurements(r_time, b_time, r_meas, b_meas)

    truth_time, global_true = extract_truth(truth_path, r_time)

    # print('done')
    #TODO Redo function call to accommodate one measurement and time set
    track_drones(r_time, b_time, r_meas, b_meas, 0, truth_time, global_true)

    plt.show()
