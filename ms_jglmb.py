import sys
import pytest
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from copy import deepcopy

import scipy.stats as stats

import gncpy.filters as gfilts
import gncpy.dynamics.basic as gdyn
import gncpy.distributions as gdistrib
import carbs.swarm_estimator.tracker as tracker
import serums.models as smodels
from serums.enums import GSMTypes, SingleObjectDistance

global_seed = 69
debug_plots = False

_meas_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)


def _state_mat_fun(t, dt, useless):
    # print('got useless arg: {}'.format(useless))
    return np.array(
        [[1.0, 0, dt, 0], [0.0, 1.0, 0, dt], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    )


def _meas_mat_fun(t, useless):
    # print('got useless arg: {}'.format(useless))
    return _meas_mat


def _setup_double_int_kf(dt):
    m_noise = 0.02
    p_noise = 0.2

    filt = gfilts.KalmanFilter()
    filt.set_state_model(state_mat_fun=_state_mat_fun)
    filt.set_measurement_model(meas_fun=_meas_mat_fun)
    filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(
        dt, np.array([[p_noise**2]])
    )
    filt.meas_noise = m_noise**2 * np.eye(2)

    return filt


def _prop_true(true_agents, tt, dt):
    out = []
    for ii, x in enumerate(true_agents):
        out.append(_state_mat_fun(tt, dt, "useless") @ x)
    return out


def _setup_double_int_gci_kf(dt):
    m_noise = 0.02
    p_noise = 0.2
    m_model1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    m_model2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    m_model_list = [m_model1, m_model2]

    meas_noise_list = [m_noise**2 * np.eye(2), 0.01 * m_noise**2 * np.eye(2)]

    doubleInt = gdyn.DoubleIntegrator()

    in_filt = gfilts.KalmanFilter()
    in_filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator())
    in_filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(
        dt, np.array([[p_noise**2]])
    )
    in_filt.meas_noise = meas_noise_list[0]
    filt = gfilts.GCIFilter(
        base_filter=in_filt,
        meas_model_list=m_model_list,
        meas_noise_list=meas_noise_list,
    )
    filt.cov = 0.25 * np.eye(4)
    return filt


def _setup_gm_glmb_double_int_birth():
    mu = [np.array([10.0, 0.0, 0.0, 1.0]).reshape((4, 1))]
    cov = [np.diag(np.array([1, 1, 1, 1])) ** 2]
    gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[1])

    return [
        (gm0, 0.003),
    ]


def _update_true_agents(true_agents, tt, dt, b_model, rng):
    out = _prop_true(true_agents, tt, dt)

    if any(np.abs(tt - np.array([0, 1, 1.5])) < 1e-8):
        x = b_model[0].means[0] + (rng.standard_normal(4) * np.ones(4)).reshape((4, 1))
        out.append(x.copy())
    return out


def _update_true_agents_pmbm_lmb_var(true_agents, tt, dt, b_model, rng):
    out = _prop_true(true_agents, tt, dt)
    if any(np.abs(tt - np.array([0, 1, 1.5])) < 1e-8):
        # if any(np.abs(tt - np.array([0, 1])) < 1e-8):
        for gm, w in b_model:
            x = gm.means[0] + (rng.standard_normal(4) * np.ones(4)).reshape((4, 1))
            out.append(x.copy())

    return out


def _gen_meas_ms(tt, true_agents, proc_noise, meas_noise, rng, meas_model_list):
    meas_in = []
    for model in meas_model_list:
        # print(model)
        sens_list = []
        for x in true_agents:
            # print(x)
            sens_list.append(model @ x)
        meas_in.append(sens_list)

    return meas_in


def test_MS_JGLMB(meas_data):  # noqa
    print("Test MS-GM-JGLMB")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    # t0, t1 = 0, 5.5 + dt
    t0, t1 = 0, 1.3 + dt

    filt = _setup_double_int_gci_kf(dt)

    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    b_model = _setup_gm_glmb_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.97,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 10**-2,
        "clutter_rate": 10**-2,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.MSJointGeneralizedLabeledMultiBernoulli(
        **JGLMB_args, **RFS_base_args
    )
    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, curr_data in enumerate(meas_data):
        print(kk)
        tt = dt * kk
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        # true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
        true_agents = _update_true_agents_pmbm_lmb_var(
            true_agents, tt, dt, b_model, rng
        )
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        jglmb.predict(tt, filt_args=pred_args)

        # meas_in = _gen_meas_ms(
        #     tt,
        #     true_agents,
        #     filt.proc_noise,
        #     filt.meas_noise_list,
        #     rng,
        #     filt.meas_model_list,
        # )

        meas_in = curr_data
        print(meas_in)
        cor_args = {"meas_fun_args": meas_fun_args}
        jglmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    jglmb.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        jglmb.plot_card_dist()
        # jglmb.plot_card_history(time_units="s", time=time)
        jglmb.plot_ospa_history()
    print("\tExpecting {} agents".format(len(true_agents)))

    # assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


if __name__ == "__main__":
    from timeit import default_timer as timer
    import matplotlib
    import pickle

    file_path = "pwease.pik"

    with open(file_path, "rb") as file:
        data = pickle.load(file)

    matplotlib.use("WebAgg")

    plt.close("all")

    debug_plots = True

    start = timer()

    test_MS_JGLMB(data)

    end = timer()
    print("{:.2f} s".format(end - start))
    print("Close all plots to exit")
    plt.show()
