import numpy as np
from matplotlib import pyplot as plt
import torch
from lifelines.utils import concordance_index as _concordance_index
from pysurvival.utils._metrics import _brier_score


def cal_time_bin_index(t, time_buckets):
    try:
        t = list(t)
        return [cal_time_bin_index(tt, time_buckets) for tt in t]
    except Exception:
        min_abs_value = [abs(a_j_1-t) for (a_j_1, a_j) in time_buckets]
        index = np.argmin(min_abs_value)
        return index

def calculate_MTLR_loss(bins, output, label):
    ############ Likelihood Calculations ############
    ## Creating the Triangular matrix
    output = torch.nan_to_num(output, nan=0, posinf=1, neginf=0)
    Triangle = np.tri(bins-1, bins, dtype=np.float32) 
    Triangle = torch.FloatTensor(Triangle)
    phi = torch.exp( torch.mm(output, Triangle) )
    phi = torch.nan_to_num(phi, nan=0, posinf=1, neginf=0)
    reduc_phi = torch.sum(phi*label, dim = 1)
    # reduc_phi[reduc_phi == 0] = 0.5
    ## normalization
    z = torch.exp( torch.mm(output, Triangle) )
    z = torch.nan_to_num(z, nan=0, posinf=1, neginf=0)
    reduc_z = torch.sum(z, dim = 1)
    # reduc_z[reduc_z == 0] = 0.5
    ############ Likelihood Calculations ############
    # MTLR cost function
    loss = - (
            torch.sum( torch.log(reduc_phi) ) \
            - torch.sum( torch.log(reduc_z) ) \
            )
    return loss

def predict(prediction, num_times, time_buckets, t = None):
    """ Predicting the hazard, density and survival functions
    
    Parameters:
    ----------å
    * `x` : **array-like** *shape=(n_samples, n_features)* --
        array-like representing the datapoints. 
        x should not be standardized before, the model
        will take care of it

    * `t`: **double** *(default=None)* --
            time at which the prediction should be performed. 
            If None, then return the function for all available t.
    """
    if torch.is_tensor(prediction):
        prediction = prediction.data.numpy()
    
    # Cretaing the time triangles
    Triangle1 = np.tri(num_times , num_times + 1 )
    Triangle2 = np.tri(num_times+1 , num_times + 1 )

    # Calculating the score, density, hazard and Survival
    phi = np.exp( np.dot(prediction, Triangle1) )
    div = np.repeat(np.sum(phi, 1).reshape(-1, 1), phi.shape[1], axis=1)
    density = (phi/div)
    Survival = np.dot(density, Triangle2)
    hazard = density[:, :-1]/Survival[:, 1:]

    # Returning the full functions of just one time point
    if t is None:
        return hazard, density, Survival
    else:
        min_abs_value = [abs(a_j_1-t) for (a_j_1, a_j) in time_buckets]
        index = np.argmin(min_abs_value)
        return hazard[:, index], density[:, index], Survival[:, index]


def predict_hazard(prediction, num_times, time_buckets, t = None):
    hazard, _, _ = predict(prediction, num_times, time_buckets, t = None)
    return hazard


def predict_density(prediction, num_times, time_buckets, t = None):
    _, density, _ = predict(prediction, num_times, time_buckets, t = None)
    return density


def predict_survival(prediction, num_times, time_buckets, t = None):
    _, _, survival = predict(prediction, num_times, time_buckets, t = None)
    return survival


def predict_cdf(prediction, num_times, time_buckets):
    survival = predict_survival(prediction, num_times, time_buckets)
    cdf = 1. - survival
    return cdf


def predict_cumulative_hazard(prediction, num_times, time_buckets):
    hazard = predict_hazard(prediction, num_times, time_buckets)
    cumulative_hazard = np.cumsum(hazard, 1)
    return cumulative_hazard


def predict_risk(prediction, num_times, time_buckets, use_log=False):
    cumulative_hazard = predict_cumulative_hazard(prediction, num_times, time_buckets)
    risk_score = np.sum(cumulative_hazard, 1)
    if use_log:
        return np.log(risk_score)
    else:
        return risk_score
    

def concordance_index(risk, T, E):
    return _concordance_index(T, risk, event_observed=E)


def integrated_brier_score(Survival, T, E, times,
                           time_buckets, t_max=None, figure_size=(20, 6.5)):
    """ The Integrated Brier Score (IBS) provides an overall calculation of 
        the model performance at all available times.
    """

    # Ordering Survival, T and E in descending order according to T
    T = np.array(T)
    E = np.array(E)
    order = np.argsort(-T)
    Survival = Survival[order, :]
    T = T[order]
    E = E[order]

    if t_max is None or t_max <= 0.:
        t_max = max(T)

    # Calculating the brier scores at each t <= t_max
    results = _brier_score(Survival, T, E, t_max, times, time_buckets, use_mean_point=False)
    
    times = results[0] 
    brier_scores = results[1] 

    # Computing the IBS
    ibs_value = np.trapz(brier_scores, times)/t_max 
    
    # Displaying the Brier Scores at different t 
    if figure_size:
        title = 'Prediction error curve with IBS(t = {:.1f}) = {:.2f}'
        title = title.format(t_max, ibs_value)
        fig, ax = plt.subplots(figsize=figure_size)
        ax.plot( times, brier_scores, color = 'blue', lw = 3)
        ax.set_xlim(-0.01, max(times))
        ax.axhline(y=0.25, ls = '--', color = 'red')
        ax.text(0.90*max(times), 0.235, '0.25 limit', fontsize=20, color='brown', 
            fontweight='bold')
        plt.title(title, fontsize=20)
        plt.show()

    return ibs_value

