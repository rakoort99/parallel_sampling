import random
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from samplers import (
    seq_gibbs,
    hog_gibbs,
    exact_async,
    local_glauber,
    chrom_gibbs,
    dist_metro,
)
import seaborn as sns


def draw_race():
    ### DEFINE DISTRIBUTION
    def samp_b(states, idx):
        if idx == 0:
            if states[1] == 1:
                return 1 if random.random() > 0.5 else 0
            else:
                return 1
        else:
            if states[0] == 1:
                return 1 if random.random() > 0.5 else 0
            else:
                return 1

    def transition_b(update, idx, states):
        # reversible so can just return 1
        static = states[1 - idx]
        if (static == 0) and (update == 0):
            return 0
        elif (static == 0) and (update == 1):
            return 1
        else:
            return 0.5

    def joint_pdf_b(states):
        if (states[0] == 0) and (states[1] == 0):
            return 0
        else:
            return 1 / 3

    def propsal_samp_b(idx, N):
        return np.random.choice([0, 1], N, replace=True)

    init = [1, 1]

    ### DRAW SAMPLES
    seq_samps = seq_gibbs(init, samp_b, 10**5)
    ea_samps = exact_async(
        init, samp_b, transition_b, joint_pdf_b, 10**5, 3, discrete=True
    )[1]
    hog_samps = hog_gibbs(init, samp_b, 10**5, 4)
    dm_samps = dist_metro(init, propsal_samp_b, transition_b, joint_pdf_b, 10**5)

    ### HISTOGRAM
    labs_seq, counts_seq = np.unique(seq_samps, return_counts=True, axis=0)
    labs_hog, counts_hog = np.unique(hog_samps, return_counts=True, axis=0)
    labs_ea, counts_ea = np.unique(ea_samps, return_counts=True, axis=0)
    labs_dm, counts_dm = np.unique(dm_samps, return_counts=True, axis=0)

    seq = pd.DataFrame(
        index=[str(i) for i in labs_seq], data={"seq": counts_seq / counts_seq.sum()}
    )
    hog = pd.DataFrame(
        index=[str(i) for i in labs_hog], data={"hog": counts_hog / counts_hog.sum()}
    )
    ea = pd.DataFrame(
        index=[str(i) for i in labs_ea], data={"ea": counts_ea / counts_ea.sum()}
    )
    dm = pd.DataFrame(
        index=[str(i) for i in labs_dm], data={"dm": counts_dm / counts_dm.sum()}
    )

    df = seq.join([hog, ea, dm], how="outer").fillna(0)
    ax = df.plot(kind="bar", title="Empirical PMF after 10^5 Samples")
    positions = range(len(df))
    bar_width = 0.8
    x_start = positions[0] - bar_width / 2
    x_end = positions[2] + bar_width / 2

    ax.plot(
        [
            x_start,
            x_end,
        ],
        [
            1 / 3,
            1 / 3,
        ],
        color="r",
        linestyle="--",
    )
    fig = ax.get_figure()
    fig.savefig("hist_race.png")

    ### MIXING
    seq_cum = np.where(np.all(seq_samps == [1, 1], axis=1), 1, 0).cumsum()
    seq_counts = np.arange(1, len(seq_cum) + 1)
    seq_cum_avg = seq_cum / seq_counts

    hog_cum = np.where(np.all(hog_samps == [1, 1], axis=1), 1, 0).cumsum()
    hog_counts = np.arange(1, len(hog_cum) + 1)
    hog_cum_avg = hog_cum / hog_counts

    ea_cum = np.where(np.all(ea_samps == [1, 1], axis=1), 1, 0).cumsum()
    ea_counts = np.arange(1, len(ea_cum) + 1)
    ea_cum_avg = ea_cum / ea_counts

    dm_cum = np.where(np.all(dm_samps == [1, 1], axis=1), 1, 0).cumsum()
    dm_counts = np.arange(1, len(dm_cum) + 1)
    dm_cum_avg = dm_cum / dm_counts

    fig, ax = plt.subplots()
    fig.suptitle("Mixing to Race Detection Distribution")

    ax.plot(seq_counts, seq_cum_avg, label="seq")
    ax.plot(hog_counts / 4, hog_cum_avg, label="hog")
    ax.plot(ea_counts / 3, ea_cum_avg, label="ea")
    ax.plot(dm_counts / 2, dm_cum_avg, label="dm")
    ax.hlines(1 / 3, xmin=1, xmax=np.max(seq_counts), linestyles="--", colors="black")
    ax.set_xlim(1, 20000)
    ax.set_xscale("log")
    ax.set_ylabel("Estimated $P(X =(1,1) )$")
    ax.set_xlabel("Time")
    ax.legend()
    fig.savefig("mixing_race.png")


def draw_mvn():
    ### DEFINE DISTRIBUTION
    a = 0.5

    def samp_mvn(states, idx):
        if idx == 0:
            return st.multivariate_normal.rvs(mean=states[1] * a, cov=1 - a**2)
        else:
            return st.multivariate_normal.rvs(mean=states[0] * a, cov=1 - a**2)

    def transition_mvn(update, idx, states):
        marginal = st.multivariate_normal(states[1 - idx] * a, cov=1 - a**2)
        return marginal.pdf(update)

    def joint_pdf_mvn(states):
        joint = st.multivariate_normal([0, 0], a)
        return joint.pdf(states)

    def propsal_samp_mvm(i, N):
        return np.random.standard_normal(N)

    init = [1, 1]

    ### DRAW SAMPLES
    seq_samps = seq_gibbs(init, samp_mvn, 10**5)
    dm_samps = dist_metro(init, propsal_samp_mvm, transition_mvn, joint_pdf_mvn, 10**5)
    ea_samps = exact_async(init, samp_mvn, transition_mvn, joint_pdf_mvn, 10**5, 3)[1]
    hog_samps = hog_gibbs(init, samp_mvn, 10**4, n_workers=4)

    ### HISTOGRAM
    fig, ax = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=True, figsize=(12, 8))
    ax[0, 0].hist2d(np.array(seq_samps)[:, 0], np.array(seq_samps)[:, 1])
    ax[0, 0].set_title("Sequential")
    ax[1, 0].hist2d(np.array(hog_samps)[:, 0], np.array(hog_samps)[:, 1])
    ax[1, 0].set_title("Hogwild")
    ax[0, 1].hist2d(np.array(ea_samps)[:, 0], np.array(ea_samps)[:, 1])
    ax[0, 1].set_title("Exact Asynchronous")
    ax[1, 1].hist2d(np.array(dm_samps)[:, 0], np.array(dm_samps)[:, 1])
    ax[1, 1].set_title("Distributed Metropolis")
    ax[0, 0].set_xlim([-3.2, 3.2])
    ax[0, 0].set_ylim([-3.2, 3.2])
    fig.suptitle("Histograms after 10^5 Samples")
    fig.savefig("hist_mvn.png")

    ### MIXING
    seq_cum = seq_samps[:, 1].cumsum()
    seq_counts = np.arange(1, len(seq_cum) + 1)
    seq_cum_avg = seq_cum / seq_counts

    hog_cum = hog_samps[:, 1].cumsum()
    hog_counts = np.arange(1, len(hog_cum) + 1)
    hog_cum_avg = hog_cum / hog_counts

    ea_cum = ea_samps[:, 1].cumsum()
    ea_counts = np.arange(1, len(ea_cum) + 1)
    ea_cum_avg = ea_cum / ea_counts

    dm_cum = dm_samps[:, 1].cumsum()
    dm_counts = np.arange(1, len(dm_cum) + 1)
    dm_cum_avg = dm_cum / dm_counts

    fig, ax = plt.subplots()
    fig.suptitle("Mixing to Multivariate Normal")

    ax.plot(seq_counts, seq_cum_avg, label="seq")
    ax.plot(hog_counts / 4, hog_cum_avg, label="hog")
    ax.plot(ea_counts / 3, ea_cum_avg, label="ea")
    ax.plot(dm_counts / 2, dm_cum_avg, label="dm")
    ax.hlines(0, xmin=1, xmax=np.max(seq_counts), linestyles="--", colors="black")
    ax.set_xlim(1, 20000)
    ax.set_xscale("log")
    ax.set_ylabel(r"Estimated $\mu_1$")
    ax.set_xlabel("Time")
    ax.legend()
    fig.savefig("mixing_mvn.png")


def draw_qcolor():
    ### DEFINE DISTRIBUTION
    q = 5

    def samp_ring(states, idx):
        n = len(states)
        options = [i for i in range(q)]
        blocked = [states[(idx - 1) % n], states[(idx + 1) % n]]
        return random.choice([elem for elem in options if elem not in blocked])

    def has_adjacent_duplicates(lst):
        n = len(lst)
        for i in range(n - 1):
            if lst[i] == lst[i + 1]:
                return True
        if lst[0] == lst[-1]:
            return True
        return False

    def transition_ring(update, idx, states):
        return 1

    def joint_pdf_ring(states):
        if has_adjacent_duplicates(states):
            return 0
        else:
            return 1

    def propsal_samp_ring(i, N):
        return np.random.choice([i for i in range(q)], N, replace=True)

    init = [1, 2, 3, 2, 1, 2, 3, 2, 1, 2]

    ### DRAW SAMPLES
    dm_samps = dist_metro(
        init, propsal_samp_ring, transition_ring, joint_pdf_ring, 10**5
    )
    seq_samps = seq_gibbs(init, samp_ring, 10**5)
    chrom_samps = chrom_gibbs(
        init, samp_ring, [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], 10**5
    )
    hog_samps = hog_gibbs(init, samp_ring, 10**5, n_workers=4)
    lg_samps = local_glauber(init, 5, 0.3, 10**5)
    ea_samps = exact_async(
        init,
        samp_ring,
        transition_ring,
        joint_pdf_ring,
        10**5,
        n_workers=3,
        discrete=True,
    )[1]

    ### MIXING
    seq_cum = np.where(seq_samps[:, 4] == 4, 1, 0).cumsum()
    seq_counts = np.arange(1, len(seq_cum) + 1)
    seq_cum_avg = seq_cum / seq_counts

    hog_cum = np.where(hog_samps[:, 4] == 4, 1, 0).cumsum()
    hog_counts = np.arange(1, len(hog_cum) + 1)
    hog_cum_avg = hog_cum / hog_counts

    ea_cum = np.where(ea_samps[:, 4] == 4, 1, 0).cumsum()
    ea_counts = np.arange(1, len(ea_cum) + 1)
    ea_cum_avg = ea_cum / ea_counts

    dm_cum = np.where(dm_samps[:, 4] == 4, 1, 0).cumsum()
    dm_counts = np.arange(1, len(dm_cum) + 1)
    dm_cum_avg = dm_cum / dm_counts

    lg_cum = np.where(lg_samps[:, 4] == 4, 1, 0).cumsum()
    lg_counts = np.arange(1, len(lg_cum) + 1)
    lg_cum_avg = lg_cum / lg_counts

    chrom_cum = np.where(chrom_samps[:, 4] == 4, 1, 0).cumsum()
    chrom_counts = np.arange(1, len(chrom_cum) + 1)
    chrom_cum_avg = chrom_cum / chrom_counts

    fig, ax = plt.subplots()
    fig.suptitle("Mixing to Uniform Proper 5-Coloring on 10 Node Ring")

    ax.plot(seq_counts, seq_cum_avg, label="seq")
    ax.plot(hog_counts / 4, hog_cum_avg, label="hog")
    ax.plot(ea_counts / 3, ea_cum_avg, label="ea")
    ax.plot(dm_counts / 10, dm_cum_avg, label="dm")
    ax.plot(lg_counts, lg_cum_avg, label="lg")
    ax.plot(chrom_counts, chrom_cum_avg, label="chrom")
    ax.set_xlim(1, 20000)
    ax.hlines(1 / 5, xmin=1, xmax=np.max(seq_counts), linestyles="--", colors="black")
    ax.set_xscale("log")
    ax.set_ylabel("Estimated $P(X_4 =4 )$")
    ax.set_xlabel("Time")
    ax.legend()
    fig.savefig("mixing_qcolor.png")

def main():
    sns.set_theme('paper')
    draw_race()
    draw_mvn()
    draw_qcolor()

if __name__ == "__main__":
    main()