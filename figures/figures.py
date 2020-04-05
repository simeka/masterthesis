import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gs
from scipy.stats import entropy
import pandas as pd

duration = 2.3  # [T] = ms -> [rate] = kHz
from matplotlib.lines import Line2D
import seaborn as sns


def save_plot(fig, name, size=(6, 3)):
    orig = fig.get_size_inches()
    fig.set_size_inches(size)

    fig.savefig(name + ".pgf", bbox_inches="tight", transparent=True)
    try:
        fig.savefig(name + ".pdf", bbox_inches="tight", transparent=True)
    except:
        print("couldnt save pdf")
    fig.set_size_inches(orig)


def pseudo_adc_conversion_analog(adc_set, round_until=1):
    """
    returns the voltage in mV measured for the value set in the c programm

    :param adc_set: int, limits: (0,1022)
    :param round_until:
    :return:
    """
    if isinstance(adc_set, (list, np.ndarray)):
        if (adc_set > 1022).any() or (adc_set < 0).any():
            raise ValueError("adc_set can only be within 0 and 1022")
    elif adc_set > 1022 or adc_set < 0:
        raise ValueError("adc_set can only be within 0 and 1022")
    return np.round(1.748648648648649 * adc_set + 14.162162162162161, decimals=round_until)


def pseudo_adc_conversion(digital_pseudo_unit):
    # converts the digital value of the adc into millivolt
    # see ipython notebook: Pseudo ADC Calibration
    return 0.681 * digital_pseudo_unit - 865.713


NUM_COLORS = 12
colors = sns.color_palette("hls", NUM_COLORS)


# ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])

def true_input_spike_rate(input_rate):
    return input_rate * 2.3e-3


def true_output_spike_rate(output_rate):
    return output_rate / 2.3e-3



files = ["output_spikes_1573574858.npy", "output_spikes_1573574960.npy",
             "output_spikes_1573575034.npy", "output_spikes_1573575179.npy"]
data = np.zeros((4, 4, 37, 32))
for i, f in enumerate(files):
    data[i] = true_output_spike_rate(np.load(f)) / 1e3

# input rates
# update: 4.4.2020, there is missconception of the true input spike rate which is at max 568kHz equal to 1306 spikes per measurement duration
input_weights = [5, 15, 30, 60]
# input_rates = true_input_spike_rate(np.linspace(-560, 560, 37)) * 1e3
input_rates = np.linspace(-560, 560, 37) # input rate in kHz


if __name__ == "__main__":

    ###################################################################################
    # single activation function with multiple weights
    mmnt = 0
    nrn = 9
    figure = plt.figure()

    plt.ylabel("$\\nu_\mathrm{output} \; (\\si{\kilo \Hz})$")
    plt.xlabel("$\\nu_\mathrm{input} \; (\\si{\kilo \Hz})$")
    for p in range(4):
        plt.plot(input_rates, data[mmnt, p, :, nrn], label="input weight %s" % input_weights[p])
        # plt.legend([Line2D([0],[0], marker='o', color="w", markerfacecolor="black")],
        #           ["input weight = {}".format(input_weights[p])])
    plt.legend()
    save_plot(figure, "single_calibrated_transfer_function_w_various_weights", (5,3))


    ###################################################################################
    # uncalibrated activation function (b=0, w=30)
    mmnt = 2
    w = 2 # corresponds to 30
    fig = plt.figure()
    plt.ylabel("$\\nu_\mathrm{output} \; (\\si{\kilo \Hz})$")
    plt.xlabel("$\\nu_\mathrm{input} \; (\\si{\kilo \Hz})$")
    plt.plot(input_rates, data[mmnt, w, :, 0:12])
    save_plot(fig, "uncalibrated_activation_function", (3,2))

    mmnt = 0
    w = 2 # corresponds to 30
    fig = plt.figure()
    plt.ylabel("$\\nu_\mathrm{output} \; (\\si{\kilo \Hz})$")
    plt.xlabel("$\\nu_\mathrm{input} \; (\\si{\kilo \Hz})$")
    plt.plot(input_rates, data[mmnt, w, :, 0:12])
    save_plot(fig, "calibrated_activation_function", (3,2))

    ###################################################################################
    # activation function with bias
    mmnts = ["output_spikes_1574183394.npy", "output_spikes_1574183702.npy",
             "output_spikes_1574183860.npy"]
    data = np.zeros((3, 3, 37, 32))

    for i, f in enumerate(mmnts):
        data[i] = true_output_spike_rate(np.load(f)) / 1e3

    # input rates
    thresholds = [(270, 300, 330), (250, 300, 350), (260, 300, 340)]
    thresholds = [(pseudo_adc_conversion_analog(thres[0]),
                   pseudo_adc_conversion_analog(thres[1]),
                   pseudo_adc_conversion_analog(thres[2]))
                  for thres in thresholds]
    input_rates = true_input_spike_rate(np.linspace(-560, 560, 37)) * 1e3
    workpoint = pseudo_adc_conversion_analog(300)

    mmnt = 0
    n_params = 0
    figure = plt.figure()
    plt.ylabel("$\\nu_\mathrm{output} \; (\\si{\kilo \Hz})$")
    plt.xlabel("$\\nu_\mathrm{input} \; (\\si{\kilo \Hz})$")
    #figure.gca().set_prop_cycle(color=colors)
    styles = ["--", "-", ":"]

    for i, thres in enumerate(thresholds[mmnt]):
        plt.plot(input_rates, data[mmnt, i, :, 0:12], linestyle=styles[i])

    plt.legend([Line2D([0], [0], marker='', linestyle=style, color="black") for style in styles],
               ["$b \propto  \delta V = \SI{%s}{\milli \V}$" % (np.round(workpoint - t, 1)) for t in thresholds[mmnt]],
               loc="upper left")
    save_plot(figure, "activation_function_w_bias", (5,3))


    ###################################################################################
    # Gaussian Free Membrane Distribution
    mmnts = ["membrane_data_0.npy", "membrane_data_1.npy"]
    input_rates = [0, true_input_spike_rate(100)]
    noise_rates = true_input_spike_rate(70)
    full_data = np.zeros((2, 26137))
    for i in range(2):
        full_data[i] = np.load(mmnts[i])[:26137]

    def gaus(x, mue, sig):
        return np.exp(-(x - mue) ** 2 / (2 * sig ** 2)) / sig / np.sqrt(2 * np.pi)

    fig, axes = plt.subplots(nrows=1)
    for i in range(1):
        ax = axes
        kullback_leibler_divergence = []
        data = pd.Series(full_data[i]).dropna()
        y, x = np.histogram(data, bins=200)
        x = x[1:] - (x[1] - x[0]) / 2
        std = data.std()
        mean = data.mean()
        kb_lb_entropy = entropy(y, gaus(x, mean, std))
        kullback_leibler_divergence += [kb_lb_entropy, ]

        y_normed, _ = np.histogram(data, bins=200, density=True)
        ax.set_ylabel("Density")

        mean_stored = mean
        ax.plot(x, gaus(x, mean, std))
        ax.axvline(mean, linestyle="--")
        ax.text(mean + 2, 0.02, '$V_{\mathrm{leak}}$')
        ax.hist(data, bins=200, alpha=0.3, density=True)
        ax.axvline(x[150 - 1], linestyle="--", color="red")
        ax.text(x[150 - 1] + 2, 0.02, '$\\vartheta$')
        ax.bar(x[150:], y_normed[150:])

        # ax.legend(["Gaussian Fit", "$V_{leak}$", "$V_{thres}$", "Histogram", "Spiking"], loc="upper left")
    ax.set_xlabel("$V_\mathrm{m} \; (\si{\milli \V})$")
    save_plot(fig, "activation_function_vmem_distr_with_thres", (5,3))