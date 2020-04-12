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

# data for circles results
duration = 2.3
points_storage = np.load("points_validation.npy")
rates_storage = np.load("rates_validation.npy")
weights_hidden_storage = np.load("weights_hidden.npy")
weights_out_storage = np.load("weights_out.npy")
bias_storage = np.load("bias.npy")
# convert rates to Hz
rates_storage /= duration

learning_steps = rates_storage.shape[0]
n_steps = rates_storage.shape[1]
n_nrns = rates_storage.shape[2]
accuracy = np.zeros(learning_steps)
target = {1: 75 / duration,
          2: 215 / duration}
VMAX = 255 / duration
VMIN = 0 / duration
error = np.zeros((learning_steps, n_steps))
targets = np.zeros((learning_steps, n_steps))
n_h_nrns = 11
n_nrns = 12
real_epochs = np.array(range(learning_steps))*5





if __name__ == "__main__":

    ###################################################################################
    # single activation function with multiple weights
    # mmnt = 0
    # nrn = 9
    # figure = plt.figure()
    #
    # plt.ylabel("$\\nu_\mathrm{output} \; (\\si{\kilo \Hz})$")
    # plt.xlabel("$\\nu_\mathrm{input} \; (\\si{\kilo \Hz})$")
    # for p in range(4):
    #     plt.plot(input_rates, data[mmnt, p, :, nrn], label="input weight %s" % input_weights[p])
    #     # plt.legend([Line2D([0],[0], marker='o', color="w", markerfacecolor="black")],
    #     #           ["input weight = {}".format(input_weights[p])])
    # plt.legend()
    # save_plot(figure, "single_calibrated_transfer_function_w_various_weights", (5,3))
    #
    #
    # ###################################################################################
    # # uncalibrated activation function (b=0, w=30)
    # mmnt = 2
    # w = 2 # corresponds to 30
    # fig = plt.figure()
    # plt.ylabel("$\\nu_\mathrm{output} \; (\\si{\kilo \Hz})$")
    # plt.xlabel("$\\nu_\mathrm{input} \; (\\si{\kilo \Hz})$")
    # plt.plot(input_rates, data[mmnt, w, :, 0:12])
    # save_plot(fig, "uncalibrated_activation_function", (3,2))
    #
    # mmnt = 0
    # w = 2 # corresponds to 30
    # fig = plt.figure()
    # plt.ylabel("$\\nu_\mathrm{output} \; (\\si{\kilo \Hz})$")
    # plt.xlabel("$\\nu_\mathrm{input} \; (\\si{\kilo \Hz})$")
    # plt.plot(input_rates, data[mmnt, w, :, 0:12])
    # save_plot(fig, "calibrated_activation_function", (3,2))
    #
    # ###################################################################################
    # # activation function with bias
    # mmnts = ["output_spikes_1574183394.npy", "output_spikes_1574183702.npy",
    #          "output_spikes_1574183860.npy"]
    # data = np.zeros((3, 3, 37, 32))
    #
    # for i, f in enumerate(mmnts):
    #     data[i] = true_output_spike_rate(np.load(f)) / 1e3
    #
    # # input rates
    # thresholds = [(270, 300, 330), (250, 300, 350), (260, 300, 340)]
    # thresholds = [(pseudo_adc_conversion_analog(thres[0]),
    #                pseudo_adc_conversion_analog(thres[1]),
    #                pseudo_adc_conversion_analog(thres[2]))
    #               for thres in thresholds]
    # input_rates = true_input_spike_rate(np.linspace(-560, 560, 37)) * 1e3
    # workpoint = pseudo_adc_conversion_analog(300)
    #
    # mmnt = 0
    # n_params = 0
    # figure = plt.figure()
    # plt.ylabel("$\\nu_\mathrm{output} \; (\\si{\kilo \Hz})$")
    # plt.xlabel("$\\nu_\mathrm{input} \; (\\si{\kilo \Hz})$")
    # #figure.gca().set_prop_cycle(color=colors)
    # styles = ["--", "-", ":"]
    #
    # for i, thres in enumerate(thresholds[mmnt]):
    #     plt.plot(input_rates, data[mmnt, i, :, 0:12], linestyle=styles[i])
    #
    # plt.legend([Line2D([0], [0], marker='', linestyle=style, color="black") for style in styles],
    #            ["$b \propto  \delta V = \SI{%s}{\milli \V}$" % (np.round(workpoint - t, 1)) for t in thresholds[mmnt]],
    #            loc="upper left")
    # save_plot(figure, "activation_function_w_bias", (5,3))
    #
    #
    # ###################################################################################
    # # Gaussian Free Membrane Distribution
    # mmnts = ["membrane_data_0.npy", "membrane_data_1.npy"]
    # input_rates = [0, true_input_spike_rate(100)]
    # noise_rates = true_input_spike_rate(70)
    # full_data = np.zeros((2, 26137))
    # for i in range(2):
    #     full_data[i] = np.load(mmnts[i])[:26137]
    #
    # def gaus(x, mue, sig):
    #     return np.exp(-(x - mue) ** 2 / (2 * sig ** 2)) / sig / np.sqrt(2 * np.pi)
    #
    # fig, axes = plt.subplots(nrows=1)
    # for i in range(1):
    #     ax = axes
    #     kullback_leibler_divergence = []
    #     data = pd.Series(full_data[i]).dropna()
    #     y, x = np.histogram(data, bins=200)
    #     x = x[1:] - (x[1] - x[0]) / 2
    #     std = data.std()
    #     mean = data.mean()
    #     kb_lb_entropy = entropy(y, gaus(x, mean, std))
    #     kullback_leibler_divergence += [kb_lb_entropy, ]
    #
    #     y_normed, _ = np.histogram(data, bins=200, density=True)
    #     ax.set_ylabel("Density")
    #
    #     mean_stored = mean
    #     ax.plot(x, gaus(x, mean, std))
    #     ax.axvline(mean, linestyle="--")
    #     ax.text(mean + 2, 0.02, '$V_{\mathrm{leak}}$')
    #     ax.hist(data, bins=200, alpha=0.3, density=True)
    #     ax.axvline(x[150 - 1], linestyle="--", color="red")
    #     ax.text(x[150 - 1] + 2, 0.02, '$\\vartheta$')
    #     ax.bar(x[150:], y_normed[150:])
    #
    #     # ax.legend(["Gaussian Fit", "$V_{leak}$", "$V_{thres}$", "Histogram", "Spiking"], loc="upper left")
    # ax.set_xlabel("$V_\mathrm{m} \; (\si{\milli \V})$")
    # save_plot(fig, "activation_function_vmem_distr_with_thres", (5,3))

    # ##################################################################################
    # # Gaussian Free Membrane Distribution
    #
    # alpha = 0.27
    # fs = 6
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 4), sharex=True, sharey="col")
    # wh_legend_labels = ["$w_{%s%s}$" % (i, j) for i in range(2) for j in range(5)]
    # wo_legend_labels = ["$w_{%s%s}$" % (i, j) for i in range(5) for j in range(1)]
    # bh_legend_labels = ["$b_%s$" % i for i in range(5)]
    # bo_legend_labels = ["$b_%s$" % i for i in range(1)]
    #
    # # hidden weights
    # axes[0, 0].plot(real_epochs, weights_hidden_storage[:, 0, :5])
    # axes[0, 0].plot(real_epochs, weights_hidden_storage[:, 1, :5], ls="--")
    # axes[0, 0].plot(real_epochs, weights_hidden_storage[:, 0, 5:], alpha=alpha)
    # axes[0, 0].plot(real_epochs, weights_hidden_storage[:, 1, 5:], ls="--", alpha=alpha)
    #
    # # hidden bias
    # axes[0, 1].plot(real_epochs, bias_storage[:, :5])
    # axes[0, 1].plot(real_epochs, bias_storage[:, 5:n_h_nrns], alpha=alpha)
    #
    # # output weights (with hidden in background)
    # axes[1, 0].plot(real_epochs, weights_out_storage[:, :5])
    # axes[1, 0].plot(real_epochs, weights_out_storage[:, 5:], alpha=alpha)
    #
    # # output bias with hidden in background
    # axes[1, 1].plot(real_epochs, bias_storage[:, -1])
    # #axes[1, 1].plot(real_epochs, bias_storage[:, :5], alpha=alpha)
    # #axes[1, 1].plot(real_epochs, bias_storage[:, 5:n_h_nrns], alpha=alpha)
    #
    # # legends
    # xlegend = 0.98
    # axes[0, 1].legend(bh_legend_labels, fontsize=fs, loc=(xlegend, 0.6))
    # axes[0, 0].legend(wh_legend_labels, fontsize=fs,
    #                   loc=(xlegend, .2), )  # title="Unit", title_fontsize=6)
    # axes[1, 1].legend(bo_legend_labels, fontsize=fs, loc=(xlegend, .9))
    # axes[1, 0].legend(wo_legend_labels, fontsize=fs,
    #                   loc=(xlegend, 0.6), )  # title="Unit", title_fontsize=6)
    #
    # # labels
    # axes[0, 1].set_ylabel("$\\vartheta \propto -b^{(\mathrm{h})} \quad (\si{\milli \V})$")
    # axes[1, 1].set_ylabel("$\\vartheta \propto -b^{(\mathrm{o})} \quad (\si{\milli \V})$")
    # axes[0, 0].set_ylabel("$W^{(\mathrm{h})}$")
    # axes[1, 0].set_ylabel("$W^{(\mathrm{o})}$")
    # for i in range(2):
    #     axes[1, i].set_xlabel("Iteration")
    #
    # plt.subplots_adjust(hspace=0.05
    #                     , wspace=0.4)
    #
    # save_plot(fig, "network_evolution_circles", (7, 4))
    #
    # # circles plots
    # def points_to_rates(points):
    #     input_rate = 500
    #     rates = input_rate * ((128 + points[:, :2]) / 255)
    #     return rates
    #
    # l_step = 1
    # rates = rates_storage[l_step]
    # points = points_storage[l_step]
    # input_rates = points_to_rates(points)
    # VMIN_INPUT = np.amin(input_rates)
    # VMAX_INPUT = np.amax(input_rates)
    #
    # # nu_x(x,y) = nu_x(x)
    # fig = plt.figure(figsize=(3, 3))
    # sc = plt.scatter(points[:, 0], points[:, 1], c=input_rates[:, 0], s=20,
    #                  vmin=VMIN_INPUT, vmax=VMAX_INPUT)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # cbar = fig.colorbar(sc, fraction=0.046, pad=0.04, label="$\\nu_{\mathrm{input, x}} \quad (\si{\kilo \Hz})$")
    # save_plot(fig, "nu_x_input", (2.5, 2.5))
    #
    # fig = plt.figure(figsize=(3, 3))
    # sc = plt.scatter(points[:, 0], points[:, 1], c=input_rates[:, 1], s=20,
    #                  vmin=VMIN_INPUT, vmax=VMAX_INPUT)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # cbar = fig.colorbar(sc, fraction=0.046, pad=0.04, label="$\\nu_{\mathrm{input, y}} \quad (\si{\kilo \Hz})$")
    # save_plot(fig, "nu_y_input", (2.5, 2.5))

    # #############################################################################
    # Deep learning activation functions
    # relu = lambda x: np.maximum(0, x)
    #
    #
    # def fsigmoid(x, a=2, b=0):
    #     return 1.0 / (1.0 + np.exp(-a * (x - b)))
    #
    #
    # def dsig(x):
    #     return fsigmoid(x) * (1 - fsigmoid(x))
    #
    #
    # def drelu(x):
    #     return (1 * (x >= 0)).astype(int)
    #
    #
    # def dtanh(x):
    #     return np.cosh(x) ** -2
    #
    #
    # # tanh from numpy
    #
    # x = np.linspace(-3, 3, 200)
    # fig = plt.figure()
    # plt.plot(x, relu(x), label="ReLu, $\Phi(x) = \mathrm{max}(0,x)$")
    # plt.plot(x, fsigmoid(x), label="Sigmoid, $\Phi(x) = \\frac{1}{1 + e^{(-\\beta x)}}$")
    # plt.plot(x, np.tanh(x), label="Tanh, $\Phi(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$")
    # plt.ylim(-1, 2)
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("Transfer function $\Phi(x)$")
    # save_plot(fig, "deeplearning_activation_functions", (3, 2.7))
    #
    # x = np.linspace(-3, 3, 200)
    # fig = plt.figure()
    # plt.plot(x, drelu(x), label="dReLu, $\Phi'(x) = \Theta(x)$")
    # plt.plot(x, dsig(x), label="dSigmoid, $\Phi'(x) = \Phi (1-\Phi)$")
    # plt.plot(x, dtanh(x), label="dTanh, $\Phi'(x) = \\frac{1}{\cosh^2(x)}$")
    # plt.legend()
    # plt.ylim(0, 1.4)
    # plt.xlabel("x")
    # plt.ylabel("$\Phi'(x) = \\frac{d\Phi}{dx}$")
    # save_plot(fig, "deeplearning_activation_functions_derivative", (3, 2.7))

    ##############################################################################################
    ######################## HX SUPER SPIKE FIGURE ###############################################
    ##############################################################################################

    pop_xor = np.loadtxt("pop_xor.data")
    pop_xor[:, 1] /= 50  # use times up to 200µs instead of 10ms
    pop_xor[:, 0] -= 1  # start at zero until 95
    pop_xor_spiketrains = {0: (pop_xor[:20], 0),
                           1: (pop_xor[20:60], 1),
                           2: (pop_xor[60:100], 1),
                           3: (pop_xor[100:160], 0)}

    colors = ["#555555", "#AF5A50", "#005B82", "#7D966E", "#D7AA50"]
    target = [0, 1, 1, 0]
    m = ["o", "o", "o", "o"]
    s = [(r) ** 2 for r in np.arange(1, 9, 2.5)]
    s = s[::-1]
    lw = 1.4
    fig = plt.figure(figsize=(3, 3))
    for p, (spiketrain, c) in pop_xor_spiketrains.items():
        if p != 3:
            plt.scatter(spiketrain[:, 1] * 1e6, spiketrain[:, 0], s=s[p], lw=lw, facecolors='none',
                        edgecolors=colors[p])
        else:
            plt.scatter(spiketrain[:, 1] * 1e6, spiketrain[:, 0], s=s[p], lw=2, color=colors[p])
    plt.legend(["$S_{%s}$" % (i) for i in range(1, 5)], loc='upper right', bbox_to_anchor=(1, 1))
    # plt.legend(["$S_{%s}$, $\mathrm{class}=%s" % (i, target[i-1]) for i in range(1, 5)], loc='upper right', bbox_to_anchor=(1, 0.75))
    plt.ylim(60, 72)
    plt.xlim(72, 120)
    #plt.title('XOR Input  raster plot')
    plt.xlabel('spike time $(\si{\micro \s})$')
    plt.ylabel('input unit')
    save_plot(fig, "superspiketasksector", (3, 3))

    s = [(r) ** 2 for r in np.arange(1, 5, 1)]
    s = s[::-1]
    lw = 0.8
    fig = plt.figure(figsize=(3, 3))
    for p, (spiketrain, c) in pop_xor_spiketrains.items():
        if p != 3:
            plt.scatter(spiketrain[:, 1] * 1e6, spiketrain[:, 0], s=s[p], lw=lw, facecolors='none',
                        edgecolors=colors[p])
        else:
            plt.scatter(spiketrain[:, 1] * 1e6, spiketrain[:, 0], s=s[p], lw=lw, color=colors[p])
    plt.legend(["$S_{%s}$" % (i) for i in range(1, 5)], loc='upper right', bbox_to_anchor=(1, 1))
    # plt.legend(["$S_{%s}$, $\mathrm{class}=%s" % (i, target[i-1]) for i in range(1, 5)], loc='upper right', bbox_to_anchor=(1, 0.75))
    plt.xlim(0, 250)
    #plt.title('XOR Input  raster plot')
    plt.xlabel('spike time $(\si{\micro \s})$')
    plt.ylabel('input unit')
    save_plot(fig, "superspiketask", (3, 3))