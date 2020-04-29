import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gs
from scipy.stats import entropy
import pandas as pd
from scipy.optimize import curve_fit

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
    colors = ["#555555", "#AF5A50", "#005B82", "#7D966E", "#D7AA50"]
    label_voutput = "output frequency $\\nu_\mathrm{out} \; (\\si{\kilo \Hz})$"
    label_vinput = "input frequency $\\nu_\mathrm{in} \; (\\si{\kilo \Hz})$"
    double_shape = (3, 2.5)
    def fsigmoid(x, vmax=1, a=2, b=0):
        return vmax / (1.0 + np.exp(-a * (x - b)))
    ###################################################################################
    # single activation function with multiple weights

    mmnt = 0
    nrn = 9
    figure = plt.figure()

    plt.ylabel(label_voutput)
    plt.xlabel(label_vinput)
    for p in range(4):
        freq_in = input_rates
        freq_out = data[mmnt, p, :, nrn]
        # popt1, pcov1 = curve_fit(fsigmoid, freq_in, freq_out, method='dogbox',
        #                          bounds=([0, 0., -10000], [500, 0.1, 10000.]))
        plt.plot(freq_in, freq_out,
                 label="$w_\\text{in} = %s$" % input_weights[p], color=colors[p])
        # plt.plot(freq_in, fsigmoid(freq_in, *popt1), colors[p])  # , label="sigmoid fit")

        # plt.legend([Line2D([0],[0], marker='o', color="w", markerfacecolor="black")],
        #           ["input weight = {}".format(input_weights[p])])
    plt.ylim(-5,136)
    plt.legend()
    save_plot(figure, "single_calibrated_transfer_function_w_various_weights", double_shape)

    #
    # ###################################################################################
    # uncalibrated activation function (b=0, w=30)
    # mmnt = 2
    # w = 2 # corresponds to 30
    # fig = plt.figure()
    # plt.ylabel(label_voutput)
    # plt.xlabel(label_vinput)
    # plt.plot(input_rates, data[mmnt, w, :, 0:12])
    # save_plot(fig, "uncalibrated_activation_function", double_shape)
    #
    # mmnt = 0
    # w = 2 # corresponds to 30
    # fig = plt.figure()
    # plt.ylabel(label_voutput)
    # plt.xlabel(label_vinput)
    # plt.plot(input_rates, data[mmnt, w, :, 0:12])
    # save_plot(fig, "calibrated_activation_function", double_shape)
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
    # input_rates = np.linspace(-560, 560, 37)#true_input_spike_rate(np.linspace(-560, 560, 37)) * 1e3
    # workpoint = pseudo_adc_conversion_analog(300)
    #
    # mmnt = 0
    # n_params = 0
    # figure = plt.figure()
    # plt.ylabel(label_voutput)
    # plt.xlabel(label_vinput)
    # plt.ylim(-5,136)
    # #figure.gca().set_prop_cycle(color=colors)
    # styles = ["--", "-", ":"]
    #
    # for i, thres in enumerate(thresholds[mmnt]):
    #     plt.plot(input_rates, data[mmnt, i, :, 0:12], linestyle=styles[i])
    #
    # plt.legend([Line2D([0], [0], marker='', linestyle=style, color="black") for style in styles],
    #            ["$\delta V = \SI{%s}{\milli \V}$" % (np.round(workpoint - t, 1)) for t in thresholds[mmnt]],
    #            loc="upper left")
    # save_plot(figure, "activation_function_w_bias", double_shape)


    # ###################################################################################
    # # Gaussian Free Membrane Distribution
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
        ax.set_ylabel("density")

        mean_stored = mean
        ax.plot(x, gaus(x, mean, std))
        ax.axvline(mean, linestyle="--")
        ax.text(mean + 2, 0.02, '$V_{\mathrm{leak}}$')
        ax.hist(data, bins=200, alpha=0.3, density=True)
        ax.axvline(x[150 - 1], linestyle="--", color="red")
        ax.text(x[150 - 1] + 2, 0.02, '$\\vartheta$')
        ax.bar(x[150:], y_normed[150:])

        # ax.legend(["Gaussian Fit", "$V_{leak}$", "$V_{thres}$", "Histogram", "Spiking"], loc="upper left")
    ax.set_xlabel("$membrane potential V_\mathrm{m} \; (\si{\milli \V})$")
    save_plot(fig, "activation_function_vmem_distr_with_thres", (5,3.5))

    # ##################################################################################
    # # Gaussian Free Membrane Distribution
    #
    n = 2500 // 5

    real_epochs = np.array(range(n)) * 5

    alpha = 0.27
    fs = 4
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 3), sharex=True, sharey="col")
    wh_legend_labels = ["$w_{%s%s}$" % (i, j) for i in range(2) for j in range(5)]
    wo_legend_labels = ["$w_{%s%s}$" % (i, j) for i in range(5) for j in range(1)]
    bh_legend_labels = ["$b_%s$" % i for i in range(5)]
    bo_legend_labels = ["$b_%s$" % i for i in range(1)]

    # hidden weights
    axes[0, 0].plot(real_epochs, weights_hidden_storage[:n, 0, :5])
    axes[0, 0].plot(real_epochs, weights_hidden_storage[:n, 1, :5], ls="--")
    axes[0, 0].plot(real_epochs, weights_hidden_storage[:n, 0, 5:], alpha=alpha)
    axes[0, 0].plot(real_epochs, weights_hidden_storage[:n, 1, 5:], ls="--", alpha=alpha)

    # hidden bias
    axes[0, 1].plot(real_epochs, bias_storage[:n, :5])
    axes[0, 1].plot(real_epochs, bias_storage[:n, 5:n_h_nrns], alpha=alpha)

    # output weights (with hidden in background)
    axes[1, 0].plot(real_epochs, weights_out_storage[:n, :5])
    axes[1, 0].plot(real_epochs, weights_out_storage[:n, 5:], alpha=alpha)

    # output bias with hidden in background
    axes[1,1].plot(real_epochs, bias_storage[:n,-1])
    #axes[1, 1].plot(real_epochs, bias_storage[:, :5], alpha=alpha)
    #axes[1, 1].plot(real_epochs, bias_storage[:, 5:n_h_nrns], alpha=alpha)

    # legends
    xlegend = 0.98
    axes[0, 1].legend(bh_legend_labels, fontsize=fs, loc=(xlegend, 0.6))
    axes[0, 0].legend(wh_legend_labels, fontsize=fs,
                      loc=(xlegend, .2), )  # title="Unit", title_fontsize=6)
    axes[1, 1].legend(bo_legend_labels, fontsize=fs, loc=(xlegend, .9))
    axes[1, 0].legend(wo_legend_labels, fontsize=fs,
                      loc=(xlegend, 0.6), )  # title="Unit", title_fontsize=6)

    # labels
    axes[0, 1].set_ylabel("$\\vartheta \propto -b^{(\mathrm{h})} \; (\si{\milli \V})$")
    axes[1, 1].set_ylabel("$\\vartheta \propto -b^{(\mathrm{o})} \;(\si{\milli \V})$")
    axes[0, 0].set_ylabel("$W^{(\mathrm{h})}$")
    axes[1, 0].set_ylabel("$W^{(\mathrm{o})}$")
    for i in range(2):
        axes[1, i].set_xlabel("Iteration")

    plt.subplots_adjust(hspace=0.05
                        , wspace=0.4)

    save_plot(fig, "network_evolution_circles", (7, 3))
    plt.subplots_adjust(hspace=0.4
                        , wspace=0.4)
    save_plot(fig, "network_evolution_circles_larger", (7, 4))
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
    #
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
    # plt.plot(x, fsigmoid(x), label="sigmoid, $\Phi(x) = \\frac{1}{1 + e^{(-\\beta x)}}$")
    # plt.plot(x, np.tanh(x), label="tanh, $\Phi(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$")
    # plt.ylim(-1, 2)
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("transfer function $\Phi(x)$")
    # save_plot(fig, "deeplearning_activation_functions", double_shape)
    #
    # x = np.linspace(-3, 3, 200)
    # fig = plt.figure()
    # plt.plot(x, drelu(x), label="dReLu, $\Phi'(x) = \Theta(x)$")
    # plt.plot(x, dsig(x), label="dsigmoid, $\Phi'(x) = \Phi (1-\Phi)$")
    # plt.plot(x, dtanh(x), label="dtanh, $\Phi'(x) = \\frac{1}{\cosh^2(x)}$")
    # plt.legend()
    # plt.ylim(0, 1.4)
    # plt.xlabel("x")
    # plt.ylabel("$\Phi'(x) = \\frac{d\Phi}{dx}$")
    # save_plot(fig, "deeplearning_activation_functions_derivative", double_shape)

    ############# theoretical sigmoid activation function ############################
    # sigmoid fit

    # ###### standalone ###########
    # sigmoid_mmnt_npz = np.load("sigmoid_mmnt_20000.npz")
    # freq_in = sigmoid_mmnt_npz["freq_input_sequence"] / 1e3
    # freq_out = sigmoid_mmnt_npz["freq_output_sequence"] / 1e3
    #
    # popt1, pcov1 = curve_fit(fsigmoid, freq_in, freq_out, method='dogbox',
    #                          bounds=([0, 0., -10000], [100, 0.1, 10000.]))
    #
    # fig = plt.figure()
    # plt.plot(freq_in, freq_out, '.', label="simulated activation function")
    # plt.plot(freq_in, fsigmoid(freq_in, *popt1), label="sigmoid fit")
    #
    # plt.ylabel(label_voutput)
    # plt.xlabel(label_vinput)
    # plt.legend()
    # save_plot(fig, "theoretical_activation_function",double_shape)
    # save_plot(fig, "theoretical_activation_function", (4, 3))

    ###### variable weight#####

    # sigmoid_mmnt_npz = np.load("sigmoid_mmnt_variable_weights_longer_final.npz")
    # freq_in = sigmoid_mmnt_npz["freq_input_sequence"] / 1e3
    # freq_out = sigmoid_mmnt_npz["freq_output_sequence_storage"] / 1e3
    # changing_parameter = sigmoid_mmnt_npz["changing_parameter"]
    #
    # fig = plt.figure()
    # for i, p in enumerate(changing_parameter):
    #     popt1, pcov1 = curve_fit(fsigmoid, freq_in, freq_out[i], method='dogbox',
    #                              bounds=([0, 0., -10000], [100, 0.1, 10000.]))
    #     plt.plot(freq_in, freq_out[i], '.', label="$w_\mathrm{in} = %s$" % p, color=colors[i])
    #     plt.plot(freq_in, fsigmoid(freq_in, *popt1), colors[i])#, label="sigmoid fit")
    #
    # plt.ylabel(label_voutput)
    # plt.xlabel(label_vinput)
    # plt.legend()
    # save_plot(fig, "theoretical_activation_function_variableweight_longer", double_shape)
    #
    # # ###### variable biases#####
    # sigmoid_mmnt_npz = np.load("sigmoid_mmnt_variable_bias.npz")
    # freq_in = sigmoid_mmnt_npz["freq_input_sequence"] / 1e3
    # freq_out = sigmoid_mmnt_npz["freq_output_sequence_storage"] / 1e3
    # changing_parameter = sigmoid_mmnt_npz["changing_parameter"]
    #
    # # freq_out = freq_out[[1,2,3,4,5],:]
    # # changing_parameter = changing_parameter[[1,2,3,4,5]]
    # fig = plt.figure()
    # for i, p in enumerate(changing_parameter):
    #     popt1, pcov1 = curve_fit(fsigmoid, freq_in, freq_out[i], method='dogbox',
    #                              bounds=([0, 0., -10000], [100, 0.1, 10000.]))
    #     plt.plot(freq_in, freq_out[i], '.', label="$\delta V = \SI{%s}{\V}$" % p, color=colors[i])
    #     plt.plot(freq_in, fsigmoid(freq_in, *popt1), colors[i])#, label="sigmoid fit")
    #
    #
    # plt.ylabel(label_voutput)
    # plt.xlabel(label_vinput)
    # plt.legend()
    # save_plot(fig, "theoretical_activation_function_variablebias_longer_final", double_shape)

    # ######### FREE MEMBRANE ###############
    # free_membrane_npz = np.load("free_membrane_mmnt_20000.npz")
    # mem_tr = free_membrane_npz["mem_tr"]
    # n_bins = 100
    # # _, x_bins = np.histogram(mem_tr, bins=n_bins)
    # bin_height, bins_edges = np.histogram(mem_tr, bins=n_bins, density=True)
    # binwidth = np.median(np.diff(bins_edges))
    # x_bins = bins_edges[1:] - 0.5*binwidth
    # def gaus(x, mue, sig):
    #     return np.exp(-(x - mue) ** 2 / (2 * sig ** 2)) / sig / np.sqrt(2 * np.pi)
    #
    # x = np.linspace(-1, 1, n_bins)
    # mean = mem_tr.mean()
    # gaus_fit = gaus(x, mean, np.std(mem_tr))
    #
    # fig = plt.figure()
    # plt.ylabel("density")
    # plt.xlabel("membrane potential $V_\mathrm{m} \; (\\si{\V})$")
    # plt.plot(x, gaus_fit, label="Gaussian fit")
    # plt.hist(mem_tr, bins=n_bins, density=True, alpha=0.5, label="free membrane")
    # # recolor and naming
    # a = 65
    # plt.bar(x_bins[a:], bin_height[a:], width=binwidth*1.05)
    # plt.axvline(mean, linestyle="--", color="black")
    # plt.text(mean+0.05, .61, '$V_{\mathrm{leak}}$')
    # plt.text(x_bins[a] + 0.05, .61, '$\\vartheta$', color="red")
    # plt.axvline(x_bins[a] - 0.5*binwidth, linestyle="--", color="red")
    # plt.legend()
    # # plt.ylim(0,1.6)
    #
    #
    # save_plot(fig, "theoretical_free_membrane_small",double_shape)
    # save_plot(fig, "theoretical_free_membrane_small",(5,3.5))


    ##############################################################################################
    ######################## HX SUPER SPIKE FIGURE ###############################################
    ##############################################################################################

    # pop_xor = np.loadtxt("pop_xor.data")
    # pop_xor[:, 1] /= 250  # use times up to 200µs instead of 10ms
    # pop_xor[:, 0] -= 1  # start at zero until 95
    # pop_xor_spiketrains = {0: (pop_xor[:20], 0),
    #                        1: (pop_xor[20:60], 1),
    #                        2: (pop_xor[60:100], 1),
    #                        3: (pop_xor[100:160], 0)}
    #
    # colors = ["#555555", "#AF5A50", "#005B82", "#7D966E", "#D7AA50"]
    # target = [0, 1, 1, 0]
    # m = ["o", "o", "o", "o"]
    # s = [(r) ** 2 for r in np.arange(1, 9, 2.5)]
    # s = s[::-1]
    # lw = 1.4
    # fig = plt.figure(figsize=(3, 3))
    # for p, (spiketrain, c) in pop_xor_spiketrains.items():
    #     if p != 3:
    #         plt.scatter(spiketrain[:, 1] * 1e6, spiketrain[:, 0], s=s[p], lw=lw, facecolors='none',
    #                     edgecolors=colors[p])
    #     else:
    #         plt.scatter(spiketrain[:, 1] * 1e6, spiketrain[:, 0], s=s[p], lw=2, color=colors[p])
    # plt.legend(["$S_{%s}$" % (i) for i in range(1, 5)], loc='upper right', bbox_to_anchor=(1, 1))
    # # plt.legend(["$S_{%s}$, $\mathrm{class}=%s" % (i, target[i-1]) for i in range(1, 5)], loc='upper right', bbox_to_anchor=(1, 0.75))
    # plt.ylim(60, 72)
    # plt.xlim(72/5, 120/5)
    # #plt.title('XOR Input  raster plot')
    # plt.xlabel('spike time $(\si{\micro \s})$')
    # plt.ylabel('input unit')
    # save_plot(fig, "superspiketasksector", (3, 3))
    #
    # s = [(r) ** 2 for r in np.arange(1, 5, 1)]
    # s = s[::-1]
    # lw = 0.8
    # fig = plt.figure(figsize=(3, 3))
    # for p, (spiketrain, c) in pop_xor_spiketrains.items():
    #     if p != 3:
    #         plt.scatter(spiketrain[:, 1] * 1e6, spiketrain[:, 0], s=s[p], lw=lw, facecolors='none',
    #                     edgecolors=colors[p], label="$S_{%s}$" % (p+1))
    #     else:
    #         plt.scatter(spiketrain[:, 1] * 1e6, spiketrain[:, 0], s=s[p], lw=lw, color=colors[p], label="$S_{%s}$" % (p+1))
    # legend_handles_labels = fig.gca().get_legend_handles_labels()
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    # # plt.legend(["$S_{%s}$, $\mathrm{class}=%s" % (i, target[i-1]) for i in range(1, 5)], loc='upper right', bbox_to_anchor=(1, 0.75))
    # plt.xlim(0, 250/5)
    # #plt.title('XOR Input  raster plot')
    # plt.xlabel('spike time $(\si{\micro \s})$')
    # plt.ylabel('input unit')
    # save_plot(fig, "superspiketask", (3, 3))
    #
    # s = [(r) ** 2 for r in np.arange(1, 5, 1)]
    # s = s[::-1]
    # lw = 0.8
    # fig = plt.figure(figsize=(3, 3))
    #
    # batchsize = 8
    # batch = np.random.randint(0, 4, size=batchsize)
    # for i, p in enumerate(batch):
    #     (spiketrain, c) = pop_xor_spiketrains[p]
    #     if p != 3:
    #         plt.scatter(spiketrain[:, 1] * 1e6 + 250*i, spiketrain[:, 0], s=s[p], lw=lw, facecolors='none',
    #                     edgecolors=colors[p])
    #     else:
    #         plt.scatter(spiketrain[:, 1] * 1e6 + 250*i, spiketrain[:, 0], s=s[p], lw=lw, color=colors[p])
    #
    # plt.legend(*legend_handles_labels, loc='upper right', bbox_to_anchor=(1, 1))
    # # plt.legend(["$S_{%s}$, $\mathrm{class}=%s" % (i, target[i-1]) for i in range(1, 5)], loc='upper right', bbox_to_anchor=(1, 0.75))
    # plt.xlim(-30, 250*batchsize)
    # #plt.title('XOR Input  raster plot')
    # plt.xlabel('spike time $(\si{\micro \s})$')
    # plt.ylabel('input unit')
    # save_plot(fig, "superspiketaskconsecutive", (6, 3))

    ###################################################################################
    # # pre post calibration
    # vleak_pre_post = np.load("vleak_pre_post_500.npz")
    # # vreset_pre_post_150 = np.load("vreset_pre_post_150.npz")
    # # vreset_pre_post_200 = np.load("vreset_pre_post_200.npz")
    # # vreset_pre_post_300 = np.load("vreset_pre_post_300.npz")
    # vreset_pre_post_400 = np.load("vreset_pre_post_400.npz")
    # vthreshold_pre_post_750 = np.load("vthreshold_pre_post_750.npz")
    # vthreshold_pre_post_800 = np.load("vthreshold_pre_post_800.npz")
    # # vthreshold_pre_post_900 = np.load("vthreshold_pre_post_900.npz")
    #
    # # vleak
    # fig = plt.figure(figsize=(3, 3))
    # sns.distplot(np.median(vleak_pre_post["pre_cadc_data_V"], axis=0), kde=False, norm_hist=True, label="pre $\SI{0.5}{\V}$")
    # sns.distplot(np.median(vleak_pre_post["post_cadc_data_V"], axis=0), kde=False, norm_hist=True, label="post $\SI{0.5}{\V}$")
    # plt.legend()
    # plt.xlim(0.35, 0.7)
    # plt.xlabel('$V_\mathrm{leak} \; (\si{\V})$')
    # plt.ylabel('density')
    # save_plot(fig, "vleak_pre_post_calibration", (2, 2))
    #
    # # vreset
    # fig = plt.figure(figsize=(3, 3))
    # # sns.distplot(vreset_pre_post_300["pre_cadc_data_V"], label="pre 0.3 V")
    # sns.distplot(vreset_pre_post_400["pre_cadc_data_V"], kde=False, norm_hist=True,  label="pre $\SI{0.4}{\V}$")
    # # sns.distplot(vreset_pre_post_300["post_cadc_data_V"], bins=5, label="post 0.3 V")
    # sns.distplot(vreset_pre_post_400["post_cadc_data_V"], bins=5, kde=False, norm_hist=True,  label="post $\SI{0.4}{\V}$")
    # plt.legend()
    # plt.xlim(0.32, 0.5)
    # plt.xlabel('$V_\mathrm{reset} \; (\si{\V})$')
    # plt.ylabel('density')
    # save_plot(fig, "vreset_pre_post_calibration", (2, 2))
    #
    # # vthreshold
    # fig = plt.figure(figsize=(3, 3))
    # sns.distplot(vthreshold_pre_post_750["pre_cadc_max_V"], kde=False, norm_hist=True, label="pre $\SI{0.75}{\V}$")
    # sns.distplot(vthreshold_pre_post_750["post_cadc_max_V"],kde=False, norm_hist=True,  bins=4, label="post $\SI{0.75}{\V}$")
    # sns.distplot(vthreshold_pre_post_800["pre_cadc_max_V"], kde=False, norm_hist=True, label="pre $\SI{0.8}{\V}$")
    # sns.distplot(vthreshold_pre_post_800["post_cadc_max_V"],kde=False, norm_hist=True,  bins=4, label="post $\SI{0.8}{\V}$")
    # plt.legend()
    # plt.xlim(0.65,1)
    # plt.xlabel('$\\vartheta \; (\si{\V})$')
    # plt.ylabel('density')
    # save_plot(fig, "vthreshold_pre_post_calibration", (2, 2))

    # # # time offset cadc ppu
    # cadcppuoffset = np.load("cadcppuoffset.npz")
    # offsets = cadcppuoffset["offsets"]
    # offsets_corrected = cadcppuoffset["offsets_corrected"]
    # a = 0.5
    # fig = plt.figure()
    # plt.hist(offsets, alpha=a, color=colors[0], label="$\Delta T$", density=True)
    # plt.axvline(offsets.mean(), color=colors[0], ls="--", label="mean")
    # plt.hist(offsets_corrected, alpha=a, color=colors[1], label="$\Delta T_\mathrm{corr}$",
    #          density=True)
    # plt.axvline(offsets_corrected.mean(), color=colors[1], ls="--", label="corr. mean")
    # plt.xlabel("$\Delta T \; (\si{\micro \s})$")
    # plt.ylabel("density")
    # plt.ylim(0,0.42)
    # plt.legend()
    # save_plot(fig, "cadcppuoffset", (2.5,2.0))

    # # cadc calibration
    # cadc_calib = np.load("cadc_calib_70.npz")
    # def cadc_to_neuron(cadc):
    #     neuron_half = (cadc // 64) + 1
    #     neuron_quad = cadc // 128
    #     vector_part = (cadc // 16) % 4
    #     vector_count = cadc % 16
    #     neuron_index = 128 * (neuron_half % 2) + vector_count + 16 * (
    #                 7 - (2 * vector_part)) - 16 * neuron_quad
    #     return neuron_index
    # idx_on_neuron = cadc_to_neuron(np.array(range(256)))
    #
    # n = cadc_calib["n"]
    # dynamic_range_in_V = cadc_calib["dac_in_V"]
    # p_per_q = cadc_calib["lin_fit_params_per_quadrant"]
    # # pre calibration
    # fig, axes = plt.subplots(1,2, figsize=(4,2))
    # for quad in range(2):
    #     mask = ((128 * quad) <= idx_on_neuron) & (idx_on_neuron < (128 * (quad + 1)))
    #     cadc_per_quad = cadc_calib["pre_calib_cadc_storage"][:, mask]
    #
    #     axes[quad].plot(dynamic_range_in_V, cadc_per_quad)
    #     axes[quad].set_xlabel("reference voltage $(\si{\V})$")
    #     axes[quad].set_ylabel("CADC lsb")
    # save_plot(fig, "pre_cadc_calib", (4,2))
    #
    # # post calibration
    # fig = plt.figure()
    # for quad in range(2):
    #     mask = ((128 * quad) <= idx_on_neuron) & (idx_on_neuron < (128 * (quad + 1)))
    #     cadc_per_quad = cadc_calib["post_calib_cadc_storage"][:, mask]
    #
    #
    #     lin_fit = lambda x: p_per_q[quad, 0] * x + p_per_q[quad, 1]
    #     # find offsets:
    #     i = n//2
    #     data = cadc_per_quad[i]
    #     plt.plot(dynamic_range_in_V, lin_fit(dynamic_range_in_V), label="linear fit q%s " % quad) #: %s *x + %s " %(np.round(p[0],2), np.round(p[1],2)))
    #     plt.plot(dynamic_range_in_V, cadc_per_quad, ls="--", alpha=0.1)
    #     plt.xlabel("reference voltage $(\si{\V})$")
    #     plt.ylabel("CADC lsb")
    #     plt.legend()
    #     plt.ylim(10, 235)
    #
    # save_plot(fig, "post_cadc_calib", (2,2))