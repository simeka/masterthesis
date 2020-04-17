import brian2 as b2
import numpy as np
from brian2 import TimedArray
from neurodynex.adex_model import AdEx
from neurodynex.leaky_integrate_and_fire import LIF
from neurodynex.tools import input_factory

from plotting.helpers.preamble import *

if __name__ == '__main__':
    b2.defaultclock.dt = 0.1 * b2.ms

    low_step = input_factory.get_step_current(t_start=100, t_end=200,
                                              unit_time=b2.ms,
                                              amplitude=500 * b2.pA)
    high_step = input_factory.get_step_current(t_start=200, t_end=400,
                                               unit_time=b2.ms,
                                               amplitude=720 * b2.pA)

    step_current = TimedArray(np.concatenate((low_step.values,
                                              high_step.values)) * 1000 * b2.mA,
                              b2.ms)

    (lif_state, _) = LIF.simulate_LIF_neuron(firing_threshold=-50 * b2.mV,
                                             v_reset=-65 * b2.mV,
                                             v_rest=-60 * b2.mV,
                                             abs_refractory_period=10 * b2.ms,
                                             membrane_time_scale=10 * b2.ms,
                                             membrane_resistance=15 * b2.Mohm,
                                             input_current=step_current,
                                             simulation_time=800 * b2.ms)

    (adex_state, _) = AdEx.simulate_AdEx_neuron(I_stim=step_current,
                                                v_rest=-60 * b2.mV,
                                                v_reset=-65 * b2.mV,
                                                tau_m=10 * b2.ms,
                                                R=15 * b2.Mohm,
                                                b=0 * b2.pA,
                                                tau_w=20 * b2.ms,
                                                a=20 * b2.nS,
                                                v_rheobase=-50 * b2.mV,
                                                v_spike=-35 * b2.mV,
                                                simulation_time=800 * b2.ms)
    current_trace = step_current.values
    lif_trace = (lif_state.v / b2.mV)[0]
    adex_trace = (adex_state.v / b2.mV)[0]
    times = lif_state.t / b2.ms

    fig = plt.figure(figsize=(5.5, 3))
    ax1 = plt.subplot(311)
    plt.plot(times, lif_trace, color=default_linecolor)
    plt.ylabel(r"$V^\mathrm{LIF}_\mathrm{m}$ [mV]")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.text(0.02, 0.85, "(a)", transform=ax1.transAxes, va='top',
             fontweight="bold")

    ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
    plt.plot(times, adex_trace, color=default_linecolor)
    plt.ylabel(r"$V^\mathrm{AdEx}_\mathrm{m}$ [mV]")
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.text(0.02, 0.85, "(b)", transform=ax2.transAxes, va='top',
             fontweight="bold")

    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(times,
             np.array([step_current(t * b2.ms, 0) for t in times]) * 1E9,
             color=default_linecolor)
    ax3.text(0.02, 0.85, "(c)", transform=ax3.transAxes, va='top',
             fontweight="bold")
    plt.ylabel(r"$I_\mathrm{stim}$ [nA]")
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r"Time [ms]")
    plt.xlim(0, 750)
    ax3.yaxis.set_label_coords(-0.1, 0.5)

    additional_global_configuration(ax1)
    additional_global_configuration(ax2)
    additional_global_configuration(ax3)

    plt.savefig(TMP_FILE,
                transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(FIG_FILE,
                transparent=True, bbox_inches='tight', pad_inches=0.02)
