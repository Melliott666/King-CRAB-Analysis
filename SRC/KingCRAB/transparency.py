"""High-level PMT window-transparency analysis."""
from dataclasses import dataclass
import glob, os
import numpy as np
from .pmt import *

def configure(**context):
    globals().update(context)

@dataclass(frozen=True)
class TransparencyConfig(PMTProcessingConfig):
    qc_percentile: float = 5.0
    charge_hist_bins: int = 1000

def get_result(results,label,voltage):
    return next((r for r in results if r["Label"]==label and r["Voltage [V]"]==voltage),None)

def process_transparency_folder(data_folder, label, voltage, config=TransparencyConfig()):
    T_PRE_SIGNAL=config.pre_signal_s; APPLY_FILTER=config.filter_cutoff_hz is not None; F_CUTOFF=config.filter_cutoff_hz
    PULSE_START_TIME=config.pulse_start_s; R_TERMINATION=config.termination_ohm; THRESHOLD_SIGMA=config.threshold_sigma
    QC_PERCENTILE=config.qc_percentile; CHARGE_HIST_BINS=config.charge_hist_bins
    target_files = sorted(glob.glob(os.path.join(data_folder, "C1C*.txt")))

    if len(target_files) == 0:
        return {
            "Label": label,
            "Voltage [V]": voltage,
            "Folder": data_folder,
            "Number of files": 0,
            "Number of waveforms": 0,
            "Accepted pulses N_acc": 0,
            "Total accepted charge [C]": np.nan,
            "Total accepted charge error [C]": np.nan,
            "Charge per waveform [C/wf]": np.nan,
            "Charge per waveform error [C/wf]": np.nan,
            "Mean accepted pulse charge [C/pulse]": np.nan,
            "Mean accepted pulse charge error [C/pulse]": np.nan,
            "Charge peak [C]": np.nan,
            "Charge peak error [C]": np.nan,
            "Charge peak pulses": 0,
            "Accepted charges": np.array([]),
            "All charges": np.array([]),
            "Average time all [s]": np.array([]),
            "Average time accepted [s]": np.array([]),
            "Average time [s]": np.array([]),
            "Average waveform all [V]": np.array([]),
            "Average waveform accepted [V]": np.array([]),
            "Average waveform all N": 0,
            "Average waveform accepted N": 0,
            "Status": "No C1C*.txt files found",
        }

    all_waveforms = []
    for filepath in target_files:
        all_waveforms.extend(processing_files(filepath))

    waveforms_norm = normalize_waveform_times(all_waveforms)
    waveforms_bs, baselines, sigmas = subtract_baseline(waveforms_norm, t_pre_signal=T_PRE_SIGNAL)

    if APPLY_FILTER:
        waveforms_proc = filter_waveforms(waveforms_bs, f_cut=F_CUTOFF)
    else:
        waveforms_proc = waveforms_bs.copy()

    H, Q_full, sigma_evt = extract_observables(
        waveforms_proc,
        sigmas,
        pulse_start_time=PULSE_START_TIME,
        R=R_TERMINATION,
    )

    Q_region = []
    for t, v in waveforms_proc:
        q_i, tl_i, tr_i = integrate_pulse_region(t, v, R=R_TERMINATION, pulse_start_time=PULSE_START_TIME)
        Q_region.append(q_i)

    Q_region = np.array(Q_region)
    sigma_mean = np.nanmean(sigma_evt)
    V_thr = THRESHOLD_SIGMA * sigma_mean
    height_pass = np.isfinite(H) & (H > V_thr)

    accepted_charge_from_height = Q_region[height_pass & np.isfinite(Q_region)]
    Qc = np.nanpercentile(accepted_charge_from_height, QC_PERCENTILE) if len(accepted_charge_from_height) else np.nan
    charge_pass = np.isfinite(Q_region) & (Q_region >= Qc) if np.isfinite(Qc) else np.zeros(len(Q_region), dtype=bool)

    Q_acc = Q_region[charge_pass]
    q_peak, q_peak_err, n_peak, counts, edges = estimate_charge_peak(Q_acc, bins=CHARGE_HIST_BINS)

    avg_t_all, avg_v_all, avg_n_all = average_filtered_waveform(waveforms_proc)
    avg_t_acc, avg_v_acc, avg_n_acc = average_filtered_waveform(waveforms_proc, keep_mask=charge_pass)

    total_charge = np.nansum(Q_acc) if len(Q_acc) else np.nan

    # Normalize by all waveforms. This measures accepted charge per trigger/window,
    # so it includes both pulse size and how often accepted pulses occur.
    mean_accepted_charge = np.nanmean(Q_acc) if len(Q_acc) else np.nan
    mean_accepted_charge_error = (
        np.nanstd(Q_acc, ddof=1) / np.sqrt(len(Q_acc))
        if len(Q_acc) > 1 else np.nan
    )
    total_charge_error = mean_accepted_charge_error * len(Q_acc) if np.isfinite(mean_accepted_charge_error) else np.nan
    charge_per_waveform = total_charge / len(waveforms_proc) if len(waveforms_proc) else np.nan
    charge_per_waveform_error = total_charge_error / len(waveforms_proc) if len(waveforms_proc) and np.isfinite(total_charge_error) else np.nan

    durations = waveform_duration(waveforms_proc)
    T_window = np.nanmean(durations) if len(durations) else np.nan
    T_window_total = len(waveforms_proc) * T_window if np.isfinite(T_window) else np.nan
    charge_rate_window = total_charge / T_window_total if np.isfinite(T_window_total) and T_window_total > 0 else np.nan

    return {
        "Label": label,
        "Voltage [V]": voltage,
        "Folder": data_folder,
        "Number of files": len(target_files),
        "Number of waveforms": len(waveforms_proc),
        "Mean noise sigma [V]": sigma_mean,
        "Voltage threshold V_thr [V]": V_thr,
        "Charge threshold Qc [C]": Qc,
        "Accepted pulses N_acc": int(len(Q_acc)),
        "Total accepted charge [C]": total_charge,
        "Total accepted charge error [C]": total_charge_error,
        "Charge per waveform [C/wf]": charge_per_waveform,
        "Charge per waveform error [C/wf]": charge_per_waveform_error,
        "Mean accepted pulse charge [C/pulse]": mean_accepted_charge,
        "Mean accepted pulse charge error [C/pulse]": mean_accepted_charge_error,
        "Window-normalized charge rate [C/s]": charge_rate_window,
        "Charge peak [C]": q_peak,
        "Charge peak error [C]": q_peak_err,
        "Charge peak pulses": int(n_peak),
        "Accepted charges": Q_acc,
        "All charges": Q_region,
        "Average time all [s]": avg_t_all,
        "Average time accepted [s]": avg_t_acc,
        "Average time [s]": avg_t_all,
        "Average waveform all [V]": avg_v_all,
        "Average waveform accepted [V]": avg_v_acc,
        "Average waveform all N": int(avg_n_all),
        "Average waveform accepted N": int(avg_n_acc),
        "Status": "OK",
    }


def add_dataset_spec(dataset_specs, seen_datasets, label, voltage, folder, file_pattern="C1C*.txt"):
    key = (label, voltage)
    if key in seen_datasets:
        return

    seen_datasets.add(key)
    dataset_specs.append({
        "label": label,
        "voltage": voltage,
        "folder": folder,
        "file_pattern": file_pattern,
    })

def result_value_and_error(label, voltage=900):
    result = get_result(label, voltage)
    if result is None or result["Status"] != "OK":
        return np.nan, np.nan
    return result["Charge per waveform [C/wf]"], result["Charge per waveform error [C/wf]"]

def plot_focused_group(group):
    values = []
    errors = []
    for label in group["labels"]:
        value, error = result_value_and_error(label, 900)
        values.append(value)
        errors.append(error)
        print(f"{label:20s} 900 V charge/wf = {value:.4e} +/- {error:.4e} C/wf")

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(
        group["labels"],
        values,
        yerr=errors,
        capsize=5,
        color=group["colors"],
        edgecolor="black",
        linewidth=0.8,
    )


def first_existing_folder(candidates):
    for folder in candidates:
        if os.path.isdir(folder):
            return folder
    return candidates[0]
