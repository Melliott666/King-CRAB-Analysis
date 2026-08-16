"""Notebook-independent helpers extracted from Gain_Curve.ipynb."""
from __future__ import annotations
from pathlib import Path
import os, re, json, glob, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
 from scipy.integrate import trapezoid, quad
 from scipy.optimize import curve_fit
 from scipy.signal import butter, filtfilt, welch
 from scipy.stats import norm, binom
except ImportError:
 pass


def listed_gain_hint(voltage):
    voltage = float(voltage)
    if voltage < _hint_voltage.min() or voltage > _hint_voltage.max():
        return 10**np.polyval(_hint_coeff, voltage)
    return 10**np.interp(voltage, _hint_voltage, np.log10(_hint_gain))

def trapezoid_integral(y, x):
    """Use the trapezoid API available in the active NumPy version."""
    integration_function = getattr(np, "trapezoid", None)
    if integration_function is None:
        integration_function = np.trapz
    return integration_function(y, x)

def estimate_spe_charge(charges, bins=80):
    # Simple fallback: largest positive histogram peak. This is not used as a
    # final gain when the pedestal + multi-PE fit fails.
    charges = np.array(charges)
    charges = charges[np.isfinite(charges) & (charges > 0)]

    if len(charges) < 10:
        return {
            "q_spe": np.nan,
            "q_spe_err": np.nan,
            "n_peak": 0,
            "hist_counts": np.array([]),
            "hist_edges": np.array([]),
            "peak_left": np.nan,
            "peak_right": np.nan,
        }

    q_low = np.nanpercentile(charges, 1)
    q_high = np.nanpercentile(charges, 95)
    charges_trim = charges[(charges >= q_low) & (charges <= q_high)]
    if len(charges_trim) < 10 or q_high <= q_low:
        return {
            "q_spe": np.nan,
            "q_spe_err": np.nan,
            "n_peak": 0,
            "hist_counts": np.array([]),
            "hist_edges": np.array([]),
            "peak_left": np.nan,
            "peak_right": np.nan,
        }

    counts, edges = np.histogram(charges_trim, bins=bins)
    idx_peak = np.argmax(counts)
    q_left = edges[max(idx_peak - 1, 0)]
    q_right = edges[min(idx_peak + 2, len(edges) - 1)]
    peak_charges = charges_trim[(charges_trim >= q_left) & (charges_trim <= q_right)]
    q_peak = np.nanmean(peak_charges) if len(peak_charges) else np.nan
    q_peak_err = np.nanstd(peak_charges, ddof=1) / np.sqrt(len(peak_charges)) if len(peak_charges) > 1 else np.nan

    return {
        "q_spe": q_peak,
        "q_spe_err": q_peak_err,
        "n_peak": int(len(peak_charges)),
        "hist_counts": counts,
        "hist_edges": edges,
        "peak_left": q_left,
        "peak_right": q_right,
    }

def pedestal_multi_pe_model(q, q_ped, q_spe, sigma_ped, sigma_pe, background, *amps):
    # amps[0] is pedestal amplitude; amps[1] is 1 PE; amps[2] is 2 PE; etc.
    y = np.full_like(q, background, dtype=float)
    for n, amp in enumerate(amps):
        center_n = q_ped + n * q_spe
        sigma_n = sigma_ped if n == 0 else np.sqrt(sigma_ped**2 + n * sigma_pe**2)
        y += amp * np.exp(-0.5 * ((q - center_n) / sigma_n)**2)
    return y

def fit_multi_pe_charge_spectrum(charges, bins=80, n_pe_max=5, min_pulses=40, q_spe_hint=None):
    charges = np.array(charges)
    charges = charges[np.isfinite(charges)]

    fallback = estimate_spe_charge(charges, bins=bins)
    fallback.update({
        "fit_method": "fit_failed_no_gain",
        "fit_success": False,
        "fit_message": "multi-PE fit was not successful; no corrected gain extracted",
        "fit_q": np.array([]),
        "fit_y": np.array([]),
        "fit_params": None,
        "fit_errors": None,
        "n_pe_max": n_pe_max,
    })

    if len(charges) < min_pulses:
        fallback["fit_message"] = f"too few charges for fit: {len(charges)} < {min_pulses}"
        return fallback

    try:
        from scipy.optimize import curve_fit
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks
    except Exception as exc:
        fallback["fit_message"] = f"scipy fitting tools unavailable: {exc}"
        return fallback

    q_low = np.nanpercentile(charges, 0.2)
    q_high = np.nanpercentile(charges, 99.5)
    charges_fit = charges[(charges >= q_low) & (charges <= q_high)]
    if len(charges_fit) < min_pulses or q_high <= q_low:
        fallback["fit_message"] = "charge percentile fit range collapsed"
        return fallback

    counts, edges = np.histogram(charges_fit, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = np.nanmedian(np.diff(edges))
    if not np.isfinite(bin_width) or bin_width <= 0:
        fallback["fit_message"] = "invalid histogram bin width"
        return fallback

    yerr = np.sqrt(np.maximum(counts, 1.0))
    q_span = q_high - q_low
    y_smooth = gaussian_filter1d(counts.astype(float), sigma=1.0)
    peak_indices, _ = find_peaks(y_smooth, prominence=max(2.0, 0.03 * np.nanmax(y_smooth)))
    peak_positions = centers[peak_indices]
    peak_positions = peak_positions[np.isfinite(peak_positions)]

    spacing_guesses = []
    if q_spe_hint is not None and np.isfinite(q_spe_hint) and q_spe_hint > bin_width:
        spacing_guesses.extend([q_spe_hint, q_spe_hint / 2, q_spe_hint * 1.5])
    if len(peak_positions) >= 2:
        diffs = np.diff(np.sort(peak_positions))
        diffs = diffs[diffs > bin_width]
        if len(diffs):
            spacing_guesses.extend(list(diffs[:5]))
            spacing_guesses.append(np.nanmedian(diffs))
    hist_peak_q = fallback.get("q_spe", np.nan)
    if np.isfinite(hist_peak_q) and hist_peak_q > bin_width:
        spacing_guesses.extend([hist_peak_q / n for n in range(1, min(n_pe_max, 5) + 1)])
    spacing_guesses.extend(list(q_span / np.arange(2, n_pe_max + 5)))

    # Deduplicate and keep reasonable spacings.
    spacing_guesses = np.array(spacing_guesses, dtype=float)
    spacing_guesses = spacing_guesses[np.isfinite(spacing_guesses)]
    spacing_guesses = spacing_guesses[(spacing_guesses > 1.5 * bin_width) & (spacing_guesses < 0.8 * q_span)]
    spacing_guesses = np.unique(np.round(spacing_guesses, decimals=24))

    if len(spacing_guesses) == 0:
        fallback["fit_message"] = "could not build any reasonable PE spacing guesses"
        return fallback

    best = None
    max_count = float(np.nanmax(counts))
    bg_guess = max(0.0, np.nanpercentile(counts, 5))

    for q_spe_guess in spacing_guesses:
        # Try both a near-zero pedestal and a pedestal at the strongest low-charge peak.
        q_ped_guesses = [max(0.0, q_low), centers[np.argmax(counts)]]
        for q_ped_guess in q_ped_guesses:
            sigma_ped_guess = max(bin_width, 0.20 * q_spe_guess)
            sigma_pe_guess = max(bin_width, 0.30 * q_spe_guess)
            amp_guesses = []
            for n in range(0, n_pe_max + 1):
                center_guess = q_ped_guess + n * q_spe_guess
                nearest = np.argmin(np.abs(centers - center_guess))
                amp_guesses.append(max(counts[nearest] - bg_guess, 1.0))

            p0 = [q_ped_guess, q_spe_guess, sigma_ped_guess, sigma_pe_guess, bg_guess] + amp_guesses
            lower = [q_low - q_span, 1.2 * bin_width, bin_width / 10, bin_width / 10, 0.0] + [0.0] * (n_pe_max + 1)
            upper = [q_high, q_span, q_span, q_span, max_count] + [20 * max_count] * (n_pe_max + 1)

            try:
                popt, pcov = curve_fit(
                    lambda q, *params: pedestal_multi_pe_model(q, *params),
                    centers,
                    counts,
                    p0=p0,
                    bounds=(lower, upper),
                    sigma=yerr,
                    absolute_sigma=True,
                    maxfev=60000,
                )
            except Exception:
                continue

            y_model = pedestal_multi_pe_model(centers, *popt)
            chi2 = np.nansum(((counts - y_model) / yerr)**2)
            ndof = max(len(counts) - len(popt), 1)
            red_chi2 = chi2 / ndof

            q_spe = popt[1]
            sigma_ped, sigma_pe = popt[2], popt[3]
            if q_spe <= 0 or sigma_ped <= 0 or sigma_pe <= 0:
                continue

            if best is None or red_chi2 < best["red_chi2"]:
                best = {"popt": popt, "pcov": pcov, "red_chi2": red_chi2}

    if best is None:
        fallback["fit_message"] = "pedestal + multi-PE fit did not converge"
        return fallback

    popt = best["popt"]
    pcov = best["pcov"]
    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan)
    q_plot = np.linspace(q_low, q_high, 800)
    y_plot = pedestal_multi_pe_model(q_plot, *popt)

    q_ped, q_spe, sigma_ped, sigma_pe = popt[:4]
    q_spe_err = perr[1]

    return {
        "q_spe": q_spe,
        "q_spe_err": q_spe_err,
        "n_peak": int(len(charges_fit)),
        "hist_counts": counts,
        "hist_edges": edges,
        "peak_left": q_ped + q_spe - np.sqrt(sigma_ped**2 + sigma_pe**2),
        "peak_right": q_ped + q_spe + np.sqrt(sigma_ped**2 + sigma_pe**2),
        "fit_method": "pedestal_plus_multi_pe_shared_spacing",
        "fit_success": True,
        "fit_message": f"reduced chi2 = {best['red_chi2']:.3g}",
        "fit_q": q_plot,
        "fit_y": y_plot,
        "fit_params": popt,
        "fit_errors": perr,
        "n_pe_max": n_pe_max,
    }

def process_voltage_folder(data_folder, voltage, file_pattern="C1C*.txt"):
    all_waveforms = []
    target_files = sorted(Path(data_folder).glob(file_pattern))

    for filepath in target_files:
        file_waveforms = processing_files(filepath)
        all_waveforms.extend(file_waveforms)

    empty_result = {
        "Voltage [V]": voltage,
        "Number of files": len(target_files),
        "Number of waveforms": 0,
        "Baseline mean [V]": np.nan,
        "Baseline std [V]": np.nan,
        "Mean noise sigma [V]": np.nan,
        "Voltage threshold V_thr [V]": np.nan,
        "Charge threshold Qc [C]": np.nan,
        "Accepted pulses N_acc": 0,
        "Accepted fraction": np.nan,
        "Accepted pulse rate [Hz]": np.nan,
        "SPE charge [C]": np.nan,
        "SPE charge error [C]": np.nan,
        "SPE peak pulses": 0,
        "Mean accepted charge [C]": np.nan,
        "Charge error [C]": np.nan,
        "Total accepted charge [C]": np.nan,
        "Charge per waveform [C/wf]": np.nan,
        "Below-threshold charge mean [C]": np.nan,
        "Below-threshold charge std [C]": np.nan,
        "Mean gain": np.nan,
        "Gain error": np.nan,
        "Accepted charges": np.array([]),
        "All charges": np.array([]),
        "SPE hist counts": np.array([]),
        "SPE hist edges": np.array([]),
        "SPE peak left [C]": np.nan,
        "SPE peak right [C]": np.nan,
        "PE fit method": "none",
        "PE fit success": False,
        "PE fit message": "no waveforms loaded",
        "PE fit q [C]": np.array([]),
        "PE fit y": np.array([]),
        "PE fit params": None,
        "PE fit errors": None,
    }

    if len(all_waveforms) == 0:
        return empty_result

    waveforms_norm = normalize_waveform_times(all_waveforms)
    waveforms_bs, baselines, sigmas = subtract_baseline(
        waveforms_norm,
        t_pre_signal=T_PRE_SIGNAL,
    )

    if APPLY_FILTER:
        waveforms_proc = filter_waveforms(waveforms_bs, f_cut=F_CUTOFF)
    else:
        waveforms_proc = waveforms_bs.copy()

    H, Q, sigma_evt = extract_observables(
        waveforms_proc,
        sigmas,
        pulse_start_time=PULSE_START_TIME,
        R=R_TERMINATION,
    )

    Q_region = []
    for t, v in waveforms_proc:
        q_i, tl_i, tr_i = integrate_pulse_region(
            t,
            v,
            R=R_TERMINATION,
            pulse_start_time=PULSE_START_TIME,
        )
        Q_region.append(q_i)

    Q_region = np.array(Q_region)
    sigma_mean = np.nanmean(sigma_evt)
    V_thr = THRESHOLD_SIGMA * sigma_mean
    height_pass = np.isfinite(H) & (H > V_thr)

    accepted_charge_from_height = Q_region[height_pass & np.isfinite(Q_region)]
    Qc = np.nanpercentile(accepted_charge_from_height, QC_PERCENTILE) if len(accepted_charge_from_height) else np.nan
    charge_pass = np.isfinite(Q_region) & (Q_region >= Qc) if np.isfinite(Qc) else np.zeros(len(Q_region), dtype=bool)

    durations = waveform_duration(waveforms_proc)
    T_window = np.nanmean(durations) if len(durations) else np.nan
    T_total = len(waveforms_proc) * T_window if np.isfinite(T_window) else np.nan

    Q_acc = Q_region[charge_pass]
    Q_below = Q_region[(~height_pass) & np.isfinite(Q_region)]
    N_acc = len(Q_acc)
    Q_mean_acc = np.nanmean(Q_acc) if N_acc else np.nan
    Q_std_acc = np.nanstd(Q_acc, ddof=1) if N_acc > 1 else np.nan
    Q_err_acc = Q_std_acc / np.sqrt(N_acc) if N_acc > 1 else np.nan
    Q_total_acc = np.nansum(Q_acc) if N_acc else np.nan
    Q_per_waveform = Q_total_acc / len(waveforms_proc) if len(waveforms_proc) and np.isfinite(Q_total_acc) else np.nan
    R_acc = N_acc / T_total if np.isfinite(T_total) and T_total > 0 else np.nan
    accepted_fraction = N_acc / len(waveforms_proc) if len(waveforms_proc) else np.nan

    # Match the original plotted gain curve: estimate the SPE charge from the
    # dominant peak of the accepted-charge histogram. This deliberately avoids
    # the later multi-PE fit, whose poor fits changed the remembered curve shape.
    spe = estimate_spe_charge(Q_acc, bins=SPE_HIST_BINS)
    spe.update({
        "fit_method": "accepted_charge_histogram_peak",
        "fit_success": True,
        "fit_message": "original gain-curve histogram-peak estimator",
        "fit_q": np.array([]), "fit_y": np.array([]),
        "fit_params": None, "fit_errors": None,
    })
    Q_spe = spe["q_spe"]
    Q_spe_err = spe["q_spe_err"]

    # This is the actual PMT gain conversion. One detected photoelectron produces
    # Q_spe Coulombs at the anode, so the multiplication gain is Q_spe / e.
    gain_mean = Q_spe / E_CHARGE if np.isfinite(Q_spe) else np.nan
    gain_err = Q_spe_err / E_CHARGE if np.isfinite(Q_spe_err) else np.nan

    return {
        "Voltage [V]": voltage,
        "Number of files": len(target_files),
        "Number of waveforms": len(waveforms_proc),
        "Baseline mean [V]": np.nanmean(baselines),
        "Baseline std [V]": np.nanstd(baselines, ddof=1) if len(baselines) > 1 else np.nan,
        "Mean noise sigma [V]": sigma_mean,
        "Voltage threshold V_thr [V]": V_thr,
        "Charge threshold Qc [C]": Qc,
        "Accepted pulses N_acc": int(N_acc),
        "Accepted fraction": accepted_fraction,
        "Accepted pulse rate [Hz]": R_acc,
        "SPE charge [C]": Q_spe,
        "SPE charge error [C]": Q_spe_err,
        "SPE peak pulses": int(spe["n_peak"]),
        "Mean accepted charge [C]": Q_mean_acc,
        "Charge error [C]": Q_err_acc,
        "Total accepted charge [C]": Q_total_acc,
        "Charge per waveform [C/wf]": Q_per_waveform,
        "Below-threshold charge mean [C]": np.nanmean(Q_below) if len(Q_below) else np.nan,
        "Below-threshold charge std [C]": np.nanstd(Q_below, ddof=1) if len(Q_below) > 1 else np.nan,
        "Mean gain": gain_mean,
        "Gain error": gain_err,
        "Accepted charges": Q_acc,
        "All charges": Q_region,
        "SPE hist counts": spe["hist_counts"],
        "SPE hist edges": spe["hist_edges"],
        "SPE peak left [C]": spe["peak_left"],
        "SPE peak right [C]": spe["peak_right"],
        "PE fit method": spe.get("fit_method", "unknown"),
        "PE fit success": spe.get("fit_success", False),
        "PE fit message": spe.get("fit_message", ""),
        "PE fit q [C]": spe.get("fit_q", np.array([])),
        "PE fit y": spe.get("fit_y", np.array([])),
        "PE fit params": spe.get("fit_params", None),
        "PE fit errors": spe.get("fit_errors", None),
    }

def interp_extrap_log_gain(V_query, V_data, logG_data, fit_coeff):
    V_query = np.array(V_query, dtype=float)
    logG_query = np.interp(V_query, V_data, logG_data)

    below = V_query < np.nanmin(V_data)
    above = V_query > np.nanmax(V_data)
    outside = below | above
    if np.any(outside):
        logG_query[outside] = np.polyval(fit_coeff, V_query[outside])

    return 10 ** logG_query

def plot_charge_spectrum_with_pe_gaussians(result, title=None):
    charges = result["Accepted charges"]
    charges = charges[np.isfinite(charges) & (charges > 0)]
    if len(charges) == 0:
        print(f"No accepted positive charges for {result['Voltage [V]']} V")
        return

    plt.figure(figsize=(9, 5.5))
    counts, edges, _ = plt.hist(
        charges,
        bins=SPE_HIST_BINS,
        histtype="step",
        lw=2.0,
        color="black",
        label="accepted charge data",
    )

    if result.get("PE fit success", False) and len(result.get("PE fit q [C]", [])):
        q_fit = result["PE fit q [C]"]
        plt.plot(q_fit, result["PE fit y"], color="tab:orange", lw=2.4, label="total fit")

        params = result.get("PE fit params")
        if params is not None:
            q_ped, q_spe, sigma_ped, sigma_pe, background = params[:5]
            amps = params[5:]
            plt.axhline(background, color="0.45", ls="--", lw=1.4, label="fit background")

            for n, amp in enumerate(amps):
                sigma_n = sigma_ped if n == 0 else np.sqrt(sigma_ped**2 + n * sigma_pe**2)
                center_n = q_ped + n * q_spe
                gaussian_n = amp * np.exp(-0.5 * ((q_fit - center_n) / sigma_n)**2)
                label = "pedestal Gaussian" if n == 0 else f"{n} PE Gaussian"
                plt.plot(q_fit, gaussian_n, ls="--", lw=1.8, label=label)
                plt.axvline(center_n, color="tab:orange", ls=":", lw=1.0, alpha=0.7)
    else:
        print(f"No successful multi-PE fit for {result['Voltage [V]']} V: {result.get('PE fit message', '')}")

    plt.axvline(result["SPE charge [C]"], color="tab:red", ls="--", lw=1.7, label="fitted PE spacing")
    plt.xlabel("Pulse charge [C]")
    plt.ylabel("Counts")
    plt.title(title or f"{result['Voltage [V]']} V accepted charge spectrum")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
