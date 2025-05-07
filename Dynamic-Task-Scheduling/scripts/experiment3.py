#!/usr/bin/env python3
import re
import subprocess
import sys
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# ------------------------------------------------------------------------------
# Matrix Generation Helper
# ------------------------------------------------------------------------------
def generate_matrix_if_needed(rows, cols, filepath, force_regenerate=False):
    if not os.path.exists(filepath) or force_regenerate:
        print(f"[INFO] Matrix file {filepath} not found or regeneration forced. Generating new matrix {rows}x{cols}...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        seed_value = rows * 100000 + cols 
        np.random.seed(seed_value)
        matrix_data = np.random.rand(rows, cols) * 20 - 10
        try:
            with open(filepath, 'w') as f:
                for i in range(rows):
                    f.write(' '.join(map(lambda x: f"{x:.6f}", matrix_data[i])) + '\n')
            print(f"[INFO] Successfully generated and saved matrix to {filepath}")
        except IOError as e:
            print(f"[ERROR] Could not write matrix file {filepath}: {e}")
            sys.exit(1)
    else:
        print(f"[DEBUG] Matrix file {filepath} found. Using existing file.")

# ------------------------------------------------------------------------------
# Global Parameters for Experiment 4 (Throughput Evaluation - Fig 5)
# ------------------------------------------------------------------------------
fixed_matrix_size = 8192
runs_per_config = 3
base_testcase_folder = "../testcase" # Relative to this script's location
executable_name = "../a.out"        # Relative to this script's location
makefile_name = "../Makefile"        # Relative to this script's location
parqr_root_dir = ".."                # Path to ParQR root from script's location

# Thread configurations for data collection (dense)
thread_configs_to_run = [2 * i for i in range(1, 53)] # 2, 4, ..., 104
# Thread configurations for plotting Figure 5 style
thread_configs_for_fig5_plot = [4, 24, 44, 64, 84, 100]

# Alpha/Beta for this experiment (as per your original Exp4 script)
# For Figure 5, the paper says "optimal alpha/beta from Exp 4.2".
# If these (32,32) and (16,16) are NOT those optimal ones, you should adjust them
# or add runs for the actual optimal ones (e.g., 12,12 for no-prio, 30,30 for prio).
# For now, using the (32,32) and (16,16) as in your provided script.
ALPHA_BETA_CONFIGS = {
    "intel_32_np": {"alpha": 32, "beta": 32, "prio": 0, "label": "Intel 32,32 (no prio)", "method_label": "Without Priority (32,32)"},
    "intel_32_wp": {"alpha": 32, "beta": 32, "prio": 1, "label": "Intel 32,32 (with prio)", "method_label": "With Priority (32,32)"},
    "intel_16_np": {"alpha": 16, "beta": 16, "prio": 0, "label": "Intel 16,16 (no prio)", "method_label": "Without Priority (16,16)"},
    "intel_16_wp": {"alpha": 16, "beta": 16, "prio": 1, "label": "Intel 16,16 (with prio)", "method_label": "With Priority (16,16)"},
    "barrier_32":  {"alpha": 32, "beta": 32, "prio": None, "label": "Barrier 32,32", "method_label": "Barrier (32,32)"},
    "barrier_16":  {"alpha": 16, "beta": 16, "prio": None, "label": "Barrier 16,16", "method_label": "Barrier (16,16)"}
}
# Which configurations to use for the main Figure 5 plot
# Paper implies optimal alpha/beta. If (32,32) is not optimal, change these keys.
FIG5_PLOT_KEYS = {
    "Without Priority": "intel_32_np", # Example: Use 32,32 no-prio for "Without Priority" line
    "With Priority": "intel_32_wp",    # Example: Use 32,32 with-prio for "With Priority" line
    "Barrier": "barrier_32"            # Example: Use 32,32 barrier for "Barrier" line
}


# ------------------------------------------------------------------------------
# Helper Functions (Identical to Experiment 3 script)
# ------------------------------------------------------------------------------
def update_makefile(source_file_name_only):
    source_path_in_makefile = f"{source_file_name_only}"
    cmd = f"sed -i 's|^MAIN_SRC * =.*|MAIN_SRC = {source_path_in_makefile}|' {makefile_name}"
    subprocess.run(cmd, shell=True, check=True)
    print(f"[DEBUG] Updated Makefile to use {source_path_in_makefile}")

def update_cpp_macro(source_file_full_path, macro_name, macro_value):
    regex = f"'s/^#define[[:space:]]\\+{macro_name}[[:space:]]\\+[0-9.]\\+/#define {macro_name} {macro_value}/'"
    cmd = f"sed -i {regex} {source_file_full_path}"
    subprocess.run(cmd, shell=True, check=True)
    print(f"[DEBUG] Updated {source_file_full_path}: {macro_name}={macro_value}")

def compile_code_cli():
    print("[DEBUG] Compiling code...")
    subprocess.run("make clean", shell=True, check=True, cwd=parqr_root_dir)
    ret = subprocess.run("make -j", shell=True, cwd=parqr_root_dir)
    if ret.returncode != 0: print("[ERROR] Compilation failed."); sys.exit(1)
    print("[DEBUG] Compilation succeeded.")

def get_matrix_file_path(current_rows, current_cols):
    matrix_file_abs_path = os.path.abspath(os.path.join(base_testcase_folder, f"matrix_{current_rows}x{current_cols}.txt"))
    generate_matrix_if_needed(current_rows, current_cols, matrix_file_abs_path)
    if not os.path.exists(matrix_file_abs_path): print(f"[ERROR] Matrix file {matrix_file_abs_path} still not found."); sys.exit(1)
    return os.path.relpath(matrix_file_abs_path, parqr_root_dir)

def run_executable_cli(current_rows, current_cols, matrix_file_path_for_exe):
    cmd_list = [os.path.relpath(executable_name, parqr_root_dir), matrix_file_path_for_exe]
    print(f"[DEBUG] Running command (from {parqr_root_dir}): {' '.join(cmd_list)}")
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True, cwd=parqr_root_dir)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error running executable: {e.stdout} {e.stderr}"); return None
    match = re.search(r"(?:Execution Time|Time taken):\s*([0-9.]+)\s*ms", result.stdout)
    if match: return float(match.group(1))
    else: print("[ERROR] Time not found in output."); print("--- STDOUT ---"); print(result.stdout); print("--- STDERR ---"); print(result.stderr); return None

def run_throughput_experiment(source_file_name_only, thread_count, priority_val, alpha_val, beta_val):
    source_file_full_path = os.path.abspath(os.path.join(parqr_root_dir, source_file_name_only))
    update_makefile(source_file_name_only)
    update_cpp_macro(source_file_full_path, "NUM_THREADS", thread_count)
    if priority_val is not None: # For intel.cpp
        update_cpp_macro(source_file_full_path, "USE_PRIORITY_MAIN_QUEUE", priority_val)
    update_cpp_macro(source_file_full_path, "ALPHA", alpha_val)
    update_cpp_macro(source_file_full_path, "BETA", beta_val)
    compile_code_cli()
    matrix_file_path_for_exe = get_matrix_file_path(fixed_matrix_size, fixed_matrix_size)
    exec_time = run_executable_cli(fixed_matrix_size, fixed_matrix_size, matrix_file_path_for_exe)
    return exec_time
# ------------------------------------------------------------------------------
# Main Experiment Execution
# ------------------------------------------------------------------------------
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(executable_name):
        print(f"[INFO] Executable {executable_name} not found. Attempting initial compile...")
        compile_code_cli()
        if not os.path.exists(executable_name): print(f"[ERROR] Executable {executable_name} still not found."); sys.exit(1)

    all_run_data = []
    intel_source_name = "intel.cpp"
    barrier_source_name = "barrier_main.cpp"

    for threads in thread_configs_to_run:
        print(f"\n[INFO] Starting experiments for {threads} THREADS\n" + "="*50)
        for config_key, params in ALPHA_BETA_CONFIGS.items():
            current_source_name = intel_source_name if "intel" in config_key else barrier_source_name
            
            avg_times_for_this_config_threads = []
            for cycle in range(1, runs_per_config + 1):
                print(f"[INFO] Cycle {cycle}/{runs_per_config} | Config: {params['label']}, Threads: {threads}")
                time_val = run_throughput_experiment(current_source_name, threads, params["prio"], params["alpha"], params["beta"])
                print(f"  => {time_val} ms")
                if time_val is not None: avg_times_for_this_config_threads.append(time_val)
            
            if avg_times_for_this_config_threads:
                all_run_data.append({
                    "MethodLabel": params["method_label"], # For easier grouping in plots
                    "ConfigKey": config_key, # Original detailed key
                    "Threads": threads,
                    "AvgTime_ms": np.mean(avg_times_for_this_config_threads)
                })
        print(f"[INFO] Completed all configurations for Threads = {threads}")

    if not all_run_data: print("[WARN] No data collected."); return
    df_results = pd.DataFrame(all_run_data)
    df_results["AvgTime_s"] = df_results["AvgTime_ms"] / 1000.0

    results_dir = "results_throughput"
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, "throughput_analysis_results.csv")
    df_results.to_csv(csv_filename, index=False)
    print(f"[INFO] Averaged results: {csv_filename}")

    # --- Plotting ---
    # Diagnostic plots (all collected data)
    plt.figure(figsize=(12, 7))
    for method_label, group_data in df_results.groupby("MethodLabel"):
        plt.plot(group_data["Threads"], group_data["AvgTime_s"], marker='o', linestyle='-', label=method_label)
    plt.xlabel("Thread Count")
    plt.ylabel("Average Execution Time (s)")
    plt.title(f"Throughput Comparison (All Configs, Matrix: {fixed_matrix_size}x{fixed_matrix_size})")
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0)) # Adjust legend position
    plt.grid(True, which="both", ls="-")
    plt.yscale('log')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(os.path.join(results_dir, "diagnostic_throughput_all_configs.png"), dpi=300)
    plt.close()
    print(f"[INFO] Generated diagnostic plot: {os.path.join(results_dir, 'diagnostic_throughput_all_configs.png')}")

    # Figure 5 style plot
    plt.figure(figsize=(10, 6))
    markers = {'Barrier': '^', 'Without Priority': 'o', 'With Priority': 's'}
    linestyles = {'Barrier': ':', 'Without Priority': '-', 'With Priority': '--'}

    for paper_label, config_key_to_use in FIG5_PLOT_KEYS.items():
        # Find the corresponding method_label from ALPHA_BETA_CONFIGS
        method_label_for_plot = ALPHA_BETA_CONFIGS[config_key_to_use]["method_label"]
        
        subset = df_results[(df_results["MethodLabel"] == method_label_for_plot) &
                            (df_results["Threads"].isin(thread_configs_for_fig5_plot))]
        if not subset.empty:
            plt.plot(subset["Threads"], subset["AvgTime_s"],
                     marker=markers.get(paper_label, 'x'),
                     linestyle=linestyles.get(paper_label, '-'),
                     label=paper_label)
        else:
            print(f"[WARN] No data for Fig5 plot: {paper_label} (using {method_label_for_plot})")
            
    plt.xlabel("Thread Count")
    plt.ylabel("Execution Time (s)")
    plt.title(f"Throughput Evaluation (Matrix: {fixed_matrix_size}x{fixed_matrix_size}) - Fig. 5 Style")
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.yscale('log')
    plt.xticks(thread_configs_for_fig5_plot)
    plt.xlim(min(thread_configs_for_fig5_plot)-2, max(thread_configs_for_fig5_plot)+2)
    plt.savefig(os.path.join(results_dir, "fig5_generated_throughput.png"), dpi=300)
    print(f"[INFO] Generated Fig. 5 style plot: {os.path.join(results_dir, 'fig5_generated_throughput.png')}")
    plt.close()

if __name__ == "__main__":
    main()
