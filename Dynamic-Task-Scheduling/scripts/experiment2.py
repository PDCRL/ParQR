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
# Global Parameters for Experiment 3 (Scalability Analysis - Fig 4a, 4b)
# ------------------------------------------------------------------------------
matrix_sizes_to_test = [300, 2400, 4800, 7200, 10800]
fixed_thread_counts = [26, 52]
runs_per_config = 3
base_testcase_folder = "../testcase" # Relative to this script's location
executable_name = "../bin/QR"        # Relative to this script's location
makefile_name = "../Makefile"        # Relative to this script's location
parqr_root_dir = ".."                # Path to ParQR root from script's location

ALPHA_BETA_NO_PRIORITY = {"alpha": 12, "beta": 12}
ALPHA_BETA_WITH_PRIORITY = {"alpha": 30, "beta": 30}
ALPHA_BETA_BARRIER = {"alpha": 12, "beta": 12} # VERIFY THIS CHOICE

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def update_makefile(source_file_name_only):
    source_path_in_makefile = f"src/{source_file_name_only}"
    cmd = f"sed -i 's|^MAIN_SRC * =.*|MAIN_SRC = {source_path_in_makefile}|' {makefile_name}"
    subprocess.run(cmd, shell=True, check=True)
    print(f"[DEBUG] Updated Makefile to use {source_path_in_makefile}")

def update_cpp_macro(source_file_full_path, macro_name, macro_value):
    # Generic macro updater
    # Regex tries to match: #define MACRO_NAME       VALUE
    # It handles various spacing and existing numeric values.
    regex = f"'s/^#define[[:space:]]\\+{macro_name}[[:space:]]\\+[0-9.]\\+/#define {macro_name} {macro_value}/'"
    cmd = f"sed -i {regex} {source_file_full_path}"
    subprocess.run(cmd, shell=True, check=True)
    print(f"[DEBUG] Updated {source_file_full_path}: {macro_name}={macro_value}")

def compile_code_cli():
    print("[DEBUG] Compiling code...")
    subprocess.run("make clean", shell=True, check=True, cwd=parqr_root_dir)
    ret = subprocess.run("make -j", shell=True, cwd=parqr_root_dir) # Use make -j for faster compiles
    if ret.returncode != 0:
        print("[ERROR] Compilation failed.")
        sys.exit(1)
    print("[DEBUG] Compilation succeeded.")

def get_matrix_file_path(current_rows, current_cols):
    matrix_file_abs_path = os.path.abspath(os.path.join(base_testcase_folder, f"matrix_{current_rows}x{current_cols}.txt"))
    generate_matrix_if_needed(current_rows, current_cols, matrix_file_abs_path)
    if not os.path.exists(matrix_file_abs_path):
        print(f"[ERROR] Matrix file {matrix_file_abs_path} still does not exist after generation attempt.")
        sys.exit(1)
    # Return path relative to parqr_root_dir for the executable
    return os.path.relpath(matrix_file_abs_path, parqr_root_dir)


def run_executable_cli(current_rows, current_cols, matrix_file_path_for_exe):
    cmd_list = [os.path.relpath(executable_name, parqr_root_dir), str(current_rows), str(current_cols), matrix_file_path_for_exe]
    print(f"[DEBUG] Running command (from {parqr_root_dir}): {' '.join(cmd_list)}")
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True, cwd=parqr_root_dir)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error running executable: {e.stdout} {e.stderr}")
        return None
    match = re.search(r"(?:Execution Time|Time taken):\s*([0-9.]+)\s*ms", result.stdout)
    if match:
        return float(match.group(1))
    else:
        print("[ERROR] Time not found in executable output."); print("--- STDOUT ---"); print(result.stdout); print("--- STDERR ---"); print(result.stderr)
        return None

def run_scalability_experiment(source_file_name_only, current_matrix_size, thread_count, priority_val, alpha_val, beta_val):
    source_file_full_path = os.path.abspath(os.path.join(parqr_root_dir, "src", source_file_name_only))

    update_makefile(source_file_name_only)
    update_cpp_macro(source_file_full_path, "NUM_THREADS", thread_count)
    if priority_val is not None: # For intel.cpp
        update_cpp_macro(source_file_full_path, "USE_PRIORITY_MAIN_QUEUE", priority_val)
    update_cpp_macro(source_file_full_path, "ALPHA", alpha_val)
    update_cpp_macro(source_file_full_path, "BETA", beta_val)

    compile_code_cli()
    matrix_file_path_for_exe = get_matrix_file_path(current_matrix_size, current_matrix_size)
    exec_time = run_executable_cli(current_matrix_size, current_matrix_size, matrix_file_path_for_exe)
    return exec_time

# ------------------------------------------------------------------------------
# Main Experiment Execution
# ------------------------------------------------------------------------------
def main():
    # Ensure script is run from its own directory for consistent relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(executable_name):
        print(f"[INFO] Executable {executable_name} not found. Attempting initial compile...")
        compile_code_cli() # Compile once initially
        if not os.path.exists(executable_name):
             print(f"[ERROR] Executable {executable_name} still not found. Exiting.")
             sys.exit(1)

    all_run_data = []
    intel_source_name = "intel.cpp"
    barrier_source_name = "barrier_main.cpp"

    for threads in fixed_thread_counts:
        print(f"\n[INFO] Starting experiments for {threads} THREADS\n" + "="*50)
        for m_size in matrix_sizes_to_test:
            print(f"[INFO] --- Matrix Size: {m_size}x{m_size} ---")
            for cycle in range(1, runs_per_config + 1):
                print(f"[INFO] Cycle {cycle}/{runs_per_config}")

                # 1. Without Priority
                ab_np = ALPHA_BETA_NO_PRIORITY
                time_val = run_scalability_experiment(intel_source_name, m_size, threads, 0, ab_np["alpha"], ab_np["beta"])
                print(f"  Without Priority ({ab_np['alpha']},{ab_np['beta']}), {threads} Thr, {m_size}x{m_size} => {time_val} ms")
                if time_val is not None: all_run_data.append({"Method": "Without Priority", "MatrixSize": m_size, "Threads": threads, "Time_ms": time_val})

                # 2. With Priority
                ab_wp = ALPHA_BETA_WITH_PRIORITY
                time_val = run_scalability_experiment(intel_source_name, m_size, threads, 1, ab_wp["alpha"], ab_wp["beta"])
                print(f"  With Priority ({ab_wp['alpha']},{ab_wp['beta']}), {threads} Thr, {m_size}x{m_size} => {time_val} ms")
                if time_val is not None: all_run_data.append({"Method": "With Priority", "MatrixSize": m_size, "Threads": threads, "Time_ms": time_val})
                
                # 3. Barrier
                ab_b = ALPHA_BETA_BARRIER
                time_val = run_scalability_experiment(barrier_source_name, m_size, threads, None, ab_b["alpha"], ab_b["beta"])
                print(f"  Barrier ({ab_b['alpha']},{ab_b['beta']}), {threads} Thr, {m_size}x{m_size} => {time_val} ms")
                if time_val is not None: all_run_data.append({"Method": "Barrier", "MatrixSize": m_size, "Threads": threads, "Time_ms": time_val})

    if not all_run_data: print("[WARN] No data collected."); return
    df_all_runs = pd.DataFrame(all_run_data)
    df_averaged = df_all_runs.groupby(["Method", "MatrixSize", "Threads"], as_index=False)["Time_ms"].mean()
    df_averaged.rename(columns={"Time_ms": "AvgTime_ms"}, inplace=True)
    df_averaged["AvgTime_s"] = df_averaged["AvgTime_ms"] / 1000.0

    results_dir = "results_scalability"
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, "scalability_analysis_results.csv")
    df_averaged.to_csv(csv_filename, index=False)
    print(f"[INFO] Averaged results: {csv_filename}")

    for threads_to_plot in fixed_thread_counts:
        df_plot = df_averaged[df_averaged["Threads"] == threads_to_plot]
        plt.figure(figsize=(10, 6))
        markers = {'Barrier': '^', 'Without Priority': 'o', 'With Priority': 's'}
        for method_name, group_data in df_plot.groupby("Method"):
            plt.plot(group_data["MatrixSize"], group_data["AvgTime_s"], marker=markers.get(method_name, 'x'), label=method_name)
        
        plt.xlabel("Matrix Size")
        plt.ylabel("Execution Time (s)")
        fig_label = 'a' if threads_to_plot == 26 else 'b'
        plt.title(f"Scalability Comparison ({threads_to_plot} Threads) - Fig 4{fig_label}")
        plt.legend()
        plt.grid(True)
        plot_filename = f"fig4{fig_label}_scalability_{threads_to_plot}threads.png"
        plt.savefig(os.path.join(results_dir, plot_filename), dpi=300)
        print(f"[INFO] Generated plot: {os.path.join(results_dir, plot_filename)}")
        plt.close()

if __name__ == "__main__":
    main()