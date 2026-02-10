#!/usr/bin/env python3
"""
Live checkpoint viewer for experiments.

This script watches a pickle checkpoint (the same format used by
`credo_comparisons.py`) and visualizes the mean and sd across completed
repetitions for the following metrics: coverage, ISL, IL and pcorr.

Usage examples:
  python comparisons/checkpoint_viewer.py --chk_file ./results/checkpoints/airfoil_checkpoint_catboost.pkl
  python comparisons/checkpoint_viewer.py --interval 5

If `--chk_file` is not given the script will try to find the newest
checkpoint in ./results/checkpoints/ and watch that.
"""
import argparse
import glob
import os
import pickle
import time
from typing import Optional

import pandas as pd
import numpy as np


METHODS = ["credo_adap", "credo_fixed", "cqr", "cqrr", "uacqrs", "uacqrp", "EPIC"]

def find_latest_checkpoint(results_dir: str = "./results/checkpoints") -> Optional[str]:
	pattern = os.path.join(results_dir, "*.pkl")
	files = glob.glob(pattern)
	if not files:
		return None
	files.sort(key=os.path.getmtime, reverse=True)
	return files[0]


def find_checkpoint_for_dataset(dataset: str, uacqr_model: Optional[str], results_dir: str = "./results/checkpoints") -> Optional[str]:
	"""Find a checkpoint for a dataset.

	If uacqr_model is provided, look for exact file `{dataset}_checkpoint_{uacqr_model}.pkl`.
	Otherwise, return the most recent file matching `{dataset}_checkpoint_*.pkl`.
	"""
	if uacqr_model:
		fname = os.path.join(results_dir, f"{dataset}_checkpoint_{uacqr_model}.pkl")
		return fname if os.path.exists(fname) else None
	pattern = os.path.join(results_dir, f"{dataset}_checkpoint_*.pkl")
	files = glob.glob(pattern)
	if not files:
		return None
	files.sort(key=os.path.getmtime, reverse=True)
	return files[0]

def load_checkpoint(chk_file: str):
	# simple load: try once, then one short retry if file was being written
	try:
		with open(chk_file, "rb") as f:
			return pickle.load(f)
	except Exception:
		# one quick retry
		time.sleep(0.1)
		with open(chk_file, "rb") as f:
			return pickle.load(f)

def summarize(arr):
	arr = np.asarray(arr)
	if arr.size == 0:
		return None, None
	# ensure 2D: (n_reps, n_methods)
	if arr.ndim == 1:
		arr = arr.reshape((-1, arr.shape[0]))
	mean = np.nanmean(arr, axis=0)
	sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
	return mean, sd

def print_stats_table(name: str, arr, methods=METHODS):
	mean, sd = summarize(arr)
	print(f"\n--- {name} (mean ± sd) ---")
	if mean is None:
		print("no data")
		return
	df = pd.DataFrame({"method": methods, "mean": mean, "sd": sd})
	# format numbers
	with pd.option_context("display.float_format", "{:.4f}".format):
		print(df.to_string(index=False))

def main():
	parser = argparse.ArgumentParser(description="Live checkpoint viewer for credo_comparisons experiments")
	parser.add_argument("--chk_file", type=str, default=None, help="path to checkpoint .pkl file")
	parser.add_argument("--dataset", type=str, default=None, help="dataset name (will look for <dataset>_checkpoint_*.pkl in results_dir)")
	parser.add_argument("--uacqr_model", type=str, default='catboost', help="uacqr_model name (e.g. 'catboost' or 'rfqr') to select exact checkpoint file")
	parser.add_argument("--results_dir", type=str, default="./results/checkpoints", help="directory to search for checkpoints if --chk_file omitted")
	parser.add_argument("--interval", type=float, default=3.0, help="polling interval in seconds (used with --watch)")
	parser.add_argument("--watch", action="store_true", help="continuously watch the checkpoint and print updates when it changes")
	args = parser.parse_args()

	# Determine checkpoint file to watch
	if args.chk_file:
		chk_file = args.chk_file
	elif args.dataset:
		chk_file = find_checkpoint_for_dataset(args.dataset, args.uacqr_model, args.results_dir)
		if chk_file is None:
			if args.uacqr_model:
				print(f"No checkpoint file found for dataset '{args.dataset}' and model '{args.uacqr_model}' in {args.results_dir}. Exiting.")
			else:
				print(f"No checkpoint files found for dataset '{args.dataset}' in {args.results_dir}. Exiting.")
			return
		else:
			print(f"Found checkpoint for dataset '{args.dataset}': {chk_file}")
	else:
		chk_file = find_latest_checkpoint(args.results_dir)
		if chk_file is None:
			print(f"No checkpoint found in {args.results_dir}. Exiting.")
			return
		print(f"Watching latest checkpoint: {chk_file}")

	try:
		# Load once and print tables (default behavior). If --watch provided,
		# continuously check modification time and re-print when changed.
		def do_print():
			data = load_checkpoint(chk_file)
			# checkpoint keys used in credo_comparisons.py: cover_results, isl_results, IL_results, pcorr_results
			cover = data.get("cover_results", [])
			isl = data.get("isl_results", [])
			IL = data.get("IL_results", [])
			pcorr = data.get("pcorr_results", [])

			# show meta info
			iteration = data.get("iteration")
			seeds = data.get("seeds")
			alpha_val = data.get("alpha")
			gamma_val = data.get("gamma")
			dataset_meta = data.get("dataset")
			n_reps = len(cover) if hasattr(cover, "__len__") else 0
			print(f"\nLoaded checkpoint: {chk_file}")
			print(f"dataset: {dataset_meta}  alpha: {alpha_val}  gamma: {gamma_val}  iteration: {iteration}  repetitions: {n_reps}")

			print_stats_table("Coverage", cover)
			print_stats_table("ISL", isl)
			print_stats_table("Interval Length", IL)
			print_stats_table("pcorr", pcorr)

		if args.watch:
			last_mtime = None
			while True:
				try:
					mtime = os.path.getmtime(chk_file)
				except FileNotFoundError:
					print(f"Checkpoint {chk_file} not found. Waiting...")
					time.sleep(args.interval)
					continue
				if last_mtime is None or mtime != last_mtime:
					try:
						do_print()
					except Exception as e:
						print(f"Failed to load/print checkpoint: {e}")
					last_mtime = mtime
				time.sleep(args.interval)
		else:
			do_print()

	except KeyboardInterrupt:
		print("Interrupted by user — exiting.")

if __name__ == "__main__":
	main()

