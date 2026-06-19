import argparse
import json
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

def scaling_function(x: float, alpha: float, beta:float):
    return alpha * (x ** beta)

def parser_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, default="./outputs/artifacts/scaling_law_fit")
    parser.add_argument("--run_path", type=str, default="./outputs/logs/training_runs.json")

    return parser.parse_args()


def main(args: argparse.Namespace):
    with open(args.run_path,"r", encoding="utf-8") as f:
        data = json.load(f)

    optimal_data : dict = {}

    for result in data:
        compute_budget = float(f"{result["compute_budget"]:.3g}")
        if optimal_data.get(compute_budget) == None:
            optimal_data[compute_budget] = result
        elif optimal_data[compute_budget]["valid_loss"] > result["valid_loss"]:
            optimal_data[compute_budget] = result

    d_xdata = []
    d_ydata = []
    n_xdata = []
    n_ydata = []

    for key, value in optimal_data.items():
        d_xdata.append(key)
        n_xdata.append(key)

        d_ydata.append(value["data"])
        n_ydata.append(value["parameters"])

        print(f"Compute Budget: {key:.3g}, Data: {value['data']:.3g}, Parameters: {value['parameters']:.3g}")


    d_popt, _ = curve_fit(scaling_function, d_xdata, d_ydata)
    n_popt, _ = curve_fit(scaling_function, n_xdata, n_ydata)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok= True)
    save_path.touch()

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"D = {d_popt[0]} C ** {d_popt[1]}\n")
        f.write(f"N = {n_popt[0]} C ** {n_popt[1]}")

    return

if __name__ == "__main__":
    main(parser_args())