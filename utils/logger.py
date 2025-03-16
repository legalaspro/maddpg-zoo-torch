import sys
import os
from datetime import datetime
import numpy as np
import time
from collections import deque
import pandas as pd


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]


class Logger:
    """
    Logging class to monitor training, supporting TensorBoard and optional CSV logging.
    
    Args:
        run_name (str): Unique identifier for the run. Defaults to current timestamp.
        folder (str): Root directory for storing logs. Defaults to 'runs'.
        algo (str): Algorithm name, used in directory structure. Defaults to 'sac'.
        env (str): Environment name, used in directory structure. Defaults to 'Env'.
        save_csv (bool): Whether to log metrics to a CSV file. Defaults to False.
    """
    
    def __init__(
         self,
        run_name=datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        folder="runs",
        algo="sac",
        env="Env",
        save_csv=False,
    ):
        self.run_name = run_name
        self.dir_name = os.path.join(folder, env, algo, run_name)
        os.makedirs(self.dir_name, exist_ok=True)

        self.writer = SummaryWriter(self.dir_name)
        self.name_to_values = {}  # Stores deque of recent values for smoothing
        self.current_env_step = 0
        self.start_time = time.time()
        self.last_csv_save = time.time()
        self.save_csv = save_csv
        self.save_every = 10 * 60  # Save CSV every 10 seconds

        if self.save_csv:
            self._data = {}  # {step: {key: val, ...}, ...} for CSV logging

    def log_all_hyperparameters(self, hyperparams: dict):
        """Log hyperparameters to TensorBoard and print them to stdout."""
        self.add_hyperparams(hyperparams)
        self.log_hyperparameters(hyperparams)

    def add_hyperparams(self, hyperparams: dict):
        """Log hyperparameters to TensorBoard."""
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in hyperparams.items()])),
        )
    
    def log_hyperparameters(self, hyperparams: dict):
        """Pretty print hyperparameters in a table format."""
        hyper_param_space, value_space = 30, 40
        format_str = "| {:<" + f"{hyper_param_space}" + "} | {:<" + f"{value_space}" + "}|"
        hbar = "-" * (hyper_param_space + value_space + 6)
    
        print(hbar)
        print(format_str.format("Hyperparams", "Values"))
        print(hbar)
    
        for key, value in hyperparams.items():
            print(format_str.format(str(key), str(value)))
    
        print(hbar)

    def add_run_command(self):
        """Log the terminal command used to start the run."""
        cmd = " ".join(sys.argv)
        self.writer.add_text("terminal", cmd)
        with open(os.path.join(self.dir_name, "cmd.txt"), "w") as f:
            f.write(cmd)
    
    def add_scalar(self, key: str, val: float, step: int, smoothing: bool = True):
        """
        Log a scalar value to TensorBoard and optionally to CSV.
        
        Args:
            key (str): Metric name (e.g., 'loss')
            val (float): Value to log
            step (int): Current training step
            smoothing (bool): Whether to smooth values for stdout logging. Defaults to True.
        """
        # Log to TensorBoard
        self.writer.add_scalar(key, val, step)

        # Update smoothing deque
        if key not in self.name_to_values:
            self.name_to_values[key] = deque(maxlen=5 if smoothing else 1)
        self.name_to_values[key].append(val)
        self.current_env_step = max(self.current_env_step, step)

        # Log to CSV if enabled
        if self.save_csv:
            if step not in self._data:
                self._data[step] = {}
            self._data[step][key] = val  # Store raw value
            
            # Periodically save CSV
            if time.time() - self.last_csv_save > self.save_every:
                self.save2csv()
                self.last_csv_save = time.time()

    def save2csv(self, file_name: str = None):
        """Save logged data to a CSV file."""
        if not self.save_csv or not self._data:
            return
        
        if file_name is None:
            file_name = os.path.join(self.dir_name, "progress.csv")
        
        # Convert to DataFrame
        steps = sorted(self._data.keys())
        rows = []
        for step in steps:
            row = {'global_step': step}
            row.update(self._data[step])
            rows.append(row)
        df = pd.DataFrame(rows)
        
        # Ensure 'global_step' is first column
        cols = ['global_step'] + [c for c in df.columns if c != 'global_step']
        df = df[cols]
        
        df.to_csv(file_name, index=False)
    
    def close(self):
        """Close the TensorBoard writer and save CSV."""
        self.writer.close()
        if self.save_csv:
            self.save2csv()
    
    def log_stdout(self):
        """Print smoothed metrics to stdout."""
        results = {k: np.mean(v) for k, v in self.name_to_values.items()}
        results['step'] = self.current_env_step
        # results['fps'] = self.fps()
        pprint(results)
    
    def fps(self) -> int:
        """Calculate frames per second (steps per second)."""
        elapsed = time.time() - self.start_time
        return int(self.current_env_step / elapsed) if elapsed > 0 else 0


def pprint(dict_data):
    """Pretty print metrics in a table format."""
    key_space, val_space = 40, 40
    border = "-" * (key_space + val_space + 5)
    row_fmt = f"| {{:<{key_space}}} | {{:<{val_space}}}|"
    
    print(f"\n{border}")
    for k, v in dict_data.items():
        k_str = truncate_str(str(k), key_space)
        v_str = truncate_str(str(v), val_space)
        print(row_fmt.format(k_str, v_str))
    print(f"{border}\n")


def truncate_str(s: str, max_len: int) -> str:
    """Truncate string with ellipsis if exceeds max length."""
    return s if len(s) <= max_len else s[:max_len-3] + "..."