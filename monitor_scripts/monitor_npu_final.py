import os
import sys
import subprocess
import time
import argparse
import logging
from datetime import datetime

class MonitorConfig:
    """Default settings for the NPU monitoring utility."""
    #XRT_SMI_PATH = "xrt-smi.exe"
    XRT_SMI_PATH = "C:\\Users\\aiene\\Downloads\\RAI_1.3.1_242_WHQL\\npu_mcdm_stack_prod\\xrt-smi.exe"
    
    DEFAULT_INTERVAL_S = 5
    DEFAULT_LOG_FILE = "npu_monitor.log"
    DEFAULT_DEVICE_BDF = ""  # Empty string targets all detected devices.

    # Reports to run. Corresponds to `xrtsmi examine --report <name>`.
    REPORTS = {
        "all": True,
    }

class NpuMonitor:
    """A utility to monitor AMD Ryzen AI NPU status using xrtsmi."""

    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.xrtsmi_path = self._validate_xrtsmi_path()
        self.base_command = self._build_base_command()
        self._setup_logging()

    def _setup_logging(self):
        """Initializes console and file logging."""
        self.logger = logging.getLogger("NpuMonitor")
        self.logger.setLevel(logging.DEBUG)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Console handler for live view
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)

        # File handler for historical data
        file_handler = logging.FileHandler(self.args.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(file_handler)

    def _validate_xrtsmi_path(self):
        """Checks for the existence and executability of xrtsmi."""
        path_to_check = self.config.XRT_SMI_PATH
        try:
            # Use '--version' as a lightweight check.
            subprocess.run([path_to_check, "--version"], capture_output=True, check=True)
            return path_to_check
        except (FileNotFoundError, subprocess.CalledProcessError):
            print(f"FATAL: Could not execute '{path_to_check}'.")
            print("Please ensure XRT is installed and 'xrt-smi' is in your system's PATH,")
            print("or set the correct full path for 'XRT_SMI_PATH' in the script.")
            sys.exit(1)

    def _build_base_command(self):
        """Constructs the base command for xrtsmi."""
        command = [self.xrtsmi_path, "examine"]
        if self.args.device:
            command.extend(["-d", self.args.device])
        return command

    def _clear_screen(self):
        """Clears the terminal screen for a clean refresh."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _run_report(self, report_name):
        """Executes a specific xrtsmi report."""
        header = f"--- [REPORT: {report_name.upper()}] ---"
        self.logger.info(f"\n{header}")
        self.logger.debug(f"Executing report: {report_name}")

        command = self.base_command + ["--report", report_name]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if result.stdout:
                self.logger.info(result.stdout.strip())
            if result.stderr:
                self.logger.warning(f"--- STDERR for {report_name} ---\n{result.stderr.strip()}")
        except Exception as e:
            self.logger.error(f"Error running report '{report_name}': {e}")
        finally:
            self.logger.info("-" * len(header))

    def run(self):
        """Starts the main monitoring loop."""
        self._clear_screen()
        self.logger.info("--- NPU Monitoring Utility ---")
        self.logger.info(f"Target Device: {self.args.device or 'All'}")
        self.logger.info(f"Interval:      {self.args.interval}s")
        self.logger.info(f"Log File:      {self.args.log_file}")
        self.logger.info("Starting monitor... Press Ctrl+C to stop.")
        
        try:
            time.sleep(2)
            while True:
                self._clear_screen()
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.logger.info(f"--- Status Update @ {timestamp} ---")

                for report, enabled in self.config.REPORTS.items():
                    if enabled:
                        self._run_report(report)
                
                self.logger.info(f"\nNext update in {self.args.interval} seconds...")
                time.sleep(self.args.interval)
        except KeyboardInterrupt:
            self.logger.info("\nMonitoring stopped by user.")
        except Exception as e:
            self.logger.critical(f"\nA critical error occurred: {e}", exc_info=True)
        finally:
            logging.shutdown()
            sys.exit(0)

def main():
    """Parses arguments and starts the monitor."""
    parser = argparse.ArgumentParser(
        description="Monitor AMD/Xilinx Ryzen AI NPU status.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i', '--interval', type=int, default=MonitorConfig.DEFAULT_INTERVAL_S,
        help=f"Refresh interval in seconds. Default: {MonitorConfig.DEFAULT_INTERVAL_S}s."
    )
    parser.add_argument(
        '-l', '--log-file', type=str, default=MonitorConfig.DEFAULT_LOG_FILE,
        help=f"Path to the log file. Default: {MonitorConfig.DEFAULT_LOG_FILE}."
    )
    parser.add_argument(
        '-d', '--device', type=str, default=MonitorConfig.DEFAULT_DEVICE_BDF,
        help="Target device BDF (Bus:Device:Function). Monitors all if unspecified."
    )
    args = parser.parse_args()
    
    monitor = NpuMonitor(MonitorConfig, args)
    monitor.run()

if __name__ == "__main__":
    main()