import os
import sys
import subprocess
import logging
import argparse # NUOVO: Aggiunto per gestire gli argomenti della riga di comando
from datetime import datetime

# --- Configuration ---
CONFIG = {
    "log_file": "pipeline.log",
    "model_dir": "models",
    "quantized_model_dir": "quantized_models",
    "base_model_name": "resnet50",
    "calib_image_source_dir": "imagenet",  
    "calib_data_dir": "calib_data_imagenet",
    "quant_mode": "int8",
    "scripts": {
        "export": "export_fp32_resnet50.py",
        "prepare_data": "prepare_calibration_data.py",
        "quantize": "model_quantization.py",
    }
}

def update_paths():
    """Aggiorna i percorsi dei modelli basati sulla configurazione."""
    CONFIG["fp32_model_path"] = os.path.join(CONFIG["model_dir"], f"{CONFIG['base_model_name']}_fp32.onnx")
    CONFIG["quantized_model_path"] = os.path.join(CONFIG["quantized_model_dir"], f"{CONFIG['base_model_name']}_quant_{CONFIG['quant_mode']}_new.onnx")

def setup_logging():
    """Configura il logging su file e console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(CONFIG["log_file"]),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging initialized.")

def run_command(command, step_name):
    """Esegue un comando shell, ne registra l'output e gestisce gli errori."""
    logging.info(f"--- Starting Step: {step_name} ---")
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command, check=True, capture_output=False, text=True
        )
        if result.stdout:
            logging.info(f"Output from {step_name}:\n{result.stdout.strip()}")
        if result.stderr:
            logging.warning(f"Standard Error from {step_name}:\n{result.stderr.strip()}")
        logging.info(f"--- Step Succeeded: {step_name} ---")
        return True
    except FileNotFoundError:
        logging.error(f"Error: The script '{command[1]}' was not found.")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"--- Step FAILED: {step_name} ---")
        logging.error(f"Return code: {e.returncode}")
        if e.stdout: logging.error(f"STDOUT:\n{e.stdout.strip()}")
        if e.stderr: logging.error(f"STDERR:\n{e.stderr.strip()}")
        raise

def export_fp32_model():
    """Esporta il modello ONNX FP32 da PyTorch."""
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    command = ["python", CONFIG["scripts"]["export"]]
    run_command(command, "Export FP32 Model")

def prepare_calibration_data():
    """Pre-elabora le immagini JPEG in file .npy per la calibrazione."""
    if not os.path.isdir(CONFIG["calib_image_source_dir"]):
        logging.error(f"Prerequisite Error: Calibration image source directory not found at '{CONFIG['calib_image_source_dir']}'")
        logging.error("Please create this directory and populate it with JPEG images.")
        raise FileNotFoundError(f"Directory not found: {CONFIG['calib_image_source_dir']}")
    
    command = [
        "python", 
        CONFIG["scripts"]["prepare_data"],
        "--input_dir", CONFIG["calib_image_source_dir"],
        "--output_dir", CONFIG["calib_data_dir"]
    ]
    run_command(command, "Prepare Calibration Data")

def quantize_model():
    """Esegue la quantizzazione del modello ONNX."""
    os.makedirs(CONFIG["quantized_model_dir"], exist_ok=True)
    command = [
        "python",
        CONFIG["scripts"]["quantize"],
        "--model_input", CONFIG["fp32_model_path"],
        "--model_output", CONFIG["quantized_model_path"],
        "--calib_data", CONFIG["calib_data_dir"],
        "--quantize", CONFIG["quant_mode"],
    ]
    run_command(command, "Quantize Model")

def main():
    """Funzione principale di esecuzione del pipeline."""
    # NUOVO: Parsing degli argomenti della riga di comando
    parser = argparse.ArgumentParser(description="Run the ONNX quantization pipeline.")
    parser.add_argument(
        '--image_dir', 
        type=str, 
        default=CONFIG['calib_image_source_dir'],
        help='Path to the directory containing calibration images (e.g., small_imagenet).'
    )
    args = parser.parse_args()

    # NUOVO: Aggiorna la configurazione con l'argomento fornito
    CONFIG["calib_image_source_dir"] = args.image_dir
    
    # NUOVO: Aggiorna i percorsi derivati dopo aver finalizzato la configurazione
    update_paths()

    setup_logging()
    start_time = datetime.now()
    logging.info("="*50)
    logging.info(f"Starting Quantization Pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Using calibration images from: {CONFIG['calib_image_source_dir']}")
    logging.info("="*50)

    try:
        export_fp32_model()
        prepare_calibration_data()
        quantize_model()

        end_time = datetime.now()
        logging.info("="*50)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY")
        logging.info(f"Total execution time: {end_time - start_time}")
        logging.info("="*50)
    except Exception as e:
        end_time = datetime.now()
        logging.critical("="*50)
        logging.critical(f"PIPELINE FAILED")
        logging.critical(f"Total execution time: {end_time - start_time}")
        logging.critical("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()