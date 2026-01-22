import logging
import os
import sys
from datetime import datetime


def setup_logging(log_dir: str = "logs", log_filename: str = "app_rag.log"):
    """
    Configures the logging system to write to a file AND to the terminal.

    Args:
        log_dir (str): Directory where logs will be saved.
        log_filename (str): Name of the log file.
    """
    # 1. Create logs folder if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Full file path (e.g., logs/app_rag.log)
    # Adds date to the name to create a new file per execution:
    file_path = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}_{log_filename}")
    # file_path = os.path.join(log_dir, log_filename)

    # 2. Define message format
    # %(asctime)s - Date/Time
    # %(name)s    - Module name (e.g., src.components.llm)
    # %(levelname)s - Level (INFO, ERROR, DEBUG)
    # %(message)s - The message itself
    log_format = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 3. Create Handlers

    # File Handler
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Terminal Handler (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Change to DEBUG to see more details in terminal
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # 4. Apply global configuration (Root Logger)
    # force=True ensures we are overwriting previous configs
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True
    )

    # Create a local logger just to notify success
    logger = logging.getLogger(__name__)
    logger.info(f"✅ Logging system initialized.")
    logger.info(f"📂 Logs being saved at: {os.path.abspath(file_path)}")
