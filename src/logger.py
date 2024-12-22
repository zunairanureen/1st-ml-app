import logging
import os
from datetime import datetime

# Generate the log file name using the current date and time
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Create the "logs" directory if it doesn't exist
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Full path to the log file (inside the "logs" directory)
LOGS_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Print the log file path to verify
print(f"Log file will be created at: {LOGS_FILE_PATH}")

# Configure logging
logging.basicConfig(
    filename=LOGS_FILE_PATH,
    format='[%(asctime)s]  %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,  # Change level to DEBUG for more verbose logging
)

# Ensure logging handlers are set
if not logging.getLogger().hasHandlers():
    print("No handlers are configured for logging!")
else:
    print("Logging handlers are properly set.")





