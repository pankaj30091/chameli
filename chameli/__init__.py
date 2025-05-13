import logging
import os
import sys
from importlib.resources import files
from logging.handlers import TimedRotatingFileHandler

__version__ = "0.1.2"

from .config import is_config_loaded, load_config

# Ensure the module's directory is in the system path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def get_default_config_path():
    """Returns the path to the default config file included in the package."""
    try:
        return files("chameli").joinpath("config/config_sample.yaml")
    except FileNotFoundError:
        # Fallback to relative path in the source directory
        return os.path.join(os.path.dirname(__file__), "config/config_sample.yaml")


def configure_logging(
    module_names=None,
    level=logging.WARNING,
    log_file=None,
    clear_existing_handlers=False,
    enable_console=True,
    backup_count=7,
):
    """
    Configure logging for specific or all modules in the pyTrade package.

    Args:
        module_names (list of str): List of module names to enable logging for.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file (str): Path to the log file. If None, logs will go to console.
        clear_existing_handlers (bool): Whether to clear existing handlers.
        enable_console (bool): Whether console logging is enabled.
        backup_count (int): Number of log files to keep.
    """
    if clear_existing_handlers:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    # Create handlers
    handlers = []
    formatter = logging.Formatter("%(asctime)s:%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    if log_file:
        file_handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=backup_count)
        file_handler.suffix = "%Y%m%d"
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # Configure logging for specific modules or globally
    if module_names:
        for module_name in module_names:
            logger = logging.getLogger(module_name)
            logger.setLevel(level)
            for handler in handlers:
                logger.addHandler(handler)
    else:
        # Configure root logger if no specific modules are mentioned
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in handlers:
            root_logger.addHandler(handler)


# Set default logging level to WARNING and log to console by default
configure_logging()


def initialize_config(config_file_path: str, force_reload=True):
    """
    Initializes the configuration from the specified file.

    Args:
        config_file_path (str): Path to the configuration file.
        force_reload (bool): Whether to force reload the configuration.

    Raises:
        RuntimeError: If the configuration is already loaded
        and force_reload is False.
    """
    if is_config_loaded() and not force_reload:
        raise RuntimeError("Configuration is already loaded.")
    else:
        load_config(config_file_path)


# Initialize configuration with the default config path
initialize_config(get_default_config_path())


class LazyModule:
    """
    Lazily loads a module when an attribute is accessed.

    Args:
        module_name (str): The name of the module to load.
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def _load_module(self):
        allowed_modules = [
            "chameli.dateutils",
            "chameli.europeanoptions",
            "chameli.interactions",
        ]
        if self.module_name not in allowed_modules:
            raise ImportError(f"Module {self.module_name} is not allowed.")
        if not self.module:
            if not is_config_loaded():
                raise RuntimeError("Configuration not loaded. Call `initialize_config` first.")
            self.module = __import__(self.module_name, fromlist=[""])

    def __getattr__(self, name):
        self._load_module()
        return getattr(self.module, name)


# Lazy loading for modules
dateutils = LazyModule("chameli.dateutils")
europeanoptions = LazyModule("chameli.europeanoptions")
interactions = LazyModule("chameli.interactions")

__all__ = [
    "dateutils",
    "europeanoptions",
    "interactions",
    "initialize_config",
    "configure_logging",
]
