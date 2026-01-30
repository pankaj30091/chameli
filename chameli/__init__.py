import inspect
import logging
import os
import sys
from importlib.resources import files
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, List

__version__ = "0.1.10"

from .config import is_config_loaded, load_config

# Ensure the module's directory is in the system path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    A TimedRotatingFileHandler that ensures the file is reopened after rotation.
    
    This fixes an issue where log rotation can cause the file handle to become stale,
    preventing new log entries from being written after rotation occurs.
    """
    
    def doRollover(self):
        """Override to ensure stream is reopened after rotation."""
        super().doRollover()
        # Ensure stream is open after rotation (parent may close it)
        if self.stream is None or self.stream.closed:
            self.stream = self._open()
    
    def emit(self, record):
        """Override to ensure stream is valid before writing."""
        try:
            if self.shouldRollover(record):
                self.doRollover()
            
            # Ensure stream is open before writing
            if self.stream is None or self.stream.closed:
                self.stream = self._open()
            
            super().emit(record)
        except Exception:
            self.handleError(record)


class StructuredFormatter(logging.Formatter):
    """Custom formatter that includes extra fields in the log output."""

    def format(self, record):
        # Create a copy of the record to avoid modifying the original
        record_copy = logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=record.pathname,
            lineno=record.lineno,
            msg=record.msg,
            args=record.args,
            exc_info=record.exc_info,
            func=record.funcName,
        )

        # Copy all attributes from the original record
        for key, value in record.__dict__.items():
            if key not in record_copy.__dict__:
                setattr(record_copy, key, value)

        # Add extra fields to the message if they exist
        extra_parts = []

        # Check for extra fields in the record (excluding standard logging fields)
        standard_fields = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "getMessage",
            "exc_info",
            "exc_text",
            "stack_info",
            "asctime",
            "message",
        }

        for key, value in record_copy.__dict__.items():
            if key not in standard_fields and not key.startswith("_"):
                extra_parts.append(f"{key}={value}")

        if extra_parts:
            record_copy.msg = f"{record_copy.msg} | {' | '.join(extra_parts)}"

        return super().format(record_copy)


def get_default_config_path():
    """Returns the path to the default config file included in the package."""
    try:
        return files("chameli").joinpath("config/config_sample.yaml")
    except FileNotFoundError:
        # Fallback to relative path in the source directory
        return os.path.join(os.path.dirname(__file__), "config/config_sample.yaml")


class ChameliLogger:
    """Centralized logger for the Chameli package with structured logging capabilities."""

    def __init__(self, name: str = "chameli"):
        self.logger = logging.getLogger(name)
        self._configured = False

    def configure(
        self,
        level: int = logging.WARNING,
        log_file: Optional[str] = None,
        clear_existing_handlers: bool = False,
        enable_console: bool = True,
        backup_count: int = 7,
        format_string: str = "%(asctime)s:%(name)s:%(filename)s:%(lineno)s - %(funcName)20s() ] %(levelname)s: %(message)s",
        enable_structured_logging: bool = True,
    ):
        """
        Configure logging for the Chameli package with enhanced features.

        Args:
            level: Logging level (e.g., logging.DEBUG, logging.INFO).
            log_file: Path to the log file. If None, logs will go to the console.
            clear_existing_handlers: Whether to clear existing handlers from the root logger.
            enable_console: Whether console logging is enabled.
            backup_count: Number of log files to keep.
            format_string: Custom format string for log messages.
            enable_structured_logging: Enable structured logging with additional context.
        """
        if self._configured and not clear_existing_handlers:
            return

        if clear_existing_handlers:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)

        # Create handlers
        handlers = []
        # Use StructuredFormatter for better extra field handling
        formatter = StructuredFormatter(format_string)

        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = SafeTimedRotatingFileHandler(log_file, when="midnight", backupCount=backup_count)
            file_handler.suffix = "%Y%m%d"
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)

        # Configure the logger
        self.logger.setLevel(level)
        for handler in handlers:
            self.logger.addHandler(handler)

        self._configured = True

        # Log configuration
        self.logger.info(
            "Chameli logging configured",
            extra={
                "log_file": log_file,
                "level": logging.getLevelName(level),
                "console_enabled": enable_console,
                "structured_logging": enable_structured_logging,
            },
        )

    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger instance with the specified name."""
        if name:
            return logging.getLogger(f"chameli.{name}")
        return self.logger

    def _get_caller_info(self):
        """Get information about the calling function."""
        try:
            # Get the caller frame (skip this method and the logging method)
            caller_frame = inspect.currentframe().f_back.f_back
            if caller_frame:
                info = inspect.getframeinfo(caller_frame)
                return {
                    "caller_filename": info.filename,
                    "caller_lineno": info.lineno,
                    "caller_function": info.function,
                }
        except Exception as e:
            pass
        return {}

    def log_error(self, message: str, error: Exception = None, context: dict = None, exc_info: bool = True):
        """Log an error with structured context."""
        extra = context or {}
        if error:
            extra["error_type"] = type(error).__name__
            extra["error_message"] = str(error)

        # Add caller information
        caller_info = self._get_caller_info()
        extra.update(caller_info)

        self.logger.error(message, extra=extra, exc_info=exc_info)

    def log_warning(self, message: str, context: dict = None):
        """Log a warning with structured context."""
        extra = context or {}

        # Add caller information
        caller_info = self._get_caller_info()
        extra.update(caller_info)

        self.logger.warning(message, extra=extra)

    def log_info(self, message: str, context: dict = None):
        """Log an info message with structured context."""
        extra = context or {}

        # Add caller information
        caller_info = self._get_caller_info()
        extra.update(caller_info)

        self.logger.info(message, extra=extra)

    def log_debug(self, message: str, context: dict = None):
        """Log a debug message with structured context."""
        extra = context or {}

        # Add caller information
        caller_info = self._get_caller_info()
        extra.update(caller_info)

        self.logger.debug(message, extra=extra)


# Global logger instance
chameli_logger = ChameliLogger()


def configure_logging(
    module_names: Optional[List[str]] = None,
    level: int = logging.WARNING,
    log_file: Optional[str] = None,
    clear_existing_handlers: bool = False,
    enable_console: bool = True,
    backup_count: int = 7,
    format_string: str = "%(asctime)s:%(name)s:%(filename)s:%(lineno)s - %(funcName)20s() ] %(levelname)s: %(message)s",
    enable_structured_logging: bool = True,
    configure_root_logger: bool = False,
):
    """
    Configure logging for specific modules or all modules in the Chameli package.

    Args:
        module_names: List of module names to enable logging for. If None, configure logging for all modules.
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Path to the log file. If None, logs will go to the console.
        clear_existing_handlers: Whether to clear existing handlers from the root logger.
        enable_console: Whether console logging is enabled.
        backup_count: Number of log files to keep.
        format_string: Custom format string for log messages.
        enable_structured_logging: Enable structured logging with additional context.
        configure_root_logger: Whether to configure the root logger (can cause duplicate logs if True).
    """
    chameli_logger.configure(
        level=level,
        log_file=log_file,
        clear_existing_handlers=clear_existing_handlers,
        enable_console=enable_console,
        backup_count=backup_count,
        format_string=format_string,
        enable_structured_logging=enable_structured_logging,
    )

    # Only configure root logger if explicitly requested to avoid duplicate logs
    if configure_root_logger:
        # Configure root logger to capture all errors
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear existing handlers if requested
        if clear_existing_handlers:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

        # Create handlers for root logger
        handlers = []
        # Use StructuredFormatter for better extra field handling
        formatter = StructuredFormatter(format_string)

        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = SafeTimedRotatingFileHandler(log_file, when="midnight", backupCount=backup_count)
            file_handler.suffix = "%Y%m%d"
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)

        # Add handlers to root logger
        for handler in handlers:
            root_logger.addHandler(handler)

    # Configure specific modules if requested
    if module_names:
        for module_name in module_names:
            logger = logging.getLogger(module_name)
            logger.setLevel(level)
            # Don't add handlers to avoid duplication


# Set default logging level to WARNING and log to console by default
# Don't configure root logger by default to avoid duplicate logs
configure_logging(configure_root_logger=False)


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
    try:
        if is_config_loaded() and not force_reload:
            raise RuntimeError("Configuration is already loaded.")
        else:
            load_config(config_file_path)
            chameli_logger.log_info(
                "Configuration initialized successfully",
                {"config_file": config_file_path, "force_reload": force_reload},
            )
    except Exception as e:
        chameli_logger.log_error(
            "Failed to initialize configuration", e, {"config_file": config_file_path, "force_reload": force_reload}
        )
        raise


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
    "chameli_logger",
]
