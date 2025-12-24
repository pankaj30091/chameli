import logging
import os
import smtplib
import zipfile
import pandas as pd
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from os.path import basename
import traceback
import psutil
import pyreadr
import requests
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import paramiko
from scp import SCPClient
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
import threading
import socket
import base64
import time
from urllib.parse import urlparse, quote
from mitmproxy import options
from mitmproxy.tools.dump import DumpMaster
from mitmproxy import http

from .config import get_config


# Define COMMASPACE explicitly
COMMASPACE = ", "


def get_dynamic_config():
    return get_config()


# Import chameli_logger lazily to avoid circular import
def get_chameli_logger():
    """Get chameli_logger instance to avoid circular imports."""
    from . import chameli_logger

    return chameli_logger


# Global variable to store the connection
remote_connection = None
is_remote = False  # Flag to indicate if the system is remote or local

# Global variable to track active proxy server
_active_proxy_server = None


def initialize_connection(remote_server=None):
    """
    Initialize the connection to the remote server or set to local.

    Args:
        remote_config (dict, optional): Configuration for the remote server. Defaults to None.
    """
    global remote_connection, is_remote

    if remote_server and remote_server.get("hostname"):
        # Remote server details
        hostname = remote_server.get("hostname")
        username = remote_server.get("username")
        password = remote_server.get("password")
        port = remote_server.get("port", 22)  # Default SSH port is 22

        try:
            # Establish the SSH connection
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname, port=port, username=username, password=password)
            ssh.get_transport().set_keepalive(30)
            remote_connection = ssh
            is_remote = True
            print(f"Connected to remote server: {hostname}")
        except Exception as e:
            raise Exception(f"Failed to connect to remote server: {e}")
    else:
        # Local system
        remote_connection = None
        is_remote = False
        print("Using local system.")


def get_os():
    """
    Get the operating system type of the current system (local or remote).

    Returns:
        str: The OS type ("Linux" or "NT").
    """
    if is_remote:
        try:
            stdin, stdout, stderr = remote_connection.exec_command("uname")
            os_type = stdout.read().decode().strip()
            return "Linux" if os_type == "Linux" else "NT"
        except Exception as e:
            get_chameli_logger().log_error(
                "Failed to get remote OS type", e, {"remote_connection": "active" if is_remote else "inactive"}
            )
            raise
    else:
        return "Linux" if os.name == "posix" else "NT"


remote_server = get_dynamic_config().get("remote_server", {})
initialize_connection(remote_server)
system_os = get_os()


def cleanup_connection():
    """
    Clean up the connection to the remote server, if any.
    """
    global remote_connection, is_remote

    if remote_connection:
        remote_connection.close()
        remote_connection = None
        is_remote = False
        print("Connection to remote server closed.")


def ensure_connection():
    """
    Ensure the SSH connection is active. Reconnect if the connection is broken.
    """
    global remote_connection, is_remote

    if is_remote:
        try:
            # Check if the connection is still active
            transport = remote_connection.get_transport()
            if transport is None or not transport.is_active():
                get_chameli_logger().log_warning(
                    "SSH connection is inactive. Reconnecting...", {"connection_status": "inactive"}
                )
                remote_server = get_dynamic_config().get("remote_server", {})
                initialize_connection(remote_server)
                get_chameli_logger().log_info("SSH connection reestablished.", {"connection_status": "reconnected"})
        except Exception as e:
            get_chameli_logger().log_error("Failed to ensure SSH connection", e, {"connection_status": "failed"})
            raise


def normalize_path(path):
    """
    Normalize file paths based on the operating system.

    Args:
        path (str): The file path to normalize.

    Returns:
        str: The normalized file path.
    """
    if is_remote:
        if system_os == "Linux":
            return str(PurePosixPath(path.replace("\\", "/")))
        elif system_os == "NT":
            return str(PureWindowsPath(path.replace("/", "\\")))
        else:
            return path
    else:
        # Local system
        if system_os == "Linux":  # Linux or macOS
            return str(PurePosixPath(path.replace("\\", "/")))
        elif system_os == "NT":  # Windows
            return str(PureWindowsPath(path.replace("/", "\\")))
        return path


def readRDS(filename, parent_request=""):
    """
    Read an RDS file from the local or remote server.

    Args:
        filename (str): Path to the RDS file.

    Returns:
        pandas.DataFrame: The data read from the RDS file.
    """
    filename = normalize_path(filename)
    if is_remote:
        try:
            # Create a temporary file in a cross-platform way
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                local_temp_file = temp_file.name

            # Download the file from the remote server to the temporary location
            with SCPClient(remote_connection.get_transport()) as scp:
                scp.get(filename, local_temp_file)

            # Log file details
            get_chameli_logger().log_debug(
                f"Downloaded RDS file to: {local_temp_file}", {"file_path": local_temp_file, "operation": "download"}
            )
            if os.path.exists(local_temp_file):
                get_chameli_logger().log_debug(
                    f"File size: {os.path.getsize(local_temp_file)} bytes",
                    {"file_path": local_temp_file, "file_size": os.path.getsize(local_temp_file)},
                )

            # Read the RDS file locally
            data = pyreadr.read_r(local_temp_file)
            if data:
                return data[None]
        except Exception as e:
            get_chameli_logger().log_error(
                "Failed to read RDS file from remote server",
                e,
                {"file_path": local_temp_file, "operation": "read_remote"},
            )
            if os.path.exists(local_temp_file):
                get_chameli_logger().log_error(
                    f"File size: {os.path.getsize(local_temp_file)} bytes",
                    e,
                    {"file_path": local_temp_file, "file_size": os.path.getsize(local_temp_file)},
                )
            raise
        finally:
            # Clean up the temporary file
            if os.path.exists(local_temp_file):
                os.remove(local_temp_file)
    else:
        try:
            # Log file details
            get_chameli_logger().log_debug(
                f"Reading RDS file locally: {filename}", {"file_path": filename, "operation": "read_local"}
            )
            if os.path.exists(filename):
                get_chameli_logger().log_debug(
                    f"File size: {os.path.getsize(filename)} bytes",
                    {"file_path": filename, "file_size": os.path.getsize(filename)},
                )

            # Read the RDS file locally
            data = pyreadr.read_r(filename)
            if data:
                return data[None]
        except Exception as e:
            get_chameli_logger().log_error(
                "Failed to read RDS file locally",
                e,
                {"file_path": filename, "operation": "read_local", parent_request: parent_request},
            )
            if os.path.exists(filename):
                get_chameli_logger().log_error(
                    f"File size: {os.path.getsize(filename)} bytes",
                    e,
                    {"file_path": filename, "file_size": os.path.getsize(filename)},
                )
            raise


def saveRDS(pd_file, path):
    """
    Save a pandas DataFrame to an RDS file locally or on a remote server.

    Args:
        pd_file (pandas.DataFrame): The DataFrame to save.
        path (str): Path to save the RDS file.
    """
    global remote_connection, is_remote

    # Normalize the path
    path = normalize_path(path)

    if is_remote:
        try:
            # Create a temporary file locally
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                local_temp_file = temp_file.name

            # Save the RDS file locally first
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_from_pd_df = ro.conversion.py2rpy(pd_file)
            ro.r["saveRDS"](r_from_pd_df, local_temp_file, version=2)

            # Upload the file to the remote server
            with SCPClient(remote_connection.get_transport()) as scp:
                scp.put(local_temp_file, path)  # Upload the file to the remote server

            get_chameli_logger().log_info(
                f"RDS file successfully saved to remote server: {path}", {"file_path": path, "operation": "save_remote"}
            )
        except Exception as e:
            get_chameli_logger().log_error(
                "Failed to save RDS file to remote server", e, {"file_path": path, "operation": "save_remote"}
            )
            raise
        finally:
            # Clean up the temporary file
            if os.path.exists(local_temp_file):
                os.remove(local_temp_file)
    else:
        try:
            # Save the RDS file locally
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_from_pd_df = ro.conversion.py2rpy(pd_file)
            ro.r["saveRDS"](r_from_pd_df, path, version=2)
            get_chameli_logger().log_info(
                f"RDS file successfully saved locally: {path}", {"file_path": path, "operation": "save_local"}
            )
        except Exception as e:
            get_chameli_logger().log_error(
                "Failed to save RDS file locally", e, {"file_path": path, "operation": "save_local"}
            )
            raise


def save_file(dest_file, content):
    """
    Save a file to a remote server or locally based on the configuration.

    Args:
        dest_file (str): The destination file path.
        content (bytes): The binary content to save.
    """
    # Normalize the destination file path
    dest_file = normalize_path(dest_file)

    if is_remote:
        try:
            # Save the file locally first as a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                local_temp_file = temp_file.name
                with open(local_temp_file, "wb") as file:
                    file.write(content)

            # Upload the file to the remote server
            with SCPClient(remote_connection.get_transport()) as scp:
                scp.put(local_temp_file, dest_file)  # Upload the file to the remote server

            get_chameli_logger().log_info(
                f"File successfully saved to remote server: {dest_file}",
                {"file_path": "{dest_file}", "operation": "file_operation"},
            )
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to save file to remote server: {e}", None, {"file_path": "{e}", "operation": "file_operation"}
            )
            raise
        finally:
            # Clean up the temporary file
            if os.path.exists(local_temp_file):
                os.remove(local_temp_file)
    else:
        try:
            # Save the file locally
            with open(dest_file, "wb") as file:
                file.write(content)
            get_chameli_logger().log_info(
                f"File successfully saved locally: {dest_file}",
                {"file_path": "{dest_file}", "operation": "file_operation"},
            )
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to save file locally: {e}", None, {"file_path": "{e}", "operation": "file_operation"}
            )
            raise


def file_exists_and_valid(file_path, min_size=1000):
    """
    Check if a file exists and meets the minimum size requirement.

    Args:
        file_path (str): The path to the file (local or remote).
        min_size (int): The minimum file size in bytes. Defaults to 1000.

    Returns:
        bool: True if the file exists and meets the size requirement, False otherwise.
    """
    # Normalize the file path
    file_path = normalize_path(file_path)

    if is_remote:
        try:
            # Check file existence and size on the remote server
            sftp = remote_connection.open_sftp()
            try:
                file_attr = sftp.stat(file_path)
                if file_attr.st_size >= min_size:
                    return True
                else:
                    get_chameli_logger().log_info(
                        f"File exists on remote server but does not meet size requirement: {file_path}",
                        {"file_path": "{file_path}", "operation": "file_operation"},
                    )
                    return False
            except FileNotFoundError:
                get_chameli_logger().log_info(
                    f"File not found on remote server: {file_path}",
                    {"file_path": "{file_path}", "operation": "file_operation"},
                )
                return False
            finally:
                sftp.close()
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to check file on remote server: {e}", None, {"file_path": "{e}", "operation": "file_operation"}
            )
            raise
    else:
        # Check file existence and size locally
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) >= min_size:
                return True
            else:
                get_chameli_logger().log_info(
                    f"File exists locally but does not meet size requirement: {file_path}",
                    {"file_path": "{file_path}", "operation": "file_operation"},
                )
                return False
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to check file locally: {e}", None, {"file_path": "{e}", "operation": "file_operation"}
            )
            raise


def read_csv_in_pandas_out(file_path, **kwargs):
    """
    Read a CSV file into a pandas DataFrame, whether the file is local or remote.

    Args:
        file_path (str): The path to the CSV file (local or remote).
        **kwargs: Additional arguments to pass to pandas.read_csv.

    Returns:
        pandas.DataFrame: The data read from the CSV file.
    """
    file_path = normalize_path(file_path)

    if is_remote:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                local_temp_file = temp_file.name

            with SCPClient(remote_connection.get_transport()) as scp:
                scp.get(file_path, local_temp_file)

            df = pd.read_csv(local_temp_file, **kwargs)
            get_chameli_logger().log_info(
                f"CSV file successfully read from remote server: {file_path}",
                {"file_path": "{file_path}", "operation": "file_operation"},
            )
            return df
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to read CSV file from remote server: {e}",
                None,
                {"file_path": "{e}", "operation": "file_operation"},
            )
            raise
        finally:
            if os.path.exists(local_temp_file):
                os.remove(local_temp_file)
    else:
        try:
            df = pd.read_csv(file_path, **kwargs)
            get_chameli_logger().log_info(
                f"CSV file successfully read locally: {file_path}",
                {"file_path": "{file_path}", "operation": "file_operation"},
            )
            return df
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to read CSV file locally: {e}", None, {"file_path": "{e}", "operation": "file_operation"}
            )
            raise


def save_pandas_in_csv_out(df, dest_file, **kwargs):
    """
    Save a pandas DataFrame to a CSV file, either locally or on a remote server.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        dest_file (str): The destination file path (local or remote).
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_csv.
    """
    dest_file = normalize_path(dest_file)

    if is_remote:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                local_temp_file = temp_file.name
                df.to_csv(local_temp_file, **kwargs)

            with SCPClient(remote_connection.get_transport()) as scp:
                scp.put(local_temp_file, dest_file)

            get_chameli_logger().log_info(
                f"CSV file successfully saved to remote server: {dest_file}",
                {"file_path": "{dest_file}", "operation": "file_operation"},
            )
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to save CSV file to remote server: {e}",
                None,
                {"file_path": "{e}", "operation": "file_operation"},
            )
            raise
        finally:
            if os.path.exists(local_temp_file):
                os.remove(local_temp_file)
    else:
        try:
            df.to_csv(dest_file, **kwargs)
            get_chameli_logger().log_info(
                f"CSV file successfully saved locally: {dest_file}",
                {"file_path": "{dest_file}", "operation": "file_operation"},
            )
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to save CSV file locally: {e}", None, {"file_path": "{e}", "operation": "file_operation"}
            )
            raise


def read_csv_from_zip(dest_file, min_size=1000, file_index=0, **kwargs):
    """
    Check if a ZIP file exists (locally or remotely), extract its contents, and read a CSV file.

    Args:
        dest_file (str): The path to the ZIP file (local or remote).
        min_size (int): The minimum file size in bytes. Defaults to 1000.
        file_index (int): The index of the CSV file in the ZIP archive. Defaults to 0.
        **kwargs: Additional arguments to pass to pandas.read_csv.

    Returns:
        pandas.DataFrame: The data read from the CSV file inside the ZIP file.
    """
    dest_file = normalize_path(dest_file)

    if is_remote:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                local_temp_file = temp_file.name

            with SCPClient(remote_connection.get_transport()) as scp:
                scp.get(dest_file, local_temp_file)

            with zipfile.ZipFile(local_temp_file, "r") as z:
                file_name = z.namelist()[file_index]
                with z.open(file_name) as f:
                    df = pd.read_csv(f, **kwargs)
            return df
        finally:
            if os.path.exists(local_temp_file):
                os.remove(local_temp_file)
    else:
        with zipfile.ZipFile(dest_file, "r") as z:
            file_name = z.namelist()[file_index]
            with z.open(file_name) as f:
                return pd.read_csv(f, **kwargs)


def read_file(source_file):
    """
    Read a file from a remote server or locally based on the configuration.

    Args:
        source_file (str): The source file path.

    Returns:
        bytes: The content of the file as bytes.
    """
    source_file = normalize_path(source_file)

    if is_remote:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                local_temp_file = temp_file.name

            with SCPClient(remote_connection.get_transport()) as scp:
                scp.get(source_file, local_temp_file)

            with open(local_temp_file, "rb") as file:
                content = file.read()
            return content
        finally:
            if os.path.exists(local_temp_file):
                os.remove(local_temp_file)
    else:
        with open(source_file, "rb") as file:
            return file.read()


def directory_exists(directory_path):
    """
    Check if a directory exists locally or on a remote server.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    directory_path = normalize_path(directory_path)

    if is_remote:
        if system_os == "Linux":  # For POSIX systems
            command = f"test -d {directory_path} && echo exists || echo not_exists"
        elif system_os == "NT":  # For Windows systems
            command = f"if exist {directory_path} (echo exists) else (echo not_exists)"
        else:
            raise Exception("Unsupported remote OS")
        # Execute the command
        stdin, stdout, stderr = remote_connection.exec_command(command)
        result = stdout.read().decode().strip()
        return result == "exists"
    else:
        # Check locally
        return os.path.isdir(directory_path)


def make_directory(directory_path):
    """
    Create a directory locally or on a remote server.

    Args:
        directory_path (str): The path to the directory to create.
    """
    directory_path = normalize_path(directory_path)

    if directory_exists(directory_path):
        return

    if is_remote:
        try:
            command = (
                f"mkdir -p {directory_path}"
                if system_os == "Linux"
                else f"powershell -Command \"New-Item -ItemType Directory -Force -Path '{directory_path}'\""
            )
            stdin, stdout, stderr = remote_connection.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                error_message = stderr.read().decode().strip()
                raise Exception(f"Failed to create directory on remote server: {error_message}")
            get_chameli_logger().log_info(
                f"Directory successfully created on remote server: {directory_path}",
                {"file_path": "{directory_path}", "operation": "file_operation"},
            )
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to create directory on remote server: {e}",
                None,
                {"file_path": "{e}", "operation": "file_operation"},
            )
            raise
    else:
        try:
            os.makedirs(directory_path, exist_ok=True)
            get_chameli_logger().log_info(
                f"Directory successfully created locally: {directory_path}",
                {"file_path": "{directory_path}", "operation": "file_operation"},
            )
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to create directory locally: {e}", None, {"file_path": "{e}", "operation": "file_operation"}
            )
            raise


def list_directory(directory_path):
    """
    List the contents of a directory, either locally or on a remote server.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        list: A list of file and directory names in the specified directory.
    """
    directory_path = normalize_path(directory_path)

    if is_remote:
        try:
            sftp = remote_connection.open_sftp()
            contents = sftp.listdir(directory_path)
            sftp.close()
            get_chameli_logger().log_info(
                f"Directory contents successfully listed on remote server: {directory_path}",
                {"file_path": "{directory_path}", "operation": "file_operation"},
            )
            return contents
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to list directory on remote server: {e}",
                None,
                {"file_path": "{e}", "operation": "file_operation"},
            )
            raise
    else:
        try:
            contents = os.listdir(directory_path)
            get_chameli_logger().log_info(
                f"Directory contents successfully listed locally: {directory_path}",
                {"file_path": "{directory_path}", "operation": "file_operation"},
            )
            return contents
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to list directory locally: {e}", None, {"file_path": "{e}", "operation": "file_operation"}
            )
            raise


def send_mail(send_from, send_to, password, subject, text, files=None, is_html=False):
    if not isinstance(send_to, list):
        raise TypeError(f"Expected 'send_to' to be a list, got {type(send_to).__name__}")

    msg = MIMEMultipart()
    msg["From"] = send_from
    msg["To"] = COMMASPACE.join(send_to)
    msg["Date"] = formatdate(localtime=True)
    msg["Subject"] = subject

    if is_html:
        msg.attach(MIMEText(text, "html"))  # Set the content type to HTML
    else:
        # Set the content type to plain text
        msg.attach(MIMEText(text, "plain"))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(fil.read(), Name=basename(f))
        part["Content-Disposition"] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(send_from, password)
    server.sendmail(send_from, send_to, msg.as_string())
    server.close()


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.70 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}


class MitmProxyServer:
    """Manages a local mitmproxy server that forwards to upstream proxy with authentication"""

    def __init__(self, upstream_proxy, proxy_user, proxy_pass, local_port=8080):
        self.upstream_proxy = upstream_proxy
        self.proxy_user = proxy_user
        self.proxy_pass = proxy_pass
        self.local_port = local_port
        self.master = None
        self.server_thread = None
        self.running = False
        self.needs_new_upstream = False  # Flag to indicate upstream proxy needs to be changed

    def _find_available_port(self, start_port=8080, max_attempts=10):
        """Find an available port starting from start_port"""
        import socket
        for offset in range(max_attempts):
            port = start_port + offset
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            try:
                result = sock.connect_ex(("127.0.0.1", port))
                sock.close()
                if result != 0:  # Port is not in use
                    return port
            except Exception:
                sock.close()
                pass
        return None  # No available port found
    
    def start(self):
        """Start the mitmproxy server in a background thread"""
        if self.running:
            return

        try:
            # Try to find an available port if default port is busy
            actual_port = self.local_port
            if actual_port == 8080:  # Only check for default port
                import socket
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_sock.settimeout(0.1)
                if test_sock.connect_ex(("127.0.0.1", 8080)) == 0:
                    # Port 8080 is in use, find alternative
                    test_sock.close()
                    alternative_port = self._find_available_port(8080, 10)
                    if alternative_port:
                        get_chameli_logger().log_warning(
                            f"Port 8080 is in use, using alternative port {alternative_port}",
                            {"original_port": 8080, "alternative_port": alternative_port, "function": "MitmProxyServer.start"}
                        )
                        actual_port = alternative_port
                        self.local_port = alternative_port  # Update instance port
                    else:
                        test_sock.close()
                        raise Exception("No available ports found (tried 8080-8089)")
                else:
                    test_sock.close()
            
            # Configure mitmproxy options
            opts = options.Options(listen_port=actual_port, listen_host="127.0.0.1")
            
            # Set upstream proxy (without auth in URL - handle auth via addon)
            upstream_host, upstream_port = self.upstream_proxy.split(':')
            upstream_url = f"http://{upstream_host}:{upstream_port}"
            opts.mode = [f"upstream:{upstream_url}"]
            
            # Set upstream proxy authentication via options (if supported)
            # Note: mitmproxy doesn't directly support upstream_auth in options,
            # so we handle it via Proxy-Authorization header in the addon
            
            # Create a custom addon to handle upstream proxy and logging
            upstream_proxy = self.upstream_proxy
            proxy_user = self.proxy_user
            proxy_pass = self.proxy_pass
            
            class ProxyAddon:
                def __init__(self, upstream_proxy, proxy_user, proxy_pass):
                    self.upstream_proxy = upstream_proxy
                    self.proxy_user = proxy_user
                    self.proxy_pass = proxy_pass
                    # Pre-compute auth header
                    auth_string = f"{self.proxy_user}:{self.proxy_pass}"
                    self.auth_header = base64.b64encode(auth_string.encode()).decode()
                
                def http_connect_upstream(self, flow: http.HTTPFlow) -> None:
                    """Handle upstream CONNECT requests - add Proxy-Authorization header"""
                    # This hook is called when mitmproxy sends a CONNECT request to the upstream proxy
                    auth_string = f"{self.proxy_user}:{self.proxy_pass}"
                    auth_header = base64.b64encode(auth_string.encode()).decode()
                    flow.request.headers["Proxy-Authorization"] = f"Basic {auth_header}"
                    get_chameli_logger().log_info(
                        f"Proxy CONNECT upstream: {flow.request.pretty_url} (auth header set)",
                        {"method": flow.request.method, "url": flow.request.pretty_url, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.ProxyAddon.http_connect_upstream"}
                    )

                def request(self, flow: http.HTTPFlow) -> None:
                    # Add Proxy-Authorization header for regular HTTP requests going through upstream proxy
                    # (CONNECT requests are handled in http_connect_upstream)
                    if flow.request.method != "CONNECT":
                        flow.request.headers["Proxy-Authorization"] = f"Basic {self.auth_header}"
                    
                    # Log CONNECT requests from client
                    if flow.request.method == "CONNECT":
                        get_chameli_logger().log_info(
                            f"Proxy CONNECT from client: {flow.request.pretty_url}",
                            {"method": flow.request.method, "url": flow.request.pretty_url, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.ProxyAddon.request"}
                        )
                
                def response(self, flow: http.HTTPFlow) -> None:
                    # Only log errors (non-2xx status codes)
                    if flow.response.status_code >= 400:
                        get_chameli_logger().log_warning(
                            f"Proxy error response: {flow.response.status_code} for {flow.request.pretty_url}",
                            {"status_code": flow.response.status_code, "method": flow.request.method, "url": flow.request.pretty_url, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.ProxyAddon.response"}
                        )
                
                def error(self, flow: http.HTTPFlow) -> None:
                    """Handle connection-level errors (like CONNECT failures)"""
                    if flow.error:
                        error_msg = flow.error.msg if hasattr(flow.error, 'msg') else str(flow.error)
                        # Log connection errors as warnings
                        get_chameli_logger().log_warning(
                            f"Proxy connection error: {error_msg}",
                            {
                                "error_msg": error_msg,
                                "url": flow.request.pretty_url if flow.request else "N/A",
                                "upstream_proxy": self.upstream_proxy,
                                "function": "MitmProxyServer.ProxyAddon.error"
                            }
                        )
            
            # Start server in background thread with its own event loop
            # Capture actual_port in closure
            captured_port = actual_port
            def run_proxy():
                import asyncio
                try:
                    get_chameli_logger().log_info(
                        f"Starting mitmproxy in thread",
                        {"local_port": captured_port, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.run_proxy"}
                    )
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Create master - needs to be in async context for get_running_loop()
                    async def create_master():
                        get_chameli_logger().log_info(
                            f"Creating DumpMaster",
                            {"local_port": captured_port, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.create_master"}
                        )
                        self.master = DumpMaster(opts)
                        self.master.addons.add(ProxyAddon(upstream_proxy, proxy_user, proxy_pass))
                        get_chameli_logger().log_info(
                            f"DumpMaster created",
                            {"local_port": captured_port, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.create_master"}
                        )
                        return self.master
                    
                    # Create master within event loop context
                    get_chameli_logger().log_info(
                        f"Starting event loop for mitmproxy",
                        {"local_port": captured_port, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.run_proxy"}
                    )
                    try:
                        # Create master in async context
                        master = loop.run_until_complete(create_master())
                        # master.run() is synchronous but needs event loop running
                        # We need to run it in a way that keeps the event loop alive
                        get_chameli_logger().log_info(
                            f"Calling master.run()",
                            {"local_port": captured_port, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.run_proxy"}
                        )
                        # Run the master - this is blocking and should start the server
                        # The event loop is already set for this thread
                        loop.run_until_complete(master.run())
                    except Exception as e:
                        get_chameli_logger().log_error(
                            f"Error in mitmproxy: {str(e)}",
                            e,
                            {"local_port": captured_port, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.run_proxy"}
                        )
                        raise
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a port binding error
                    if "address already in use" in error_str.lower() or "errno 98" in error_str.lower():
                        get_chameli_logger().log_error(
                            f"Port binding error - port {captured_port} is already in use. Try cleaning up existing proxy servers.",
                            e,
                            {"local_port": captured_port, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.run_proxy"}
                        )
                    else:
                        get_chameli_logger().log_error(
                            f"mitmproxy server error: {str(e)}",
                            e,
                            {"local_port": captured_port, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.run_proxy"}
                        )
                    raise
            
            # Start the thread with the event loop
            self.server_thread = threading.Thread(target=run_proxy, daemon=True)
            self.server_thread.start()
            
            get_chameli_logger().log_info(
                f"mitmproxy server thread started",
                {"local_port": self.local_port, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.start"}
            )
            
            # Verify server is listening
            # Give mitmproxy more time to start (it may take a moment)
            import time
            max_attempts = 20  # Increased attempts
            for attempt in range(max_attempts):
                time.sleep(0.3)  # Increased sleep time
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_sock.settimeout(1.0)  # Increased timeout
                try:
                    result = test_sock.connect_ex(("127.0.0.1", actual_port))
                    test_sock.close()
                    if result == 0:
                        self.running = True
                        get_chameli_logger().log_info(
                            f"mitmproxy server started and verified on port {actual_port}",
                            {"local_port": actual_port, "upstream_proxy": self.upstream_proxy, "attempt": attempt+1, "function": "MitmProxyServer.start"}
                        )
                        return
                except Exception:
                    test_sock.close()
                    pass
            
            # Check if thread is still alive (server might be starting)
            if self.server_thread.is_alive():
                get_chameli_logger().log_warning(
                    f"mitmproxy server thread is alive but port {actual_port} not yet accepting connections, marking as running anyway",
                    {"local_port": actual_port, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.start"}
                )
                self.running = True
                return
            
            raise Exception(f"mitmproxy server started but port {actual_port} is not accepting connections after {max_attempts} attempts")
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to start mitmproxy server",
                e,
                {"local_port": self.local_port, "upstream_proxy": self.upstream_proxy, "function": "MitmProxyServer.start"}
            )
            raise

    def stop(self):
        """Stop the mitmproxy server"""
        if self.master and self.running:
            try:
                self.master.shutdown()
            except:
                pass
            self.running = False
            get_chameli_logger().log_info(
                f"mitmproxy server stopped",
                {"local_port": self.local_port, "function": "MitmProxyServer.stop"}
            )

    def get_local_proxy_url(self):
        """Get the local proxy URL for Firefox configuration"""
        return f"127.0.0.1:{self.local_port}"


# Keep LocalProxyServer as alias for backward compatibility, but use MitmProxyServer
LocalProxyServer = MitmProxyServer


def cleanup_proxy_server():
    """Stop the active proxy server if running"""
    global _active_proxy_server
    if _active_proxy_server:
        _active_proxy_server.stop()
        _active_proxy_server = None


def get_session_or_driver(
    url_to_test,
    get_session=True,
    headless=False,
    desktop_session=4,
    proxy_source=None,
    api_key=None,
    proxy_user=None,
    proxy_password=None,
    country_code=None,
    webdriver_path=None,
    default_timeout=10,
):
    """
    Retrieve a session or WebDriver instance configured with proxy and authentication settings.
    
    Logs the credentials being used for debugging purposes.

    This function provides the ability to either return a `requests.Session` object or a Selenium WebDriver instance
    configured with proxy settings, user authentication, and other options. It also supports fetching proxies from
    external sources and validating them.

    Args:
        url_to_test (str): The URL to test the proxy or session configuration.
        get_session (bool, optional): If True, return a `requests.Session` object; otherwise, return a WebDriver instance. Defaults to True.
        headless (bool, optional): If True, run the WebDriver in headless mode. Defaults to False.
        desktop_session (int, optional): The desktop session number for display when not in headless mode. Defaults to 4.
        proxy_source (str, optional): The source of proxies to use. Options include "webshare" or "sslhosts". Defaults to None.
        api_key (str, optional): API key for accessing proxy services like Webshare. Required if `proxy_source` is "webshare". Defaults to None.
        proxy_user (str, optional): Username for proxy authentication. Defaults to None.
        proxy_password (str, optional): Password for proxy authentication. Defaults to None.
        country_code (str, optional): The country code to filter proxies by location (e.g., "US", "IN"). Defaults to None.
        webdriver_path (str, optional): Path to the WebDriver executable. If not provided, it falls back to the configuration file. Defaults to None.
        default_timeout (int, optional): timeout in seconds before next proxy attempt. Defaults to 10 seconds.
    Returns:
        requests.Session or selenium.webdriver.Firefox: A configured session or WebDriver instance, depending on the `get_session` parameter.

    Raises:
        Exception: If there are issues with proxy configuration, WebDriver setup, or fetching proxies.

    Notes:
        - If `proxy_source` is "webshare", the function fetches proxies from the Webshare API.
        - If `proxy_source` is "sslhosts", the function scrapes proxies from sslproxies.org using Selenium.
        - The function validates proxies by testing them against the provided `url_to_test`.
        - WebRTC is disabled in the WebDriver to prevent IP leaks when using proxies.
    """
    # Log credentials being used (for debugging webshare account issues)
    if proxy_source == "webshare" and api_key and proxy_user:
        get_chameli_logger().log_info(
            "get_session_or_driver: Using webshare credentials",
            {
                "proxy_source": proxy_source,
                "proxy_user": proxy_user,
                "proxy_pass_masked": f"{proxy_password[:3]}***" if proxy_password else None,
                "api_key_masked": f"{api_key[:3]}***{api_key[-3:]}" if api_key else None,
                "function": "get_session_or_driver"
            }
        )

    def setup_driver_with_proxy_auth(proxy=None, proxy_user=None, proxy_pass=None):
        proxy_ip = None
        proxy_port = None
        if proxy:
            proxy_ip, proxy_port = proxy.split(":")

        # Configure Firefox options
        options = Options()
        options.set_preference("network.proxy.type", 1)  # Manual proxy config
        options.set_preference(
            "general.useragent.override",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.70 Safari/537.36",
        )

        if proxy_ip is not None and proxy_port is not None:
            # Use local proxy server if credentials are provided
            if proxy_user is not None and proxy_pass is not None:
                global _active_proxy_server
                
                # Track local port to reuse if we need to restart
                local_port_to_use = 8080  # Default port

                # Check if we need to restart proxy server (different upstream proxy or marked as failed)
                if _active_proxy_server and _active_proxy_server.running:
                    needs_restart = (
                        _active_proxy_server.upstream_proxy != f"{proxy_ip}:{proxy_port}" or
                        _active_proxy_server.proxy_user != proxy_user or
                        _active_proxy_server.proxy_pass != proxy_pass or
                        _active_proxy_server.needs_new_upstream
                    )
                    
                    if needs_restart:
                        get_chameli_logger().log_info(
                            f"Restarting proxy server with new upstream proxy (keeping same local port)",
                            {
                                "old_upstream": _active_proxy_server.upstream_proxy,
                                "new_upstream": f"{proxy_ip}:{proxy_port}",
                                "local_port": _active_proxy_server.local_port,
                                "function": "setup_driver_with_proxy_auth"
                            }
                        )
                        # Keep the same local port to avoid Firefox reconfiguration
                        local_port_to_use = _active_proxy_server.local_port
                        _active_proxy_server.stop()
                        _active_proxy_server = None
                        # Will create new server below with same local port
                    else:
                        # Reuse existing proxy server
                        get_chameli_logger().log_info(
                            f"Reusing existing local proxy server",
                            {
                                "upstream_proxy": f"{proxy_ip}:{proxy_port}",
                                "local_proxy": f"127.0.0.1:{_active_proxy_server.local_port}",
                                "function": "setup_driver_with_proxy_auth"
                            }
                        )

                # Start local proxy server if not already running
                if not _active_proxy_server or not _active_proxy_server.running:
                    try:
                        print(f"[DEBUG] Starting local proxy server for {proxy_ip}:{proxy_port} on port {local_port_to_use}")
                        _active_proxy_server = LocalProxyServer(
                            upstream_proxy=f"{proxy_ip}:{proxy_port}",
                            proxy_user=proxy_user,
                            proxy_pass=proxy_pass,
                            local_port=local_port_to_use
                        )
                        _active_proxy_server.start()
                        print(f"[DEBUG] Proxy server started, running={_active_proxy_server.running}")
                        # Verify it's actually running
                        if not _active_proxy_server.running:
                            raise Exception("Proxy server failed to start")
                        
                        # Additional health check - verify proxy server is actually accepting connections
                        import time
                        time.sleep(0.3)  # Give it a moment to be ready
                        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        test_sock.settimeout(1.0)
                        try:
                            result = test_sock.connect_ex(("127.0.0.1", 8080))
                            test_sock.close()
                            if result != 0:
                                get_chameli_logger().log_warning(
                                    f"Proxy server may not be accepting connections - connect_ex returned {result}",
                                    {"local_port": 8080, "result": result, "function": "setup_driver_with_proxy_auth"}
                                )
                        except Exception as e:
                            test_sock.close()
                            get_chameli_logger().log_warning(
                                f"Could not verify proxy server connection: {str(e)}",
                                {"local_port": 8080, "error": str(e), "function": "setup_driver_with_proxy_auth"}
                            )
                        
                        # Give the server a moment to be fully ready for connections
                        import time
                        time.sleep(0.5)
                        
                        # Test that the proxy server is actually accepting connections
                        try:
                            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            test_sock.settimeout(2)
                            local_port = _active_proxy_server.local_port
                            result = test_sock.connect_ex(("127.0.0.1", local_port))
                            test_sock.close()
                            if result == 0:
                                get_chameli_logger().log_info(
                                    f"Proxy server verified - accepting connections on port {local_port}",
                                    {"local_port": local_port, "function": "setup_driver_with_proxy_auth"}
                                )
                            else:
                                get_chameli_logger().log_warning(
                                    f"Proxy server may not be accepting connections - connect_ex returned {result}",
                                    {"local_port": local_port, "result": result, "function": "setup_driver_with_proxy_auth"}
                                )
                        except Exception as e:
                            get_chameli_logger().log_warning(
                                f"Could not verify proxy server connection: {str(e)}",
                                {"local_port": local_port if '_active_proxy_server' in locals() and _active_proxy_server else 8080, "error": str(e), "function": "setup_driver_with_proxy_auth"}
                            )
                    except Exception as e:
                        print(f"[DEBUG] Failed to start proxy server: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        get_chameli_logger().log_error(
                            f"Failed to start local proxy server: {str(e)}",
                            e,
                            {
                                "upstream_proxy": f"{proxy_ip}:{proxy_port}",
                                "local_port": 8080,
                                "function": "setup_driver_with_proxy_auth"
                            }
                        )
                        # Fall back to direct proxy (will show auth popup)
                        get_chameli_logger().log_warning(
                            "Falling back to direct proxy configuration (auth popup may appear)",
                            {"upstream_proxy": f"{proxy_ip}:{proxy_port}", "function": "setup_driver_with_proxy_auth"}
                        )
                        # Fall back to direct proxy configuration
                        options.set_preference("network.proxy.type", 1)
                        options.set_preference("network.proxy.http", proxy_ip)
                        options.set_preference("network.proxy.http_port", int(proxy_port))
                        options.set_preference("network.proxy.ssl", proxy_ip)
                        options.set_preference("network.proxy.ssl_port", int(proxy_port))
                        options.set_preference("network.proxy.no_proxies_on", "")
                        options.set_preference("media.peerconnection.enabled", False)
                        # Continue with driver creation - don't return here

                # Configure Firefox to use local proxy (no auth needed)
                # Use the actual port from the proxy server (may be different from 8080 if port conflict)
                actual_proxy_port = _active_proxy_server.local_port if _active_proxy_server else 8080
                options.set_preference("network.proxy.type", 1)  # Manual proxy config
                options.set_preference("network.proxy.http", "127.0.0.1")
                options.set_preference("network.proxy.http_port", actual_proxy_port)
                options.set_preference("network.proxy.ssl", "127.0.0.1")
                options.set_preference("network.proxy.ssl_port", actual_proxy_port)
                options.set_preference("network.proxy.ftp", "127.0.0.1")
                options.set_preference("network.proxy.ftp_port", actual_proxy_port)
                options.set_preference("network.proxy.socks", "")
                options.set_preference("network.proxy.socks_port", 0)
                options.set_preference("network.proxy.socks_version", 0)
                options.set_preference("network.proxy.socks_remote_dns", False)
                options.set_preference("network.proxy.no_proxies_on", "")  # No exclusions
                options.set_preference("network.proxy.share_proxy_settings", True)
                options.set_preference("network.proxy.allow_hijacking_localhost", True)  # Allow localhost proxies
                options.set_preference("media.peerconnection.enabled", False)  # Disable WebRTC to prevent IP leaks
                get_chameli_logger().log_info(
                    f"Firefox proxy preferences configured",
                    {"local_proxy": f"127.0.0.1:{actual_proxy_port}", "function": "setup_driver_with_proxy_auth"}
                )

                actual_proxy_port = _active_proxy_server.local_port if _active_proxy_server else 8080
                get_chameli_logger().log_info(
                    f"Using local proxy server for authentication",
                    {
                        "upstream_proxy": f"{proxy_ip}:{proxy_port}",
                        "local_proxy": f"127.0.0.1:{actual_proxy_port}",
                        "function": "setup_driver_with_proxy_auth"
                    }
                )
            else:
                # No credentials, use proxy directly (will show popup)
                options.set_preference("network.proxy.type", 1)  # Manual proxy config
                options.set_preference("network.proxy.http", proxy_ip)
                options.set_preference("network.proxy.http_port", int(proxy_port))
                options.set_preference("network.proxy.ssl", proxy_ip)
                options.set_preference("network.proxy.ssl_port", int(proxy_port))
                options.set_preference("network.proxy.no_proxies_on", "")  # No exclusions
                options.set_preference("media.peerconnection.enabled", False)  # Disable WebRTC to prevent IP leaks
                
        if headless:
            options.add_argument("--headless")
        else:
            os.environ["DISPLAY"] = f":{str(desktop_session)}"

        # Use the user-provided WebDriver path if available, otherwise fall back to the YAML config
        driver_path = webdriver_path if webdriver_path else get_dynamic_config().get("driver_path", "")
        service = Service(driver_path)
        
        try:
            get_chameli_logger().log_info(
                f"Creating Firefox driver with proxy configuration",
                {"proxy_ip": proxy_ip, "proxy_port": proxy_port, "proxy_user": proxy_user is not None, "function": "setup_driver_with_proxy_auth"}
            )
            driver = webdriver.Firefox(service=service, options=options)
            get_chameli_logger().log_info(
                f"Firefox driver created successfully",
                {"function": "setup_driver_with_proxy_auth"}
            )
        except Exception as e:
            get_chameli_logger().log_error(
                f"Failed to create Firefox driver: {str(e)}",
                e,
                {"driver_path": driver_path, "function": "setup_driver_with_proxy_auth"}
            )
            raise

        # Inject headers using JavaScript (if needed)
        for key, value in headers.items():
            driver.execute_script(f"Object.defineProperty(navigator, '{key}', {{get: () => '{value}'}});")
        return driver

    def setup_session_with_proxy_auth(
        proxy,
        proxy_user=None,
        proxy_password=None,
    ):
        if proxy_user and proxy_password:
            proxy_url = f"http://{proxy_user}:{proxy_password}@{proxy}"
            proxies = {"http": proxy_url, "https": proxy_url}
        elif proxy_source:
            proxy_url = f"http://{proxy}"
            proxies = {"http": proxy_url, "https": proxy_url}
        else:
            proxies = None
        s = requests.Session()
        if proxies:
            s.proxies.update(proxies)
        s.headers.update(headers)
        driver = setup_driver_with_proxy_auth(proxy, proxy_user, proxy_password)
        try:
            driver.get(url_to_test)  # Navigate to the target URL
            cookies = driver.get_cookies()
            for cookie in cookies:
                s.cookies.set(cookie["name"], cookie["value"], domain=cookie.get("domain"))
        finally:
            driver.quit()  # Ensure the WebDriver is closed
        return s

    def test_proxies(
        proxies,
        proxy_user=None,
        proxy_password=None,
        get_session=True,
        test_url="https://httpbin.org/ip",
    ):
        """
        Test a proxy with both WebDriver and requests.
        """

        # Test with WebDriver
        def test_with_webdriver(proxy, proxy_user, proxy_password, default_timeout=default_timeout):
            driver = None
            try:
                driver = setup_driver_with_proxy_auth(proxy, proxy_user, proxy_password)
                if driver is None:
                    get_chameli_logger().log_error(
                        f"Failed to create WebDriver for proxy {proxy}",
                        None,
                        {"proxy": proxy, "test_url": test_url, "function": "test_with_webdriver"}
                    )
                    return None
                # Use reasonable timeout for proxy testing - increased from 5s to 15s for slow proxies
                # Allow up to 15 seconds for proxy to respond, but respect default_timeout if it's higher
                test_timeout = min(default_timeout, 15) if default_timeout else 15
                driver.set_page_load_timeout(test_timeout)
                
                # Add retry logic for proxy testing (some proxies are slow to respond)
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        driver.get(test_url)
                        break  # Success, exit retry loop
                    except Exception as retry_error:
                        if attempt < max_retries - 1:
                            # Log retry attempt
                            get_chameli_logger().log_warning(
                                f"Proxy test attempt {attempt + 1} failed for {proxy}, retrying...",
                                {"proxy": proxy, "attempt": attempt + 1, "error": str(retry_error), "function": "test_with_webdriver"}
                            )
                            time.sleep(1)  # Brief delay before retry
                            continue
                        else:
                            # Last attempt failed, re-raise
                            raise
                get_chameli_logger().log_info(
                    f"Successfully tested proxy {proxy} with WebDriver",
                    {"proxy": proxy, "test_url": test_url, "function": "test_with_webdriver"}
                )
                return driver
            except Exception as e:
                # Check if it's a timeout error - these are common and expected for bad proxies
                error_type = type(e).__name__
                is_timeout = "timeout" in str(e).lower() or "Timeout" in error_type
                
                if is_timeout:
                    get_chameli_logger().log_warning(
                        f"Proxy {proxy} timed out during test (this is normal for slow/bad proxies)",
                        {"proxy": proxy, "test_url": test_url, "timeout": test_timeout, "function": "test_with_webdriver"}
                    )
                else:
                    get_chameli_logger().log_error(
                        f"WebDriver test failed for proxy {proxy}: {str(e)}",
                        e,
                        {"proxy": proxy, "test_url": test_url, "error_type": error_type, "function": "test_with_webdriver"}
                    )
                
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
                return None

        # Test with requests
        def test_with_requests(proxy, proxy_user, proxy_password, default_timeout=default_timeout):
            s = setup_session_with_proxy_auth(proxy, proxy_user, proxy_password)
            response = s.get(test_url, timeout=default_timeout)
            if response.status_code == 200:
                return s
            else:
                return None

        # Run both tests
        for proxy in proxies:
            if not get_session:
                out = test_with_webdriver(proxy, proxy_user, proxy_password)
            else:
                out = test_with_requests(proxy, proxy_user, proxy_password)
            if out:
                return out
        
        # If we get here, all proxies failed
        get_chameli_logger().log_error(
            f"All proxies failed for {len(proxies)} proxy(ies)",
            None,
            {"proxy_count": len(proxies), "test_url": test_url, "get_session": get_session, "function": "test_proxies"}
        )
        return None

    def get_proxy_country(ip_address):
        """
        Validate the actual country of a proxy using a third-party service.
        """
        try:
            # Use a third-party service like ipinfo.io or ip-api.com
            response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("countryCode")  # Returns the country code (e.g., "IN")
            else:
                get_chameli_logger().log_error(
                    f"Failed to fetch country for IP {ip_address}. Status code: {response.status_code}",
                    None,
                    {"file_path": "{response.status_code}", "operation": "file_operation"},
                )
                return None
        except Exception as e:
            get_chameli_logger().log_error(
                f"Error while fetching country for IP {ip_address}: {e}",
                None,
                {"file_path": "{e}", "operation": "file_operation"},
            )
            return None

    def get_random_webshare_proxy(country_code=None, mode="direct"):
        # Log API key being used for fetching proxies
        get_chameli_logger().log_info(
            "Fetching proxies from webshare API",
            {
                "api_key_masked": f"{api_key[:3]}***{api_key[-3:]}" if api_key else None,
                "country_code": country_code,
                "mode": mode,
                "function": "get_random_webshare_proxy"
            }
        )
        if country_code:
            url = f"https://proxy.webshare.io/api/v2/proxy/list/?mode={mode}&page=1&page_size=100&country_code={country_code}"
        else:
            url = f"https://proxy.webshare.io/api/v2/proxy/list/?mode={mode}&page=1&page_size=100"

        proxies = []
        response = requests.get(url, headers={"Authorization": api_key}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for proxy in data["results"]:
                if proxy["valid"]:
                    proxy_address = f"{proxy['proxy_address']}:{proxy['port']}"
                    # Validate the proxy's actual country
                    actual_country = get_proxy_country(proxy["proxy_address"])
                    if country_code is None or actual_country == country_code:
                        proxies.append(proxy_address)
        else:
            get_chameli_logger().log_error(
                f"Failed to retrieve proxies. Status code: {response.status_code}",
                None,
                {"file_path": "{response.status_code}", "operation": "file_operation"},
            )
        return proxies

    def get_free_proxy(
        country_code=None,
    ):
        def kill_firefox_processes():
            for process in psutil.process_iter():
                try:
                    if process.name().lower() in ["firefox", "geckodriver"]:
                        process.kill()
                except psutil.NoSuchProcess:
                    continue

        def fetch_proxies(country_code=None):
            """Fetch proxies from sslproxies.org using Selenium."""
            options = Options()
            options.add_argument("--headless")
            driver_path = webdriver_path if webdriver_path else get_dynamic_config().get("driver_path", "")
            service = Service(driver_path)
            driver = webdriver.Firefox(service=service, options=options)

            try:
                get_chameli_logger().log_info("Fetching proxies from sslproxies.org...")
                driver.get("https://sslproxies.org")

                # Wait for the proxy table to load
                table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
                rows = table.find_elements(By.XPATH, ".//tbody/tr")

                # Parse the proxies
                proxies = []
                for row in rows:
                    columns = row.find_elements(By.TAG_NAME, "td")
                    if country_code is None or (
                        country_code is not None and get_proxy_country(columns[0].text.strip()) == country_code
                    ):
                        proxy = f"{columns[0].text.strip()}:{columns[1].text.strip()}"
                        proxies.append(proxy)
                return proxies

            except Exception as e:
                get_chameli_logger().log_info(
                    f"Error fetching proxies: {e}", {"file_path": "{e}", "operation": "file_operation"}
                )
                return []

            finally:
                driver.quit()
                kill_firefox_processes()

        """Get a list of proxies from sslproxies.org"""
        proxies = fetch_proxies(country_code)
        return proxies

    if proxy_source and proxy_source.lower() == "webshare":
        proxies = get_random_webshare_proxy(country_code=country_code)
    elif proxy_source and proxy_source.lower() == "sslhosts":
        proxies = get_free_proxy(country_code=country_code)
    else:
        proxies = [None]
    return test_proxies(proxies, proxy_user, proxy_password, get_session, url_to_test)
