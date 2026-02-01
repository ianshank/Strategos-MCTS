"""
Pytest configuration for chess UI Selenium tests.

This module provides fixtures and configuration for running Selenium tests
against the chess web UI.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver


# Check if Selenium is available
try:
    import selenium
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "selenium: marks tests as Selenium browser tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "ui: marks tests as UI tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip Selenium tests if selenium is not installed."""
    if not SELENIUM_AVAILABLE:
        skip_selenium = pytest.mark.skip(reason="Selenium not installed")
        for item in items:
            if "selenium" in item.keywords:
                item.add_marker(skip_selenium)


@pytest.fixture(scope="session")
def ui_server_port() -> int:
    """Get the port for the UI server."""
    return int(os.getenv("CHESS_UI_PORT", "7861"))


@pytest.fixture(scope="session")
def ui_server(ui_server_port: int) -> Generator[subprocess.Popen | None, None, None]:
    """Start the UI server for testing.

    This fixture starts a Gradio server in the background and yields
    the subprocess. The server is terminated after all tests complete.
    """
    if not SELENIUM_AVAILABLE:
        yield None
        return

    # Get the project root directory
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(__file__))
            )
        )
    )

    # Start the UI server
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root

    server_process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"""
import sys
sys.path.insert(0, '{project_root}')
from src.games.chess.ui import create_chess_ui
demo = create_chess_ui()
demo.launch(server_port={ui_server_port}, share=False, show_error=True, prevent_thread_lock=False)
            """,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=project_root,
    )

    # Wait for server to start
    max_wait = 30
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < max_wait:
        try:
            import urllib.request
            urllib.request.urlopen(f"http://localhost:{ui_server_port}", timeout=1)
            server_ready = True
            break
        except Exception:
            time.sleep(0.5)

    if not server_ready:
        server_process.terminate()
        pytest.skip("UI server failed to start")

    yield server_process

    # Cleanup
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()


@pytest.fixture(scope="session")
def browser_options() -> dict:
    """Get browser configuration options."""
    return {
        "headless": os.getenv("HEADED", "").lower() not in ("1", "true", "yes"),
        "slow_mo": float(os.getenv("SLOW_MO", "0.5")),
        "browser": os.getenv("BROWSER", "chrome"),
        "window_size": (1920, 1080),
    }


@pytest.fixture(scope="function")
def driver(browser_options: dict, ui_server: subprocess.Popen | None) -> Generator[WebDriver, None, None]:
    """Create and configure WebDriver for each test.

    This fixture creates a new browser instance for each test function,
    ensuring test isolation.
    """
    if not SELENIUM_AVAILABLE:
        pytest.skip("Selenium not installed")

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.firefox.service import Service as FirefoxService

    driver_instance: WebDriver | None = None

    try:
        if browser_options["browser"] == "chrome":
            options = ChromeOptions()
            if browser_options["headless"]:
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument(f"--window-size={browser_options['window_size'][0]},{browser_options['window_size'][1]}")

            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = ChromeService(ChromeDriverManager().install())
                driver_instance = webdriver.Chrome(service=service, options=options)
            except ImportError:
                driver_instance = webdriver.Chrome(options=options)

        else:  # firefox
            options = FirefoxOptions()
            if browser_options["headless"]:
                options.add_argument("--headless")

            try:
                from webdriver_manager.firefox import GeckoDriverManager
                service = FirefoxService(GeckoDriverManager().install())
                driver_instance = webdriver.Firefox(service=service, options=options)
            except ImportError:
                driver_instance = webdriver.Firefox(options=options)

        driver_instance.implicitly_wait(10)
        driver_instance.set_page_load_timeout(30)

        yield driver_instance

    except Exception as e:
        pytest.skip(f"Could not create browser driver: {e}")

    finally:
        if driver_instance:
            driver_instance.quit()


@pytest.fixture(scope="function")
def chess_url(ui_server_port: int) -> str:
    """Get the URL for the chess UI."""
    return f"http://localhost:{ui_server_port}"
