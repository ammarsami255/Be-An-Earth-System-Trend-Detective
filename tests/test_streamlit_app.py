"""
Streamlit AppTest Verification
EarthTrend Explorer - NASA Space Apps 2026
"""

import os
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))


def test_streamlit_app_renders():
    """Verify that the Streamlit app loads and runs with default state without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App threw exception during default run: {at.exception}"


def test_streamlit_app_navigation():
    """Verify navigation through all available pages in the application."""
    at_init = AppTest.from_file(APP_PATH, default_timeout=30)
    at_init.run()
    assert len(at_init.radio) > 0, "Navigation radio widget not found."
    pages = list(at_init.radio[0].options)

    for p in pages:
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.radio[0].set_value(p).run()
        assert not at.exception, f"Page '{p}' threw an exception: {at.exception}"
