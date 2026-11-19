"""Where IVRV keeps its files. Code reads paths.BASE_DIR at call time, so tests can point it elsewhere."""
from pathlib import Path

PROJECT_DIR = LIVE_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parents[1]
BASE_DIR = PROJECT_DIR  # relative log paths in live_config.json resolve here
DASHBOARD_HTML = PROJECT_DIR / "dashboard" / "live_dashboard.html"
DEMO_DASHBOARD_HTML = PROJECT_DIR / "dashboard" / "demo_positions_dashboard.html"
SERVICE_DASHBOARD_HTML = REPO_DIR / "voltr" / "service" / "pipelines_dashboard.html"
DEFAULT_CONFIG = REPO_DIR / "config" / "live_config.json"
