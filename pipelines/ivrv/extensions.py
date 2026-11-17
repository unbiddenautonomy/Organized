"""Other pipelines' dashboard routes and in-process bridges, as listed in config/service.json.

IVRV's dashboard is the service's main page, so it serves the "web" routes each
pipeline declares (PCR at /PCR, 43Weeks at /firstTwoWeeks). The older
single-process session also runs PCR's "bridge" in-process. Nothing here
imports another pipeline by name; the registry entries do.
"""
import logging

from voltr.service import registry

from . import paths

LOGGER = logging.getLogger("fixed_strike_live")
_PIPELINES = registry.pipelines(registry.service_settings(paths.DEFAULT_CONFIG))


def _load(name, role):
    spec = _PIPELINES.get(name, {}).get(role)
    return registry.import_object(spec) if spec else None


def weeks43_module():
    """43Weeks' dashboard routes and in-process bridge, or None when it is not registered."""
    return _load("43weeks", "web")


# Existing models, routing, paper positions, configuration and logs are untouched by these extensions.
PCR_BRIDGE = PCR_WEB = None
try:
    PCR_BRIDGE, PCR_WEB = _load("pcr", "bridge"), _load("pcr", "web")
except Exception:
    LOGGER.exception("Optional PCR extension unavailable; existing strategy continues")
