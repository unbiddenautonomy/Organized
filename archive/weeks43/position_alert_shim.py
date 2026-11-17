"""Access the copied runtime's shared position sound from standalone 43Weeks."""
from pathlib import Path
import sys

inference = Path(__file__).resolve().parent.parent / 'ivrv' / 'engine' / 'Inference'
if str(inference) not in sys.path:
    sys.path.append(str(inference))
from live.position_alert import notify_position_action, notify_signal
