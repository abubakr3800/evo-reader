import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app as flask_app

# Passenger on this cPanel host doesn't pass SCRIPT_NAME through to the WSGI
# environ for an app mounted under a subdirectory (here, /evo-reader). Without
# it, Flask has no idea it isn't mounted at "/", so every url_for() call
# (redirects after upload, the dashboard/file links on the result page,
# form-less links, etc.) generates root-relative URLs like /result/<id>
# instead of /evo-reader/result/<id> -- which is exactly the 404s and the
# "sent to / instead of /evo-reader/" behavior you're seeing.
#
# Fix: force SCRIPT_NAME on every request so Flask (and url_for) knows the
# real mount point, and strip that prefix from PATH_INFO if the web server
# ever does start including it (keeps this safe either way).
MOUNT_PREFIX = "/evo-reader"


class PrefixFix:
    def __init__(self, app, prefix):
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        path_info = environ.get("PATH_INFO", "")
        if path_info.startswith(self.prefix):
            path_info = path_info[len(self.prefix):]
        environ["PATH_INFO"] = path_info or "/"
        environ["SCRIPT_NAME"] = self.prefix
        return self.app(environ, start_response)


application = PrefixFix(flask_app, MOUNT_PREFIX)
