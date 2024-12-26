import logging
from contextvars import ContextVar

session_id: ContextVar[str] = ContextVar('session_id', default='NO_SESSION')

class SessionFilter(logging.Filter):
    def filter(self, record):
        record.session_id = session_id.get()
        return True