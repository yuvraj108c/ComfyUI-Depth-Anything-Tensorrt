class ColoredLogger:
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def __init__(self, name):
        self.name = name

    def _log(self, level, message):
        color = self.COLORS.get(level, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        print(f"{color}[{self.name}|{level}]{reset} - {message}")

    def debug(self, message):
        self._log('DEBUG', message)

    def info(self, message):
        self._log('INFO', message)

    def warning(self, message):
        self._log('WARNING', message)

    def error(self, message):
        self._log('ERROR', message)

    def critical(self, message):
        self._log('CRITICAL', message)