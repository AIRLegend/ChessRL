"""
Defines a custom Logger to be used in all the distributed processes.
"""
import logging


class Logger():
    """
    Custom logger (with colors). MUST be obtained with Logger.get_instance()
    """

    __the_instance = None

    def get_instance():
        if Logger.__the_instance is None:
            Logger.__the_instance = Logger()
        return Logger.__the_instance

    def __init__(self):
        # Setup logger
        self.logger = logging.getLogger("ChessRL")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColorFormatter())
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def set_level(self, level):
        if level == 0:
            self.logger.setLevel(logging.DEBUG)
        elif level == 1:
            self.logger.setLevel(logging.INFO)
        elif level == 2:
            self.logger.setLevel(logging.ERROR)


class ColorFormatter(logging.Formatter):
    """ Internal class to make the outputs colored """
    l_blue = "\x1b[94m"
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmat = "%(asctime)s - [%(levelname)s] - %(message)s"

    FORMATS = {
        logging.DEBUG: l_blue + fmat + reset,
        logging.INFO: grey + fmat + reset,
        logging.WARNING: yellow + fmat + reset,
        logging.ERROR: red + fmat + reset,
        logging.CRITICAL: bold_red + fmat + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
