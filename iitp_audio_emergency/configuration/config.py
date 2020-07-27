import logging.config

logging.config.fileConfig('./configuration/logging.conf')
logger = logging.getLogger('Audio Threat Classification')
