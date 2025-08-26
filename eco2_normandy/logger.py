import logging

class Logger:
    """Logger personnalisé pour l'application.
    Il enregistre les messages dans la console et dans un fichier.
    Les niveaux de log, handeler et formater sont configurables.
    """
    def __init__(self, name="eco2_normandy", log_file="app.log"):
        self.name = name
        self.log_file = log_file
        self._logger = self.get_logger()
    
    def get_logger(self):
        """
        Retourne le logger configuré.
        Si le logger n'a pas de handlers, il le configure.
        """
        logger = logging.getLogger(self.name)
        if not logger.hasHandlers():
            return self.setup_handler(logger)
        return logger

    def setup_handler(self, logger):
        # Crée un logger
        logger.setLevel(logging.DEBUG)  # Niveau global (le plus bas pour capter tous les messages)

        # ----- Handler console -----
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.DEBUG)  # Affiche dans la console

        # ----- Handler fichier -----
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(logging.WARNING)  # Enregistre dans le fichier.

        # ----- Format commun -----
        self.set_default_format()

        # ----- Ajout des handlers -----
        logger.addHandler(self.console_handler) 
        logger.addHandler(self.file_handler)

        # debug, info, warning, error, critical
        return logger

    def set_default_format(self):
        """Change le format des handlers."""
        self.set_console_format()
        self.set_file_format()
    
    def set_console_format(self, format="\033[1m%(levelname)s\033[0m :\t%(message)s",end="\n"):
        """Change le format du handler console."""
        formatter = logging.Formatter(format + end)
        self.console_handler.setFormatter(formatter)

    def set_file_format(self, format="-- %(levelname)s -- %(asctime)s\n%(message)s", end="\n"):
        """Change le format du handler fichier."""
        formatter = logging.Formatter(format + end)
        self.file_handler.setFormatter(formatter)

    def setLevel(self, level):
        self._logger.setLevel(level)
        self.console_handler.setLevel(level)
        self.file_handler.setLevel(level)
    
    def set_console_level(self, level):
        self.console_handler.setLevel(level)
    def set_file_level(self, level):
        self.file_handler.setLevel(level)

    def close(self):
        for handler in self._logger.handlers:
            handler.close()
            self._logger.removeHandler(handler)
    
    def __getattr__(self, name):
        # Éviter la récursion infinie lors de l'accès à _logger
        if name == '_logger':
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Éviter les méthodes de copie/sérialisation pour deepcopy
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
        try:
            return getattr(self._logger, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")