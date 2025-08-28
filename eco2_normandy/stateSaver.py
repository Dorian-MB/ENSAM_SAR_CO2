import simpy
from .factory import Factory
from .storage import Storage
from .ship import Ship


class StateSaver:
    def __init__(
        self,
        env: simpy.Environment,
        factory: Factory,
        storages: list[Storage],
        ships: list[Ship],
    ) -> None:
        self.env = env
        self.factory = factory
        self.storages = storages
        self.ships = ships

        self.action = env.process(self.run())

    def run(self):
        while True:
            self.factory._save_state()
            for s in self.storages:
                s._save_state()
            for s in self.ships:
                s._save_state()

            yield self.env.timeout(1)
