
# Regulare import
import sys
import math
import random
from pathlib import Path

import pygame
import numpy as np

sys.path.append(str(Path.cwd()))

# Personal import 
from KPIS import LiveKpisGraphsGenerator  
from eco2_normandy import Factory, Storage, Simulation, shipState
from eco2_normandy.logger import Logger

class AnimeCfg:

    NAME = None
    # Couleurs (red, green, bleu, alpha)
    WHITE = [255, 255, 255, 255]
    BLACK = [0, 0, 0, 255]
    BLUE = [0, 0, 255, 255]
    BLUE_ROYAL = [65, 105, 225, 255]
    GREEN = [0, 150, 0, 255]
    SEAGREEN = [46, 139, 87, 255]
    RED = [255, 0, 0, 255]
    ORANGE = [255, 165, 0, 255]
    PURPLE = [148, 0, 211, 255]
    SHIP_COLORS = [RED, ORANGE, PURPLE, BLUE, GREEN]
    
    # Chemins vers les assets
    assets_path = Path().cwd() / "assets"
    IM_SHIP_PATH = assets_path / "ship.png"
    STORAGE_PATH = assets_path / "storage.png"
    FACTORY_PATH = assets_path / "factory.png"
    PORT_PATH = assets_path / "port 2.png"
    SMOKE_PATH = assets_path / "fumee.png"

    # Tailles et dimensions
    SHIP_SIZE = 50
    PORT_SIZE = 70
    ANIM_WIDTH, ANIM_HEIGHT = 500, 500
    WINDOW_WIDTH, WINDOW_HEIGHT = 900, 900
    WINDOW_SIZE = WINDOW_WIDTH, WINDOW_HEIGHT

    MIN_FPS = 0 
    MAX_FPS = 60
    DEFAULT_FPS = 30


class PGAnime:
    num_anime = 0
    def __init__(self, simulation_config, config_name=None, random_pos=False, logger=None,
                anime_cfg=AnimeCfg(), Simulation=Simulation,
                **kw
                ):
        """
        Initialise l’animation à partir de la configuration de simulation.

        Args:
            config (dict): paramètres de la simulation (ex. nombre de périodes, distances, etc.).
            config_name (str): nom utilisé pour identifier cette instance d’animation.
            random_pos (bool): si True, place aléatoirement les ports dans la fenêtre.
            logger : instance de logger pour enregistrer les événements.
            anime_cfg (AnimeCfg): objet contenant les constantes d’affichage (tailles, couleurs, chemins).
            Simulation : classe de simulation à instancier.
            **kw: arguments supplémentaires passés à l’objet.
        """
        PGAnime.num_anime += 1
        self.logger = logger or Logger()
        self.config = simulation_config
        self.cfg = anime_cfg
        self.cfg.NAME = config_name or str(PGAnime.num_anime)
        self.SHIP_SIZE = self.cfg.SHIP_SIZE
        self.PORT_SIZE = self.cfg.PORT_SIZE
        self.WIDTH, self.HEIGHT =  self.cfg.WINDOW_WIDTH, self.cfg.WINDOW_HEIGHT
        self.PERIODE = self.config["general"]["num_period_per_hours"]
        self.simulation = Simulation(config_name=self.cfg.NAME, config=self.config)
        self.env = self.simulation.env
        self.random_pos = random_pos
        self.__dict__.update(kw)
        self._init()
        self._init_port_position()
        self._init_img()

    def _init(self):  
        """Configure Pygame, initialise l’environnement de simulation et les contrôles UI."""
        # Initialisation simulation et positions

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Animation de la Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.all_ports = self.simulation.storages + [self.simulation.factory]
        self.wast_production_overtime = 0

        # UI controls
        # Slider pour la vitesse de simulation
        slider_width, slider_height = 200, 10
        self.slider_rect = pygame.Rect(0, 0, slider_width, slider_height) # Position (x, y) def dans 'dessine_UI'
        slider_knob_width = 10
        self.knob_rect = pygame.Rect(0, 0, slider_knob_width, slider_height)
        self.dragging = False

        # Bouton pause
        self.paused = True 
        self.pause_button_rect = pygame.Rect(10, 40, 100, 30)

        # bouton pour afficher/masquer les KPIs
        self.kpi_button_rect = pygame.Rect(
            self.pause_button_rect.right + 10,
            self.pause_button_rect.y,
            80, self.pause_button_rect.height
        )
        self.show_kpis = False
        self.kpis_graphs = []

        # FPS
        self.current_fps = min(max(self.cfg.DEFAULT_FPS, self.cfg.MIN_FPS), self.cfg.MAX_FPS)

        # scroll / pan
        self.view_offset = [0, 0]
        self.panning = False
        self.pan_anchor = (0, 0)
        self.offset_anchor = (0, 0)
        self.scroll_speed = 30
        
        self.kpis_generator = self._get_kpis_generator()

        self.ui_height = 80
        self.anim_height = self.cfg.ANIM_HEIGHT
        self.surfaces_pos = {"ui":(0, 0), "anime":(0, self.ui_height), "kpis":(0, self.anim_height+self.ui_height)}
        self._init_surfaces()
        self.all_surfs = [self.ui_surf, self.anim_surf, self.kpis_surf]

    def _init_surfaces(self):
        """Crée les surfaces Pygame pour l’UI, l’animation et les KPIs."""
        self.ui_surf = pygame.Surface((self.WIDTH, self.ui_height), pygame.SRCALPHA) # UI fixe
        self.anim_surf = pygame.Surface((self.WIDTH, self.anim_height), pygame.SRCALPHA)
        # todo anim_surf.subsurface (width fix /p ) pour une zone dedié a l'animation ? cplipping integré
        self._set_kpis_surface()
        
    def _init_port_position(self):
        if self.random_pos:
            self._set_random_port_positions()
        else:
            self._set_fixed_port_positions()
    
    @staticmethod
    def _get_image(path, size):
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.scale(img, (size, size))
        return img

    def _init_img(self):
        """Charge et prépare les images (bateaux, usines, stockages, fumée), 
        et assigne une couleur/une image à chaque navire et port."""
        # Initialisation des images
        self.ship_img = self._get_image(self.cfg.IM_SHIP_PATH, self.SHIP_SIZE)
        self.factory_img = self._get_image(self.cfg.FACTORY_PATH, self.PORT_SIZE)
        self.storage_img = self._get_image(self.cfg.STORAGE_PATH, self.PORT_SIZE)
        self.port_img = self._get_image(self.cfg.PORT_PATH, self.PORT_SIZE)
        self.smoke_img = self._get_image(self.cfg.SMOKE_PATH, self.PORT_SIZE)

        # Affecte à chaque bateau son image (avec un point de couleur) sans changer l'image de base
        for ship in self.simulation.ships:
            ship.color = random.choice(self.cfg.SHIP_COLORS)
            if self.cfg.SHIP_COLORS:
                self.cfg.SHIP_COLORS.remove(ship.color)
            ship.img = self.ship_img.copy()
            dot_size = 5
            pygame.draw.circle(ship.img, ship.color, (dot_size, dot_size), dot_size)
            ship.base_img = ship.img.copy()
        
        for storage in self.simulation.storages:
            storage.img = self.storage_img.copy()
            storage.color = self.cfg.SEAGREEN
        self.simulation.factory.img = self.factory_img.copy()
        self.simulation.factory.color = self.cfg.BLUE_ROYAL
        self.simulation.factory.smoke_img = self.smoke_img

    def _set_fixed_port_positions(self):
        x, _ = self.anim_surf.get_size()
        y = self.cfg.ANIM_HEIGHT
        off_set_x, off_set_y = x // 10, y // 2
        self.simulation.factory.position = (off_set_x + self.PORT_SIZE, off_set_y)

        off_set_y = 2 * self.PORT_SIZE
        storages = self.simulation.storages
        n = len(storages)
        pos = np.linspace(off_set_y, y - off_set_y, n) if n > 1 else [y // 2]
        for i, s in enumerate(storages):
            s.position = ( x - self.PORT_SIZE, pos[i])

    def _set_random_port_positions(self):
        x, y = self.anim_surf.get_size()
        off_set_x, off_set_y = x // 10, y // 2
        random_pos = lambda: (random.randint(off_set_x, self.WIDTH - off_set_x), random.randint(off_set_y, self.HEIGHT - off_set_y))
        factory_pos = random_pos()
        self.simulation.factory.position = factory_pos

        for s in self.simulation.storages:
            storage_pos = random_pos()
            while math.dist(storage_pos, factory_pos) < 300:
                storage_pos = random_pos()
            s.position = storage_pos
        return self.config

    def _get_ship_position_when_docking_or_waiting(self, pos):
        """Calcule la position écran d’un navire lorsqu’il est à quai ou en attente. C'est a dire,
        lorsqu'il est sur la gauche d'un port.

        Args:
            pos (tuple[int, int]): coordonnées du port d’attache.
        """
        pos = self._get_img_port_position(pos)
        off_set = self.PORT_SIZE // 2 + self.SHIP_SIZE // 2
        x, y = pos
        pos = (x - off_set, y)
        return [i - self.SHIP_SIZE // 2 for i in pos]
    
    def _get_img_port_position(self, position, margin=5):
        """Décale la position d’un port vers le bas pour y afficher l’image.

        Args:
            position (tuple[int, int]): position du bâtiment.
            margin (int): marge verticale entre bâtiment et port.
        """
        x, y = position
        return (x, y + self.PORT_SIZE + margin)
    
    def _get_smoke_pos_for_factory_wasted_production(self, factory_position):
        """Calcule où dessiner la fumée de production gaspillée de l’usine.

        Args:
            position (tuple[int, int]): position de l’usine.
        """
        x, y = factory_position
        margin = 20
        return x + margin, y - self.PORT_SIZE - margin

    def _blit_port_image(self, img, pos, size):
        """Colle une image centrée sur `pos` dans la surface d’animation.

        Args:
            img (Surface): image à afficher.
            pos (tuple[int, int]): point central.
            size (int): taille de l’image.
        """
        # Permet de centrer le port au coordoner dans "pos"
        self.anim_surf.blit(img, (pos[0] - size // 2, pos[1] - size // 2, size, size))

    def _blit_text(self, txt, position):
        """Colle un `Surface` texte à l’emplacement donné.

        Args:
            txt (Surface): texte rendu.
            position (tuple[int, int]): coordonnées écran.
        """
        self.anim_surf.blit(txt, position)

    def _world_to_screen(self, pos):
        """Transforme des coordonnées monde en coordonnées écran.

        Args:
            pos (tuple[int, int]): coord. dans le monde de simulation.

        Returns:
            list[int, int]: coordonnées à l’écran (tenu compte du scroll).
        """
        x, y = pos
        screen_x = x
        screen_y = y - self.view_offset[1] # "-" car on scroll vers le bas (limite haute en y=0)
        return [screen_x, screen_y]

    def _get_screen_position(self):
        """Calcule les positions d’affichage de chaque sous-surface (UI, anim, KPIs).

        Returns:
            dict[str, tuple[int, int]]: positions écran pour 'ui', 'anime' et 'kpis'. Ex:{'anime':(0, 120), ...}
        """
        surfaces_pos = self.surfaces_pos.copy()
        for surf_name, pos in surfaces_pos.items(): 
            if surf_name == "ui":continue # ui reste fixe
            surfaces_pos[surf_name] = self._world_to_screen(pos)
        return surfaces_pos 
    
    def _clamp_view_offset(self):
        """Limite le scroll vertical entre 0 et le maximum calculé."""
        total_h = self.anim_surf.get_height() + self.kpis_surf.get_height()
        visible_h = self.HEIGHT - self.ui_height
        max_off = max(0, total_h - visible_h)
        # Clamp entre 0 (haut) et max_off (bas)
        return max(0, min(self.view_offset[1], max_off))

    def draw_ui(self):
        """Dessine la barre de contrôle (slider de vitesse, boutons Pause/KPIs, affichage du temps)."""
        # Slider / vitesse de simulation
        slider_width = self.slider_rect.width
        slider_height = self.slider_rect.height
        slider_x = self.WIDTH - slider_width - 40
        slider_y = slider_height + 40
        slider_knob_width = self.knob_rect.width
        self.slider_rect.x, self.slider_rect.y = slider_x, slider_y
        self.knob_rect.x = slider_x + slider_width * (self.current_fps - self.cfg.MIN_FPS) / self.cfg.MAX_FPS - slider_knob_width / 2
        self.knob_rect.y = slider_y

        pygame.draw.rect(self.ui_surf, self.cfg.BLACK, self.slider_rect, 2, border_radius=10)
        pygame.draw.rect(self.ui_surf, self.cfg.RED, self.knob_rect, border_radius=10)
        fps_text = self.font.render(f"Vitesse de simulation: {'x' + str(self.current_fps) if self.current_fps != 0 else 'uncap'}", True, self.cfg.BLACK)
        self.ui_surf.blit(fps_text, (self.slider_rect.x, self.slider_rect.y - 20))
        
        # Bouton Pause/Resume
        pygame.draw.rect(self.ui_surf, self.cfg.ORANGE, self.pause_button_rect, border_radius=10)
        btn_text = "Resume" if self.paused else "Pause"
        btn_surface = self.font.render(btn_text, True, self.cfg.BLACK)
        btn_rect = btn_surface.get_rect(center=self.pause_button_rect.center)
        self.ui_surf.blit(btn_surface, btn_rect)

        # bouton KPIs
        pygame.draw.rect(self.ui_surf, self.cfg.PURPLE, self.kpi_button_rect, border_radius=10)
        txt = self.font.render("KPIs", True, self.cfg.BLACK)
        txt_rect = txt.get_rect(center=self.kpi_button_rect.center)
        self.ui_surf.blit(txt, txt_rect)
        
        time_text = self.font.render(f"Time: {int(self.env.now / self.PERIODE)} heures", True, self.cfg.BLACK)
        self.ui_surf.blit(time_text, (10, 10))

    def draw_ship(self, ship):
        """Affiche un bateau sur la surface d’animation selon son état (navigating, docked, waiting).
        
        Args:
            ship (Ship): instance du navire à dessiner, avec ses attributs (état, positions, image, couleur).
        """
        if ship.state == shipState.NAVIGATING:
            # Utiliser la position de départ (du port d'où il vient) et la position de destination en tant que port
            if isinstance(ship.destination, Factory):
                start = ship.former_destination.position
                end = ship.destination.position
                storage_name = ship.former_destination.name
            else:
                start = ship.former_destination.position
                end = ship.destination.position
                storage_name = ship.destination.name

            total_distance = self.config["general"]["distances"]["Le Havre"][storage_name]
            progress = 1 - (ship.distance_to_go / total_distance)

            # Permet de demarer au meme endroit ou le bateau est en docking/waiting
            # => plus fluide
            start = self._get_ship_position_when_docking_or_waiting(start)
            end = self._get_ship_position_when_docking_or_waiting(end)
            
            current_x = int(start[0] + (end[0] - start[0]) * progress)
            current_y = int(start[1] + (end[1] - start[1]) * progress)
            current_pos = (current_x, current_y)
            self.anim_surf.blit(ship.img, current_pos)
        
        elif ship.state == shipState.DOCKED:
            # Lorsque le navire est docké, on l'affiche au point d'arrivée avec une barre de chargement.
            pos = ship.destination.position
            port_pos = self._get_img_port_position(pos)  # utiliser la position du port
            ship_pos = [i - self.SHIP_SIZE // 2 for i in port_pos] # affiche le bateau au milieu du port 
            self.anim_surf.blit(ship.img, ship_pos)
            
            match ship.destination:
                case Factory():
                    progress = ship.capacity / ship.capacity_max if ship.capacity_max != 0 else 0
                case Storage():
                    progress = 1 - ship.capacity / ship.capacity_max if ship.capacity_max != 0 else 0

            bar_height = 10
            bar_width = self.PORT_SIZE
            # Les lignes suivante permetent d'afficher les bar de chargement horizontal, les un en dessous des autres, automatiquement.
            # Ex: {"Le Havre": {"ship1":1, "ship2":2}, "Bergen":{"ship3":1}} etc
            try: # Block try, pour pouvoir utiliser l'ancienne simulation : debug
                port_name = [s.name for s in self.simulation.storages] + [self.simulation.factory.name]
                all_ship_order = { 
                        destination_name:{s.name:s.dock_req.order 
                                        for s in self.simulation.ships 
                                        if s.is_docked and s.destination.name==destination_name}
                        for destination_name in port_name
                                }
                all_ship_order = dict(sorted(all_ship_order[ship.destination.name].items(), key=lambda x:x[1]))
                order = [i for i, k in enumerate(all_ship_order.keys()) if k == ship.name][0]
                marge = 1
                offset = (bar_height+marge)*order
            except :
                offset = 0

            bar_pos = (port_pos[0] - self.PORT_SIZE // 2, 1 + port_pos[1] + self.PORT_SIZE // 2 + offset)
            pygame.draw.rect(self.anim_surf, self.cfg.BLACK, (*bar_pos, bar_width, bar_height), 2, border_radius=10)
            pygame.draw.rect(self.anim_surf, ship.color, (*bar_pos, int(bar_width * progress), bar_height), border_radius=10)

        elif ship.state in (shipState.WAITING, shipState.DOCKING):
            alpha = 0.6 if ship.capacity == 0 else 1
            ship.img = ship.base_img.copy()
            ship.img.set_alpha(int(255 * alpha))
            ship_pos = self._get_ship_position_when_docking_or_waiting(ship.destination.position)
            self.anim_surf.blit(ship.img, ship_pos)

    def _draw_smoke(self, port, pos):
        alpha = 0.8 if port.capacity == port.capacity_max else 0
        if alpha == 0: self.wast_production_overtime = 0
        elif not self.paused: self.wast_production_overtime += port.wasted_production
        txt = f"Wasted production: {int(self.wast_production_overtime)}" if alpha!=0 else ""
        txt = self.font.render(txt, True, self.cfg.RED)
        smoke_pos = self._get_smoke_pos_for_factory_wasted_production(factory_position=pos)
        port.smoke_img.set_alpha(int(255*alpha))
        self._blit_port_image(port.smoke_img, smoke_pos, self.PORT_SIZE)
        self._blit_text(txt,(smoke_pos[0]- 60, smoke_pos[1] - 60))

    def draw_port(self, port):
        """Affiche un port (usine ou stockage) avec sa barre de progression et, pour l’usine, la fumée de production.

        Args:
            port (Factory|Storage): instance du port à dessiner, avec positions, capacités et images.
        """
        pos = port.position
        self._blit_port_image(port.img, pos, self.PORT_SIZE)
        # Calculer la position du port sous l'usine
        port_pos = self._get_img_port_position(pos)
        self._blit_port_image(self.port_img, port_pos, self.PORT_SIZE)
        if isinstance(port, Factory):
            self._draw_smoke(port, pos)
        ratio = port.capacity / port.capacity_max
        text = self.font.render(f"{port.name}: {int(ratio * 100)}%", True, self.cfg.BLACK)
        self._blit_text(text, (pos[0] - 40, pos[1] -60))
        self._draw_vertical_progress_bar(port, port.color, pos=pos)

    def _draw_vertical_progress_bar(self, port, color, offset_x=4, border_radius=10, pos=None):
        bar_height = self.PORT_SIZE 
        bar_width = 12
        progress = (port.capacity / port.capacity_max) if port.capacity_max > 0 else 0
        if pos is None : pos = port.position
        bar_pos = [ i + sign * bar_height // 2 for sign, i in zip([1, -1], pos)]
        bar_pos[0] += offset_x
        x, y = bar_pos
        pygame.draw.rect(self.anim_surf, self.cfg.BLACK, (*bar_pos, bar_width, bar_height), 2, border_radius=border_radius)
        pygame.draw.rect(self.anim_surf, color, (x, y + bar_height * (1 - progress), bar_width, bar_height * progress), border_radius=border_radius)

    def draw_ports_and_ships(self):
        # --- Affichage des bâtiments et port ---
        for port in self.all_ports:
                self.draw_port(port)
        # Dessine les bateaux
        for ship in self.simulation.ships:
            self.draw_ship(ship)

    def _draw_dynamic_border(self, surf, pos):
        border_rect = surf.get_rect()
        x, y = pos
        border_rect.x = x
        border_rect.y = y
        pygame.draw.rect(self.screen, self.cfg.BLACK, border_rect, 2)

    def blit_all_surfaces(self):
        # ui reste fixe
        positions = self._get_screen_position()
        self.screen.blit(self.anim_surf, positions["anime"])
        self._draw_dynamic_border(self.anim_surf, positions["anime"])
        if self.show_kpis:
            self.screen.blit(self.kpis_surf, positions["kpis"])
            self._draw_dynamic_border(self.kpis_surf, positions["kpis"])
        self.screen.blit(self.ui_surf, (0, 0)) # fixe, blit en denier pour qu'il soit devant
        self._draw_dynamic_border(self.ui_surf, positions["ui"])

    def fill_surf(self, color):
        """Remplit toutes les surfaces (screen, UI, anim, KPIs) avec la couleur donnée.

        Args:
            color (list[int, int, int, int]): couleur RGBA.
        """
        self.screen.fill(color)
        self.ui_surf.fill(color)
        self.anim_surf.fill(color)
        if self.show_kpis:
            self.kpis_surf.fill(color)

    def handle_pygame_event(self, event):
        """Gère un `pygame.Event` (scroll, resize, pan, UI click, drag knob).

        Args:
            event (pygame.Event): événement capturé.
        """
        # gestion de la molette pour scroll vertical
        if event.type == pygame.MOUSEWHEEL:
            self.view_offset[1] += -event.y * self.scroll_speed
            # Calcul du défilable total et de la fenêtre visible
            self.view_offset[1] = self._clamp_view_offset()
        # redimensionnement de la fenêtre
        if event.type == pygame.VIDEORESIZE:
            # Force une taille mini de 800×800
            new_w = max(800, event.w)
            new_h = max(800, event.h)
            self.WIDTH, self.HEIGHT = new_w, new_h
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
            self.kpis_generator.on_resize(self.screen.get_size(), self.cfg.WINDOW_SIZE)
            self._init_surfaces()
            self._init_port_position()

        # démarrage du pan au clic-droit(3) / click-gauche(1)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (3, 1):
            # si on n’est pas sur un control UI (optionnel : tester collision avec slider_rect etc)
            self.panning = True
            self.pan_anchor = event.pos
            self.offset_anchor = tuple(self.view_offset)

        # arrêt du pan
        if event.type == pygame.MOUSEBUTTONUP and event.button in (3, 1):
            self.panning = False

        # mouvement de la souris pendant pan
        if event.type == pygame.MOUSEMOTION and self.panning:
            dy = event.pos[1] - self.pan_anchor[1]
            self.view_offset[1] = self.offset_anchor[1] - dy
            self.view_offset[1] = self._clamp_view_offset()

        # Gestion de l'ui
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.knob_rect.collidepoint(event.pos):
                self.dragging = True
            elif self.pause_button_rect.collidepoint(event.pos):
                self.paused = not self.paused
            elif self.kpi_button_rect.collidepoint(event.pos):
                self.show_kpis = not self.show_kpis
                self._set_kpis_surface()
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x, _ = event.pos
            new_knob_x = max(self.slider_rect.x, min(mouse_x - self.knob_rect.width/2, self.slider_rect.x + self.slider_rect.width - self.knob_rect.width))
            self.knob_rect.x = new_knob_x
            relative_pos = (self.knob_rect.x - self.slider_rect.x) / (self.slider_rect.width - self.knob_rect.width)
            self.current_fps = int(self.cfg.MIN_FPS + relative_pos * (self.cfg.MAX_FPS - self.cfg.MIN_FPS))

    def _step(self, step):
        """Fait avancer la simulation jusqu’à `step` ou marque la pause si fin atteinte.

        Args:
            step (int): dernière période simulée.

        Returns:
            tuple[int, bool]: (nouveau step, état running).
        """
        if self.env.peek() < self.simulation.NUM_PERIOD:
            while self.env.peek() <= step:
                self.simulation.step()
            step = self.env.peek()
        else:
            self.paused = True
        return step

    def _run(self):
        """Lance la boucle principale Pygame : gère les événements, met à jour la simulation et rafraîchit l’affichage.

        Returns:
            bool: False quand la fenêtre est fermée.
        """
        step, running = 1, True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.handle_pygame_event(event)

            if not self.paused:
                step = self._step(step)
            self.fill_surf(self.cfg.WHITE)
            
            self.draw_ports_and_ships() 

            self.draw_ui()

            if self.show_kpis:
                self.draw_kpis()

            self.blit_all_surfaces()
            pygame.display.flip()
             
            self.clock.tick(self.current_fps)
        # --- FIN DE LA BOUCLE WHILE ---
        pygame.quit()
        return False

    def _get_kpis_generator(self):
        paused = self.paused
        self.paused = False # pas de pause pour faire de step dans le kpis
        self._step(1) # do one step to init data
        self.paused = paused # restore pause state
        return LiveKpisGraphsGenerator(
            self.simulation.result,
            self.config,
        )
        
    def _update_data_to_kpis_generator(self):
        self.kpis_generator.upload_data(self.simulation.result)
  
    def _load_kpi_graphs(self):
        self._update_data_to_kpis_generator()
        self.kpis_graphs = []

        for name, updater in self.kpis_generator.graphs.items():
            raw, (w, h) = updater()
            surf = pygame.image.frombuffer(raw, (w, h), "RGBA")
            self.kpis_graphs.append(surf)

    def draw_kpis(self):
        self._load_kpi_graphs()
        height_pos = 0
        for  surf in self.kpis_graphs:
            pos = [0, height_pos]
            height_pos += surf.get_height()
            self.kpis_surf.blit(surf, pos)

    def _set_kpis_surface(self):
        """Recrée la surface KPI avec la hauteur nécessaire selon les graphes chargés."""
        if self.show_kpis:self._load_kpi_graphs()
        else: self.kpis_graphs = []
        height = sum([surf.get_height() for surf in self.kpis_graphs])
        self.kpis_surf = pygame.Surface((self.WIDTH, height), pygame.SRCALPHA)

    def run(self):
        """Point d’entrée public pour démarrer l’animation.

        Returns:
            bool: False quand l’animation se termine (fenêtre fermée).
        """
        return self._run()


if __name__ == "__main__":
    from eco2_normandy.tools import get_simlulation_variable
    # from eco2_normandy.simulation import Simulation
    file_path = "scenarios/dev/phase3_bergen_18k_2boats.yaml" 
    # file_path = "scenarios/dev/phase3_rotterdam_18k_2boats_3tanks.yaml"
    simulation_variables = get_simlulation_variable(file_path)[0]

    anim = PGAnime(simulation_variables, Simulation=Simulation)
    anim.run()
