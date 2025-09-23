

from optimizer.CPModel.utils import dominates
data = []

front = []
for metrics in data:
  non_dom = []
  for entry in front:
      if dominates(entry, metrics):  # Si une entry est dominé par la nouvelle metrics → on ne l'ajoute pas
        break  # on garde les meme solutions -> on quitte
      if not dominates(metrics, entry):
          non_dom.append(entry)  # ni l'une ni l'autre ne domine → on garde entry
  # on n'a pas quitté → la nouvelle solution est non dominée
  non_dom.append(metrics)
  front = non_dom



"""
$ poetry run python GUI/PGAnime.py 
C:\Users\doria\AppData\Local\pypoetry\Cache\virtualenvs\eco2-normandy-uzCgPYJL-py3.13\Lib\site-packages\pygame\pkgdata.py:25: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_stream, resource_exists
pygame 2.6.1 (SDL 2.28.4, Python 3.13.7)
Hello from the pygame community. https://www.pygame.org/contribute.html
Traceback (most recent call last):
  File "C:\Users\doria\Documents\CODE\repo-git\ENSAM_SAR_CO2\GUI\PGAnime.py", line 991, in <module>
    anim = PGAnime(simulation_variables, Simulation=Simulation)
  File "C:\Users\doria\Documents\CODE\repo-git\ENSAM_SAR_CO2\GUI\PGAnime.py", line 92, in __init__
    self._init()
    ~~~~~~~~~~^^
  File "C:\Users\doria\Documents\CODE\repo-git\ENSAM_SAR_CO2\GUI\PGAnime.py", line 146, in _init
    self.kpis_generator = self._get_kpis_generator()
                          ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\doria\Documents\CODE\repo-git\ENSAM_SAR_CO2\GUI\PGAnime.py", line 733, in _get_kpis_generator
    return LiveKpisGraphsGenerator(
        self.simulation.result,
        self.config,
    )
  File "C:\Users\doria\Documents\CODE\repo-git\ENSAM_SAR_CO2\KPIS\LiveKpisGraphsGenerator.py", line 54, in __init__
    self._init_graphs()
    ~~~~~~~~~~~~~~~~~^^
  File "C:\Users\doria\Documents\CODE\repo-git\ENSAM_SAR_CO2\KPIS\LiveKpisGraphsGenerator.py", line 120, in _init_graphs
    plot_factory_capacity_evolution = self._init_plot_factory_capacity_evolution(figsize)
  File "C:\Users\doria\Documents\CODE\repo-git\ENSAM_SAR_CO2\KPIS\LiveKpisGraphsGenerator.py", line 339, in _init_plot_factory_capacity_evolution
    x, y, capa_max = self._get_data_factory_capacity_evolution(init)
                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Users\doria\Documents\CODE\repo-git\ENSAM_SAR_CO2\KPIS\LiveKpisGraphsGenerator.py", line 320, in _get_data_factory_capacity_evolution
    df = self.dfs[self.factory_name]
         ~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "C:\Users\doria\AppData\Local\pypoetry\Cache\virtualenvs\eco2-normandy-uzCgPYJL-py3.13\Lib\site-packages\pandas\core\frame.py", line 4106, in __getitem__
    return self._getitem_multilevel(key)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
  File "C:\Users\doria\AppData\Local\pypoetry\Cache\virtualenvs\eco2-normandy-uzCgPYJL-py3.13\Lib\site-packages\pandas\core\frame.py", line 4164, in _getitem_multilevel
    loc = self.columns.get_loc(key)
  File "C:\Users\doria\AppData\Local\pypoetry\Cache\virtualenvs\eco2-normandy-uzCgPYJL-py3.13\Lib\site-packages\pandas\core\indexes\multi.py", line 3059, in get_loc
    loc = self._get_level_indexer(key, level=0)
  File "C:\Users\doria\AppData\Local\pypoetry\Cache\virtualenvs\eco2-normandy-uzCgPYJL-py3.13\Lib\site-packages\pandas\core\indexes\multi.py", line 3437, in _get_level_indexer
    raise KeyError(key)
KeyError: 'Le Havre'    
"""