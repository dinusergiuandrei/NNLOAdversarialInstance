def get_import_targets():
    """
    Returns the packet names to be imported. This method is only intended to be used in the `import_all` function.
    :return: a tuple with 3 dictionaries: package-to-package_local_name (import <numpy> as <np>); function_module-to-function_name (from <scipy.stats> import <pearsonr>); function_name-to-function_local_name.
    """
    packages = {
        'os': 'os',
        'sys': 'sys',
        'numpy': 'np',
        'pandas': 'pd',
        'seaborn': 'sns',
        'matplotlib.pyplot': 'plt',
        'gurobipy':'gp'
    }
    functions = {
        'pprint': 'pprint',
        'scipy.stats': 'pearsonr',
        'scipy.stats.mstats': 'winsorize',
        'IPython.core.interactiveshell': 'InteractiveShell',
        'pathlib': 'Path',
        'datetime': 'datetime',
        'tqdm.auto':'tqdm',
        'utils.nn_utils':['load_model', 'load_data', 'relu']
    }
    return packages, functions


def import_all(g):
    """
    This method imports multiple necessary packages with the purpose of replacing the long list of imports from notebooks.
    :param g: The globals() dictionary called in the script/notebook it is desired to import the packages.
    """
    import importlib

    packages, functions = get_import_targets()

    for package, alias in packages.items():
        g[alias] = importlib.import_module(package)

    for function_path, function_name in functions.items():
        if isinstance(function_name, str):
            function_name = [function_name]
        for f in function_name:
            g[f] = getattr(importlib.import_module(function_path), f)


def setup(g):
    import_all(g)
    g['InteractiveShell'].ast_node_interactivity = 'all'
    g['plt'].style.use('ggplot')
    g['pd'].set_option('display.max_columns', None)
    g['pd'].set_option('display.max_rows', 200)
    g['pd'].set_option('display.width', 1000)
    g['plt'].rcParams['figure.figsize'] = (15, 5)
    g['plt'].rcParams['axes.titlesize'] = 24
    g['plt'].rcParams['axes.labelsize'] = 22
    g['plt'].rcParams['ytick.labelsize'] = 20
    g['plt'].rcParams['xtick.labelsize'] = 20
