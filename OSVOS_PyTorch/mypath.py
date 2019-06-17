from OSVOS_PyTorch.util.path_abstract import PathAbstract
import os

class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        basedirname = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(basedirname, 'DAVIS_2016/DAVIS')

    @staticmethod
    def save_root_dir():
        basedirname = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(basedirname, 'OSVOS_PyTorch/models')

    @staticmethod
    def models_dir():
        basedirname = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(basedirname, 'OSVOS_PyTorch/models')