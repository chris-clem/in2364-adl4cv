from OSVOS_PyTorch.util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/home/christoph/in2364-adl4cv/DAVIS_2016/DAVIS'

    @staticmethod
    def save_root_dir():
        return '/home/christoph/in2364-adl4cv/OSVOS_PyTorch/models'

    @staticmethod
    def models_dir():
        return "/home/christoph/in2364-adl4cv/OSVOS_PyTorch/models"

