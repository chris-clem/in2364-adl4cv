from util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/home/max/in2364-adl4cv/DAVIS_2016/DAVIS'

    @staticmethod
    def save_root_dir():
        return './models'

    @staticmethod
    def models_dir():
        return "./models"

