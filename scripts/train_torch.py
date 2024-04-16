import yaml

def merge_dictionaries_recursively(dict1, dict2):
  if dict2 is None: return

  for k, v in dict2.items():
    if k not in dict1:
      dict1[k] = dict()
    if isinstance(v, dict):
      merge_dictionaries_recursively(dict1[k], v)
    else:
      dict1[k] = v
class Config(object):
    def __init__(self, config_path = 'config.yaml', default_path=None):
        with open(config_path) as cf_file:
            cfg = yaml.safe_load(cf_file.read())
        if default_path is not None:
            with open(default_path) as def_cf_file:
                default_cfg = yaml.safe_load(def_cf_file.read())

            merge_dictionaries_recursively(default_cfg, cfg)

        self._data = cfg

    def get(self, path=None, default=None):
        # we need to deep-copy self._data to avoid over-writing its data
        sub_dict = dict(self._data)

        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)

            value = sub_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default



if __name__ == '__main__':

    config = Config(config_path='config.yaml')
    print(config.get())

