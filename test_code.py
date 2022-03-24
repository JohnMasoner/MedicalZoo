from transforms import common
import configparser
config = configparser.ConfigParser()
config.read('config/sample.ini')
a = common.transform(config)
print(a)
