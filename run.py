from step import train
import sys
import configparser

config = configparser.ConfigParser()
config.read(sys.argv[1])

if __name__ == '__main__':
    train.trainer(config)
