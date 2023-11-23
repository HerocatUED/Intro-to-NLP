import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logfile = './log.txt'
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


def save_log(train: bool = True, cnt: int = 0, f1: float = 0, ac: float = 0):
    if train:
        logger.info('Train |epoch:{epoch}, macro-F1:{f1}, accuracy:{ac}'.format(
            epoch=cnt, f1=f1, ac=ac))
    else:
        logger.info('Test: macro-F1:{f1}, accuracy:{ac}'.format(f1=f1, ac=ac))
