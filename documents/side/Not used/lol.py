import logging
import logging.handlers
import logging
def main():

    logger = logging.getLogger()
    logpath = r'C:\Users\sofu0\PycharmProjects\BACHELOR-ITU-2022\models\modellog.log'

    logging.basicConfig(filename=logpath, encoding='utf-8', level=logging.DEBUG, filemode='a', format="%(message)s")
    logging.debug('This message should go to the log file')
    logging.info('So should this')
    logging.warning('fisk')
    logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
    logger.setLevel(logging.INFO)


if __name__ == "__main__":
    main()
    # logger = logging.getLogger()
    # fh = logging.handlers.RotatingFileHandler('./logtest.log', maxBytes=10240, backupCount=5)
    # fh.setLevel(logging.DEBUG)#no matter what level I set here
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.info('INFO')
    # logger.error('ERROR')