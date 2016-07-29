# -*- coding: utf-8 -*-
"""
Script to extract data from Excel Workbooks.

@author = 'JR Minnaar <jr.minnaar@gmail.com>'
@updated = '2015-09-14 9:43'
@version = 'v0.5'
"""

# IMPORTS
# ---------------------------------------------------------------------------------------------------------------------
import importlib
import logging

from utils.files import get_files
from utils.data import DataStore, process_data
from utils.excel import get_tables, get_excel_data


# LOGGING
# ---------------------------------------------------------------------------------------------------------------------
logger = logging.getLogger('j_rep')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# log = os.path.join(os.getcwd(), '../Logs/log.txt')
# handler = logging.FileHandler(log, mode='a')
# handler.setFormatter(formatter)
# logger.addHandler(handler)


# MAIN
# ---------------------------------------------------------------------------------------------------------------------
def main(config: dict, data_set_name: str):
    development_mode = False

    logger.info(config['data_path'])
    store = DataStore(path=config['data_path'], store_type='csv', store_name=data_set_name, freshness=0)

    logger.debug('Start data extraction process...')
    if not store.check(key='data'):
        if not development_mode or not store.check(key='raw_data'):
            if not store.check(key='tables'):
                if not store.check(key='files'):
                    logger.debug('Retrieving all files...')
                    files = get_files(**config['file_config'])
                    saved = store.save(files, key='files')
                    if not saved:
                        logger.warning('No files found. Check logs.')
                        return False
                else:
                    files = store.load(key='files')
                logger.debug('Retrieving all tables...')
                tables = get_tables(files, **config['sheet_config'])
                saved = store.save(tables, key='tables')
                if not saved:
                    logger.warning('No files could be opened. Check logs.')
                    return False
            else:
                tables = store.load(key='tables')
            logger.debug('Retrieving raw data...')
            raw_data = get_excel_data(tables, config['sheet_config'], config['mappings'], config['meta_range'])

            if development_mode:
                store.save(raw_data['values'], key='raw_data')
                store.save(raw_data['type_info'], key='type_info')
        else:
            raw_data = {
                'values': store.load(key='raw_data'),
                'type_info': store.load(key='type_info'),
            }
        logger.debug('Processing raw data...')
        data = process_data(raw_data['values'], raw_data['type_info'], config['mappings'])
        store.save(data, key='data')
    else:
        data = store.load(key='data')

    logger.debug('Done!')

    if config['special']:
        try:
            logger.info('Writing the Excel Report for %s...' % data_set_name)
            config['report_function'](data, data_set_name)
        except:
            logger.exception('Writing the Excel Report failed:')

    return data


if __name__ == '__main__':
    logger.critical('START')
    logger.info('-------------------------------------------------')
    # for data_set in ['recrush', 'mtp', 'red_area']:
    for data_set in ['questions']:
        logger.info('New data set: %s' % data_set)
        logger.info('-------------------------------------------------')
        module = importlib.import_module('config.' + data_set)

        df_out = main(module.CONFIG, data_set)

        logger.info('Done!')
        logger.info('-------------------------------------------------')
    logger.critical('END')
