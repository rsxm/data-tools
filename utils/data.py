# IMPORTS
# ---------------------------------------------------------------------------------------------------------------------
import os
import datetime
import re
import math
import numpy as np
import pandas as pd
import xlrd
from utils.excel import get_column_types
from logging import getLogger


# LOGGING
# ----------------------------------------------------------------------------------------------------------------------
logger = getLogger('j_rep')


# GLOBALS
# ---------------------------------------------------------------------------------------------------------------------
YEAR_2000 = 36526
YEAR_2015 = 42005


# FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------
def to_date(x, year, null_values=('TBA', 'TBC', '-')):
    """Attempt to convert input to datetime, else return pd.NaT

    :param object x:
    :param str|int year:
    :return:
    """

    if x and not pd.isnull(x):
        try:
            x = float(x)
        except (ValueError, TypeError):
            pass

        if isinstance(x, int) or isinstance(x, float):
            if YEAR_2000 < x < YEAR_2015:
                return datetime.datetime(*xlrd.xldate_as_tuple(x, 0))

        elif isinstance(x, str):
            if not x.strip() in null_values:
                d = re.findall('\d+', x)
                s = '-'.join(['{:02d}'.format(int(e)) for e in d])
                s = s.replace('00', '-00')
                try:
                    date_time = datetime.datetime.strptime('{0}-{1}'.format(year, s), '%Y-%d-%m-%H-%M')
                    logger.debug('Fixed {} -> {} -> {}'.format(x, s, date_time))
                    return date_time
                except ValueError:
                    d = re.findall('\d{2}', x)
                    s = '-'.join(['{:02d}'.format(int(e)) for e in d])
                    try:
                        date_time = datetime.datetime.strptime('{0}-{1}'.format(year, s), '%Y-%d-%m-%H-%M')
                        logger.debug('T2:Fixed {} -> {} -> {}'.format(x, s, date_time))
                        return date_time
                    except ValueError:
                        logger.debug('Failed {} -> {}'.format(x, s))

        # TODO Find datetime64 location
        # elif isinstance(x, np.datetime64) or isinstance(x, pd.Timestamp):
        elif isinstance(x, pd.Timestamp):
            return pd.to_datetime(x)

        elif isinstance(x, datetime.datetime):
            return x

        else:
            logger.debug('Unknown type: {}'.format(type(x)))

        logger.debug('Could not parse {} of type {}.'.format(x, type(x)))

    return pd.NaT


def to_int(x):
    """Attempt to convert input to integer, else return 0.00

    """
    try:
        x = to_float(x)
        return int(x)

    except (TypeError, ValueError):
        if x:
            logger.warning('Not an integer: {}'.format(x))
        return None


def to_float(x):
    """Attempt to convert input to float, else return 0.00

    """
    try:
        x = float(x)
        if math.isnan(x):
            x = 0.00
        return x

    except (TypeError, ValueError):
        if x:
            logger.warning('Not a float: {}'.format(x))
        return 0.00


class DataStore:
    def __init__(self, path, store_type, store_name='data_store', freshness=12):
        self.path = path
        self.store_type = store_type
        self.store_name = store_name
        self.freshness = freshness

    def check(self, key):
        """Checks freshness of cache

        :param str key:
        :return bool:
        """
        if self.store_type == 'csv':
            try:
                file_path = os.path.join(self.path, self.store_name, '{}.csv'.format(key))
                last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))

                if (datetime.datetime.now() - last_modified).total_seconds() / 3600 > self.freshness:
                    return False

            except FileNotFoundError:
                logger.warning('Data with key `{}` does not exist!'.format(key))
                return False

        logger.debug('Cache is fresh')
        return True

    def save(self, df, key):
        """Utility function to save data

        :param pandas.DataFrame df: the pandas.DataFrame to be exported
        :param str key: The name of the table, file or sheet
        :return boolean:
        """
        if df is None:
            logger.debug('Not saving NoneType')
            return False
        if df.empty:
            logger.debug('Not saving empty DataFrame')
            return False

        if not df.index.name:
            df.index.name = 'id'

        if self.store_type == 'csv':
            store = os.path.abspath(os.path.join(self.path, self.store_name, '{}.csv'.format(key)))
            os.makedirs(os.path.dirname(store), exist_ok=True)
            try:
                df.to_csv(store, encoding='utf-8', sep=',', index=True)
                logger.debug('Wrote: {}'.format(store))
            except FileNotFoundError:
                logger.exception('Could not save data, could not find:\n    {}'.format(store))
                return False

        elif self.store_type == 'hdf5':
            store = os.path.join(self.path, '.h5'.format(self.store_name))
            df.to_hdf(store, key=key, format='fixed', mode='a', complevel=9, complib='bzip2')

        elif self.store_type == 'excel':
            store = os.path.join(self.path, '.xlsx'.format(self.store_name))
            df.to_excel(store, sheet_name=key)

        logger.debug(
            'Exported data to `{}` using key `{}` and store_type `{}`'.format(self.store_name, key, self.store_type))

        return True

    def load(self, key):
        """Utility function to save data

        :param str key: The name of the table, file or sheet
        :return pandas.DataFrame:
        """
        df = pd.DataFrame()

        if self.store_type == 'csv':
            store = os.path.join(self.path, self.store_name, '{}.csv'.format(key))
            df = pd.read_csv(store, encoding='utf-8', sep=',', index_col=0)

        elif self.store_type == 'hdf5':
            store = os.path.join(self.path, '{}.h5'.format(self.store_name))
            df = pd.read_hdf(store, key=key)

        elif self.store_type == 'excel':
            store = os.path.join(self.path, '{}.xlsx'.format(self.store_name))
            df = pd.read_excel(store, sheet_name=key)

        if not df.empty:
            logger.info(
                'Imported data from `{}` using key `{}` and store_type `{}`'.format(
                    self.store_name, key, self.store_type))

        return df


def post_process(df, strings=None, date_times=None, integers=None, floats=None):
    """

    :param df:
    :param [str] strings:
    :param [str] date_times:
    :param [str] integers:
    :return:
    """

    if strings:
        logger.info('Enforcing string columns...')
        df.loc[:, strings] = df[strings].applymap(str)
    if floats:
        logger.info('Enforcing float columns...')
        df.loc[:, floats] = df[floats].applymap(np.float64)
    if integers:
        logger.info('Enforcing integer columns...')
        df.loc[:, integers] = df[integers].applymap(np.int64)
    if date_times:
        logger.info('Enforcing date_time columns...')
        df.loc[:, date_times] = df.loc[:, date_times].applymap(pd.to_datetime)

    return df


def process_data(df, type_info, mappings):
    logger.debug('Fixing column types..')

    fix_columns = {}
    new_from_mapping = {}
    new_from_concat = {}

    column_types = get_column_types(type_info, mappings['column_type_overrides'])

    for k, v in fix_columns.items():
        df[k] = df.loc[:, k].apply(v)

    for k, v in column_types.items():
        if v == 'int':
            df[k] = df.loc[:, k].apply(to_int)
        if v == 'Decimal' or v == 'float':
            df[k] = df.loc[:, k].apply(to_float)
        if v == 'Datetime':
            df[k] = df.loc[:, k].apply(lambda dt: to_date(dt, 2014))
        if v == 'Empty':
            logger.debug('Dropping empty column: {}'.format(k))
            df.drop(k, axis=1, inplace=True)

    logger.debug('Adding new columns...')
    for k, v in new_from_mapping.items():
        df[k] = df.loc[:, v].map(mappings[k])

    for k, v in new_from_concat.items():
        df[k] = pd.concat([df[column].dropna() for column in v]).reindex_like(df)

    df = mappings['custom_post_processing'](df)

    return df


# TESTS
# ---------------------------------------------------------------------------------------------------------------------
import unittest


class TestCaching(unittest.TestCase):
    def setUp(self):
        self.year = 2014
        self.date_time = datetime.datetime(2014, 9, 11, 6, 0)
        self.data = [
            {
                'string_property': 'Bob',
                'integer_property': 12,
                'date_time_property': datetime.datetime.now(),
                'float_property': 0.4634,
                'boolean_property': False,
            },
            {
                'string_property': 'Amy',
                'integer_property': 9,
                'date_time_property': datetime.datetime.now(),
                'float_property': 11.6384,
                'boolean_property': True,
            },
            {
                'float_property': 11.6384,
                'boolean_property': True,
            },
        ]

        self.store = DataStore(path='/tmp', store_type='csv')

    def test_data_export(self):
        self.assertTrue(self.store.save(df=pd.DataFrame(self.data), key='test'))

    def test_data_import(self):
        df = self.store.load(key='test')
        self.assertTrue(df.shape == pd.DataFrame(self.data).shape)

    def test_data_check(self):
        self.store.save(df=pd.DataFrame(self.data), key='test')
        self.store.freshness = 12
        self.assertTrue(self.store.check(key='test'))
        self.store.freshness = 0
        self.assertFalse(self.store.check(key='test'))

    def test_to_datetime_string(self):
        dt = '11/9 600'
        self.assertEqual(to_date(dt, year=self.year), self.date_time)

    def test_to_datetime_xl_date(self):
        dt = YEAR_2000 + 1
        self.assertEqual(to_date(dt, year=self.year), datetime.datetime(2000, 1, 2))

    def test_to_datetime_xl_date_string(self):
        dt = str(YEAR_2000 + 1)
        self.assertEqual(to_date(dt, year=self.year), datetime.datetime(2000, 1, 2))


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaching)
    unittest.TextTestRunner(verbosity=2).run(suite)
