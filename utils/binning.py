# -*- coding: utf-8 -*-
"""
Lib to do downtime analysis

Adapted from [HSE_LAB](http://hselab.org/content/occupancy-analysis-python-pandas-part-1-create-date-data-frame

"""

# LOGGING
# ---------------------------------------------------------------------------------------------------------------------
import logging

logger = logging.getLogger('j_rep')
logger.setLevel(logging.DEBUG)


# IMPORTS
# ---------------------------------------------------------------------------------------------------------------------
import math
import time
import numpy as np
import datetime
import pandas as pd
from dateutil.parser import parse


# FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------
def td_to_minutes(x):
    """
    Converts a timedelta object to minutes
    """
    return x.days * 24.0 * 60 + x.seconds / 60.0 + x.microseconds / 6000.0


vtd_to_minutes = np.vectorize(td_to_minutes)  # Make usable with list like things


def day_bin(dt, bin_size):
    """
    Computes bins of day based on bins size for a datetime.

    Parameters
    ----------
    dt : datetime.datetime object, default now.
    bin_size : Size of bins in minutes; default 30 minutes.

    Returns
    -------
    0 to (n-1) where n is number of bins per day.

    Examples
    --------
    dt = datetime(2013,1,7,1,45)
    bins = day_bin(dt,30)
    # bins = 3

    """
    if dt is None:
        dt = datetime.datetime.now()

    if not isinstance(dt, datetime.datetime):
        dt = pd.Timestamp(dt)

    minutes = (dt.hour * 60) + dt.minute
    bins = math.trunc(minutes / bin_size)

    return bins


v_day_bin = np.vectorize(day_bin)


def week_bin(dt, bin_size):
    """
    Computes dt_bin of week based on dt_bin size for a datetime.

    Based on .weekday() convention of 0=Monday.

    Parameters
    ----------
    dt : datetime.datetime object, default now.
    bin_size : Size of dt_bin in minutes; default 60 minutes.

    Returns
    -------
    0 to (n-1) where n is number of bins per week.

    Examples
    --------
    dt = datetime(2013,1,7,1,45)
    dt_bin = week_bin(dt,30)
    # dt_bin = ???

    """
    if dt is None:
        dt = datetime.datetime.now()

    if not isinstance(dt, datetime.datetime):
        dt = pd.Timestamp(dt)

    minutes = (dt.weekday() * 1440) + (dt.hour * 60) + dt.minute
    bins = math.trunc(minutes / bin_size)

    return bins


v_week_bin = np.vectorize(week_bin)


def round_down_time(dt, round_minutes_to, shift=5):
    """
   Find floor of a datetime object to specified number of minutes.

   dt : datetime.datetime object
   round_minutes_to : Closest number of minutes to round to.
   """
    dt = dt + datetime.timedelta(0, shift * 60 * 60)

    round_seconds_to = round_minutes_to * 60
    seconds = (dt - dt.min).seconds

    floor_time = seconds // round_seconds_to * round_seconds_to

    return dt + datetime.timedelta(0, floor_time - seconds, -dt.microsecond) - datetime.timedelta(0, shift * 60 * 60)


v_round_down_time = np.vectorize(round_down_time)


def round_up_time(dt, round_minutes_to, shift=5):
    """
   Find ceiling of a datetime object to specified number of minutes

   dt : datetime.datetime object
   round_minutes_to : Closest number of minutes to round to.
   """
    dt = dt + datetime.timedelta(0, shift * 60 * 60)

    round_seconds_to = round_minutes_to * 60.0
    seconds = (dt - dt.min).seconds

    ceiling_time = math.ceil(seconds / round_seconds_to) * round_seconds_to

    return dt + datetime.timedelta(0, ceiling_time - seconds, -dt.microsecond) - datetime.timedelta(0, shift * 60 * 60)


v_roundup_time = np.vectorize(round_up_time)


def is_greater_two_bins(bin_left, bin_right, bin_size):
    return (bin_right - bin_left) > datetime.timedelta(minutes=bin_size)


def downtime_fraction(stop_record_range, bin_size, rectype):
    """
    Computes fractional downtime in bin_left and bin_right.

    Parameters
    ----------
    stop_record_range: list consisting of [record_from, record_to]
    bin_size: dt_bin size in minutes
    rectype: One of 'inner', 'outer', 'left', 'right'. See
             stop_record_analysis_relationship() doc for details.

    Returns
    -------
    [bin_left_downtime_fraction, bin_right_downtime_fraction] where each is a real number in [0.0,1.0]

    """
    record_from, record_to = stop_record_range

    if rectype:
        pass

    bin_left = round_down_time(record_from, bin_size)
    bin_right = round_down_time(record_to, bin_size)

    # bin_left occupancy
    right_edge = min(bin_left + datetime.timedelta(minutes=bin_size), record_to)
    bin_left_downtime_seconds = (right_edge - record_from).seconds
    bin_left_downtime_fraction = bin_left_downtime_seconds / (bin_size * 60.0)

    # bin_right occupancy
    if bin_left == bin_right:
        bin_right_downtime_fraction = 0.0  # Use bin_left_downtime_fraction
    else:
        left_edge = max(bin_right, record_from)
        bin_right_downtime_seconds = (record_to - left_edge).seconds
        bin_right_downtime_fraction = bin_right_downtime_seconds / (bin_size * 60.0)

    assert 1.0 >= bin_left_downtime_fraction >= 0.0, \
        "bad bin_left_downtime_fraction={:.3f} in={} out={}".format(
            bin_left_downtime_fraction, record_from, record_to)

    assert 1.0 >= bin_right_downtime_fraction >= 0.0, \
        "bad bin_right_downtime_fraction={:.3f} in={} out={}".format(
            bin_right_downtime_fraction, record_from, record_to)

    if is_greater_two_bins(bin_left, bin_right, bin_size):
        rt_rf = (record_to - record_from).total_seconds()
        rt_br = (record_to - bin_right).total_seconds()
        rf_bl = bin_size * 60.0 - (record_from - bin_left).total_seconds()

        full_bins = (rt_rf - rt_br - rf_bl) / (bin_size * 60.0)

        # logger.info('\n\nRecord range:\t{} -- {}\nBin range:\t{} -- {}\n {}\n'.format(
        # record_from, record_to, bin_left, bin_right, bin_size)
        # )

        # logger.info('{} - {} - {} = {}'.format(rt_rf, rt_br, rf_bl, full_bins))

    else:
        full_bins = 0

    return [bin_left_downtime_fraction, bin_right_downtime_fraction, int(full_bins)]


def stop_record_analysis_relationship(stop_record_range, analysis_range):
    """
    Determines relationship type of stop record to analysis date range.

    Parameters
    ----------
    stop_record_range: list consisting of [record_from, record_to]
    analysis_range: list consisting of [analysis_start, analysis_end]

    Returns
    -------
    Returns a string, either 'inner', 'left', 'right, 'outer',
    'backwards', 'none' depending
    on the relationship between the stop record being analyzed and the
    analysis date range.

    |---|---|---|---|---|---|---|---|---|---|---|---\---|---|  ~ time bins

    Type 'inner':

             |-------------------------|                       ~ analysis
                    |--------------|                           ~ record

    Type 'left':
                             analysis
                    |-------------------------|
              |--------------|
                  record

    Type 'right':

                    |-------------------------|
            analysis_start            analysis_end
                                       |--------------|
                                 record_from    record_to

    Type 'outer':
                        analysis
              |-------------------------|
       |-------------------------------------|
                    record


    Type 'backwards':
        The exit time is BEFORE the entry time. This is a BAD record.

    Type 'none':
        Ranges do not overlap
    """
    record_from, record_to = stop_record_range
    analysis_start, analysis_end = analysis_range

    if record_from > record_to:
        return 'backwards'
    elif (analysis_start <= record_from < analysis_end) and (analysis_start <= record_to < analysis_end):
        return 'inner'
    elif (analysis_start <= record_from < analysis_end) and (record_to >= analysis_end):
        return 'right'
    elif (record_from < analysis_start) and (analysis_start <= record_to < analysis_end):
        return 'left'
    elif (record_from < analysis_start) and (record_to >= analysis_end):
        return 'outer'
    else:
        return 'none'


def build_skeleton(df, bin_size, analysis_start_dt, analysis_end_dt, category_field_name):
    """Creates skeleton pandas.DataFrame for analysing downtime

    :param pandas.DataFrame df:
    :param int bin_size:
    :param datetime.datetime analysis_start_dt:
    :param datetime.datetime analysis_end_dt:
    :param str category_field_name:
    :param str from_field_name:
    :param str to_field_name:
    :return:
    """
    logger.info('Building skeleton DataFrame...')
    bin_freq = str(bin_size) + 'min'
    range_by_date = pd.date_range(analysis_start_dt, analysis_end_dt,
                                  freq=bin_freq).to_pydatetime()

    categories = [c for c in df[category_field_name].unique()]

    columns = ['category', 'datetime', 'off-line', 'on-line', 'downtime']

    # Create an empty ByDate data frame
    df_by_date = pd.DataFrame(columns=columns)

    len_by_date = len(range_by_date)
    for category in categories:
        by_date_data = {
            'category': [category] * len_by_date,
            'datetime': range_by_date,
            'off-line': [0.0] * len_by_date,
            'on-line': [0.0] * len_by_date,
            'downtime': [0.0] * len_by_date,
        }
        by_date_df_cat = pd.DataFrame(by_date_data, columns=columns)
        df_by_date = pd.concat([df_by_date, by_date_df_cat])

    df_by_date['dayofweek'] = df_by_date['datetime'].map(lambda x: x.weekday())
    df_by_date['day_bin'] = df_by_date['datetime'].map(lambda x: day_bin(x, bin_size))
    df_by_date['week_bin'] = df_by_date['datetime'].map(lambda x: week_bin(x, bin_size))
    df_by_date = df_by_date.set_index(['category', 'datetime'], drop=False)

    logger.info('Done: {}'.format(df_by_date.shape))

    return df_by_date


def bin_stop_record(df_by_date, analysis_range, bin_size, by_date_record):
    """

    :param df_by_date:
    :param analysis_range:
    :param bin_size:
    :param by_date_record:

    :return:
    """
    record_from, record_to, category = by_date_record
    bin_left = round_down_time(record_from, bin_size)
    bin_right = round_down_time(record_to, bin_size)
    rectype = stop_record_analysis_relationship([record_from, record_to], analysis_range)

    # print "{} {} {} {} {:.3f} {:.3f} {:.3f}".format(record_from, record_to, category,
    # rectype, time.clock(), from_to_downtime_fraction[0], from_to_downtime_fraction[1])

    if rectype == 'backwards':
        logger.warning('{}: {} {} {}'.format(rectype, record_from, record_to, category))

    elif rectype != 'none':

        # tt = 0
        from_to_downtime_fraction = downtime_fraction([record_from, record_to], bin_size, rectype)

        if rectype == 'inner':
            df_by_date.ix[(category, pd.Timestamp(bin_left)), 'downtime'] = from_to_downtime_fraction[0]
            df_by_date.ix[(category, pd.Timestamp(bin_right)), 'downtime'] = from_to_downtime_fraction[1]
            df_by_date.ix[(category, pd.Timestamp(bin_left)), 'off-line'] += 1.0
            df_by_date.ix[(category, pd.Timestamp(bin_right)), 'on-line'] += 1.0

            bin_dt = bin_left
            for i in range(from_to_downtime_fraction[2]):
                bin_dt = bin_dt + datetime.timedelta(minutes=bin_size)
                df_by_date.ix[(category, pd.Timestamp(bin_dt)), 'downtime'] = 1.0

                # if is_greater_two_bins(bin_left, bin_right, bin_size):

                # for i in range(from_to_downtime_fraction[0])

                #    dt_bin = bin_left + datetime.timedelta(minutes=bin_size)
                #    while dt_bin < bin_right:
                #        df_by_date.ix[(category, pd.Timestamp(dt_bin)), 'downtime'] += 1.0
                #        dt_bin += datetime.timedelta(minutes=bin_size)

                # logger.debug("{} {} {} {} {} {:.3f} {:.3f}".format(record_from, record_to, gt2bins, category, rectype,
                #                                                   from_to_downtime_fraction[0],
                #                                                   from_to_downtime_fraction[1],))

        elif rectype == 'right':
            # to is outside analysis window
            df_by_date.ix[(category, bin_left), 'downtime'] = from_to_downtime_fraction[0]
            df_by_date.ix[(category, bin_left), 'off-line'] += 1.0

            bin_dt = bin_left
            for i in range(from_to_downtime_fraction[2]):
                bin_dt = bin_dt + datetime.timedelta(minutes=bin_size)
                df_by_date.ix[(category, pd.Timestamp(bin_dt)), 'downtime'] = 1.0

                # if is_greater_two_bins(bin_left, bin_right, bin_size):
                #    dt_bin = bin_left + datetime.timedelta(minutes=bin_size)
                #    while dt_bin <= analysis_range[1]:
                #        df_by_date.ix[(category, pd.Timestamp(dt_bin)), 'downtime'] += 1.0
                #        dt_bin += datetime.timedelta(minutes=bin_size)

        elif rectype == 'left':
            # start is outside analysis window
            df_by_date.ix[(category, bin_right), 'downtime'] = from_to_downtime_fraction[1]
            df_by_date.ix[(category, bin_right), 'on-line'] += 1.0

            bin_dt = bin_left
            for i in range(from_to_downtime_fraction[2]):
                bin_dt = bin_dt + datetime.timedelta(minutes=bin_size)
                df_by_date.ix[(category, pd.Timestamp(bin_dt)), 'downtime'] = 1.0

                # if is_greater_two_bins(bin_left, bin_right, bin_size):
                #    dt_bin = analysis_range[0] + datetime.timedelta(minutes=bin_size)
                #    while dt_bin < bin_right:
                #        df_by_date.ix[(category, pd.Timestamp(dt_bin)), 'downtime'] += 1.0
                #        dt_bin += datetime.timedelta(minutes=bin_size)

        elif rectype == 'outer':
            # start and end sandwich analysis window

            bin_dt = bin_left
            for i in range(from_to_downtime_fraction[2]):
                bin_dt = bin_dt + datetime.timedelta(minutes=bin_size)
                df_by_date.ix[(category, pd.Timestamp(bin_dt)), 'downtime'] = 1.0

                # if is_greater_two_bins(bin_left, bin_right, bin_size):
                #    dt_bin = analysis_range[0]
                #    while dt_bin <= analysis_range[1]:
                #        df_by_date.ix[(category, pd.Timestamp(dt_bin)), 'downtime'] += 1.0
                #        dt_bin += datetime.timedelta(minutes=bin_size)

        else:
            pass

        return True


def time_span_to_time_bins(df, from_field_name, to_field_name, category_field_name,
                           analysis_start, analysis_end, bin_size):
    """

    :param pandas.DataFrame df: 
    :param str from_field_name:
    :param str to_field_name:
    :param str category_field_name:
    :param int bin_size: 
    :param str analysis_start:
    :param str analysis_end:
    """

    analysis_start_dt = parse(analysis_start)
    analysis_end_dt = parse(analysis_end)
    analysis_range = (analysis_start_dt, analysis_end_dt)
    logger.info('Analysing range: {}'.format(analysis_range))

    df_by_date = build_skeleton(df, bin_size, analysis_start_dt, analysis_end_dt, category_field_name)

    # return df_by_date

    # # Compute LOS - the results is a timedelta value
    df['DowntimeTimeDelta'] = df[to_field_name] - df[from_field_name]
    df['Downtime'] = (df.DowntimeEnd - df.DowntimeStart).astype('timedelta64[m]')

    by_date_records = zip(
        df[from_field_name].astype(datetime.datetime),
        df[to_field_name].astype(datetime.datetime),
        df[category_field_name]
    )

    # MAIN PROCESSING LOOP
    for by_date_record in by_date_records:
        if by_date_record[0] > datetime.datetime(2013, 8, 1) and by_date_record[1] > datetime.datetime(2013, 8, 1):
            bin_stop_record(df_by_date, analysis_range, bin_size, by_date_record)
        else:
            logger.warning('Bad date!: {}'.format(by_date_record))

    logger.info("Done processing stop recs: {}".format(time.clock()))
    return df_by_date


def benchmark():
    index = pd.date_range(start='2013-12-30 7:00', end='2014-12-31 19:00', freq='12H')
    df = pd.DataFrame(data=[1] * len(index), index=index, columns=['one', ])
    df.resample('M', how='sum')
    return df


import unittest


class TestDowntimeMagic(unittest.TestCase):
    def setUp(self):

        self.bin_size = 60 * 12  # bin_size in minutes

        self.base = datetime.datetime(2014, 10, 1, 13)
        self.delta = datetime.timedelta(0, 1 * 60 * 60)  # 1 hours

        self.analysis_range = [self.base - self.delta * 10, self.base + self.delta * 10]

        self.record_range = {
            'inner': [self.base - self.delta, self.base + self.delta],
            'outer': [self.base - self.delta * 20, self.base + self.delta * 20],
            'right': [self.base - self.delta, self.base + self.delta * 12],
            'left': [self.base - self.delta * 12, self.base + self.delta],
            'backwards': [self.base + self.delta, self.base - self.delta],
            'none': [self.base - self.delta * 12, self.base - self.delta * 11]}

        self.inner_record_range = {
            'zero_range_on_left_edge': ([self.base + self.delta * 6, self.base + self.delta * 6], [0, 0, 0]),
            'zero_range_on_right_edge': ([self.base - self.delta * 6, self.base - self.delta * 6], [0, 0, 0]),
            'single_bin_third': ([self.base, self.base + self.delta * 4], [1 / 3, 0, 0]),
            'single_bin_half_to_right': ([self.base, self.base + self.delta * 6], [1 / 2, 0, 0]),
            'single_bin_full': ([self.base - self.delta * 6, self.base + self.delta * 6], [1, 0, 0]),
            'two_bin_half_half': ([self.base, self.base + self.delta * 12], [0.5, 0.5, 0]),
            'two_bin_full_half': ([self.base - self.delta * 6, self.base + self.delta * 12], [1, 0.5, 0]),
            'two_bin_half_full': ([self.base, self.base + self.delta * 18], [0.5, 0, 1]),
            'two_bin_full_full': ([self.base - self.delta * 6, self.base + self.delta * 18], [1, 0, 1]),
            'mutli_bin_half_third': ([self.base, self.base + self.delta * 22], [0.5, 1 / 3, 1]),
            'mutli_bin_third_half': ([self.base + self.delta * 2, self.base + self.delta * 24], [1 / 3, 0.5, 1]),

            # 'mutli_bin_half_half': ([self.base, self.base + self.delta * 86], [0.5, 0.5, 1]),

        }

        self.bin_left = datetime.datetime(2014, 1, 11, 7, 0, 0)
        self.bin_right = self.bin_left + datetime.timedelta(0, self.bin_size * 60)

    def test_stop_record_analysis_relationship(self):
        for key, value in self.record_range.items():
            self.assertEqual(key, stop_record_analysis_relationship(value, self.analysis_range))

    def test_round_down_time(self):

        # bin_left tests
        self.assertEqual(round_down_time(self.bin_left + self.delta, self.bin_size), self.bin_left)
        self.assertEqual(
            round_down_time(self.bin_left - self.delta, self.bin_size), self.bin_right - datetime.timedelta(1))

        # bin_right tests
        self.assertEqual(round_down_time(self.bin_right + self.delta, self.bin_size), self.bin_right)
        self.assertEqual(round_down_time(self.bin_right - self.delta, self.bin_size), self.bin_left)

        # bin_left = 07:00, bin_size = 12h, round_down_time(record_end) = 07:00 (not 19:00)
        self.assertEqual(round_down_time(self.bin_right, self.bin_size), self.bin_right)
        # bin_left = 07:00, record_end = 07:00, round_down_time(record_end) = 07:00 (not 19:00)
        # self.assertEqual(round_down_time(self.bin_left, self.bin_size), self.bin_right - datetime.timedelta(1))
        self.assertEqual(round_down_time(self.bin_left, self.bin_size), self.bin_left)

    def test_round_up_time(self):

        self.assertEqual(
            round_up_time(self.bin_left + self.delta, self.bin_size), self.bin_right)

        self.assertEqual(
            round_up_time(self.bin_left - self.delta, self.bin_size), self.bin_left)

        self.assertEqual(
            round_up_time(self.bin_right + self.delta, self.bin_size), self.bin_left + datetime.timedelta(1))

        self.assertEqual(
            round_up_time(self.bin_right - self.delta, self.bin_size), self.bin_right)

    def test_downtime_fraction(self):
        # logger.info('\n')
        for key, values in self.inner_record_range.items():
            # logger.info(key)
            self.assertEqual(
                downtime_fraction(values[0], self.bin_size, 'inner'),
                values[1])

    def test_is_greater_two_bins(self):
        bl = datetime.datetime(2014, 10, 1, 7)
        br = datetime.datetime(2014, 10, 1, 7)
        self.assertFalse(is_greater_two_bins(bl, br, self.bin_size))

        bl = datetime.datetime(2014, 10, 1, 7)
        br = datetime.datetime(2014, 10, 1, 19)
        self.assertFalse(is_greater_two_bins(bl, br, self.bin_size))

        bl = datetime.datetime(2014, 10, 1, 7)
        br = datetime.datetime(2014, 10, 2, 7)
        self.assertTrue(is_greater_two_bins(bl, br, self.bin_size))


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDowntimeMagic)
    unittest.TextTestRunner(verbosity=2).run(suite)

    # run_tests()

    # dt = datetime.datetime(2014, 10, 1, 13, 0)
    # dt + datetime.timedelta(0, 24*60*60)