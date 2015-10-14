# IMPORTS
# ---------------------------------------------------------------------------------------------------------------------
import os
import re
import fnmatch
import datetime
import collections
import pathspec
import pandas as pd
from logging import getLogger

# LOGGING
# ---------------------------------------------------------------------------------------------------------------------
logger = getLogger('j_rep')


# FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------
def locate(root_path, file_specs, new_style=False):
    """ Locate files matching file_pattern in root_path

    :param str root_path: root path to start search
    :param str file_specs: glob wildcard matching to filter files

    :rtype : collections.Iterable[str]
    """

    if not new_style:
        for path, dirs, files in os.walk(os.path.abspath(root_path)):
            for filename in fnmatch.filter(files, file_specs):
                yield os.path.join(path, filename)
    else:
        specs = pathspec.PathSpec.from_lines(pathspec.GitIgnorePattern, file_specs.splitlines())
        matches = specs.match_tree(root_path)

        yield from (os.path.join(root_path, match) for match in matches)


def file_stats(file_path, file_path_pattern=None):
    """ Return file information extracted from name or directory

    :param file_path: full path to file
    :param file_path_pattern: re.compile pattern instance

    :rtype : dict
    """
    info = {'file_path': file_path}
    if file_path_pattern:
        e = [m.groupdict() for m in file_path_pattern.finditer(file_path)]
        if e:
            info.update(e[0])

    info.update({
        'size': os.path.getsize(file_path),
        'date_modified': datetime.datetime.fromtimestamp(os.path.getmtime(file_path)),
    })
    return info


def get_files(root_path='./', file_filter='*', file_path_pattern='(?P<name>.+)',
              period_from=None, period_to=None, post_processing=None, unique_index=True):
    """Function to get file information and store in pandas DataFrame

    :param list files:
    :param list unique_stats:
    :param datetime.datetime|string from_date:
    :param datetime.datetime|string to_date:

    :rtype : pandas.DataFrame
    """

    file_path_pattern = re.compile(file_path_pattern)
    file_index = list(file_path_pattern.groupindex.keys())

    files = list(map(lambda x: file_stats(x, file_path_pattern), locate(root_path, file_filter, new_style=True)))

    if not files:
        raise ValueError('No files found!')

    df_files = pd.DataFrame(files)
    file_count = df_files.shape[0]

    logger.debug('Found  {} files with\n\troot_path:\t{}\n\tfile_pattern:\t{}'.format(
        file_count, root_path, file_filter.replace('\n', '; ')))

    # if unique_stats:
    # df = df.sort(columns=unique_stats + ['date_modified', 'size'], ascending=False)
    # df.drop_duplicates(subset=unique_stats, inplace=True)
    # #df[~df.duplicated(subset=['month', 'day', 'hour'])]

    # logger.info('Dropped: {}'.format(file_count - df.shape[0]))

    file_count = df_files.shape[0]
    logger.info('Found {0} files matching criteria'.format(file_count))

    df_files = df_files[df_files['size'] > 0]
    logger.info('Dropped {} empty files'.format(file_count - df_files.shape[0]))

    if period_from:
        df_files = df_files[df_files.date_modified > period_from]
    if period_to:
        df_files = df_files[df_files.date_modified < period_to]

    logger.info('Dropped {} files based on date modified'.format(file_count - df_files.shape[0]))

    if unique_index:
        df_files.sort(columns=['date_modified', 'size'], inplace=True, ascending=False)
        df_files.drop_duplicates(subset=file_index, inplace=True)
        df_files.dropna(subset=file_index, inplace=True)
        df_files.set_index(file_index, inplace=True)
        df_files.sort_index(inplace=True)

    logger.info('Dropped {} duplicates based on index data'.format(file_count - df_files.shape[0]))

    if post_processing:
        df_files = post_processing(df_files)

    df_files.reset_index(inplace=True)
    return df_files


# from pathlib import Path

# def new_locate(base_path, glob):
#     """Advanced file finder
#
#     :param base_path:
#     :param glob:
#     :return:
#     """
#     path = Path(base_path)
#     files = path.glob(glob)
#     return files
#

def _stats(path, extract_pattern=''):
    """

    :param pathlib.Path path:
    :param str extract_pattern:
    :return:
    """
    _os_stats = path.stat()
    _path = str(path)

    stats = {
        'path': _path,
        'size': _os_stats.st_size,
        'date_modified': datetime.datetime.fromtimestamp(_os_stats.st_mtime),
        'date_created': datetime.datetime.fromtimestamp(_os_stats.st_mtime),
    }

    _extract_pattern = re.compile(extract_pattern)

    if extract_pattern:
        data = [m.groupdict() for m in _extract_pattern.finditer(_path)]
        if data:
            stats.update(data[0])

    return stats


import tempfile
import unittest


class TestFiles(unittest.TestCase):

    def setUp(self):
        self.root_path = os.path.join(tempfile.gettempdir(), 'Data')
        self.files = [
            'Workbook_1.xlsx',
            'Workbook_2.xlsx',
            'Folder_1/CSV_File_11.csv',
            'Folder_1/JSON_File_11.json',
            'Folder_1/SubFolder_11/CSV_File_111.csv',
            'Folder_1/SubFolder_11/Workbook_111.xlsx',
            'Folder_2/TextFile_21.txt',
            'Folder_2/SubFolder_21/TextFile_211.txt',
        ]

        for file in self.files:
            file_path = os.path.join(self.root_path, file)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(b'Some content!')

    def test_locate_patterns(self):
        # file_patterns = ['*', '.xlsx', '[C|J]*.*']
        cases = [('*', 8), ('*.csv', 2), ('[C|J]*.*', 3), ('*_[2-9][1-9]*', 2)]

        for case in cases:
            with self.subTest('Search All', case=case):
                files = list(locate(self.root_path, file_specs=case[0]))
                self.assertEqual(len(files), case[1])

    def test_get_files(self):

        # file_patterns = ['*', '.xlsx', '[C|J]*.*']
        cases = [('*', 8), ('*.csv', 2), ('[C|J]*.*', 3), ('*_[2-9][1-9]*', 2)]

        for case in cases:
            with self.subTest('Search All', case=case):
                files = get_files(self.root_path, file_filter=case[0])
                self.assertEqual(files.shape[0], case[1])


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFiles)
    unittest.TextTestRunner(verbosity=2).run(suite)
