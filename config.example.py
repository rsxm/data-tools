# IMPORTS
# ---------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd


# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def custom_post_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add code here to custom process columns
    :param df:
    :return:
    """
    return df


def custom_report(df: pd.DataFrame) -> bool:
    """
    Add code here to custom process columns
    :param df:
    :return:
    """
    return True


# CONFIG
# ---------------------------------------------------------------------------------------------------------------------
IGNORE = """
*{keyword}*.xls[x|m|]
!.*
!._*
!~*
!Templates/
!Do not use/
"""

CONFIG = {
    'data_set_name': 'PTP QC Planned Work',
    'data_path': os.path.join(os.getcwd(), '../Exports/'),
    'meta_range': 'E6:E6',
    'file_config': dict(
        root_path='/Volumes/Data/Temp/PTP2015/QC/',
        file_filter=IGNORE.format(keyword='QC-Overall'),
        file_path_pattern=r'^Week\s+(?P<week>\d+).+$',
        period_from=None,
        period_to=None,
        post_processing=None,
        unique_index=False,
    ),
    'sheet_config': dict(
        sheet_filter='Planned Work',
        table_start_keyword='RECORD',
        header_offset=1,
        header_rows=1,
        table_width=13,
    ),
    'special': False,
    'mappings': {
        'columns': [
            'EQ. ID.',
            'DONE BY',
            'PLANNED / SCHEDULED',
            'U1',
            'ACTUAL START',
            'JOB / TASK DESCRIPTION',
            'JOB STATUS',
            'ACTUAL TIME COMPLETION',
            'DOWNTIME      (IN HOUR)',
            'JOB CODE',
            'WHARF MARK',
            'REMARK',
            'W. REQUEST & W. ORDER NO.',
        ],
        'column_type_overrides': {
            'DOWNTIME      (IN HOUR)': 'float',
        },
        'custom_post_processing': custom_post_processing,
    },
    'report_function': custom_report,
}

