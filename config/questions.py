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
    df.subcategory.fillna(method='ffill', inplace=True)

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
    'data_set_name': 'Assessment Questions',
    'data_path': os.path.join(os.getcwd(), 'export/'),
    'meta_range': None,
    'file_config': dict(
        root_path='./data',
        file_filter=IGNORE.format(keyword=''),
        file_path_pattern=r'^./data/(?P<category>.+)\.xlsx$',
        period_from=None,
        period_to=None,
        post_processing=None,
        unique_index=False,
    ),
    'sheet_config': dict(
        sheet_filter='.+',
        table_start_keyword='Question',
        header_offset=0,
        header_rows=1,
        table_width=3,
        probe_col=3,
        grace_rows=10,
    ),
    'special': False,
    'mappings': {
        'columns': [
            'subcategory',
            'number',
            'question',
        ],
        'column_type_overrides': {
            'number': 'string',
        },
        'custom_post_processing': custom_post_processing,
    },
    'report_function': custom_report,
}

