# IMPORTS
# ---------------------------------------------------------------------------------------------------------------------
from IPython.html import widgets  # Widget definitions
from IPython.html.widgets import interact  # , interactive, fixed
# from IPython.display import display  # Used to display widgets in the notebook
# from IPython.utils.traitlets import Unicode  # Used to declare attributes of our widget

import datetime
import seaborn as sns


# GLOBALS
# ---------------------------------------------------------------------------------------------------------------------
PIVOT_CONF = {}


# FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------
def total_pivot(df, show_plot=True, index=None, columns=None, values=None, drop_columns=None, subset=None,
                period_end='2014-12-31', period_start='2014-01-01',
                aggfunc='sum', aggregate=None, query='',
                show_values_as='% of calendar time',
                stacked=True, grouping=None, group_other=True, transposed=False, ylim=None):
    """
    :param df:
    :param boolean show_plot:
    :param index:
    :param columns:
    :param values:
    :param drop_columns:
    :param subset:
    :param period_end:
    :param period_start:
    :param aggfunc:
    :param aggregate:
    :param string query:
    :param show_values_as:
    :param boolean stacked:
    :return : dict
    """

    if aggregate:
        pass

    data = df.query('' +
                    'DowntimeEnd < "{}" & '.format(period_end) +
                    'DowntimeEnd >= "{}" & '.format(period_start) +
                    'Downtime >= 0 & ' +
                    'Downtime < 1600000' if not query else query) \
        .pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc) \
        .sort(inplace=False, axis=1, ascending=False)

    if grouping:
        grouped = []
        for k, v in grouping.items():
            data[k] = data.loc[:, v].sum(axis=1)
            grouped = grouped + v

            data.drop(v, axis=1, inplace=True)

        other = list(c for c in data.columns if c not in grouped and c not in list(grouping.keys()))

        if group_other:
            data['Ungrouped'] = data.loc[:, other].sum(axis=1)
            data.drop(other, axis=1, inplace=True)

    if drop_columns:
        data = data.drop(drop_columns, axis=1)

    if subset:
        data = data[subset]

    divisor = 1
    if not ylim:
        ylim = None
    axis = 1

    if show_values_as == '% of calendar time':
        fmt = '%Y-%m-%d'  # %H:%M:%S'
        d2 = datetime.datetime.strptime(period_end, fmt)
        d1 = datetime.datetime.strptime(period_start, fmt)
        divisor = ((d2 - d1).days * 24 * 60) / 100
        ylim = [0, 100]
    elif show_values_as == '% of column total':
        divisor = data.sum(axis=1) / 100
        ylim = [0, 100]
        axis = 0
    elif show_values_as == '% of row total':
        divisor = data.sum(axis=0) / 100
        # ylim = [0, 100]
        axis = 1
    elif show_values_as == 'no calculation':
        divisor = 1

    data = data.div(divisor, axis=axis)

    with sns.color_palette("Paired", n_colors=10):
        if show_plot:
            if transposed:
                if ylim:
                    data.plot(kind='barh', stacked=stacked, figsize=(18, 6), width=0.9, fontsize=16, xlim=ylim)
                else:
                    data.plot(kind='barh', stacked=stacked, figsize=(18, 6), width=0.9, fontsize=16)
            else:
                if ylim:
                    data.plot(kind='bar', stacked=stacked, figsize=(18, 6), width=0.9, fontsize=16, ylim=ylim)
                else:
                    data.plot(kind='bar', stacked=stacked, figsize=(18, 6), width=0.9, fontsize=16)
        else:
            return data


def pivot_config(index, columns, values, subset, drop_columns,
                 aggfunc, show_values_as, show_plot,
                 stacked, period_start='2014-09-01', period_end='2014-10-30', query=''):
    """Function to update the global PIVOT_CONF dictionary
        Used with the pivot widget.
    :param index:
    :param columns:
    :param values:
    :param subset:
    :param drop_columns:
    :param aggfunc:
    :param show_values_as:
    :param show_plot:
    :param stacked:
    :param period_start:
    :param period_end:
    :param query:
    :return:
    """
    global PIVOT_CONF
    PIVOT_CONF.update(locals())

    print(PIVOT_CONF)

    return True


def pivot_widget(index, columns, values, subset, drop_columns):
    """
    :param pandas.DataFrame df:
    :return : IPython.html.widgets.interact
    """
    widget = interact(
        pivot_config,
        index=index,
        columns=widgets.SelectWidget(values=columns),
        values=widgets.SelectWidget(values=values),
        subset=widgets.SelectWidget(values=subset),
        drop_columns=widgets.SelectWidget(values=drop_columns),
        aggfunc=('sum', 'mean', 'count'),
        show_values_as=('no calculation', '% of column total', '% of row total', '% of calendar time'),
        show_plot=True,
        stacked=True,
        query=widgets.TextareaWidget(),
    )
    return widget