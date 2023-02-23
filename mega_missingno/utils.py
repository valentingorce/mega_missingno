"""Utility functions for missingno."""
from pyspark.sql.functions import sort_array, array, col, sum, when, size


def nullity_sort(df, sort=None, axis='columns'):
    """
    Sorts a DataFrame according to its nullity, in either ascending or descending order.
    :param df: The DataFrame object being sorted.
    :param sort: The sorting method: either "ascending", "descending", or None (default).
    :return: The nullity-sorted DataFrame.
    """
    if sort is None:
        return df
    elif sort not in ['ascending', 'descending']:
        raise ValueError('The "sort" parameter must be set to "ascending" or "descending".')

    if axis not in ['rows', 'columns']:
        raise ValueError('The "axis" parameter must be set to "rows" or "columns".')

    if axis == 'columns':
        if sort == 'ascending':
            sorted_df = df.select('*',
                                    sort_array(array([df[c].isNotNull() for c in df.columns])).alias('count_array')) \
                            .orderBy('count_array')
            result_df = sorted_df.drop('count_array')
            return result_df
        elif sort == 'descending':
            sorted_df = df.select('*',
                                    sort_array(array([df[c].isNotNull() for c in df.columns]), asc=False).alias('count_array')) \
                            .orderBy('count_array')
            result_df = sorted_df.drop('count_array')
            return result_df
    elif axis == 'rows':
        if sort == 'ascending':
            sorted_cols = sort_array(array([col(c).isNotNull() for c in df.columns]))
            sorted_df = df.select(*[col(c) for c in df.columns[sorted_cols.asc().expr()]])
            return sorted_df
        elif sort == 'descending':
            sorted_cols = sort_array(array([col(c).isNotNull() for c in df.columns]))
            sorted_df = df.select(*[col(c) for c in df.columns[sorted_cols.desc().expr()]])
            return sorted_df


def nullity_filter(df, filter=None, p=0, n=0):
    """
    Filters a DataFrame according to its nullity, using some combination of 'top' and 'bottom' numerical and
    percentage values. Percentages and numerical thresholds can be specified simultaneously: for example,
    to get a DataFrame with columns of at least 75% completeness but with no more than 5 columns, use
    `nullity_filter(df, filter='top', p=.75, n=5)`.
    :param df: The DataFrame whose columns are being filtered.
    :param filter: The orientation of the filter being applied to the DataFrame. One of, "top", "bottom",
    or None (default). The filter will simply return the DataFrame if you leave the filter argument unspecified or
    as None.
    :param p: A completeness ratio cut-off. If non-zero the filter will limit the DataFrame to columns with at least p
    completeness. Input should be in the range [0, 1].
    :param n: A numerical cut-off. If non-zero no more than this number of columns will be returned.
    :return: The nullity-filtered `DataFrame`.
    """
    if filter == 'top':
        if p:
            row_counts = sum(when(col(c).isNotNull(), 1).otherwise(0) for c in df.columns) # Count non-null values in each column
            selected_cols = [c for c in df.columns if row_counts[c]/size(df) >= p] # Select columns with non-null count >= threshold
            df = df.select(*selected_cols)
        if n:
            sorted_cols = sorted(df.columns, key=lambda c: df.where(col(c).isNotNull()).count(), reverse=True)[:n]
            df = df.select(*[col(c) for c in sorted_cols])
            # maybe use approxQuantile?
    elif filter == 'bottom':
        if p:
            sorted_cols = [c for c in df.columns if df.where(col(c).isNotNull()).count() / df.count() <= p]
            df = df.select(*[col(c) for c in sorted_cols])
        if n:
            sorted_cols = sorted(df.columns, key=lambda x: df.where(col(x).isNotNull()).count())[:n]
            df = df.select(*[col(c) for c in sorted_cols])
    return df