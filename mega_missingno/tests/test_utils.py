"""
Utilities test module. Asserts that utility functions are correct.
"""

import unittest
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import Row

import sys; sys.path.append("../")
import mega_missingno.mega_missingno as msno

def _are_equals(df1,df2):
    return ((df1.exceptAll(df2).count() == 0) and (df2.exceptAll(df1).count() == 0))


class TestNullitySort(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({'A': [0, np.nan, np.nan], 'B': [0, 0, np.nan]})
        self.spark = SparkSession.builder.appName('test').getOrCreate()
        self.df = self.spark.createDataFrame(df)

    def test_no_op(self):
        expected = self.df
        result = msno.nullity_sort(self.df, sort=None)

        assert _are_equals(result, expected)

    def test_ascending_sort(self):
        result = msno.nullity_sort(self.df, sort='ascending')
        indices = [2, 1, 0]
        expected = self.spark.createDataFrame([Row(*[t for t in self.df.select('*').collect()[i]]) for i in indices],
                      self.df.schema)
        assert _are_equals(result, expected)

    def test_descending_sort(self):
        result = msno.nullity_sort(self.df, sort='descending')
        indices = [0, 1, 2]
        expected = self.spark.createDataFrame([Row(*[t for t in self.df.select('*').collect()[i]]) for i in indices],
                      self.df.schema)
        assert _are_equals(result, expected)


class TestNullityFilter(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({'A': [0, np.nan, np.nan], 'B': [0, 0, np.nan], 'C': [0, 0, 0]})
        self.spark = SparkSession.builder.appName('test').getOrCreate()
        self.df = self.spark.createDataFrame(df)

    def test_no_op(self):
        assert _are_equals(self.df,msno.nullity_filter(self.df))
        assert _are_equals(self.df,msno.nullity_filter(self.df, filter='top'))
        assert _are_equals(self.df,msno.nullity_filter(self.df, filter='bottom'))

    def test_percentile_cutoff_top_p(self): 
        expected = self.df.select('B','C')
        result = msno.nullity_filter(self.df, p=0.6, filter='top')
        assert _are_equals(result, expected)

    def test_percentile_cutoff_bottom_p(self): 
        expected = self.df.select('A')
        result = msno.nullity_filter(self.df, p=0.6, filter='bottom')
        assert _are_equals(result, expected)

    def test_percentile_cutoff_top_n(self): 
        expected = self.df.select('C')
        result = msno.nullity_filter(self.df, n=1, filter='top')
        assert _are_equals(result, expected)

    def test_percentile_cutoff_bottom_n(self):
        expected = self.df.select('A')
        result = msno.nullity_filter(self.df, n=1, filter='bottom')
        assert _are_equals(result, expected)

    def test_combined_cutoff_top(self): 
        expected = self.df.select('C')
        result = msno.nullity_filter(self.df, n=2, p=0.7, filter='top')
        assert _are_equals(result, expected)

    def test_combined_cutoff_bottom(self): 
        expected = self.df.select('A')
        result = msno.nullity_filter(self.df, n=2, p=0.4, filter='bottom')
        assert _are_equals(result, expected)