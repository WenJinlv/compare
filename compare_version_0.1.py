#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 02:03:34 2018
@author: wen
"""

import os
import argparse
import warnings
from abc import ABCMeta, abstractmethod
import numpy as np
warnings.filterwarnings('ignore')
import h5py
import prettytable


def parse_commandline():
    """
    Get user input & Parsing command line strings into Python object
    """
    parser = argparse.ArgumentParser(
        add_help=True,
        description="This program is compare field data which dump \
        in different moment")
    parent_parser = argparse.ArgumentParser(add_help=False)
    group = parent_parser.add_argument_group('arguments')
    group.add_argument("src_file", help="The source file")
    group.add_argument("target_file", help="The target file")
    group.add_argument(
        "--error_distributed",
        action="store",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Output the distributed error,\
                    1 for global statistical\
                    2 for local statistical,\
                    3 for detail statistical.")

    strategy_parser = argparse.ArgumentParser(parents=[parent_parser])
    strategy_parser.add_argument(
        "--relate_error",
        action="store_true",
        help="Output the average relate error")
    strategy_parser.add_argument(
        "--absolute_error",
        action="store_true",
        help="Output the average absolute error")
    strategy_parser.add_argument(
        "--norm2_error", action="store_true", help="Output the 2-norm error")
    strategy_parser.add_argument(
        "--inf_norm_error", action="store_true", help="Output the inf norm error")
    strategy_parser.add_argument(
        "--bins", action="store", type=int, default=8, help="Set histogram bins")
    config = strategy_parser.parse_args()

    args = {'is_abs_err': config.absolute_error, 'is_rel_err': config.relate_error,
            'is_norm2_err': config.norm2_error, 'is_infnorm_err': config.inf_norm_error,
            'src_file': config.src_file, 'dst_file': config.target_file, 'pwd': os.getcwd(),
            'bins': config.bins, 'strategy': config.error_distributed }
    return args


class ReadHdf5(object):
    def __init__(self, filename):
        self.filename = filename
        self.data_item = None
        self.index_space = None
        self.grid_type = None
        self.elem_type = None
        self.depth = None
        self.depthDict = None
        self.fields = None
        self.indexs = None

        self.get_hdf5_info()

    def get_hdf5_info(self):
        with h5py.File(self.filename, "r") as hdf:
            # read data_item & index_space from hdf5
            self.data_item = np.array(hdf.get('data_item'))
            self.index_space = np.array(hdf.get('index_space'))
            if 'grid_type' not in list(hdf.get('meta').attrs.keys()):
                self.grid_type = None
            else:
                self.grid_type = hdf.get('meta').attrs['grid_type']  # .decode('ascii')
                self.depth = hdf.get('meta').attrs['depth']  # .decode('ascii')
            self.depthDict = dict([(n.decode('ascii'), i) for n, i in zip(
                hdf.get('data_item')['name'],
                hdf.get('data_item')['id'])])
            # get field name & index range from data_item and index_space
            self.fields = [item[0].decode('ascii') for item in self.data_item]
            self.indexs = [(index[0].decode('ascii'), index[2], index[3], index[4])
                               for index in self.index_space]
    def get_data(self, field, index):
        with h5py.File(self.filename, 'r') as hdf:
            raws_dataset = hdf.get(field + '@' + index)
            self.elem_type = raws_dataset.attrs.get('elem_type')
            return np.array(raws_dataset)


class ConcateIndex(object):
    """
    Concate all index and data for specific field

    """
    def __init__(self, hdf5_info):
        self.hdf5_info = hdf5_info
        self.field_dict = {}
        self.index_values = []
        self.index_space_val = []

    def get_data_and_index(self):
        # get concatenate field data & index for specific field
        for field in self.hdf5_info.fields:
            depth_index = self.hdf5_info.depthDict[field]
            self.field_dict[field] = np.array([])
            for index, dim, lower, upper in self.hdf5_info.indexs:
                # add index data value to field dict
                clean_data = self.proprocess_data(field, index, True)
                self.field_dict[field] = np.concatenate(
                    (self.field_dict[field], clean_data), axis=0)
                # append index to index value list
                # handle complex index
                if self.hdf5_info.elem_type == 4:
                    upper[1] = int(upper[1] / 2)
                else:
                    upper = upper
                    # 3D,2D or 1D index process
                if dim == 5:
                    for i in np.arange(upper[0] - lower[0]):
                        for j in np.arange(upper[1] - lower[1]):
                            for k in range(upper[2]):
                                for depth in np.arange(upper[-2]):
                                    self.index_values.append(
                                        (lower[0] + i, lower[1] + j,
                                            lower[2] + k, lower[3] + depth,
                                            lower[4] + 0))
                                    self.index_space_val.append(
                                        (index,
                                            len(self.hdf5_info.depthDict) - depth_index - 1))
                elif dim == 4:
                    for i in np.arange(upper[0] - lower[0]):
                        for j in np.arange(upper[1] - lower[1]):
                            for depth in np.arange(upper[-2]):
                                self.index_values.append(
                                    (lower[0] + i, lower[1] + j,
                                        lower[2] + depth, lower[3] + 0))
                                self.index_space_val.append(
                                    (index, len(self.hdf5_info.depthDict) - depth_index - 1))
                elif dim == 3:
                    for i in np.arange(upper[0] - lower[0]):
                        for depth in np.arange(upper[-2]):
                            self.index_values.append(
                                (lower[0] + i, lower[1] + depth, lower[2] + 0))
                            self.index_space_val.append(
                                (index, len(self.hdf5_info.depthDict) - depth_index - 1))
        return self.field_dict, self.index_values, self.index_space_val, self.hdf5_info.grid_type

    def proprocess_data(self, field, index, is_proprocess=True):
        # Process raw data include missing data
        raws_dataset = self.hdf5_info.get_data(field, index)
        if not is_proprocess:
            return raws_dataset.ravel()
        else:
            # set missing data value is zero
            # raw_data[np.isnan(raw_data)] = 0
            if self.hdf5_info.elem_type == 5:
                raw_data = raws_dataset.ravel()
                clean_data = []
                for i in range(int(len(raw_data))):
                    if i % 2 == 0:
                        clean_data.append(np.complex(raw_data[i], raw_data[i + 1]))
                return np.array(clean_data)
            else:
                return raws_dataset.ravel()



class ConcateField(object):
    """  Concate all field data for specific hdf5 file
    """
    def __init__(self, field_dict):
        self.field_dict = field_dict

    def concate_field(self):
        total_data = np.array([])
        for k in self.field_dict:
            total_data = np.concatenate((self.field_dict[k], total_data), axis=0)
        return total_data





class Measure_error(object):
    """ Measure the error of source & target file
    """
    def __init__(self, src_data, dst_data, args):
        self.args = args
        self.src_data = src_data
        self.dst_data = dst_data

    def error_rule(self):
        diff = np.abs(self.src_data - self.dst_data)
        # choice for absolute error
        if self.args['is_abs_err']:
            print("display: absolute error")
            yield diff
        # choice for relate error
        if self.args['is_rel_err']:
            print("display: relate error")
            # +1e-14 to avoid divided by zero
            self.src_data[np.where(self.src_data == 0)] = 1e-14
            yield diff / np.abs(self.src_data)
        # choice for 2-norm error
        if self.args['is_norm2_err']:
            print("display: norm-2 error")
            print(np.linalg.norm(diff[~np.isnan(diff)], 2))
        # choice for inf-norm error
        if self.args['is_infnorm_err']:
            print("display: norm-inf error")
            print(np.linalg.norm(diff[~np.isnan(diff)], np.inf))
        # absolute error by default
        if not (self.args['is_abs_err'] or self.args['is_rel_err'] or
                self.args['is_norm2_err'] or self.args['is_infnorm_err']):
            print("display: absolute error")
            yield diff


def _statstics_quantity(diff, bins):
    """  Statstic field quantity include max, min, average, sum and histogram
    """
    nan_num = np.isnan(diff).sum()
    abs_diff = np.abs(diff[~np.isnan(diff)])
    np.seterr(invalid='ignore')
    # range: +1e-20 for avoid array element is identity.eg.arr.min() == \
    # arr.max() so that can't work well
    hist = np.histogram(
        abs_diff, bins=int(bins), range=(abs_diff.min(), abs_diff.max() + 1e-20))
    return np.max(abs_diff), np.min(abs_diff), np.average(
        abs_diff), nan_num, diff.size, hist


class Strategy(metaclass=ABCMeta):

    def __init__(self, args, src_data, dst_data,
                 src_field, src_index, src_index_space,
                 dst_field, dst_index, dst_index_space,
                 grid_type):
        self.args = args
        self.src_data = src_data
        self.dst_data = dst_data
        self.src_field = src_field
        self.src_index = src_index
        self.src_index_space = src_index_space
        self.dst_field = dst_field
        self.dst_index = dst_index
        self.dst_index_space = dst_index_space
        self.grid_type  = grid_type

    @abstractmethod
    def strategy(self):
        pass


class ClobalStrategy(Strategy):

    def strategy(self):
        print("\n" + "=" * 25 + "compare two hdf5 file=" + "=" * 25)
        print("\nexe path: {:^16}  \nsrc path: {:^16}  \ntarget path: {:^16} \n".
              format(self.args['pwd'], self.args['src_file'], self.args['dst_file']))

        diff_gen = Measure_error(self.src_data, self.dst_data, self.args).error_rule()
        for diff in diff_gen:
            output = _statstics_quantity(diff, self.args['bins'])
            print("    statstics info: ")
            table = prettytable.PrettyTable([
                'max', 'min', 'average', 'missing value count', 'element count'
            ])
            histogram_table = prettytable.PrettyTable()
            table.add_row(
                [output[0], output[1], output[2], output[3], output[4]])
            histogram_table.add_column("Bin",
                                       [(output[5][1][i], output[5][1][i + 1])
                                        for i in range(self.args['bins'])
                                        if output[5][0][i] > 0])
            histogram_table.add_column(
                "Count",
                [output[5][0][i] for i in range(self.args['bins']) if output[5][0][i] > 0])
            print(table)
            print("    histogram:")
            print(histogram_table)


class ItemStrategy(Strategy):

    def strategy(self):
        print("\n" + "=" * 25 + "compare two hdf5 file=" + "=" * 25)
        print("\nexe path: {:^16}  \nsrc path: {:^16}  \ntarget path: {:^16} \n".
              format(self.args['pwd'], self.args['src_file'], self.args['dst_file']))

        for src_field_key, dst_field_key in zip(self.src_field.keys(),
                                                self.dst_field.keys()):
            diff_gen = Measure_error(self.src_field[src_field_key],
                                     self.dst_field[dst_field_key], self.args).error_rule()
            for diff in diff_gen:
                output = _statstics_quantity(diff, self.args['bins'])
                table = prettytable.PrettyTable([
                    'max', 'min', 'average', 'missing value count',
                    'element count'
                ])
                histogram_table = prettytable.PrettyTable()
                print("    data item: {} ".format(src_field_key))
                print("    statstics info: ")
                table.add_row(
                    [output[0], output[1], output[2], output[3], output[4]])
                histogram_table.add_column(
                    "Bin", [(output[5][1][i], output[5][1][i + 1])
                            for i in range(self.args['bins']) if output[5][0][i] > 0])
                histogram_table.add_column("Count", [
                    output[5][0][i] for i in range(self.args['bins']) if output[5][0][i] > 0
                ])
                print(table)
                print("    histogram:")
                print(histogram_table)
                print("\n")


class Detail_strategy(Strategy):

    def strategy(self):
        print("\n" + "=" * 25 + "compare two hdf5 file=" + "=" * 25)
        print("\nexe path: {:^16}  \nsrc path: {:^16}  \ntarget path: {:^16} \n".
              format(self.args['pwd'], self.args['src_file'], self.args['dst_file']))

        diff_gen = Measure_error(self.src_data, self.dst_data, self.args).error_rule()
        for diff in diff_gen:
            table = prettytable.PrettyTable(
                ['max', 'min', 'average', 'missing value count', 'element count'])
            histogram_table = prettytable.PrettyTable()
            detail_table = prettytable.PrettyTable(
                ['error', 'src_data', 'dst_data', 'index', 'index space'])
            if len(diff[diff > 1e-12]) == 0:
                print("    NO ERROR    ")
            else:
                output = _statstics_quantity(diff[diff > 1e-12], self.args['bins'])
                print("    statstics info: ")
                table.add_row(
                    [output[0], output[1], output[2], output[3], output[4]])
                histogram_table.add_column(
                    "Bin", [(output[5][1][i], output[5][1][i + 1])
                            for i in range(self.args['bins']) if output[5][0][i] > 0])
                histogram_table.add_column("Count", [
                    output[5][0][i] for i in range(self.args['bins']) if output[5][0][i] > 0
                ])
                if self.grid_type == b'Unstructured Mesh':
                    [
                        detail_table.add_row([
                            d, src, dst, (index[1], index_space[1], index[0]),
                            index_space[0]
                        ]) for d, src, dst, index, index_space in zip(
                        diff, self.src_data, self.dst_data, self.src_index,
                        self.src_index_space) if d > 1e-12
                    ]
                else:
                    [
                        detail_table.add_row(
                            [d, src, dst, index, index_space[0]])
                        for d, src, dst, index, index_space in zip(
                        diff, self.src_data, self.dst_data, self.src_index,
                        self.src_index_space) if d > 1e-12
                    ]
                print(table)
                print("    histogram(only display count>0):")
                print(histogram_table)
                print("    detail info:")
                print(detail_table)



class DisplayTable(object):
    """
    Display the info which is user-defined error_distributed strategy
    """
    def __init__(self, args):
        self.args = args




def main():
    args = parse_commandline()
    concate_src_index = ConcateIndex(ReadHdf5(args['src_file']))
    concate_dst_index = ConcateIndex(ReadHdf5(args['dst_file']))
    src_field, src_index, src_index_space, grid_type = concate_src_index.get_data_and_index()
    dst_field, dst_index, dst_index_space, grid_type = concate_dst_index.get_data_and_index()
    src_data = ConcateField(src_field).concate_field()
    dst_data = ConcateField(dst_field).concate_field()

    if args['strategy'] == 1:
        Strategy = ClobalStrategy

    if args['strategy'] == 2:
        Strategy = ItemStrategy

    if args['strategy'] == 3:
        Strategy = ItemStrategy

    Strategy(args, src_data, dst_data,
                 src_field, src_index, src_index_space,
                 dst_field, dst_index, dst_index_space,
                 grid_type).strategy()


if __name__ == "__main__":
    main()
