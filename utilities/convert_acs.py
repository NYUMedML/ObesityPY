import os
import sys

import csv
import xlrd
import argparse

def csv_from_excel(excel_file):
    workbook = xlrd.open_workbook(excel_file)
    worksheet = workbook.sheet_by_index(0)
    with open(excel_file.replace('.xls','.csv'), 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        for i in range(worksheet.nrows):
            wr.writerow(map(str, worksheet.row_values(i)))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process ACS Template file names from given folder path')
    parser.add_argument('-p', '--path', required=True, type=str, metavar='path', dest='path', help='path to data folder (excluding final "/")')
    args = parser.parse_args()

    files = ['/'.join((args.path, f)) for f in os.listdir(args.path) if f.endswith('.xls')]
    print('Converting {} files from {}'.format(len(files), args.path))
    for f in files:
        csv_from_excel(f)
    print('Converted all xls files to csv in {}'.format(args.path))
