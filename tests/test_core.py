'''Test core data classes and functions.'''

from pathlib import Path

import os
import time
import shutil

import pandas as pd

import skrt.core


def test_defaults():

    assert skrt.core.Defaults().print_depth == 0
    skrt.core.Defaults(opts={'print_depth': 2, 'test_item': 1})
    assert skrt.core.Defaults().print_depth == 2
    assert skrt.core.Defaults().test_item == 1

def test_data():
    data = skrt.core.Data(thing=10)
    assert data.thing == 10
    assert data.get_print_depth() == skrt.core.Defaults().print_depth
    data.set_print_depth(10)
    assert data.get_print_depth() == 10
    data.print()

def test_pathdata():
    for path in [".", Path(".")]:
        pdata = skrt.core.PathData(path)
        assert pdata.path == skrt.core.fullpath(path)
        assert pdata.create_objects(None) == []

def test_dated():
    for path in [".", Path(".")]:
        dated = skrt.core.Dated(path)
        assert not dated.date
        assert not dated.time

def test_dated_sorting():
    timestamp = "19990405_120394"
    for path in [timestamp, Path(timestamp)]:
        dated1 = skrt.core.Dated(path)
        dated2 = skrt.core.Dated(auto_timestamp=True)
        assert dated2 > dated1
        assert dated2 >= dated1
        assert dated1 < dated2
        assert dated1 <= dated2
        assert dated1 == dated1
        assert dated1 != dated2 

def test_dated_interval():
    timestamp = "20010502_120358"
    for path in [timestamp, Path(timestamp)]:
        dated = skrt.core.Dated(timestamp)
        assert dated.in_date_interval("19991203", "20030402")

def test_time_separated_objects():
    # Test filtering of dated objects based on time separation.
    dated1 = skrt.core.Dated("19990405_120324")
    dated2 = skrt.core.Dated("20010502_120358")
    dated3 = skrt.core.Dated("19990405_130324")
    objs1 = [dated1, dated2, dated3]

    # Keep most recent, requiring default time separation (4 hours).
    objs2 = skrt.core.get_time_separated_objects(objs1)
    assert len(objs2) == 2
    assert dated1 not in objs2
    assert dated3 == objs2[0]
    assert dated2 == objs2[1]

    # Keep least recent, requiring default time separation (4 hours).
    objs3 = skrt.core.get_time_separated_objects(objs1, most_recent=False)
    assert len(objs3) == 2
    assert dated3 not in objs3
    assert dated1 == objs3[0]
    assert dated2 == objs3[1]

    # Keep most recent, requiring time separation of 10 seconds.
    objs4 = skrt.core.get_time_separated_objects(objs1, min_delta=10, unit='s')
    assert len(objs4) == 3
    assert dated1 == objs4[0]
    assert dated3 == objs4[1]
    assert dated2 == objs4[2]

def test_files():
    for path1 in ("afile.txt", Path("afile.txt")):
        for path2 in ("bfile.txt", Path("bfile.txt")):
            for path3 in ("afile.txt", Path("afile.txt")):
                file1 = skrt.core.File(path1)
                file2 = skrt.core.File(path2)
                assert file2 > file1
                file3 = skrt.core.File(path3)
                assert file3 == file1
                assert file3 != file2

def test_archive():
    timestamp = '19990405_230203'
    tdir = f'tmp/{timestamp}'
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
    os.makedirs(tdir)
    os.makedirs(f'{tdir}/some_dir')
    with open(f'{tdir}/some_file.txt', 'w') as file:
        file.write('testing')
    with open(f'{tdir}/.hidden', 'w') as file:
        file.write('testing')
    for path in [tdir, Path(tdir)]:
        archive = skrt.core.Archive(path)
        assert archive.timestamp == timestamp
        assert len(archive.files) == 1
        assert 'some_file' in archive.files[0].path
        archive2 = skrt.core.Archive(path, allow_dirs=True)
        assert len(archive2.files) == 2

def test_logger(capsys):
    logger = skrt.core.get_logger("test", "INFO")
    test_msg = "Test message"
    logger.info(test_msg)
    captured = capsys.readouterr()
    assert test_msg in captured.err
    logger.debug(test_msg)
    captured = capsys.readouterr()
    assert test_msg not in captured.err

def test_null_data():
    x = skrt.core.Data()
    assert(type(x).__name__ == 'Data')

def test_null_path_data():
    x = skrt.core.PathData()
    assert(type(x).__name__ == 'PathData')
    assert(x.path == '')
    assert(x.subdir == '')

def test_null_dated():
    x = skrt.core.Dated()
    assert(type(x).__name__ == 'Dated')
    assert(x.date == '')
    assert(x.path == '')
    assert(x.subdir == '')
    assert(x.time == '')
    assert(x.timestamp == '')

def test_null_archive():
    x = skrt.core.Archive()
    assert(type(x).__name__ == 'Archive')
    assert(x.date == '')
    assert(x.files == [])
    assert(x.path == '')
    assert(x.subdir == '')
    assert(x.time == '')
    assert(x.timestamp == '')

def test_null_file():
    x = skrt.core.File()
    assert(type(x).__name__ == 'File')
    assert(x.date == '')
    assert(x.path == '')
    assert(x.subdir == '')
    assert(x.time == '')
    assert(x.timestamp == '')

def test_data_by_filename():
    timestamp = skrt.core.generate_timestamp()
    data_objects = []
    n_object = 5
    for idx in range(n_object):
        data_objects.append(skrt.core.PathData(
            f'/datastore/{timestamp}_object{idx:03}.dat'))

    data_by_filename = skrt.core.get_data_by_filename(data_objects)
    assert len(data_by_filename) == len(data_objects)
    for filename, data_object in data_by_filename.items():
        assert filename in data_object.path
        filename_from_path = (Path(data_object.path).stem
                .split(f'{timestamp}_')[-1])
        assert filename == filename_from_path

def test_file_info():
    timestamp = '19990405_230203'
    tdir = Path(f'tmp/{timestamp}')
    if tdir.exists():
        shutil.rmtree(tdir)
    tdir.mkdir(parents=True)
    (tdir / 'some_dir').mkdir()
    path1 = tdir / 'some_file.txt'
    with open(path1, 'w') as file:
        file.write('testing')
    path2 = tdir / '.hidden'
    with open(path2, 'w') as file:
        file.write('testing')
    path_data = skrt.core.PathData(path1)
    archive = skrt.core.Archive(tdir)
    archive2 = skrt.core.Archive(tdir, allow_dirs=True)
    assert path_data.get_n_file() == 1
    assert archive.get_n_file() == 1
    size = Path(archive.files[0].path).stat().st_size
    assert path_data.get_file_size() == size
    assert archive.get_file_size() == size
    assert archive2.get_n_file() == 2
    assert skrt.core.get_n_file(archive) == 1
    assert skrt.core.get_n_file([archive, archive2]) == 3
    for file in archive2.files:
        size += Path(file.path).stat().st_size
    assert skrt.core.get_file_size([archive, archive2]) == size

def test_intervals():
    '''Test determinations of intervals.'''

    timestamp1 = 1
    timestamp2 = 1
    assert skrt.core.get_interval_in_days(timestamp1, timestamp2) == None
    assert skrt.core.get_interval_in_whole_days(timestamp1, timestamp2) == None

    timestamp1 = pd.Timestamp("20220121183500")
    timestamp2 = pd.Timestamp("20220205123500")

    assert (skrt.core.get_interval_in_days(timestamp1, timestamp2) ==
            14 + 18 / 24)
    assert skrt.core.get_interval_in_whole_days(timestamp1, timestamp2) == 15

    timestamp1 = pd.Timestamp("20220205115900")
    timestamp2 = pd.Timestamp("20220205120100")
    assert skrt.core.get_interval_in_whole_days(timestamp1, timestamp2) == 0

    timestamp1 = pd.Timestamp("20220205235900")
    timestamp2 = pd.Timestamp("20220206000100")
    assert skrt.core.get_interval_in_whole_days(timestamp1, timestamp2) == 1

def test_hour_in_day():
    '''Test determinations of hour in day.'''

    timestamp = 1
    assert skrt.core.get_hour_in_day(timestamp) == None

    timestamp = pd.Timestamp("20220205123522")
    assert skrt.core.get_hour_in_day(timestamp) == 12 + 35 / 60 + 22 / 3600

def test_year_fraction():
    '''Test conversion from timestamp to year fraction'''

    assert skrt.core.year_fraction(1) == None
    assert (skrt.core.year_fraction(pd.Timestamp("20220121"))
            == (2022 + 20 / 365))
    assert (skrt.core.year_fraction(pd.Timestamp("20200229"))
            == (2020 + 59 / 366))
