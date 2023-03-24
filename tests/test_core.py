'''Test core data classes and functions.'''

from pathlib import Path
from pytest import approx

import os
import time
import timeit
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
    timestamp1 = "19990405_120394"
    timestamp2 = "19990405_120395"
    for path in [timestamp1, Path(timestamp1)]:
        dated1 = skrt.core.Dated(path)
        for dated2 in [skrt.core.Dated(timestamp2),
                       skrt.core.Dated(auto_timestamp=True)]:
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
    assert test_msg in captured.out
    logger.debug(test_msg)
    captured = capsys.readouterr()
    assert test_msg not in captured.out

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

small_number = 1.e-10
def test_hour_in_day():
    '''Test determinations of hour in day.'''

    timestamp = 1
    assert skrt.core.get_hour_in_day(timestamp) == None

    timestamp = pd.Timestamp("20220205123522")
    assert skrt.core.get_hour_in_day(timestamp) == approx(
            12 + 35 / 60 + 22 / 3600, small_number)

def test_hour_in_week():
    '''Test determinations of hour in week.'''

    timestamp = 1
    assert skrt.core.get_hour_in_week(timestamp) == None

    # 20220205 is a Saturday (day 6).
    timestamp = pd.Timestamp("20220205123522")
    assert skrt.core.get_hour_in_week(timestamp) == approx(
            5 * 24 + skrt.core.get_hour_in_day(timestamp), small_number)

def test_day_in_week():
    '''Test determinations of day in week.'''

    timestamp = 1
    assert skrt.core.get_day_in_week(timestamp) == None

    # 20220205 is a Saturday (day 6).
    timestamp = pd.Timestamp("20220205123522")
    assert skrt.core.get_day_in_week(timestamp) == approx(
            5 + skrt.core.get_hour_in_day(timestamp) / 24, small_number)

def test_year_fraction():
    '''Test conversion from timestamp to year fraction'''

    assert skrt.core.year_fraction(1) == None
    assert (skrt.core.year_fraction(pd.Timestamp("20220121"))
            == (2022 + 20 / 365))
    assert (skrt.core.year_fraction(pd.Timestamp("20200229"))
            == (2020 + 59 / 366))

def test_relative_path():
    """Test relative path."""
    user_path = Path("~").expanduser()
    non_user_path = Path("/non/user/path")
    sub_path = Path("path/to/file")
    assert skrt.core.relative_path(user_path / sub_path) == str(sub_path)
    assert skrt.core.relative_path(non_user_path) == str(non_user_path)
    assert skrt.core.relative_path(non_user_path / sub_path, 3) == str(sub_path)
    assert skrt.core.relative_path(user_path / sub_path, -3) == str(sub_path)

def test_make_dir():
    """Text directory creation, with and without overwriting allowed."""
    tdir = Path(skrt.core.fullpath("tmp/make_dir_test"))
    if tdir.exists():
        shutil.rmtree(tdir)
    path = skrt.core.make_dir(tdir)
    assert path == tdir

    # Check different values for parameters overwrite and require_empty.
    for overwrite in [True, False]:
        for require_empty in [True, False]:
            (tdir / "tmp.txt").touch(exist_ok=True)
            path = skrt.core.make_dir(tdir, overwrite, require_empty)
            if not overwrite and require_empty:
                assert path is None
            else:
                assert path == tdir

def test_tictoc():
    # Define number of iterations, sleep time per iteration,
    # and level of agreement required in timing tests.
    n_iteration = 3
    sleep_time = 0.25
    small_number = 1.e-2

    # Start timer.
    t1 = timeit.default_timer()
    skrt.core.tic()

    # Check time between consecutive calls to tic() and toc().
    for idx in range(n_iteration):
        t2 = timeit.default_timer()
        skrt.core.tic()
        time.sleep(sleep_time)
        tic_toc = skrt.core.toc()
        t3 = timeit.default_timer()
        assert tic_toc == approx(t3 - t2, abs=small_number)

    # Check nesting - time relative to the timer start time.
    tic_toc = skrt.core.toc()
    t4 = timeit.default_timer()
    assert tic_toc == approx(t4 - t1, abs=small_number)

    # Check accumulation - time relative to the last call to tic().
    for idx in range(n_iteration):
        time.sleep(sleep_time)
        tic_toc = skrt.core.toc()
        t5 = timeit.default_timer()
        assert tic_toc == approx(t5 - t2, abs=small_number)

def test_tictoc_messages(capsys):
    """Test output from toc() command."""
    # Check that TicToc().default_message is output
    # when TicToc().message is True.
    skrt.core.TicToc(True)
    assert skrt.core.TicToc().message is True
    skrt.core.tic()
    skrt.core.toc()
    assert skrt.core.TicToc().default_message
    assert skrt.core.TicToc().default_message in capsys.readouterr().out

    # Check that empty string is output
    # when TicToc().message is False.
    skrt.core.TicToc(False)
    assert skrt.core.TicToc().message is False
    skrt.core.toc()
    assert capsys.readouterr().out == ""

def test_qualified_name():
    """Test determination of qualified name for a class."""
    from skrt.core import Archive
    assert "skrt.core.Archive" == skrt.core.qualified_name(Archive)

def test_get_stat():
    """Test calculation of mean for values of a dictionary."""
    # Check that the value returned for an empty dictionary is None.

    nval = 29
    tests = (
            # Null input.
            (None, None, None, {}, None),
            # Insufficent inputs for statistic.
            ([1], None, "stdev", {}, None),
            ([[1, 1, 1]], None, "stdev", {}, [None, None, None]),
            # Median of consecutive integers from zero (dictionary values).
            ({val: val for val in range(nval)},
                None, "mean", {}, (nval - 1) / 2),
            # Median of consecutive integers from zero (list values).
            (list(range(nval)), None, "median", {}, (nval - 1) / 2),
            # Mean of consecutive integers from zero,
            # after substituting None for odd values.
            ({val: (val if val % 2 == 0 else None) for val in range(nval)},
                None, "mean", {}, (nval - 1) / 2),
            # Mean of consecutive integers from zero,
            # after substituting 0 substituted for odd values.
            ({val: (val if val % 2 == 0 else None) for val in range(nval)},
                0, "mean", {}, (nval - 1) * (nval + 1) / (4 * nval)),
            # Quantiles of consecutive integers from zero.
            (list(range(nval)), None, "quantiles", {"n": 2}, [(nval - 1) / 2]),
            # Medians of tuple components.
            ({val: (val, -val) for val in range(nval)},
                None, "mean", {}, [(nval - 1) / 2, -(nval - 1) / 2]),
            )

    for values, value_for_None, stat, kwargs, result in tests:
        assert (skrt.core.get_stat(values, value_for_None, stat, **kwargs)
                == result)

def test_get_stat_functions():
    """
    Check that example names of functions defined by Python statistics module
    are returned by skrt.core.get_stat_functions().
    """
    example_stats = ["mean", "median", "mode", "stdev"]
    assert all([stat in skrt.core.get_stat_functions()
                for stat in example_stats])

def test_get_dict_permutations():
    """
    Test extraction of lists of dictionaries from dictionary of lists.
    """
    # Check output when input isn't a non-empty dictionary.
    for not_non_empty_dict in [None, 4, {}]:
        assert skrt.core.get_dict_permutations(None) == [{}]

    # Check output for non-empty dictionary.
    in_dict = {"A": [1, 2, 3], "B": [4, 5]}
    out_list = [{"A": 1, "B": 4}, {"A": 1, "B": 5}, {"A": 2, "B": 4},
                {"A": 2, "B": 5}, {"A": 3, "B": 4}, {"A": 3, "B": 5}]
    assert skrt.core.get_dict_permutations(in_dict) == out_list

def test_qualified_name():
    """Test determination of a class's qualified name."""

    # Check value returned for class.
    assert skrt.core.qualified_name(skrt.core.Data) == "skrt.core.Data"

    # Check value returned for non-class.
    assert skrt.core.qualified_name(5) == None

def test_compress_user():
    """Test replacement by '~' of user home directory at start of path."""
    home = Path(skrt.core.fullpath("~"))
    not_home = "/not/home"
    # Protect against unlikely case where home directory is "/not/home"...
    if not_home in str(home):
        not_home = "/not/second/home"
    assert skrt.core.compress_user(home) == "~/."
    assert skrt.core.compress_user(not_home) == skrt.core.fullpath(not_home)
