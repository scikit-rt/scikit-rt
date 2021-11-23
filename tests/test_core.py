'''Test core data classes and functions.'''

import os
import time
import shutil

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
    pdata = skrt.core.PathData('.')
    assert pdata.path == skrt.core.fullpath('.')
    assert pdata.get_dated_objects(None) == []

def test_dated():
    dated = skrt.core.Dated(path='.')
    assert not dated.date
    assert not dated.time

def test_dated_sorting():
    dated1 = skrt.core.Dated("19990405_120394")
    dated2 = skrt.core.Dated(auto_timestamp=True)
    assert dated2 > dated1

def test_dated_interval():
    dated = skrt.core.Dated("20010502_120358")
    assert dated.in_date_interval("19991203", "20030402")

def test_files():
    file1 = skrt.core.File('afile.txt')
    file2 = skrt.core.File('bfile.txt')
    assert file2 > file1
    file3 = skrt.core.File('afile.txt')
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
    archive = skrt.core.Archive(tdir)
    assert archive.timestamp == timestamp
    assert len(archive.files) == 1
    assert 'some_file' in archive.files[0].path
    archive2 = skrt.core.Archive(tdir, allow_dirs=True)
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
