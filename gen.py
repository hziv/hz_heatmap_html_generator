#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2023, Hedi Ziv @ Optalert

"""
HTML generator, populating heatmap data from CSV.
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from logging import basicConfig, StreamHandler, Formatter, getLogger, debug, info, error, DEBUG
from multiprocessing.pool import ThreadPool
from os import cpu_count
from os.path import isdir, isfile, split
from typing import Union, List

from glob2 import glob
from numpy import ndarray
from pandas import read_csv, DataFrame, Series

"""
=========
CONSTANTS
=========
"""

DESCRIPTION = \
    "HTML generator, populating heatmap data from CSV."

CONFIG_VERSION = 0.1  # must specify minimal configuration file version here

# noinspection SpellCheckingInspection
DEFAULT_CFG = (
    f"# gen.py configuration file.\n"
    f"# use # to mark comments.  Note # has to be first character in line.\n"
    f"\n"
    f"config_version = {CONFIG_VERSION}\n"
    f"\n"
    f"############################\n"
    f"# important file locations #\n"
    f"############################\n"
    f"# use slash (/) or single back-slash (\\) as path separators.\n"
    f"\n"
    f"default_base_html_path = templates\\index.ts\n"
    f"default_csv_path = data\\tiny.csv\n"
    f"default_dest_html_path = .\\index.ts\n"
    f"\n"
    f"##########################\n"
    f"# further configurations #\n"
    f"##########################\n"
    f"\n"
    f"google_api_key = <populate Google API key here>"
)

REPLACEMENT_STRING = "    [location: new google.maps.LatLng({}, {}), weight: {}],\n"

"""
================
GLOBAL VARIABLES
================
"""


"""
=========
FUNCTIONS
=========
"""


def find_all_files(base_path: str, search_string: str = '*.odf') -> List[str]:
    assert isinstance(base_path, str)
    assert isdir(base_path)
    assert isinstance(search_string, str)
    files = glob(f"{base_path}/**/{search_string}", case_sensitive=False)
    # debug(f"found {len(files)} files in {base_path}")
    return files


def multiprocessing_wrapper(func, inputs: Union[list, ndarray, Series], enable_multiprocessing: bool = False):
    assert isinstance(inputs, (list, ndarray, Series))
    assert isinstance(enable_multiprocessing, bool)
    if func is not None:
        if isinstance(func, list):  # list of functions
            for f in func:
                multiprocessing_wrapper(f, inputs, enable_multiprocessing)  # recursively call for each function
        if enable_multiprocessing:
            cpu_cnt = cpu_count()
            debug(f"{cpu_cnt} processors found")
            process_pool = ThreadPool(cpu_cnt)
            process_pool.map(func, sorted(inputs))
        else:
            for inp in sorted(inputs):
                func(inp)
        debug(f"multiprocessing_command finished running {len(inputs)}")
    else:
        error("func argument to multiprocessing_command is None")


"""
==================
PROGRESS BAR CLASS
==================
"""


class ProgressBar:
    """ Parse arguments. """

    # class globals
    _title_width = 20
    _width = 32
    _bar_prefix = ' |'
    _bar_suffix = '| '
    _empty_fill = ' '
    _fill = '#'
    progress_before_next = 0
    debug = False
    verbose = False
    quiet = False

    _progress = 0  # between 0 and _width -- used as filled portion of progress bar
    _increment = 0  # between 0 and (_max - _min) -- used for X/Y indication right of progress bar

    def __init__(self, text, maximum=10, minimum=0, verbosemode=''):
        """ Initialising parsing arguments.
        :param text: title of progress bar, displayed left of the progress bar
        :type text: str
        :param maximum: maximal value presented by 100% of progress bar
        :type maximum: int
        :param minimum: minimal value, zero by default
        :type minimum: int
        :param verbosemode: 'debug', 'verbose' or 'quiet'
        :type verbosemode: str
        """

        self.log = getLogger(self.__class__.__name__)
        assert isinstance(text, str)
        assert isinstance(maximum, int)
        assert isinstance(minimum, int)
        assert maximum > minimum
        self._text = text
        self._min = minimum
        self._max = maximum
        self._progress = 0
        self._increment = 0
        # LOGGING PARAMETERS
        assert isinstance(verbosemode, str)
        assert verbosemode in ['', 'debug', 'verbose', 'quiet']
        if verbosemode == 'debug':
            self.debug = True
        elif verbosemode == 'verbose':
            self.verbose = True
        elif verbosemode == 'quiet':
            self.quiet = True
        debug('{} started'.format(self._text))
        self.update()

    def __del__(self):
        """ Destructor. """

        # destructor content here if required
        debug('{} destructor completed.'.format(str(self.__class__.__name__)))

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        """
        Set length of the progress bar in characters.
        :param value: number of characters
        :type value: int
        """
        assert isinstance(value, int)
        assert 0 < value < 80
        self._width = value

    @property
    def title_width(self):
        return self._title_width

    @title_width.setter
    def title_width(self, value):
        """
        Set padding width for text before the progress bar.
        :param value: padding width in number of characters
        :type value: int
        """
        assert isinstance(value, int)
        assert 0 < value < 80
        self._title_width = value

    def next(self, n=1):
        """ Increment progress bar state.
        :param n: increment progress bar by n
        :type n: int
        """

        assert isinstance(n, int)
        assert n >= 0
        if n > 0:
            self._progress += 1 / (n * (self._max - self._min) / self._width)
            if self._progress > self._width:
                self._progress = self._width
            self._increment += n
            if float(self._progress) >= self.progress_before_next + 1 / self._width:
                self.progress_before_next = self._progress
                self.update()

    def update(self, end_char='\r'):
        """ Update progress bar on console.
        :param end_char: character used to command cursor to get back to beginning of line without carriage return.
        :type end_char: str
        """

        assert isinstance(end_char, str)
        diff = self._max - self._min
        bar = self._fill * int(self._progress)
        empty = self._empty_fill * (self._width - int(self._progress))
        if not self.debug and not self.verbose and not self.quiet:
            print("{:<{}.{}s}{}{}{}{}{}/{}".format(self._text, self._title_width, self._title_width,
                                                   self._bar_prefix, bar, empty, self._bar_suffix,
                                                   str(self._increment), str(diff)), end=end_char)

    def finish(self):
        """ Clean up and release handles. """

        self._progress = self._width
        self._increment = self._max - self._min
        if self._increment < 0:
            self._increment = 0
        self.update('\n')
        debug('{} finished'.format(self._text))


"""
==================
CONFIG FILE PARSER
==================
"""


class Config:
    """
    Configuration file parser.
    Note all return values in string format.
    """

    _config = {}

    def __init__(self, config_version: float, path: Union[None, str] = None, default_cfg: str = DEFAULT_CFG):
        """
        Initialisations
        @param path: Path to config file.  Local directory by default
        @param config_version: must specify minimum config file version here
        @param default_cfg: Default CFG
        """

        self.log = getLogger(self.__class__.__name__)
        assert isinstance(config_version, float)
        assert isinstance(default_cfg, str)
        if path is None:
            path = f'./{self.__class__.__name__}.cfg'
            debug(f'Path to configuration file not specified.  Using: {path}')
        else:
            assert isinstance(path, str)
            if isfile(path):
                debug(f'Configuration file detected as {path}')
            else:
                debug(f"Configuration file path specified {path} does not exist, creating default")
                try:
                    with open(path, 'wt') as config_file:
                        config_file.write(default_cfg)
                    config_file.close()
                    debug(f'{path} file created')
                except PermissionError as e:
                    error(f"can not access file {path}. Might be opened by another application. "
                          f"Error returned: {e}")
                    raise PermissionError(f"can not access file {path}. Might be opened by another application."
                                          f"Error returned: {e}")
                except OSError as e:
                    error(f"can not access file {path}. Might be opened by another application. "
                          f"Error returned: {e}")
                    raise OSError(f"can not access file {path}. Might be opened by another application."
                                  f"Error returned: {e}")
                except UserWarning as e:
                    debug(f"{path} file empty - {e}")
                    raise UserWarning(f"can not access file {path}. Might be opened by another application. "
                                      f"Error returned: {e}")
        # read file
        try:
            with open(path, 'rt') as config_file:
                for line in config_file:
                    # skip comment or empty lines
                    if not (line.startswith('#') or line.startswith('\n')):
                        var_name, var_value = line.split('=')
                        var_name = var_name.strip(' \t\n\r')
                        var_value = var_value.strip(' \t\n\r')
                        if ',' in var_value:
                            self._config[var_name] = [x.strip(' \t\n\r') for x in var_value.split(',')]
                        else:
                            self._config[var_name] = var_value
            config_file.close()
            info(f'Configuration file {path} read.')
            debug('Config file contents:')
            # log config file content
            for key in self._config.keys():
                debug(f"config[{key}] = {self._config[key]}")
            debug('End of Config file content.')
        except PermissionError as e:
            error(f"can not access file {path}. Might be opened by another application. "
                  f"Error returned: {e}")
            raise PermissionError(f"can not access file {path}. Might be opened by another application."
                                  f"Error returned: {e}")
        except OSError as e:
            error(f"can not access file {path}. Might be opened by another application. "
                  f"Error returned: {e}")
            raise OSError(f"can not access file {path}. Might be opened by another application."
                          f"Error returned: {e}")
        except UserWarning as e:
            debug(f"{path} file empty - {e}")
            raise UserWarning(f"can not access file {path}. Might be opened by another application. "
                              f"Error returned: {e}")
        # verify config_version
        file_version = self.__getitem__("config_version")
        fault_msg = f"Config file {split(path)[1]} version ({file_version}) is lower than " \
                    f"expected {config_version}. Consider deleting and re-run code to " \
                    f"generate default config file "\
                    f"with latest version."
        try:
            file_version = float(file_version)
        except ValueError as e:
            error(f"config_version value in file is not a valid float. error: {e}")
        if not isinstance(file_version, (float, int)):
            raise ValueError(fault_msg)
        if file_version < config_version:
            raise ValueError(fault_msg)

    def __del__(self):
        """ Destructor. """

        # destructor content here if required
        debug(f'{str(self.__class__.__name__)} destructor completed.')

    def __getitem__(self, item: str) -> Union[str, list]:
        """
        return parameter from configuration file
        @param item: name of parameter
        @return: value of parameter from configuration file
        """
        assert isinstance(item, str)
        if item in self._config:
            try:
                ret = self._config[item]
            except KeyError as e:
                error(f'parameter requested {item} not in config file, error: {e}')
                return ''
            if isinstance(ret, str) and ret.lower() == 'none':
                ret = None
            return ret
        else:
            info(f'parameter requested {item} not in config file')
            return ''


"""
===============
GENERATOR CLASS
===============
"""


class HtmlGenerator:
    """
    HTML Generator Class.
    """

    _src_html = ''
    _coordinates = DataFrame()
    _dest_dir = ''

    def __init__(self, src_path: str, csv_path: str, dest_path: str) -> None:
        """
        Initialisations
        """

        _ = getLogger(self.__class__.__name__)

        assert isinstance(src_path, str)
        assert isfile(src_path)
        progress_bar = ProgressBar('reading', 2)
        with open(src_path) as src_file:
            self._src_html = src_file.readlines()
        progress_bar.next()
        assert isinstance(csv_path, str)
        self._coordinates = read_csv(csv_path, index_col=False)
        debug(f"{split(csv_path)[1]} read with size {self._coordinates.shape} and columns: "
              f"{list(self._coordinates.columns)}")
        progress_bar.finish()
        assert isinstance(dest_path, str)
        self._dest_path = dest_path

    def __del__(self):
        """ Destructor. """
        # destructor content here if required
        debug(f'{str(self.__class__.__name__)} destructor completed.')

    @staticmethod
    def get_coordinate_as_string_lines(df: DataFrame) -> Series:
        #    {location: new google.maps.LatLng(###LATITUDE###, ###LONGITUDE###), weight: ###JDS###},
        assert isinstance(df, DataFrame)
        assert all(col in df.columns for col in ["Longitude", "Latitude", "Data"]), "invalid coordinate CSV format"

        def populate_coordinates_into_html_line(row: Series) -> str:
            progress_bar.next()
            ret = REPLACEMENT_STRING.format(row["Latitude"],
                                            row["Longitude"],
                                            row["Data"] / 100).replace('[', '{').replace(']', '}')
            return ret

        progress_bar = ProgressBar('converting', df.shape[0])
        df["line_as_string"] = df.apply(populate_coordinates_into_html_line, axis=1)
        return df["line_as_string"]

    def run(self):
        """
        Main program.
        """
        split_idx = self._src_html.index("REPLACE_THIS_LINE\n")
        begin_text = self._src_html[:split_idx]
        end_text = self._src_html[split_idx + 1:]
        coordinate_series = self.get_coordinate_as_string_lines(self._coordinates)
        with open(self._dest_path, 'wt') as dest_file:
            progress_bar = ProgressBar('writing', 3)
            dest_file.writelines(begin_text)
            progress_bar.next()
            dest_file.writelines(coordinate_series.to_numpy(dtype=str))
            progress_bar.next()
            dest_file.writelines(end_text)
            progress_bar.finish()


"""
========================
ARGUMENT SANITY CHECKING
========================
"""


class ArgumentsAndConfigProcessing:
    """
    Argument parsing and default value population (from config).
    """

    _src_path = ''
    _csv_path = ''
    _dest_path = ''

    def __init__(self, src_path: str, csv_path: str, dest_path: str, config_path: str) -> None:
        """
        Initialisations
        """

        _ = getLogger(self.__class__.__name__)

        assert isinstance(config_path, str)
        config = Config(CONFIG_VERSION, config_path, DEFAULT_CFG)

        assert isinstance(src_path, str)
        if src_path == '':  # get default from config
            src_path = config["default_base_html_path"]
            debug(f"default SRC path read from config: {src_path}")
        self._src_path = src_path

        assert isinstance(csv_path, str)
        if csv_path == '':  # get default from config
            csv_path = config["default_csv_path"]
            debug(f"default CSV path read from config: {csv_path}")
        self._csv_path = csv_path

        assert isinstance(dest_path, str)
        if dest_path == '':  # get default from config
            dest_path = config["default_dest_html_path"]
            debug("default dest directory path read from config")
        self._dest_path = dest_path

    def __del__(self):
        """ Destructor. """
        # destructor content here if required
        debug(f'{str(self.__class__.__name__)} destructor completed.')

    def run(self):
        """
        Main program.
        """
        generator = HtmlGenerator(src_path=self._src_path,
                                  csv_path=self._csv_path,
                                  dest_path=self._dest_path)
        generator.run()


"""
======================
COMMAND LINE INTERFACE
======================
"""


def main():
    """ Argument Parser and Main Class instantiation. """

    # ---------------------------------
    # Parse arguments
    # ---------------------------------

    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)

    no_extension_default_name = parser.prog.rsplit('.', 1)[0]
    parser.add_argument('--src', dest='src_path', nargs=1, type=str, default=[''],
                        help='path to template HTML file, default (unspecified) from config')
    parser.add_argument('--csv', dest='csv_path', nargs=1, type=str, default=[''],
                        help='path to heatmap CSV file, default (unspecified) from config')
    parser.add_argument('--dest', dest='dest_path', nargs=1, type=str, default=[''],
                        help='path to destination HTML file, default (unspecified) from config')
    parser.add_argument('-c', dest='config', nargs=1, type=str, default=[f"./{no_extension_default_name}.cfg"],
                        help=f"path to config file. \"./{no_extension_default_name}.cfg\" by default")

    parser.add_argument('-d', '--debug', help='sets verbosity to display debug level messages',
                        action="store_true")

    args = parser.parse_args()

    # ---------------------------------
    # Preparing LogFile formats
    # ---------------------------------

    assert all(
        isinstance(field, list) and len(field) == 1 and isinstance(field[0], str)
        for field in [args.src_path, args.csv_path, args.dest_path, args.config]
    )

    log_filename = f'{no_extension_default_name}.log'
    try:
        basicConfig(filename=log_filename, filemode='a', datefmt='%Y/%m/%d %I:%M:%S %p', level=DEBUG,
                    format='%(asctime)s, %(threadName)-8s, %(name)-15s %(levelname)-8s - %(message)s')
    except PermissionError as err:
        raise PermissionError(f'Error opening log file {log_filename}. File might already be opened by another '
                              f'application. Error: {err}\n')

    console = StreamHandler()
    if args.debug:
        console.setLevel(DEBUG)
    formatter = Formatter('%(threadName)-8s, %(name)-15s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    getLogger('').addHandler(console)

    getLogger('main')
    info(f"Successfully opened log file named: {log_filename}")
    debug(f"Program run with the following arguments: {str(args)}")

    # ---------------------------------
    # Debug mode
    # ---------------------------------

    assert isinstance(args.debug, bool)

    # ---------------------------------
    # Instantiation
    # ---------------------------------

    arg_processing = ArgumentsAndConfigProcessing(src_path=args.src_path[0],
                                                  csv_path=args.csv_path[0],
                                                  dest_path=args.dest_path[0],
                                                  config_path=args.config[0])
    arg_processing.run()
    debug('Program execution completed. Starting clean-up.')


if __name__ == "__main__":
    main()
