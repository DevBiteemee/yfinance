#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# yfinance - market data downloader
# https://github.com/ranaroussi/yfinance
#
# Copyright 2017-2019 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import logging
import traceback

import concurrent.futures as _futures
import pandas as _pd

from . import Ticker, utils
from .data import YfData
from . import shared


@utils.log_indent_decorator
def download(tickers, start=None, end=None, actions=False, threads=True,
             ignore_tz=None, group_by='column', auto_adjust=False, back_adjust=False,
             repair=False, keepna=False, progress=True, period="max", interval="1d",
             prepost=False, proxy=None, rounding=False, timeout=10, session=None,
             multi_level_index=True):
    """Download yahoo tickers
    :Parameters:
        tickers : str, list
            List of tickers to download
        period : str
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Either Use period parameter or use start and end
        interval : str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
        start: str
            Download start date string (YYYY-MM-DD) or _datetime, inclusive.
            Default is 99 years ago
            E.g. for start="2020-01-01", the first data point will be on "2020-01-01"
        end: str
            Download end date string (YYYY-MM-DD) or _datetime, exclusive.
            Default is now
            E.g. for end="2023-01-01", the last data point will be on "2022-12-31"
        group_by : str
            Group by 'ticker' or 'column' (default)
        prepost : bool
            Include Pre and Post market data in results?
            Default is False
        auto_adjust: bool
            Adjust all OHLC automatically? Default is False
        repair: bool
            Detect currency unit 100x mixups and attempt repair
            Default is False
        keepna: bool
            Keep NaN rows returned by Yahoo?
            Default is False
        actions: bool
            Download dividend + stock splits data. Default is False
        threads: bool / int
            How many threads to use for mass downloading. Default is True
        ignore_tz: bool
            When combining from different timezones, ignore that part of datetime.
            Default depends on interval. Intraday = False. Day+ = True.
        proxy: str
            Optional. Proxy server URL scheme. Default is None
        rounding: bool
            Optional. Round values to 2 decimal places?
        timeout: None or float
            If not None stops waiting for a response after given number of
            seconds. (Can also be a fraction of a second e.g. 0.01)
        session: None or Session
            Optional. Pass your own session object to be used for all requests
        multi_level_index: bool
            Optional. Always return a MultiIndex DataFrame? Default is False
    """
    logger = utils.get_yf_logger()

    if logger.isEnabledFor(logging.DEBUG):
        if threads:
            # With DEBUG, each thread generates a lot of log messages.
            # And with multi-threading, these messages will be interleaved, bad!
            # So disable multi-threading to make log readable.
            logger.debug('Disabling multithreading because DEBUG logging enabled')
            threads = False
        if progress:
            # Disable progress bar, interferes with display of log messages
            progress = False

    if ignore_tz is None:
        # Set default value depending on interval
        if interval[1:] in ['m', 'h']:
            # Intraday
            ignore_tz = False
        else:
            ignore_tz = True

    # create ticker list
    tickers = tickers if isinstance(
        tickers, (list, set, tuple)) else tickers.replace(',', ' ').split()

    # accept isin as ticker
    shared._ISINS = {}
    _tickers_ = []
    for ticker in tickers:
        if utils.is_isin(ticker):
            isin = ticker
            ticker = utils.get_ticker_by_isin(ticker, proxy, session=session)
            shared._ISINS[ticker] = isin
        _tickers_.append(ticker)

    tickers = _tickers_

    tickers = list(set([ticker.upper() for ticker in tickers]))

    if progress:
        shared._PROGRESS_BAR = utils.ProgressBar(len(tickers), 'completed')

    # reset dfs
    dfs = {}
    shared._ERRORS = {}
    shared._TRACEBACKS = {}

    # Ensure data initialised with session.
    YfData(session=session)

    # download using threads
    if threads:
        with _futures.ThreadPoolExecutor() as executor:
            futures = []
            for i, ticker in enumerate(tickers):
                futures.append(
                    executor.submit(_download_one_threaded, ticker=ticker, period=period,
                                    interval=interval, start=start, end=end, prepost=prepost,
                                    actions=actions, auto_adjust=auto_adjust,
                                    back_adjust=back_adjust,
                                    progress=(progress and i > 0), proxy=proxy,
                                    rounding=rounding
                    )
                )

            for future in _futures.as_completed(futures):
                ticker, data = future.result()
                print(f'Ticker {ticker}, data {data}')
                dfs[ticker.upper()] = data
                if progress:
                    shared._PROGRESS_BAR.animate()
    # download synchronously
    else:
        for i, ticker in enumerate(tickers):
            ticker, data = _download_one(ticker, period=period, interval=interval,
                                 start=start, end=end, prepost=prepost,
                                 actions=actions, auto_adjust=auto_adjust,
                                 back_adjust=back_adjust, repair=repair, keepna=keepna,
                                 proxy=proxy,
                                 rounding=rounding, timeout=timeout)
            dfs[ticker.upper()] = data
            if progress:
                shared._PROGRESS_BAR.animate()

    if progress:
        shared._PROGRESS_BAR.completed()

    if shared._ERRORS:
        # Send errors to logging module
        logger = utils.get_yf_logger()
        logger.error('\n%.f Failed download%s:' % (
            len(shared._ERRORS), 's' if len(shared._ERRORS) > 1 else ''))

        # Log each distinct error once, with list of symbols affected
        errors = {}
        for ticker in shared._ERRORS:
            err = shared._ERRORS[ticker]
            err = err.replace(f'{ticker}', '%ticker%')
            if err not in errors:
                errors[err] = [ticker]
            else:
                errors[err].append(ticker)
        for err in errors.keys():
            logger.error(f'{errors[err]}: ' + err)

        # Log each distinct traceback once, with list of symbols affected
        tbs = {}
        for ticker in shared._TRACEBACKS:
            tb = shared._TRACEBACKS[ticker]
            tb = tb.replace(f'{ticker}', '%ticker%')
            if tb not in tbs:
                tbs[tb] = [ticker]
            else:
                tbs[tb].append(ticker)
        for tb in tbs.keys():
            logger.debug(f'{tbs[tb]}: ' + tb)

    if ignore_tz:
        for tkr in dfs.keys():
            if (dfs[tkr] is not None) and (dfs[tkr].shape[0] > 0):
                dfs[tkr].index = dfs[tkr].index.tz_localize(None)

    try:
        data = _pd.concat(dfs.values(), axis=1, sort=True,
                          keys=dfs.keys(), names=['Ticker', 'Price'])
    except Exception:
        _realign_dfs(dfs)
        data = _pd.concat(dfs.values(), axis=1, sort=True,
                          keys=dfs.keys(), names=['Ticker', 'Price'])
    data.index = _pd.to_datetime(data.index, utc=True)
    # switch names back to isins if applicable
    data.rename(columns=shared._ISINS, inplace=True)

    if group_by == 'column':
        data.columns = data.columns.swaplevel(0, 1)
        data.sort_index(level=0, axis=1, inplace=True)

    if not multi_level_index and len(tickers) == 1:
        data = data.droplevel(0 if group_by == 'ticker' else 1, axis=1).rename_axis(None, axis=1)

    return data


def _realign_dfs(dfs):
    idx_len = 0
    idx = None

    for df in dfs.values():
        if len(df) > idx_len:
            idx_len = len(df)
            idx = df.index

    for key in dfs.keys():
        try:
            dfs[key] = _pd.DataFrame(
                index=idx, data=dfs[key]).drop_duplicates()
        except Exception:
            dfs[key] = _pd.concat([
                utils.empty_df(idx), dfs[key].dropna()
            ], axis=0, sort=True)

        # remove duplicate index
        dfs[key] = dfs[key].loc[
            ~dfs[key].index.duplicated(keep='last')]


def _download_one_threaded(ticker, start=None, end=None,
                           auto_adjust=False, back_adjust=False, repair=False,
                           actions=False, progress=True, period="max",
                           interval="1d", prepost=False, proxy=None,
                           keepna=False, rounding=False, timeout=10):
    data = _download_one(ticker, start, end, auto_adjust, back_adjust, repair,
                         actions, period, interval, prepost, proxy, rounding,
                         keepna, timeout)
    if progress:
        shared._PROGRESS_BAR.animate()
    return data


def _download_one(ticker, start=None, end=None,
                  auto_adjust=False, back_adjust=False, repair=False,
                  actions=False, period="max", interval="1d",
                  prepost=False, proxy=None, rounding=False,
                  keepna=False, timeout=10):
    data = utils.empty_df()
    try:
        data = Ticker(ticker).history(
                period=period, interval=interval,
                start=start, end=end, prepost=prepost,
                actions=actions, auto_adjust=auto_adjust,
                back_adjust=back_adjust, repair=repair, proxy=proxy,
                rounding=rounding, keepna=keepna, timeout=timeout,
                raise_errors=True
        )
    except Exception as e:
        # glob try/except needed as current thead implementation breaks if exception is raised.
        shared._ERRORS[ticker.upper()] = repr(e)
        shared._TRACEBACKS[ticker.upper()] = traceback.format_exc()

    return ticker, data
