#!/usr/bin/env python

# Copyright (c) 2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a simple watchdog timer to detect timeouts
It is for example used in the ScenarioManager
"""
import signal
import os
import traceback

class Watchdog(object):
    """
    Simple watchdog timer to detect timeouts

    Args:
        timeout (float): Timeout value of the watchdog [seconds]. If triggered, raises a KeyboardInterrupt.
        interval (float): Time between timeout checks [seconds]. Defaults to 1% of the timeout.

    Attributes:
        _timeout (float): Timeout value of the watchdog [seconds].
        _interval (float): Time between timeout checks [seconds].
        _failed (bool): True if watchdog exception occured, false otherwise
    """

    def __init__(self, timeout=1.0, interval=None):
        """Class constructor"""
        self._timeout = timeout + 1.0
        self._watchdog_stopped = False
        self._failed = False
        signal.signal(signal.SIGALRM, self.handler)

    def handler(self, signum, frame):
        print(f'Watchdog exception - Timeout of {self._timeout} seconds occured')
        traceback.print_stack(frame)
        self._failed = True
        # The code catches exceptions. We need to force a shutdown, so that the restart scripts can do their work
        os._exit(7)
        raise Exception("Watchdog timeout")
    def start(self):
        """Start the watchdog"""
        signal.alarm(int(self._timeout))

    def stop(self):
        """Stop the watchdog"""
        signal.alarm(0)

    def get_status(self):
        """returns False if watchdog exception occured, True otherwise"""
        return not self._failed
