from __future__ import division
from collections import defaultdict


def time2ind(time):
    hour, minute, second = [int(time[i:i + 2]) for i in range(0, 6, 2)]
    index = 120 * hour + 2 * minute + (1 if second > 30 else 0)
    return index


def time2sec(time):
    hour, minute, second = [int(time[i:i + 2]) for i in range(0, 6, 2)]
    index = 3600 * hour + 60 * minute + second
    return index


def sort(users):
    unique_users = set()
    for user_id, days in users.items():
        for date, day in days.items():
            unique_users.add(day.user)
    sorted_users = list(unique_users)
    sorted_users.sort(key=lambda user: user.id_)
    return sorted_users


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.iteritems()}
    return d
