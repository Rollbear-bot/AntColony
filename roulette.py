# -*- coding: utf-8 -*-
# @Time: 2020/11/11 14:22
# @Author: Rollbear
# @Filename: roulette.py

import random


class Roulette:
    def __init__(self, weight: dict):
        items = [(key, value) for key, value in weight.items()]
        self.index = [elem[0] for elem in items]
        self.weight_lt = [elem[1] for elem in items]

    def roll(self):
        hit = None
        volume = sum(self.weight_lt)
        key = random.random() * volume
        for i, sample in enumerate(self.weight_lt):
            if sum(self.weight_lt[:i + 1]) >= key:
                hit = i
        return self.index[hit]
