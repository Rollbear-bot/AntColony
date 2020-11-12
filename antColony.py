# -*- coding: utf-8 -*-
# @Time: 2020/11/11 14:32
# @Author: Rollbear
# @Filename: antColony.py

import numpy as np
import random

from entity.roulette import Roulette


class AntColony:
    def __init__(self, alpha, beta, rho, m, num_iter):
        # 算法参数
        self.alpha = alpha  # 权重alpha
        self.beta = beta  # 权重beta
        self.rho = rho  # 信息素更新权重
        self.m = m  # 族群规模
        self.num_iter = num_iter  # 迭代次数

        # 维护的数据结构
        self.taboo = None  # 禁忌表，路径记忆向量
        self.tau = None  # 信息素表
        self.paths = None  # 路径长度表（邻接表）
        self.ant_tract = None  # 记录蚂蚁走过的路径
        # self.global_shortest = None  # 全局最短路径
        self.iter_shortest = []  # 记录每轮迭代的局部最优路径

    def run(self, adj_mat, debug=False):
        self.paths = np.array(adj_mat)

        # 使用贪婪法初始化信息素
        self.alg_init(debug)

        # 执行num iter次迭代
        for i_iter in range(self.num_iter):
            # 在每轮迭代，重新初始化禁忌表与蚂蚁路径
            self.taboo = {}
            self.ant_tract = {}
            if debug:
                print("\n" + "=" * 10 + f"iter {i_iter + 1}" + "=" * 10)
            self.run_iter(debug)

    def run_iter(self, debug):
        # 为每只蚂蚁随机选取出发城市
        ant_cur_cities = random.choices(range(len(self.paths)), k=self.m)
        # 保存蚂蚁的出发城市，在遍历其他城市后需要回来
        ant_start_cities = ant_cur_cities.copy()
        if debug:
            self.print_title("Set Off City")
            for i_ant, city in enumerate(ant_cur_cities):
                print(f"Ant{i_ant}: {city}")

        end = False  # 算法到某个蚂蚁的禁忌表满（没有可用城市）时停止
        while not end:
            # 为每只蚂蚁选择下一城市
            for i_ant, cur_city in enumerate(ant_cur_cities):
                # 当前城市（已到达过的城市）加入禁忌表
                if i_ant not in self.taboo:
                    self.taboo[i_ant] = [cur_city]
                else:
                    self.taboo[i_ant].append(cur_city)

                # 计算每个允许前往的城市的路径权重
                city_weight = {}
                avail_cities = [i_city for i_city, _ in enumerate(self.paths[cur_city])
                                if i_city not in self.taboo.get(i_ant, [])]
                if len(avail_cities) == 0:
                    end = True
                    break
                for avail_city in avail_cities:
                    # 计算访问某个城市的概率（权重）
                    city_weight[avail_city] = self.probability(cur_city, avail_city)

                # 轮盘赌选择下一城市
                next_city = Roulette(city_weight).roll()
                # 蚂蚁前往了下一城市，更新记录蚂蚁当前位置的表ant_cur_cities
                ant_cur_cities[i_ant] = next_city
                # 在蚂蚁路径记录中添加这条记录
                if i_ant not in self.ant_tract:
                    self.ant_tract[i_ant] = [(cur_city, next_city)]
                else:
                    self.ant_tract[i_ant].append((cur_city, next_city))

                if debug:
                    self.print_title("Ant Report")
                    print(f"Ant {i_ant}: {city_weight}")
                    print(f"next city: {next_city}")

        # 每只蚂蚁回到出发的城市
        for i_ant in range(self.m):
            cur_city = ant_cur_cities[i_ant]
            next_city = ant_start_cities[i_ant]
            self.ant_tract[i_ant].append((cur_city, next_city))

            if debug:
                self.print_title("Ant Report")
                print(f"Ant {i_ant}: back to start city.")
                print(f"next city: {next_city}")

        # 本轮迭代结束，更新信息素，并计算本次迭代的局部最优
        local_shortest = self.update_tau(debug)
        # 存储本轮迭代的局部最优路径
        self.iter_shortest.append(local_shortest)

    def alg_init(self, debug):
        """初始化信息素：首先使用贪婪法计算次短路径信息素"""
        self.tau = np.zeros(shape=self.paths.shape)
        greed_taboo = []
        cur_city = 0
        path_length = 0

        if debug:
            self.print_title("Alg Init")
            print("Greedy Path:", end=" ")

        while True:
            greed_taboo.append(cur_city)  # 走过的城市加入禁忌表（局部变量）
            if debug:
                print(cur_city, end=" ")
            next_city_cluster = sorted([(i, elem) for i, elem in enumerate(self.paths[cur_city])
                                        if i not in greed_taboo], key=lambda item: item[1])
            if len(next_city_cluster) == 0:
                break
            next_city = sorted([(i, elem) for i, elem in enumerate(self.paths[cur_city])
                                if i not in greed_taboo], key=lambda item: item[1])[0][0]
            path_length += self.paths[cur_city, next_city]
            cur_city = next_city

        path_length += self.paths[cur_city, 0]

        tau_0 = self.m / path_length  # 使用贪婪最短路径计算初始信息素浓度
        if debug:
            print()
            print(f"τ0: {tau_0}")
        self.tau[:, :] = tau_0  # 所有路径上的信息素浓度初始化为tau0

    def update_tau(self, debug):
        """更新信息素，并返回本次迭代的局部最优路径"""
        if debug:
            self.print_title("Updating tau")

        ant_path_length = []
        ant_delta_tau = {}  # 存储每只蚂蚁对每条路径的信息素增量

        # 计算每只蚂蚁在一轮迭代中走过的路径长度，计算信息素更新量
        for i_ant, tracts in self.ant_tract.items():
            # 计算蚂蚁在这次迭代走过的路径长
            path_length = sum([self.paths[tract[0], tract[1]] for tract in tracts])
            if debug:
                print(f"Ant {i_ant} path length: {path_length}")
            ant_path_length.append(path_length)
            # 信息素增量等于蚂蚁这次迭代走过的路径长的倒数
            delta_tau = {(i, j): 1 / path_length for i, j in tracts}

            ant_delta_tau[i_ant] = delta_tau

        # 更新信息素
        for i in range(len(self.tau)):
            for j in range(len(self.tau[0])):
                # 计算所有蚂蚁对这条边的更新量
                delta_sum = sum([delta.get((i, j), 0) for delta in ant_delta_tau.values()])
                self.tau[i, j] = (1 - self.rho) * self.tau[i, j] + delta_sum

        if debug:
            print("tau:")
            print(self.tau)

        # 返回局部最优路径
        return min(ant_path_length)

    def eta(self, i, j):
        """以路径距离的倒数作为η（课本方法）"""
        return 1 / self.paths[i, j]

    def probability(self, cur_city, target_city):
        """计算蚂蚁从城市A前往城市B的概率"""
        return (self.tau[cur_city, target_city] ** self.alpha) * \
               (self.eta(cur_city, target_city) ** self.beta)

    @staticmethod
    def print_title(message):
        print("-" * 10 + message + "-" * 10)
