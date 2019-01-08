from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np

class ScoreLogger:

    def __init__(self, env_name, version_name='.', average_score_to_solve=1000,
                 consecutive_runs_to_solve=10):
        self.env_name = env_name
        self.scores_csv_path = "{}/scores/scores.csv".format(version_name)
        self.scores_png_path = "{}/scores/scores.png".format(version_name)
        self.solved_csv_path = "{}/scores/solved.csv".format(version_name)
        self.solved_png_path = "{}/scores/solved.png".format(version_name)
        self.average_score_to_solve = average_score_to_solve
        self.consecutive_runs_to_solve = consecutive_runs_to_solve
        self.scores = deque(maxlen=self.consecutive_runs_to_solve)

        if os.path.exists(self.scores_png_path):
            os.remove(self.scores_png_path)
        if os.path.exists(self.scores_csv_path):
            os.remove(self.scores_csv_path)

    def add_score(self, score, run):
        self._save_csv(self.scores_csv_path, score)
        self._save_png(input_path=self.scores_csv_path,
                       output_path=self.scores_png_path,
                       x_label="runs",
                       y_label="scores",
                       average_of_n_last=self.consecutive_runs_to_solve,
                       show_goal=True,
                       show_trend=True,
                       show_legend=True)
        self.scores.append(score)
        mean_score = mean(self.scores)
        print("Scores: (min: " + str(min(self.scores)) + ", avg: " + str(mean_score) + ", max: " + str(max(self.scores)) + ")\n")
        if mean_score >= self.average_score_to_solve and len(self.scores) >= self.consecutive_runs_to_solve:
            solve_score = run-self.consecutive_runs_to_solve
            print("Solved in " + str(solve_score) + " runs, " + str(run) + " total runs.")
            self._save_csv(self.solved_csv_path, solve_score)
            self._save_png(input_path=self.solved_csv_path,
                           output_path=self.solved_png_path,
                           x_label="trials",
                           y_label="steps before solve",
                           average_of_n_last=None,
                           show_goal=False,
                           show_trend=False,
                           show_legend=False)
            exit()

    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))

        plt.subplots()
        plt.plot(x, y, label="score per run")

        average_range = average_of_n_last if average_of_n_last is not None else len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--", label="last " + str(average_range) + " runs average")

        if show_goal:
            plt.plot(x, [self.average_score_to_solve] * len(x), linestyle=":", label=str(self.average_score_to_solve) + " score average goal")

        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend")

        plt.title(self.env_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])
