print('Loading dependencies we have already seen...')
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.set_palette(sns.color_palette("bright", 8))

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

print('Importing ray...')
import ray
from ray import tune

print('Done...')

from scipy.stats import norm

def plot_some_tune_results(df, rng=(0.85, 1.0)):
    fig, ax = plt.subplots(1, 1, figsize=(16,6))
    x = np.linspace(rng[0], rng[1], 100)

    lines = []
    for mu, sigma in zip(df['mean_f1_score'], df['std_f1_score']):
        pdf = norm.pdf(x, mu, sigma)
        line, = ax.plot(x, pdf, alpha=0.6)
        ax.axvline(mu, color=line.get_color())
        ax.text(mu, pdf.max(), f"{mu:.3f}", color=line.get_color(), fontsize=14)
        lines.append(line)

    ax.set_title(f"Average F1 Scores")