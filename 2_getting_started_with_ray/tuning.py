from dependencies import *

import ray
from ray import tune

# differences from what we've seen before, this is an end to end training function
# where we are loading the dataset running our complete train and test loop whilst
# 
def e2e_simple_training(config):
    
    #threadsafe
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    
    # chose your CV strategy
    splitter = StratifiedKFold(n_splits=5)
    
    # run k fold training and testing
    f1_scores = [] # keep hold of all individual scores
    for train_ind, test_ind in splitter.split(X, y):
        pipeline = make_pipeline(RobustScaler(),
                                  RandomForestClassifier(random_state=42))

        pipeline.set_params(**config)
        pipeline.fit(X[train_ind], y[train_ind])
        
        y_pred = pipeline.predict(X[test_ind])
        
        f1_scores.append(f1_score(y_pred, y[test_ind]))
    
    # use tunes reporter
    tune.track.log(mean_f1_score=np.array(f1_scores).mean(),
                std_f1_score=np.array(f1_scores).std(),
                # and we can actually add any metrics we like
                done=True)
    
    
from scipy.stats import norm

def plot_some_tune_results(df):
    fig, ax = plt.subplots(1, 1, figsize=(16,6))
    x = np.linspace(0.85, 1.0, 100)

    n_estimators = df['config/randomforestclassifier__n_estimators'].values.tolist()

    lines = []
    for mu, sigma in zip(df['mean_f1_score'], df['std_f1_score']):
        pdf = norm.pdf(x, mu, sigma)
        line, = ax.plot(x, pdf, alpha=0.6)
        ax.axvline(mu, color=line.get_color())
        ax.text(mu, pdf.max(), f"{mu:.3f}", color=line.get_color(), fontsize=14)
        lines.append(line)

    plt.legend(handles=lines, labels=n_estimators, title="n estimators")
    ax.set_title(f"Average F1 Scores")