import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loaders import load_cancer_data
from utils import pairplot



df = load_cancer_data(only_data=True)


data = pd.read_csv("./results.csv", index_col=False)
fit_time = data.loc[0, :][1:]
score_time = data.loc[1, :][1:]
test_score = 100 * data.loc[2, :][1:]


pairplot(df, hue='target', 
    vars=['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'mean symmetry'], 
    show=False, save='docs/images/pairplot.png')

plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(), annot=True)
plt.savefig('docs/images/heatmap.png')

plt.figure()
plt.plot(fit_time)
plt.xlabel('n_components')
plt.ylabel('Secs.')
plt.title('Fit time')
plt.savefig('docs/images/fit_time.png')
plt.grid(True)


plt.figure()
plt.plot(score_time)
plt.xlabel('n_components')
plt.ylabel('Secs.')
plt.title('Score time')
plt.savefig('docs/images/score_time.png')
plt.grid(True)


plt.figure()
plt.plot(test_score)
plt.xlabel('n_components')
plt.ylabel('%')
plt.title('Test Score')
plt.savefig('docs/images/score.png')
plt.grid(True)

plt.show()