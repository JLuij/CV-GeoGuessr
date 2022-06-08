import seaborn as sns


def plot_confusion_matrix(confusion_matrix):
    sns.set(rc = {'figure.figsize':(10,10)})
    ax = sns.heatmap(confusion_matrix.to('cpu'), annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');