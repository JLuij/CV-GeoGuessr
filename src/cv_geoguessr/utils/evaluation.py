import torch as th


# Running the model evaluation
def create_confusion_matrix(model, partitions, dataloader, predictions=True, device='cuda:0'):
    n = len(partitions.cells)
    confusion_matrix = th.zeros((n, n)).to(device)
    samples = n * [0]

    i = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with th.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = th.max(outputs, 1)

        for label, output, pred in zip(labels, outputs, preds):
            l = th.argmax(label)
            # print(l)

            samples[l] += 1

            if predictions:
                confusion_matrix[l, :] += pred
            else:
                confusion_matrix[l, :] += output

            i += 1

    return confusion_matrix, samples


def test_coverage(partitions, dataloader):
    n = len(partitions)
    samples = n * [0]

    for inputs, labels in dataloader:
        for label in labels:
            l = th.argmax(label)
            samples[l] += 1

    return samples