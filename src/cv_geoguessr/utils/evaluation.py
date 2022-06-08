import torch as th


# Running the model evaluation
def create_confusion_matrix(model, partitions, dataloader, predictions=True, device='cuda:0'):
    n = len(partitions.cells)
    confusion_matrix = th.zeros((n, n)).to(device)

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

            if predictions:
                confusion_matrix[l, :] += pred
            else:
                confusion_matrix[l, :] += output

            i += 1

    return confusion_matrix
