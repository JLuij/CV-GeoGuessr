import time
import copy
import torch
import wandb


def train_model(model, criterion, optimizer, scheduler, data_loaders, data_set_sizes, grid_partitioning, CHECKPOINT_FOLDER, device, num_epochs=25):
    """
    Trains a model, based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    :param model: the model to train
    :param criterion: the criterion to use
    :param optimizer: the optimizer to use
    :param scheduler: torch.optim.lr_scheduler
    :param num_epochs:
    :return: a trained model
    """

    since = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        train_acc = 0
        test_acc = 0

        distance_error = {}
        distance_error_count = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            i = 0
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                i += inputs.size(0)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward: track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    _, actual_label_index = torch.max(labels, 1)

                    loss = criterion(outputs, actual_label_index)

                    # Add distance error metric
                    for index, label in enumerate(actual_label_index[preds != actual_label_index].tolist()):
                        distance_error.setdefault(label, 0)
                        distance_error[label] += (grid_partitioning.cells[label].centroid).distance(
                            grid_partitioning.cells[preds[index]].centroid)

                        distance_error_count.setdefault(label, 0)
                        distance_error_count[label] += 1

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                _, actual_label_index = torch.max(labels, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == actual_label_index)
                print(f'{phase} {i}/{data_set_sizes[phase]} - loss: {(running_loss / i):.4f} | accuracy: {(running_corrects.double() / i):.4f}\r', end='')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data_set_sizes[phase]
            epoch_acc = running_corrects.double() / data_set_sizes[phase]

            # Append avg distance error
            avg_distance = 0
            for k in distance_error:
                avg_distance += distance_error[k] / distance_error_count[k]
            avg_distance /= len(distance_error.keys())

            # writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            wandb.log({f"Loss/{phase}": epoch_loss, "epoch": epoch})
            # writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)
            wandb.log({f"Accuracy/{phase}": epoch_acc, "epoch": epoch})
            wandb.log({f"Distance/{phase}": avg_distance, "epoch": epoch})

            if phase == 'train':
                train_acc = epoch_acc
            else:
                test_acc = epoch_acc

            print(f'{phase} loss: {epoch_loss:.4f} | accuracy: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

        print(f"{train_acc}\t{test_acc}")

        torch.save(model.state_dict(),
                   CHECKPOINT_FOLDER + f"epoch_{epoch}.ckpt")

    time_elapsed = time.time() - since

    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model)

    return model