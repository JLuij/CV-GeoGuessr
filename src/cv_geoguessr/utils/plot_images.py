import matplotlib.pyplot as plt

def plot_images(images, mean, std):
    fig = plt.figure()

    for i in range(len(images)):
        image = images[i] * std[:, None, None] + mean[:, None, None]

        print(image.shape)

        ax = plt.subplot(1, 10, i + 1)
        fig.set_size_inches(18.5, 10.5)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')

        plt.imshow(image.numpy().transpose(1, 2, 0))

    plt.show()
