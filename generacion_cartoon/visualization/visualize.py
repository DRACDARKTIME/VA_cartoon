import matplotlib.pyplot as plt


def plot_img_original_generated(model, num_img=10,test_ds=None):
    for img in test_ds:
        mean, logvar = model.encode(img)
        z = model.reparameterize(mean, logvar)
        generated_image = model.decode(z, apply_sigmoid=True)
        plt.imshow(img[num_img], cmap='gray')
        plt.axis('off')
        plt.show()
        plt.imshow(generated_image[num_img].numpy().squeeze(), cmap='gray')
        plt.axis('off')
        plt.show()
        break