import Datasets
import Architectures
import Optimizers
import Losses
import Evaluators
import Utils

dataset = Datasets.MNIST()
model = Architectures.DenseNetwork(img_shape=dataset.img_shape, latent_dim=100)
#model = Architectures.DCGan(img_shape=dataset.img_shape, latent_dim=100)
loss = Losses.BCE()
trainer = Optimizers.Adam(model=model, loss=loss, dataset=dataset)
#trainer = Optimizers.SGDA(step_size=0.02, model=model, loss=loss, dataset=dataset)

filename = 'MNIST-Dense-BCE-Adam-10000-CPU'
iterations = 10000
batch_size = 32
save_every = 500
metrics = [Evaluators.MNIST_IS(), Evaluators.DiscriminatorAccuracy()]

Utils.run(folder='runs', filename=filename, iterations=iterations, save_every=save_every, batch_size=batch_size, metrics=metrics, model=model, dataset=dataset, loss=loss, trainer=trainer, resume=False)