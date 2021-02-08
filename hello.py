import Datasets
import Architectures
import Optimizers
import Losses
import Evaluators
import Utils

dataset = Datasets.CIFAR10()
#model = Architectures.DenseNetwork(img_shape=dataset.img_shape, latent_dim=100, sigmoid=False)
model = Architectures.OptimisticMirrorDescent2018(img_shape=dataset.img_shape, latent_dim=100, sigmoid=False)
loss = Losses.WGANGP(model=model)
trainer = Optimizers.Adam(model=model, loss=loss, dataset=dataset)
#trainer = Optimizers.SGDA(step_size=0.02, model=model, loss=loss, dataset=dataset)

filename = 'MNIST-DCGan-WGANGP-Adam'
iterations = 100
batch_size = 32
save_every = 10
metrics = []

Utils.run(folder='runs', filename=filename, iterations=iterations, save_every=save_every, batch_size=batch_size, metrics=metrics, model=model, dataset=dataset, loss=loss, trainer=trainer, resume=False)