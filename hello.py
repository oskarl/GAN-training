import Datasets
import Architectures
import Optimizers
import Losses
import Evaluators
import Utils

dataset = Datasets.MNIST()
model = Architectures.DenseNetwork(img_shape=dataset.img_shape, latent_dim=100, sigmoid=True)
#model = Architectures.OptimisticMirrorDescent2018(img_shape=dataset.img_shape, latent_dim=100, sigmoid=False)
loss = Losses.BCE(model=model)
trainer = Optimizers.Nesterov(step_size=0.02, momentum_weight=0.9 , model=model, loss=loss, dataset=dataset)
#trainer = Optimizers.SGDA(step_size=0.02, model=model, loss=loss, dataset=dataset)
mnistis = Evaluators.MNIST_IS(samples=500)
da = Evaluators.DiscriminatorAccuracy(samples=500)

filename = 'MNIST-BCE-Nesterov'
iterations = 10000
batch_size = 32
save_every = 500
metrics = [mnistis, da]

Utils.run(folder='runs', filename=filename, iterations=iterations, save_every=save_every, batch_size=batch_size, metrics=metrics, model=model, dataset=dataset, loss=loss, trainer=trainer, resume=False)