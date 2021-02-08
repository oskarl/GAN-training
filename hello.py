import Datasets
import Architectures
import Optimizers
import Losses
import Evaluators
import Utils

dataset = Datasets.CelebA(zip_file_path='files/32x32.pickle')
#model = Architectures.DenseNetwork(img_shape=dataset.img_shape, latent_dim=100, sigmoid=False)
model = Architectures.OptimisticMirrorDescent2018(img_shape=dataset.img_shape, latent_dim=100, sigmoid=False)
loss = Losses.WGANGP(model=model, gradient_penalty=1.0)
trainer = Optimizers.OGDA(step_size=0.002, model=model, loss=loss, dataset=dataset)
#trainer = Optimizers.SGDA(step_size=0.02, model=model, loss=loss, dataset=dataset)

filename = 'CelebA-WGANGP-OGDA'
iterations = 1000
batch_size = 64
save_every = 1000
metrics = []

Utils.run(folder='runs', filename=filename, iterations=iterations, save_every=save_every, batch_size=batch_size, metrics=metrics, model=model, dataset=dataset, loss=loss, trainer=trainer, resume=False)