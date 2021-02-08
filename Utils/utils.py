import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
from tensorflow.python.client import device_lib
import time

def devices():
    ds = []
    for d in device_lib.list_local_devices():
        dname = d.device_type
        if d.physical_device_desc != '':
            dname += ' ' + d.physical_device_desc
        ds.append(dname)
    return ds

def dir_exists(folder):
    return os.path.isdir(folder+'/')

def make_dir(folder):
    results_dir = folder+'/'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

def save_json(folder, filename, data):
    with open(folder+'/'+filename+'.json', 'w+') as handle:
        jsontext = json.dumps(data)
        handle.write(jsontext)

def sample_images(folder, model, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, model.latent_dim))
    gen_imgs = model.generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if model.img_shape[2] == 1:
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            else:
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1    

    filename = folder+"/"+str(epoch)+".png"

    fig.savefig(filename)
    plt.close()

    return filename

def save_plots(folder, filename, data, metrics):
    nplots = 1+len(metrics)
    fig, plots = plt.subplots(1+len(metrics), constrained_layout=True, figsize=(6,4*nplots), dpi=200)
    fig.suptitle(data['dataset']+', '+data['architecture']+', '+data['loss']+', '+data['optimizer'])

    plots[0].set_title('Loss')
    plots[0].set_ylabel('Loss')
    plots[0].set_xlabel('Iterations')
    plots[0].plot(data['iteration'], data['d_loss'], label='D')
    plots[0].plot(data['iteration'], data['g_loss'], label='G')
    plots[0].legend()

    for i in range(len(metrics)):
        plots[1+i].set_title(metrics[i].name)
        plots[1+i].set_ylabel(metrics[i].name)
        plots[1+i].set_xlabel('Iterations')

        plots[1+i].plot(data['iteration'], data[metrics[i].name])       

    fig.savefig(folder+'/'+filename+'.png', dpi=200)
    plt.close()


def save_video(folder, filename, images):
    video_name = folder+'/'+filename+'.avi'

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, 24, (width,height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

def run(folder, filename, iterations, save_every, batch_size, metrics, model, dataset, loss, trainer, resume=False):
    originalfolder = folder
    folder = folder + '/' + filename

    if resume and dir_exists(folder):
        model.generator.load_weights(folder+'/Checkpoints/Generator')
        model.discriminator.load_weights(folder+'/Checkpoints/Discriminator')
    else:
        i = 1
        originalfilename = filename
        while dir_exists(folder):
            filename = originalfilename + ' (' + str(i) + ')'
            folder = originalfolder + '/' + filename
            i += 1

    training_results = {
        'filename': filename, 
        'iterations': iterations, 
        'batch_size': batch_size, 
        'dataset': dataset.name, 
        'architecture': model.name,
        'loss': loss.name,
        'optimizer': trainer.name,
        'iteration': [], 
        'd_loss': [], 
        'g_loss': [],
        'execution_time': [],
        'environment': devices(),
        'restarts': []
    }

    for metric in metrics:
        training_results[metric.name] = []

    start_i = 0
    end_i = iterations+1
    if resume and dir_exists(folder):
        handle = open(folder+'/Results.json', 'r')
        data = json.load(handle)
        training_results = data
        start_i = training_results['iteration'][-1]+1
        end_i = start_i+iterations
        training_results['iterations'] = start_i+iterations
        training_results['restarts'].append(start_i)

    images = []

    start_time = time.time()

    make_dir(folder)
    make_dir(folder+'/Images')
    make_dir(folder+'/Checkpoints')

    for i in range(start_i,end_i):
        r = trainer.train_step(batch_size=batch_size)
        print(i,"d:",r['d_loss'],"g:",r['g_loss'])
        if i%save_every == 0:
            training_results['iteration'].append(i)
            training_results['d_loss'].append(float(r['d_loss']))
            training_results['g_loss'].append(float(r['g_loss']))
            training_results['execution_time'].append((time.time() - start_time))

            image_file = sample_images(folder=folder+'/Images', model=model, epoch=i)
            images.append(image_file)

            for metric in metrics:
                value = metric.calculate(model=model, dataset=dataset)
                print(metric.name,str(value))
                training_results[metric.name].append(value)

            save_json(folder=folder, filename='Results', data=training_results)
            
            model.generator.save_weights(folder+'/Checkpoints/Generator')
            model.discriminator.save_weights(folder+'/Checkpoints/Discriminator')

            start_time = time.time()

    save_plots(folder=folder, filename='Results', data=training_results, metrics=metrics)
    save_video(folder=folder, filename='Training-'+str(len(training_results['restarts'])), images=images)