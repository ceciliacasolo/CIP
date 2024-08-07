import json
import os
import numpy as np
from torch import nn, optim
from absl import app, logging, flags
import sys
sys.path.append('')

from utils.utils import setup_directories, get_loaders, counterfactual_simulations_img, cf_output, process_image_data
from utils.data_utils import find_causal_dataset, latent_to_index
from utils.model import  NeuralNetworkImage
from utils.training import ImageTraining


flags.DEFINE_integer("epochs", 1000, "epochs")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("number_samples", 1000, "number samples")
flags.DEFINE_float("beta", 0, "HSCIC regularization parameter")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_integer("nh", 8, "number hidden layers")
flags.DEFINE_integer("count_samples", 100, "number counterfactual samples")
flags.DEFINE_integer("data_points_count", 100, "number data points used for counterfactual samples")
flags.DEFINE_string("output_dir", "results_image/",
                    "Path to the output directory (for results).")
flags.DEFINE_string("output_name", "",
                    "Name for result folder. Use timestamp if empty.")
flags.DEFINE_integer("seed", 0, "The random seed.")

flags.DEFINE_integer("init_n", 200, "initial number data points used for generating dataset")
flags.DEFINE_integer("match_n", 300, "n datapoints of dsprites used for matching")
flags.DEFINE_string("data_dir", "data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
                    "Directory of the input data.")

FLAGS = flags.FLAGS

def main(_):
    out_dir = setup_directories(FLAGS)
    FLAGS.log_dir = out_dir

    logging.info("Save FLAGS (arguments)...")
    with open(os.path.join(out_dir, 'flags.json'), 'w') as fp:
        json.dump(FLAGS.flag_values_dict(), fp, sort_keys=True, indent=2)

    logging.info(f"Set random seed {FLAGS.seed}...")
    np.random.seed(FLAGS.seed)

    # ---------------------------------------------------------------------------
    # Load data, create causal data and counterfactual dataset
    # ---------------------------------------------------------------------------
    results_dict = {}

    logging.info("Loading data...")
    dataset_zip = np.load(FLAGS.data_dir, allow_pickle=True, encoding="latin1") 
    imgs = dataset_zip['imgs']
    metadata = dataset_zip['metadata'][()]

    logging.info("Processing data...")
    latents_sizes = metadata['latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    # Generate new dataset of latent variables and outcome via matching 
    df_total = find_causal_dataset(FLAGS.init_n, len(imgs), latents_sizes, latents_bases, imgs)
    logging.info("Created Dataset ...")

    # Store respective images in imgs_df, drop unobserved variables from df_total_final and process data
    indices_sampled = latent_to_index(df_total.drop(['output', 'noise_scale'], axis=1), latents_bases)
    imgs_df = imgs[indices_sampled]

    df_total, df_no_unobs, stats = process_image_data(df_total)

    loader_train, loader_test = get_loaders(df_no_unobs, imgs_df, FLAGS.batch_size)

    # Generate counterfactual samples
    loader_count_images, loader_count_tabular = counterfactual_simulations_img(
        FLAGS.data_points_count, FLAGS.count_samples, stats, imgs, df_total, latents_bases)
    logging.info("Created counterfactual dataset...")

    # ---------------------------------------------------------------------------
    # Train, test and save results
    # ---------------------------------------------------------------------------
    cnet = NeuralNetworkImage(ndim_features=6)
    optimizer = optim.Adam(cnet.parameters(), lr=FLAGS.lr)
    loss_function = nn.MSELoss()
    
    trainer = ImageTraining(model=cnet,
                           optimizer=optimizer,
                           loss_function=loss_function,
                           num_epochs=FLAGS.epochs, 
                           beta_hscic=FLAGS.beta, 
                           beta_l=1.0)    
    
    training_stats = trainer.train(loader_train, loader_test)
    logging.info("Training finished ...")

    # Find counterfactual outcomes Yhat and VCF
    count_results = [cf_output(cnet, zip(loader_count_images[i], loader_count_tabular[i]), img_example = True) for i in range(len(loader_count_images))]
    var_res = [np.var(count_results[i]['values'], ddof=1) for i in range(len(count_results))] 
    VCF = np.mean(var_res)

    results = [
        training_stats['test_acc'][-1],   
        training_stats['test_hscic'][-1], 
        VCF                              
    ]

    results_dict = {}
    results_dict[str(FLAGS.beta)] = results

    logging.info(f"Store results...")
    result_path = os.path.join(out_dir, "results.npz")
    np.savez(result_path, **results_dict)

    logging.info(f"DONE")


if __name__ == "__main__":
    app.run(main)
