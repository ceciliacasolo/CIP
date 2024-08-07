from absl import app, logging, flags
import numpy as np
import json
import os
from torch import nn, optim
import sys
sys.path.append('')

from utils.utils import setup_directories, cf_output, data_processing, counterfactual_simulations
from utils.data_utils import simulation_data
from utils.training import StandardTraining
from utils.model import Model

flags.DEFINE_integer("epochs", 1000, "epochs")
flags.DEFINE_integer("batch_size", 512, "batch size")
flags.DEFINE_integer("number_samples", 10000, "number samples")
flags.DEFINE_float("beta", 0, "HSCIC regularization parameter")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_integer("dim_h", 20, "hidden dimension")
flags.DEFINE_integer("dim_input", 3, "input dimension")
flags.DEFINE_integer("nh", 8, "number hidden layers")
flags.DEFINE_integer("count_samples", 500, "number counterfactual samples")
flags.DEFINE_integer("data_points_count", 1000, "number data points used for counterfactual samples")
flags.DEFINE_string("output_dir", "results/",
                    "Path to the output directory (for results).")
flags.DEFINE_string("output_name", "",
                    "Name for result folder. Use timestamp if empty.")
flags.DEFINE_integer("seed", 0, "The random seed.")


FLAGS = flags.FLAGS

def main(_):
    out_dir = setup_directories(FLAGS)
    FLAGS.log_dir = out_dir

    logging.info("Save FLAGS (arguments)...")
    with open(os.path.join(out_dir, 'flags.json'), 'w') as fp:
        json.dump(FLAGS.flag_values_dict(), fp, sort_keys=True, indent=2)

    logging.info(f"Set random seed {FLAGS.seed}...")
    np.random.seed(FLAGS.seed)
    
    # Simulation of the data
    data = simulation_data(FLAGS.number_samples, FLAGS.sim)
    
    # generate counterfactual samples
    # in each element of count_loader_tot, count_samples counterfactual datapoints are generated
    count_loader_tot = counterfactual_simulations(FLAGS.data_points_count, FLAGS.count_samples, data)
    train_dataloader, test_dataloader, _ = data_processing(data, FLAGS.batch_size)

    loss_function = nn.MSELoss()
    cnet = Model(dim_in=FLAGS.dim_input, nh=FLAGS.nh, dim_h=FLAGS.dim_h)
    optimizer = optim.Adam(cnet.parameters(), lr=FLAGS.lr)
    
    trainer = StandardTraining(model=cnet,
                           optimizer=optimizer,
                           loss_function=loss_function,
                           num_epochs=FLAGS.epochs, 
                           beta_hscic=FLAGS.beta, 
                           beta_l=1.0)    
    
    training_stats = trainer.train(train_dataloader, test_dataloader)


    # find counterfactuals outcomes
    # in each element of count_loader_tot, the variance of the counterfactual outcomes of the count_samples datapoints is stored in var_res
    count_results = [cf_output(cnet, count_loader_tot[i]) for i in range(len(count_loader_tot))]
    var_res = [np.var(count_results[i]['values'], ddof=1) for i in range(len(count_results))]  # variance CF outcomes
    VCF = np.mean(var_res)

    results = [
        training_stats['test_acc'][-1],   
        training_stats['test_hscic'][-1], 
        VCF                              
    ]

    # Prepare results dictionary; assuming FLAGS.beta or similar variable is set elsewhere
    results_dict = {}
    results_dict[str(FLAGS.beta)] = results

    logging.info(f"Store results...")
    result_path = os.path.join(out_dir, "results.npz")
    np.savez(result_path, **results_dict)
    logging.info(f"DONE")


if __name__ == "__main__":
    app.run(main)
