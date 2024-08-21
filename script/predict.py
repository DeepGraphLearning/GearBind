import os
import sys
import math
import pprint
import pickle

from tqdm import tqdm

import numpy as np

import torch

from torchdrug import core, models, data, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gearbind import dataset, layer, model, task, util


def dump(cfg, dataset, solver):
    dataloader = data.DataLoader(dataset, solver.batch_size, shuffle=False, num_workers=0)
    device = torch.device(solver.gpus[0])
    solver.model.eval()
    preds = []
    for batch in tqdm(dataloader):
        batch = utils.cuda(batch, device=device)
        with torch.no_grad():
            output = solver.model.predict(batch)
            preds.append(output.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return pred


def test(cfg, dataset):
    if "checkpoints" in cfg:
        preds = []
        for i in range(len(cfg.checkpoints)):
            cfg.checkpoint = cfg.checkpoints[i]
            solver = util.build_solver(cfg, dataset)
            pred = dump(cfg, solver.test_set, solver)
            preds.append(pred)
        pred = np.stack(preds, axis=0)

        test_split = cfg.dataset.split.test_set
        pdb_files = [pdb_file for pdb_file in dataset.pdb_files]

        if cfg.task.model["class"] == "BindModel":
            model_class = "GearBind"
        elif cfg.task.model["class"] == "DDGPredictor":
            model_class = "BindDDG"
        with open("%s_%s_%s.csv" % (model_class, cfg.dataset["class"], test_split), "w") as fout:
            model_name = cfg.task.model["class"]
            fout.write("pdb,%s\n" % (
                ",".join([
                    "%s_%d" % (model_name, i) for i in range(len(cfg.checkpoints))
                ] + ["%s_mean" % model_name]
            )))
            for i in range(pred.shape[1]):
                fout.write('"%s",%s\n' % (pdb_files[i], (
                    ",".join([
                        "%.5f" % pred[j, i] for j in range(pred.shape[0])
                    ] + ["%.5f" % pred[:, i].mean(axis=0)]
                ))))


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    if "test_sets" in cfg.dataset.split:
        test_sets = cfg.dataset.split.pop("test_sets")
        for test_set in test_sets:
            cfg.dataset.split.test_set = test_set
            test(cfg, dataset)
    else:
        test(cfg, dataset)
