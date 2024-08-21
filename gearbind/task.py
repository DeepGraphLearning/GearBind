import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torch_data

from torchdrug import core, tasks, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("tasks.BindingAffinityChange")
class BindingAffinityChange(tasks.PropertyPrediction, core.Configurable):

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), normalization=True, 
                graph_construction_model=None, verbose=0):
        super(BindingAffinityChange, self).__init__(model, task, criterion, metric, num_mlp_layer=0, 
                                                    normalization=normalization, num_class=1, 
                                                    graph_construction_model=graph_construction_model, verbose=verbose)

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                num_class.append(value.max().item() + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class

    def predict(self, batch, all_loss=None, metric=None):
        if self.graph_construction_model:
            batch["wild_type"] = self.graph_construction_model(batch["wild_type"])
            batch["mutant"] = self.graph_construction_model(batch["mutant"])
        pred = self.model(batch, all_loss=all_loss, metric=metric)['ddG']        
        return pred


@R.register("tasks.ContrastiveLearning")
class ContrastiveLearning(tasks.Task, core.Configurable):

    def __init__(self, model, num_mlp_layer=2, graph_construction_model=None, temp=1.0, criterion="categorical"):
        super().__init__()
        self.model = model
        self.graph_construction_model = graph_construction_model
        self.temp = temp

        hidden_dims = [model.output_dim // 2] * (num_mlp_layer - 1) + [1]
        self.mlp = layers.MultiLayerPerceptron(
            input_dim=model.output_dim,
            hidden_dims=hidden_dims
        )
        assert criterion in ["categorical", "bce"]
        self.criterion = criterion

    def forward(self, batch):
        all_loss = 0.
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss=all_loss, metric=metric)
        metric.update(self.evaluate(pred, target))

        if self.criterion == "categorical":
            all_loss -= metric["log_likelihood"]
        elif self.criterion == "bce":
            all_loss += metric["bce"]
        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        if self.graph_construction_model:
            wild_type = self.graph_construction_model(batch["wild_type"])
        wt_output = self.model(wild_type, wild_type.node_feature.float())["graph_feature"]
        mt_outputs = []
        for mutant in batch["mutants"]:
            mutant = self.graph_construction_model(mutant)
            mt_output = self.model(mutant, mutant.node_feature.float())["graph_feature"]
            mt_outputs.append(mt_output)
        mt_output = torch.stack(mt_outputs, dim=1)
        energy = self.mlp(torch.cat([wt_output[:, None, :], mt_output], dim=1)).squeeze(-1)   # (batch_size, 1 + num_mutant)
        return energy

    def target(self, batch):
        num_mutant = len(batch["mutants"])
        return torch.cat([
            torch.ones((batch["wild_type"].batch_size, 1), dtype=torch.float, device=batch["wild_type"].device),
            torch.zeros((batch["wild_type"].batch_size, num_mutant), dtype=torch.float, device=batch["wild_type"].device),
        ], dim=-1)

    def evaluate(self, pred, target):
        log_prob = torch.log_softmax(-pred / self.temp, dim=-1)  # low energy -> high prob
        log_prob = log_prob[:, 0].mean()

        ranking = pred.argsort(dim=-1)[:, 0].float() + 1
        hits1 = (ranking <= 1).float().mean()
        hits5 = (ranking <= 5).float().mean()
        hits10 = (ranking <= 10).float().mean()
        mean_rank = ranking.mean()
        mrr = (1 / ranking).mean()
        # (pred[:, 0] < pred[:, 1]).float()
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        bce = loss_fn(-pred / self.temp, target)
        num_mutant = pred.shape[1] - 1
        weight = torch.tensor([1.0] + [1.0 / num_mutant] * num_mutant, dtype=torch.float, device=bce.device)
        bce = (bce * weight[None, :]).mean()
        # print(pred)
        return {"log_likelihood": log_prob, 
                "bce": bce, 
                "hits@1": hits1, 
                "hits@10": hits10,
                "hits@5": hits5,
                "mean_rank": mean_rank, 
                "mrr": mrr}
