import torch
import torch.nn as nn
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.models.base import Stage
from up_cnn_model import CNN_Class


class ChemEmbedRetrieval(RetrievalMassSpecGymModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn = CNN_Class()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.cnn(x)

    def step(self, batch: dict, stage: Stage) -> dict:
        x = batch["spec"]
        mol_true = batch["mol"]
        cands = batch["candidates_mol"]
        batch_ptr = batch["batch_ptr"]

        mol_pred = self.forward(x)

        loss = nn.functional.mse_loss(mol_pred, mol_true)

        mol_pred_repeated = mol_pred.repeat_interleave(batch_ptr, dim=0)
        scores = nn.functional.cosine_similarity(mol_pred_repeated, cands)

        return dict(loss=loss, scores=scores)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)