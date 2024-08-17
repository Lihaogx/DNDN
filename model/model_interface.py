import pytorch_lightning as pl
from model.network.dynamic_dowker_model import DNDN, DowkerMetrics
from utils.utils import create_optimizer, create_scheduler

class MInterface(pl.LightningModule):
    def __init__(self, model_config, optim_config):
        super().__init__()
        self.model_config = model_config
        self.optim_config = optim_config
        self.loss_type = model_config.loss_type
        self.test_metric = DowkerMetrics(loss_type=self.loss_type)
        self.load_model()
        
    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        batch = self(batch)
        loss = self.model.loss(batch, self.loss_type)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     batch = self(batch)
    #     loss = self.model.loss(batch, self.loss_type)
    #     self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
    #     self.val_metric.update(batch)
        
    #     return loss

    def test_step(self, batch, batch_idx):
        batch = self(batch)
        loss = self.model.loss(batch, self.loss_type)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.test_metric.update(batch)
        
        return loss
    
    
    # def on_validation_epoch_end(self):
    #     results = self.val_metric.compute()
    #     self.log_dict({'test_wd': results['wd'], 
        #                'test_pi': results['pi']
        #                }, prog_bar=True)
    #     self.val_metric.reset()
    #     # Make the Progress Bar leave there
    #     self.print('')

    def on_test_epoch_end(self):
        results = self.test_metric.compute()
        self.log_dict({'test_wd': results['wd'], 
                    'test_pi': results['pi'],
                    }, prog_bar=True)
        self.test_metric.reset()
        # Make the Progress Bar leave there
        self.print('')
        
    def configure_optimizers(self):
        optimizer = create_optimizer(self.model.parameters(), self.optim_config)
        scheduler = create_scheduler(optimizer, self.optim_config)
        return [optimizer], [scheduler]
    
    
    def load_model(self):
        self.model = DNDN(num_classes = self.model_config.num_class, 
                          in_dim=1, 
                          hidden_dim=self.model_config.hidden_dim, 
                          num_layers=self.model_config.num_layers, 
                          dropout=self.model_config.dropout, 
                          new_node_feat=self.model_config.new_node_feat, 
                          use_edge_attn=self.model_config.use_edge_attn, 
                          combine=self.model_config.fusion,)