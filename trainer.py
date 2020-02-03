import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from evaluation import create_evaluation

class Trainer:
    
    def __init__(self,               
                 model,                     # Model to be trained.
                 crit,                      # Loss function
                 optim = None,              # Optimiser
                 train_dl = None,           # Training data set (dl means data loader)
                 val_test_dl = None,        # Validation (or test) data set
                 pos_weight_train = None,   # pos_weight against imbalance of labels in train dataset
                 pos_weight_val = None,     # pos_weight against imbalance of labels in validation dataset
                 cuda = True,               # Whether to use the GPU
                 early_stopping_cb = None): # The stopping criterion. 
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self.pos_weight_train = pos_weight_train
        self.pos_weight_val = pos_weight_val
        self._cuda = cuda
        self._early_stopping_cb = early_stopping_cb

        self._mode = None
        
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            self.pos_weight_train = pos_weight_train.cuda()
            self.pos_weight_val = pos_weight_val.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss

        self._optim.zero_grad()
        out = self._model(x)
        loss = self._crit(out, y)
        loss = (loss * self.pos_weight_train).mean()
        loss.backward()
        self._optim.step()
        return loss
        
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions

        # care: you must also tell gpu tensor[0.5,0.5] by cuda()
        out = self._model(x)
        loss = self._crit(out, y)
        loss = (loss * self.pos_weight_val).mean()
        out = t.ge(out, t.tensor([0, 0]).cuda()).float()   # If out is greater than 0, give it corresponding label (imagine sigmoid)
        return loss, out

        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it

        self._mode = 'train'
        iter_num = len(self._train_dl)
        average_loss = 0
        for img, label in tqdm(self._train_dl):
            if self._cuda:
                img = img.cuda()
                label = label.cuda()
            loss = self.train_step(img, label)      # loss type is tensor
            average_loss += loss.item() / iter_num
        print("\nrain loss: ", average_loss)
        return average_loss


    
    def val_test(self):
        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics

        self._mode = "val"
        with t.no_grad():           # disable gradient computation during with sentence
            iter_num = len(self._val_test_dl)
            pred_list = []
            label_list = []
            average_loss = 0
            for img, label in tqdm(self._val_test_dl):
                if self._cuda:
                    img = img.cuda()
                    label = label.cuda()

                loss, pred = self.val_test_step(img, label)
                average_loss += loss.item() / iter_num

                label_list.append(label.cpu())
                pred_list.append(pred.cpu())
            print("\nvalidation loss: ", average_loss)
            create_evaluation(label_list, pred_list, self._mode)

        return average_loss


        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_cb is not None or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        train_loss_list = []
        validation_loss_list = []
        num_epoch = 0

        # If you want to restore a checkpoint, you can set here.
        # num_epoch = the file you want restore
        # self.restore_checkpoint(num_epoch)

        while True:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists 
            # use the save_checkpoint function to save the model for each epoch
            # check whether early stopping should be performed using the early stopping callback and stop if so
            # return the loss lists for both training and validation
            num_epoch += 1

            train_loss = self.train_epoch()
            validation_loss = self.val_test()

            train_loss_list.append(train_loss)
            validation_loss_list.append(validation_loss)

            self.save_checkpoint(num_epoch)

            self._early_stopping_cb.step(validation_loss)
            if self._early_stopping_cb.should_stop():
                return train_loss_list, validation_loss_list
