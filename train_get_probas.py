print('importing...')
import argparse
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
from functools import reduce
from operator import concat
import numpy as np
import torch
import logging
import mlflow
from torch.utils.data import DataLoader

from tqdm import tqdm, trange
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from conformer import Conformer
from data.dataloader import ClassificationDataset, collate_fn

logger = logging.getLogger(__name__)

def flatten(iterable):
    iterator, sentinel, stack = iter(iterable), object(), []
    while True:
        value = next(iterator, sentinel)
        if value is sentinel:
            if not stack:
                break
            iterator = stack.pop()
        elif isinstance(value, str):
            yield value
        else:
            try:
                new_iterator = iter(value)
            except TypeError:
                yield value
            else:
                stack.append(iterator)
                iterator = new_iterator

def set_seed(args):
    """
    Set the seed for NumPy and pyTorch
    :param args: Namespace
        Arguments passed to the script
    """
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate_model(model, loader, device, n_gpu, loss_fn, global_step, run_name, threshold):
    model.eval()
    print(threshold)
    eval_loss = 0.0
    nb_eval_steps = 0
    correct = 0
    total = 0
    classes = ['UNM','h5mC','pU','m5C','m6A','unknown']
    y_pred=[]
    y_pred_scores=pd.DataFrame(columns=classes)
    y_true=[]
    total_per_class = [0] * len(classes)
    probas_df=pd.DataFrame()
    correct_per_class = [0] * len(classes)
    for batch in tqdm(loader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'inputs': batch[0].to(device),
                      'input_lengths': batch[1].to(device)}
            pred = model(**inputs)
            output=pred.cpu().detach().numpy()
            y_true.append(list(batch[2].tolist()))
            targets = batch[2].to(device)
            loss = loss_fn(pred, targets)
            # the class with the highest energy is what we choose as prediction
            
            prob, predicted = torch.max(torch.exp(pred.data), 1)
            for i in range(len(prob)):
                    if prob[i].item() < threshold:
                        predicted[i]=len(classes)-1
                       
            probas_df=probas_df.append(pd.DataFrame(torch.exp(pred.data).cpu().numpy()), ignore_index=True)
            y_pred.append(list(predicted.tolist()))
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            for i in range(len(classes)):
                total_per_class[i] += targets[targets == i].size(0)
                correct_per_class[i] += (predicted[targets == i] == targets[targets == i]).sum().item()

            tmp_eval_loss = loss
            quit()
            if n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
    print(f'total_per_class {total_per_class}')
    y_pred=list(flatten(y_pred))
    y_true=list(flatten(y_true))
    probas_df['true']=y_true
    probas_df.to_csv('probas_df.csv')
    #np.savetxt('y_true.txt',y_true)
    #y_pred_score.reshape((152,5))
    #y_pred_scores.to_csv('y_pred_scores.csv')
    confmat=pd.DataFrame(confusion_matrix(y_true,y_pred,normalize='true'))
    print(confmat)
    confmat.to_csv('confmat'+run_name+'.csv')
    eval_loss = eval_loss / nb_eval_steps
    mlflow.log_metric("validation loss", eval_loss, global_step)

    for i, name in enumerate(classes):
        if total_per_class[i] !=0:
            mlflow.log_metric(f"validation accuracy {name}",
                              100 * correct_per_class[i]/total_per_class[i], global_step)

    mlflow.log_metric("validation accuracy total", 100 * correct/total, global_step)
    print(f"eval loss: {eval_loss}")
    print(f"eval accuracy: {100 * correct/total} %")
    model.train()


def train(args, model, train_dataloader, test_dataloader, loss_fn):
    """
    Train the model
    :param args: Namespace
    :param model:  model
         model to train
    :param train_dataloader: DataLoader
        The DataLoader for training data
    :return: int, float
        Returns the current step and loss
    """
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {'inputs': batch[0].to(args.device),
                      'input_lengths': batch[1].to(args.device)}
            #if global_step!=31:  print(inputs)
            pred = model(**inputs)
            targets = batch[2].to(args.device)
            loss = loss_fn(pred, targets)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

            if args.logging_steps > 0 and (global_step + 1) \
                    % args.logging_steps == 0:
                # Log metrics
                mlflow.log_metric("training loss", (tr_loss - logging_loss) / args.logging_steps,
                                  global_step)
                print(f"training loss: {(tr_loss - logging_loss) / args.logging_steps}")
                logging_loss = tr_loss
            if args.evaluate_during_training and (global_step + 1) \
                    % args.evaluation_steps == 0:
                evaluate_model(model, test_dataloader, args.device, args.n_gpu, loss_fn, global_step,args.run_name,args.decision_threshold)
            global_step += 1
            
        model.eval()

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    # Required parameter
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where the model predictions and checkpoints "
                             "will be written.")

    # Other parameters
    parser.add_argument("--model_path", default=None, type=str,
                        help="Path to pre-trained model")
    parser.add_argument("--mlflow_path", default="/scratch/izar/szalata/mlflow", type=str)
    parser.add_argument("--dataroot", default="classification_dataset", type=str,
                        help="Path to directory with test_features.lmdb and train_features.lmdb")
    parser.add_argument("--train", action='store_true', help="Whether to run training.")
    parser.add_argument("--eval", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5,
                        help="Number of updates steps to accumulate before performing a "
                             "backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-7, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.2, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=45, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--evaluation_steps", type=int, default=1000,
                        help="Evaluate every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--class_weights", action='store_true')
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    # conformer params
    parser.add_argument("--input_dim", type=int, default=10, help="conformer input dimension")
    parser.add_argument("--decoder_dim", type=int, default=256)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--num_encoder_layers", type=int, default=3)

    parser.add_argument("--batch_size", default=60, type=int, help="Batch size per GPU/CPU.")
    parser.add_argument("--run_name", default=None, type=str, help="name of the mlflow run")
    parser.add_argument("--experiment_name", default="default", type=str)
    parser.add_argument("--decision_threshold",default=0, type=float)
    
    args = parser.parse_args()
    mlflow.set_tracking_uri(f"file://{args.mlflow_path}")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to "
            "overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    num_classes = 5
    model = Conformer(num_classes=num_classes, input_dim=args.input_dim,
                      encoder_dim=args.encoder_dim, num_encoder_layers=args.num_encoder_layers,
                      decoder_dim=args.decoder_dim, device=args.device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    args.num_workers = 1 * args.n_gpu
    val_dataloader = DataLoader(
        ClassificationDataset("val.pkl", args.dataroot, args.input_dim),
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    global_step = 0

    n_samples = val_dataloader.dataset.target.bincount()
    normed_weights = [1 - (x / sum(n_samples)) for x in n_samples]
    normed_weights = torch.FloatTensor(normed_weights).to(args.device)
    if args.class_weights:
        loss_fn = torch.nn.NLLLoss(weight=normed_weights)
    else:
        loss_fn = torch.nn.NLLLoss()
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "pytorch version": torch.__version__,
            "cuda version": torch.version.cuda,
        })
        mlflow.log_params(vars(args))

        # Training
        if args.train:
            args.batch_size = args.batch_size * max(1, args.n_gpu)
            
            '''
            files=os.listdir(traindatafolder)
            datasets=list(ClassificationDataset(file,input_dim) for  file in files)
            train_dataloader=Dataloader(ConcatDataset(datasets) ...)
            but here, each dataset will load its file in memory... should change class.
            '''
            
            train_dataloader = DataLoader(
                ClassificationDataset("train.pkl", args.dataroot, args.input_dim),
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )

            global_step, tr_loss = train(args, model, train_dataloader, val_dataloader, loss_fn)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

            # Create output directory if needed
            output_dir = os.path.join(args.output_dir, 'final-model')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            logger.info("Saving final model checkpoint to %s", output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "final_model.pt"))

            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            mlflow.log_artifacts(args.output_dir)

        if args.eval:
            model.eval()
            evaluate_model(model, val_dataloader, args.device, args.n_gpu, loss_fn, global_step,args.run_name,args.decision_threshold)


if __name__ == "__main__":
    main()
