from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from datasets import load_dataset, load_metric
import torch

from utils import preprocess_function, postprocess_text
from model import get_model, set_seed

import argparse

def get_datasets(datasets):
    column_names = datasets["train"].column_names

    # 전체 train dataset을 사용하는 예제가 아니고, sampling된 데이터를 사용하는 코드입니다. 적절하게 코드를 수정하여 사용하셔도 좋습니다.

    train_dataset = datasets["train"]
    train_dataset = train_dataset.select(range(train_arg.max_train_samples))
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=train_arg.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )

    # 전체 데이터로 평가
    eval_examples = datasets["validation"]

    # 샘플 데이터로 평가
    # eval_examples = eval_examples.select(range(max_val_samples)) 

    eval_dataset = eval_examples.map(
        preprocess_function,
        batched=True,
        num_proc=train_arg.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    return train_dataset, eval_dataset



def main(train_arg, data_arg):

    print ("PyTorch version:[%s]."%(torch.__version__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ("device:[%s]."%(device))

    datasets = load_dataset(data_arg.dataset_name)
    metric = load_metric(data_arg.metric_name)


    train_dataset, eval_dataset = get_datasets(datasets)

    model, tokenizer = get_model(train_arg)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 간단한 post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"].select(range(train_arg.max_val_samples)))]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"].select(range(train_arg.max_val_samples))]

        result = metric.compute(predictions=formatted_predictions, references=references)
        return result

    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )

    args = Seq2SeqTrainingArguments(
        output_dir=train_arg.output_dir, 
        do_train=True, 
        do_eval=True, 
        predict_with_generate=True,
        per_device_train_batch_size=train_arg.batch_size,
        per_device_eval_batch_size=train_arg.batch_size,
        num_train_epochs=train_arg.num_train_epochs,
        save_strategy='epoch',
        save_total_limit=2 # 모델 checkpoint를 최대 몇개 저장할지 설정
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train(resume_from_checkpoint=None)
    trainer.save_model()
    trainer.save_state()
    print(train_result)


if __name__ == "__main__":
    train_parser = argparse.ArgumentParser(description="Training Arguments")
    train_parser.add_argument("--model_name", type=str, help="model name or path", default="google/mt5-small")
    train_parser.add_argument("--batch_size", type=int, help="per device batch size", default=4)
    train_parser.add_argument("--num_train_epoch", type=int, help="num train epochs", default=30)
    train_parser.add_argument("--learning_rate", type=float, help="train learning rate", default=1e-5)
    train_parser.add_argument("--output_dir", type=str, help="model save directory", default="outputs")
    train_parser.add_argument("--max_train_samples", type=int, help="maximum number of train samples", default=16)
    train_parser.add_argument("--max_val_samples", type=int, help="maximum number of validation samples", default=16)
    train_parser.add_argument("--num_beams", type=int, help="number of beam search", default=2)
    train_parser.add_argument("--preprocessing_num_workers", type=int, help="number of preprocessing workers", default=12)
    train_arg = train_parser.parse_args()

    data_parser = argparse.ArgumentParser(description="Data Arguments")
    data_parser.add_argument("--dataset_name", type=str, help="dataset name or path", default="squad_kor_v1")
    data_parser.add_argument("--metric_name", type=str, help="metric name or path", default="squad")
    data_parser.add_argument("--max_source_length", type=int, help="maximum length of source data", default=1024)
    data_parser.add_argument("--max_target_length", type=int, help="maximum length of target corpus", default=128)
    data_parser.add_argument("--padding", type=bool, help="pad to remaining length", default=None)
    data_arg = data_parser.parse_args()
    set_seed(42)
    main(train_arg, data_arg)