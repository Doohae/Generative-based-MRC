# import argparse
# from transformers import AutoModel, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
# from tqdm.auto import tqdm

# model = AutoModel.from_pretrained(arg.model_path)
# tokenizer = AutoTokenizer.from_pretrained(arg.model_path)

# metrics = trainer.evaluate(
#     max_length=max_target_length,
#     num_beams=num_beams,
#     metric_key_prefix="eval"
# )

# print(metrics)

# document = "세종대왕님은 언제 태어났어?"
# print(document)
# input_ids = tokenizer(document, return_tensors='pt').input_ids
# outputs = model.generate(input_ids.to('cuda'))
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str, help="model(tokenizer) name or path", default="outputs")
