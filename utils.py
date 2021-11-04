import nltk
# nltk는 postprocess를 위해 import합니다 - postprocess_text 함수 참고
nltk.download('punkt')


def preprocess_function(examples, arg):
    inputs = [f"question: {q}  context: {c} </s>" for q, c in zip(examples["question"], examples["context"])]
    targets = [f'{a["text"][0]} </s>' for a in examples['answers']]
    model_inputs = arg.tokenizer(
        inputs,
        max_length=arg.max_source_length,
        padding=arg.padding,
        truncation=True
    )

    # targets(label)을 위해 tokenizer 설정
    with arg.tokenizer.as_target_tokenizer():
        labels = arg.tokenizer(
            targets,
            max_length=arg.max_target_length,
            padding=arg.padding,
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["example_id"] = []
    for i in range(len(model_inputs["labels"])):
        model_inputs["example_id"].append(examples["id"][i])
    return model_inputs


def postprocess_text(preds, labels, datasets, metric, arg):
    """
    postprocess는 nltk를 이용합니다.
    Huggingface의 TemplateProcessing을 사용하여
    정규표현식 기반으로 postprocess를 진행할 수 있지만
    해당 미션에서는 nltk를 이용하여 간단한 후처리를 진행합니다
    """

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
