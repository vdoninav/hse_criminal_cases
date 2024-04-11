import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines.token_classification import TokenClassificationPipeline


class TokenClassificationPipe(TokenClassificationPipeline):
    def preprocess(self, sentence, offset_mapping=None, **preprocess_params):
        enc = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,  # Return multiple chunks
            max_length=self.tokenizer.model_max_length,
            padding=True
        )

        num_chunks = len(enc["input_ids"])

        for i in range(num_chunks):
            model_inputs = {k: v[i].unsqueeze(0) for k, v in enc.items()}
            model_inputs["sentence"] = sentence if i == 0 else None
            model_inputs["is_last"] = (i == num_chunks - 1)

            yield model_inputs

    def _forward(self, model_inputs):
        offset_mapping = model_inputs.pop("offset_mapping", None)

        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        sentence = model_inputs.pop("sentence")
        is_last = model_inputs.pop("is_last")

        overflow_to_sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        output = self.model(**model_inputs)
        logits = output["logits"] if isinstance(output, dict) else output[0]

        model_outputs = {
            "offset_mapping": offset_mapping,
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "sentence": sentence,
            "overflow_to_sample_mapping": overflow_to_sample_mapping,
            "is_last": is_last,
            **model_inputs,
        }

        # Reshape outputs to fit with the postprocess inputs
        model_outputs["input_ids"] = model_outputs["input_ids"].reshape(1, -1)
        model_outputs["token_type_ids"] = model_outputs["token_type_ids"].reshape(1, -1)
        model_outputs["attention_mask"] = model_outputs["attention_mask"].reshape(1, -1)
        model_outputs["special_tokens_mask"] = model_outputs["special_tokens_mask"].reshape(1, -1)
        model_outputs["offset_mapping"] = model_outputs["offset_mapping"].reshape(1, -1, 2)

        return model_outputs