import tensorflow as tf
from transformers import TokenClassificationPipeline
import torch

class TokenClassificationChunkPipeline(TokenClassificationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, sentence, offset_mapping=None, **preprocess_params):
        tokenizer_params = preprocess_params.pop("tokenizer_params", {})
        truncation = True if self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0 else False
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=truncation,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,  # Return multiple chunks
            max_length=self.tokenizer.model_max_length,
            padding=True
        )
        num_chunks = len(inputs["input_ids"])

        for i in range(num_chunks):
            if self.framework == "tf":
                model_inputs = {k: tf.expand_dims(v[i], 0) for k, v in inputs.items()}
            else:
                model_inputs = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
            if offset_mapping is not None:
                model_inputs["offset_mapping"] = offset_mapping
            model_inputs["sentence"] = sentence if i == 0 else None
            model_inputs["is_last"] = i == num_chunks - 1
            yield model_inputs

    def _forward(self, model_inputs):
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        is_last = model_inputs.pop("is_last")

        overflow_to_sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        output = self.model(**model_inputs)
        logits = output["logits"] if isinstance(output, dict) else output[0]


        model_outputs = {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "overflow_to_sample_mapping": overflow_to_sample_mapping,
            "is_last": is_last,
            **model_inputs,
        }

        # We reshape outputs to fit with the postprocess inputs
        model_outputs["input_ids"] = torch.reshape(model_outputs["input_ids"], (1, -1))
        model_outputs["token_type_ids"] = torch.reshape(model_outputs["token_type_ids"], (1, -1))
        model_outputs["attention_mask"] = torch.reshape(model_outputs["attention_mask"], (1, -1))
        model_outputs["special_tokens_mask"] = torch.reshape(model_outputs["special_tokens_mask"], (1, -1))
        model_outputs["offset_mapping"] = torch.reshape(model_outputs["offset_mapping"], (1, -1, 2))

        return model_outputs
    