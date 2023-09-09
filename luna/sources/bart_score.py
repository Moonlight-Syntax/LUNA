# This is a version of official BARTScore code base with restricted functionality to better serve our purposes.
# Please see the original code here: https://github.com/neulab/BARTScore (licensed under Apache License 2.0)

import logging
import traceback
import typing as tp

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


class BARTScorer:
    def __init__(
        self,
        device: str = "cuda:0",
        max_length: int = 1024,
        checkpoint: str = "facebook/bart-large-cnn",
        tokenizer_checkpoint: tp.Optional[str] = None,
    ):
        # Set up model
        self.device = device
        self.max_length = max_length
        logging.info("Initializing and loading tokenizer...")
        self.tokenizer = BartTokenizer.from_pretrained(
            tokenizer_checkpoint if tokenizer_checkpoint is not None else checkpoint
        )
        logging.info("Done!\nInitializing and loading model...")
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        logging.info("Done!")
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction="none", ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def score(self, srcs, tgts, batch_size=4):
        """Score a batch of examples"""
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i : i + batch_size]
            tgt_list = tgts[i : i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt"
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt"
                    )
                    src_tokens = encoded_src["input_ids"].to(self.device)
                    src_mask = encoded_src["attention_mask"].to(self.device)

                    tgt_tokens = encoded_tgt["input_ids"].to(self.device)
                    tgt_mask = encoded_tgt["attention_mask"]
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens)
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                logging.info(f"source: {src_list}")
                logging.info(f"target: {tgt_list}")
                exit(0)
        return score_list
