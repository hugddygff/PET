# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch
import os
import numpy as np

from IPython import embed


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
                ("pos", fields[5]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}

        #train_caption_path = "/data2/jintao/slt/train_caption.txt"
        #train_caption_path2 = "/data2/jintao/slt/train_caption2.txt"

        #dev_caption_path = "/data2/jintao/slt/dev_caption.txt"
        #dev_caption_path2 = "/data2/jintao/slt/dev_caption2.txt"

        #test_caption_path = "/data2/jintao/slt/test_caption.txt"
        #test_caption_path2 = "/data2/jintao/slt/test_caption2.txt"


        train_pos = "/home/jintao/slt-master/train_save.txt"
        dev_pos = "/home/jintao/slt-master/dev_save.txt"
        test_pos = "/home/jintao/slt-master/test_save.txt"


        train_i3d = "/data2/jintao/slt/image_train"
        dev_i3d = "/data2/jintao/slt/image_dev"
        test_i3d = "/data2/jintao/slt/image_test"




        if path[0].split(".")[-1] == "train":

            pos_file = open(train_pos, "r")
            pos_file = pos_file.readlines()
            dic = {}

            for row in pos_file:
                row_ = row.strip().split("   ")
                dic[row_[0]] = row_[1]

            i3d_path = train_i3d

        elif path[0].split(".")[-1] == "dev":

            pos_file = open(dev_pos, "r")
            pos_file = pos_file.readlines()
            dic = {}

            for row in pos_file:
                row_ = row.strip().split("   ")
                dic[row_[0]] = row_[1]

            i3d_path = dev_i3d


        elif path[0].split(".")[-1] == "test":

            pos_file = open(test_pos, "r")
            pos_file = pos_file.readlines()
            dic = {}

            for row in pos_file:
                row_ = row.strip().split("   ")
                dic[row_[0]] = row_[1]

            i3d_path = test_i3d





        #    caption = open(train_caption_path, "w")
        #    caption2 = open(train_caption_path2, "w")

        #elif path[0].split(".")[-1] == "dev":
        #    caption = open(dev_caption_path, "w")
        #    caption2 = open(dev_caption_path2, "w")

        #else:
        #    caption = open(test_caption_path, "w")
        #    caption2 = open(test_caption_path2, "w")

        #embed()

        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]

                #i3d_feat = np.load(os.path.join(i3d_path, seq_id.split("/")[-1] + ".npy"))
                #i3d_feat = torch.from_numpy(i3d_feat)
                #embed()

                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    assert samples[seq_id]["pos"] == s["pos"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                        #"sign":i3d_feat,
                        "pos": dic[s["name"]],
                    }


                #caption.write(s["name"] + "\n")
                #caption2.write(s["text"].strip() + "\n")
                #embed()


        #caption.close()
        #caption2.close()
        #embed()

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                        sample["pos"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)

class SignTranslationDataset2(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}

        #train_caption_path = "/data2/jintao/slt/train_caption.txt"
        #train_caption_path2 = "/data2/jintao/slt/train_caption2.txt"

        #dev_caption_path = "/data2/jintao/slt/dev_caption.txt"
        #dev_caption_path2 = "/data2/jintao/slt/dev_caption2.txt"

        #test_caption_path = "/data2/jintao/slt/test_caption.txt"
        #test_caption_path2 = "/data2/jintao/slt/test_caption2.txt"

        #if path[0].split(".")[-1] == "train":
        #    caption = open(train_caption_path, "w")
        #    caption2 = open(train_caption_path2, "w")

        #elif path[0].split(".")[-1] == "dev":
        #    caption = open(dev_caption_path, "w")
        #    caption2 = open(dev_caption_path2, "w")

        #else:
        #    caption = open(test_caption_path, "w")
        #    caption2 = open(test_caption_path2, "w")

        #embed()

        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }


                #caption.write(s["name"] + "\n")
                #caption2.write(s["text"].strip() + "\n")




                #embed()


        #caption.close()
        #caption2.close()
        #embed()

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
