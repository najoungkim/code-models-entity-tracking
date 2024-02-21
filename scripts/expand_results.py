"""Expand results."""

import argparse
import json
import re

import pandas as pd


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
    )

    parser.add_argument("--expanded_dataset", type=str, required=True)

    parser.add_argument("--num_boxes", type=int, default=7)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    out_filename = args.input_file.replace(".jsonl", "-expanded.jsonl")

    with open(args.input_file, encoding="utf-8") as in_f, \
            open(args.expanded_dataset, encoding="utf-8") as exp_f:
        pred_df = pd.read_json(in_f, lines=True, orient="records")
        cleaned_indiv_predictions = []
        targets = []
        contexts = []
        for i, exp_line in enumerate(exp_f):
            d = json.loads(exp_line)
            targets.append(d["masked_content"].replace(
                "<extra_id_0>", "").strip())
            contexts.append(d["sentence_masked"].replace(" <extra_id_0>", ""))

            if i % args.num_boxes == 0:
                cols = pred_df.iloc[i // args.num_boxes]
                pred, context = (
                    # cols["target"],
                    cols["prediction"],
                    cols["input"],
                )

                if "Statement:" in pred:
                    pred = pred.split("Statement:")[1]
                    pred = pred.replace("Box 0 contains", "")

                if "." in pred:
                    stop_idx = pred.index(".")
                    pred = pred[0:stop_idx]

                if "Box " in pred:
                    indiv_predictions = re.split(
                        r", and Box|, Box| and Box|; Box", pred
                    )
                else:
                    if "is in" in pred or "are in" in pred or "is nothing in" in pred:
                        pred = re.sub(
                            r"in Container ([A-Z0-9]) (and )?the",
                            "in Container \g<1>, the",
                            pred,
                        )
                        pred = re.sub(
                            r"(, ([a-zA-Z- ]+) (is|are) in Container ([A-Z0-9]))",
                            ", Container \g<4> contains \g<2>",
                            pred,
                        )
                        pred = re.sub(
                            r"(, )?[Tt]here is nothing in Container ([A-Z0-9])",
                            ", Container \g<2> is empty",
                            pred,
                        )
                    indiv_predictions = re.split(
                        r", and Container|, Container| and Container", pred
                    )

                if indiv_predictions[0].startswith("is empty,"):
                    indiv_predictions[0] = "is empty"

                indiv_predictions = indiv_predictions[0:1] + sorted(
                    indiv_predictions[1:]
                )

                if len(indiv_predictions) < args.num_boxes:
                    print("Missing predictions:" +
                          "||".join(indiv_predictions))
                    boxes_mentioned = {0}
                    for indiv_pred in list(indiv_predictions[1:]):
                        if len(indiv_pred.strip()) < 1:
                            continue
                        box_name = indiv_pred.strip()[0]
                        # box numbers
                        if ord(box_name) >= 48 and ord(box_name) < 48 + args.num_boxes:
                            if (ord(box_name) - 48) in boxes_mentioned:
                                indiv_predictions.remove(indiv_pred)
                            boxes_mentioned.add(ord(box_name) - 48)
                        # box A, box B, ...
                        elif (
                            ord(box_name) >= 65 and ord(
                                box_name) < 65 + args.num_boxes
                        ):
                            if (ord(box_name) - 65) in boxes_mentioned:
                                indiv_predictions.remove(indiv_pred)
                            boxes_mentioned.add(ord(box_name) - 65)

                    for i in range(args.num_boxes):
                        if i not in boxes_mentioned:
                            indiv_predictions.insert(
                                i, f"{i} contains invalid")
                    # indiv_predictions.extend(["contains invalid"] * (args.num_boxes - len(indiv_predictions)))
                    print("Missing predictions after:" +
                          "||".join(indiv_predictions))

                if len(indiv_predictions) > args.num_boxes:
                    print("Extra predictions: " + "||".join(indiv_predictions))
                    indiv_predictions = indiv_predictions[0: args.num_boxes]

                assert (
                    len(indiv_predictions) == args.num_boxes
                ), f"wrong number of predictions:\n{'||'.join(indiv_predictions)}"

                # reformat zero-shot/few-shot examples:
                if "Description: " in context:
                    context = context.split("Description: ")[-1]
                    context = (
                        context.replace("\\nStatement:", "").replace(
                            " Statement:", "")
                        + " ."
                    )
                    context = context.replace(
                        "\\n\\nLet's think step by step.\\n", "")

                # assert contexts[-1].replace("Box 0 contains .", ".") == context.replace(
                #    "Box 0 contains .", "."
                # ), f"contexts do not match, likely because --input_file and --expanded_dataset are different datasets\n{contexts[-1]}\n{context}"
                # the first prediction has the format of the individual predictions
                # e.g., "contains the plane" or "is empty".
                cleaned_indiv_predictions.append(
                    indiv_predictions[0].strip().strip(".")
                )
                with_verb = indiv_predictions[0].strip().startswith(
                    "contains"
                ) or indiv_predictions[0].strip().startswith("is")
                for indiv_pred in indiv_predictions[1:]:
                    if with_verb:
                        if "contains " in indiv_pred:
                            idx = indiv_pred.index("contains ")
                        elif "is " in indiv_pred:
                            idx = indiv_pred.index("is ")
                        else:
                            print(
                                f"Unsupported prediction format: {indiv_pred}")
                            idx = 0
                            indiv_pred = "contains invalid"
                    else:
                        if "contains " in indiv_pred:
                            idx = indiv_pred.index(
                                "contains ") + len("contains ")
                        elif "is " in indiv_pred:
                            idx = 0
                            indiv_pred = "nothing"
                        else:
                            print(
                                f"Unsupported prediction format: {indiv_pred}")
                            idx = 0
                            indiv_pred = "invalid"
                            # raise ValueError(f"Unsupported prediction format: {indiv_pred}")
                    cleaned_indiv_predictions.append(
                        indiv_pred[idx:].strip("."))

        final_df = pd.DataFrame(
            {'target': targets, 'prediction': cleaned_indiv_predictions, 'input': contexts})
        with open(out_filename, 'w', encoding="UTF-8") as wf:
            wf.write(final_df.to_json(orient='records',
                     lines=True, force_ascii=False))
