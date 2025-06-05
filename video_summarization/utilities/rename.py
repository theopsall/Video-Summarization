import argparse
import os
import re

from video_summarization.utilities.utils import crawl_directory

grToLat = {
    "Α": "A",
    "Ά": "A",
    "α": "a",
    "ά": "a",
    "Β": "B",
    "β": "b",
    "Γ": "G",
    "γ": "g",
    "Δ": "D",
    "δ": "d",
    "Ε": "E",
    "Έ": "E",
    "έ": "e",
    "ε": "e",
    "Ζ": "Z",
    "ζ": "z",
    "Η": "H",
    "Ή": "H",
    "η": "h",
    "ή": "h",
    "Θ": "U",
    "θ": "u",
    "Ι": "I",
    "Ί": "I",
    "ι": "i",
    "ί": "i",
    "Κ": "K",
    "κ": "k",
    "Λ": "L",
    "λ": "l",
    "Μ": "M",
    "μ": "m",
    "Ν": "N",
    "ν": "n",
    "Ξ": "J",
    "ξ": "j",
    "Ο": "O",
    "Ό": "O",
    "ο": "o",
    "ό": "o",
    "Π": "P",
    "π": "p",
    "Ρ": "R",
    "ρ": "r",
    "Σ": "S",
    "ς": "s",
    "σ": "s",
    "Τ": "T",
    "τ": "t",
    "Υ": "Y",
    "Ύ": "Y",
    "ύ": "y",
    "υ": "y",
    "Φ": "F",
    "φ": "f",
    "Χ": "X",
    "χ": "x",
    "Ψ": "C",
    "ψ": "c",
    "ω": "v",
    "ώ": "v",
    "Ω": "V",
    "Ώ": "V",
}


def deEmojify(text):
    regrex_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002500-\U00002bef"  # chinese char
        "\U00002702-\U000027b0"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2b55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    return regrex_pattern.sub(r"", text)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input folder with Videos")

    return parser.parse_args()


def rename(input_dir):
    tree = crawl_directory(input_dir)
    for filename in tree:
        removed = filename.maketrans(grToLat)
        dst = filename.translate(removed)
        dst = deEmojify(dst)
        os.rename(filename, dst)


if __name__ == "__main__":
    print("Starting renaming process")
    parser = parse_arguments()
    if not os.path.isdir(parser.input):
        assert f"Error {parser.input} not found"
    rename(parser.input)
    print("Renaming process Finished!")
