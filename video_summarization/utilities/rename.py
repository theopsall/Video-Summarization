import argparse
import os
import re

from utils import crawl_directory

grToLat = {
    'Α': 'A', 'Ά': 'A', 'α': 'a', 'ά': 'a', 'Β': 'B',
    'β': 'b', 'Γ': 'G', 'γ': 'g', 'Δ': 'D', 'δ': 'd',
    'Ε': 'E', 'Έ': 'E', 'έ': 'e', 'ε': 'e', 'Ζ': 'Z',
    'ζ': 'z', 'Η': 'H', 'Ή': 'H', 'η': 'h', 'ή': 'h',
    'Θ': 'U', 'θ': 'u', 'Ι': 'I', 'Ί': 'I', 'ι': 'i',
    'ί': 'i', 'Κ': 'K', 'κ': 'k', 'Λ': 'L', 'λ': 'l',
    'Μ': 'M', 'μ': 'm', 'Ν': 'N', 'ν': 'n', 'Ξ': 'J',
    'ξ': 'j', 'Ο': 'O', 'Ό': 'O', 'ο': 'o', 'ό': 'o',
    'Π': 'P', 'π': 'p', 'Ρ': 'R', 'ρ': 'r', 'Σ': 'S',
    'ς': 's', 'σ': 's', 'Τ': 'T', 'τ': 't', 'Υ': 'Y',
    'Ύ': 'Y', 'ύ': 'y', 'υ': 'y', 'Φ': 'F', 'φ': 'f',
    'Χ': 'X', 'χ': 'x', 'Ψ': 'C', 'ψ': 'c', 'ω': 'v',
    'ώ': 'v', 'Ω': 'V', 'Ώ': 'V'
}


def deEmojify(text):
    regrex_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", re.UNICODE)
    return regrex_pattern.sub(r'', text)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input folder with Videos")

    return parser.parse_args()


def main():
    parser = parse_arguments()
    tree = crawl_directory(parser.input)
    for filename in tree:
        removed = filename.maketrans(grToLat)
        dst = filename.translate(removed)
        dst = deEmojify(dst)
        os.rename(filename, dst)


if __name__ == "__main__":
    print("Starting renaming process")
    main()
    print("Renaming process Finished!")
