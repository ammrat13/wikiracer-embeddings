import argparse
import typing
import xml.etree.ElementTree as ET

import jsonlines
import mwparserfromhell
import tiktoken
from tqdm import tqdm

NAMESPACE = "{http://www.mediawiki.org/xml/export-0.11/}"
"""Namespace prefix for the MediaWiki XML format

The XML parser returns node names with the namespace prefix included, so we have
to know this. This was taken from the root node of the XML file.
"""

TOKENIZER = tiktoken.get_encoding("cl100k_base")
"""The tokenizer used by OpenAI's embedding models

This is used to find out how many tokens are in each article.
"""

MAX_TOKENS = 8191
"""The maximum number of tokens that can be passed to the model at once"""


def get_pages(input_file) -> typing.Iterator[ET.Element]:
    """Yield each page element from the XML file

    We make sure to delete each page from the parse tree after we're done with
    it to prevent high memory use.
    """

    root = None
    for event, elem in ET.iterparse(input_file, events=["start", "end"]):

        if event == "start" and root is None:
            root = elem
            continue

        if event != "end":
            continue

        if elem.tag == f"{NAMESPACE}page":
            yield elem

        if elem != root and elem == root[0]:
            del root[0]
            continue


def filter_page(page: ET.Element) -> bool:
    """Return True if the article should be included in the output

    We don't want to count redirects, nor do we want to count pages outside of
    namespace 0.
    """

    if page.find(f"{NAMESPACE}redirect") is not None:
        return False
    if page.find(f"{NAMESPACE}ns").text != "0":
        return False
    return True


def main(args: argparse.Namespace) -> None:

    pages = filter(filter_page, get_pages(args.input))
    output = jsonlines.Writer(args.output)

    for i, page in tqdm(enumerate(pages)):

        title = page.find(f"{NAMESPACE}title").text
        text = page.find(f"{NAMESPACE}revision").find(f"{NAMESPACE}text").text

        parsed_text = mwparserfromhell.parse(text)
        if args.lead_only:
            parsed_text = parsed_text.get_sections(include_lead=True, include_headings=True)[0]

        compiled_text = "= " + title + " =\n\n" + parsed_text.strip_code()
        cropped_text = TOKENIZER.decode(TOKENIZER.encode(compiled_text)[:MAX_TOKENS])

        output.write(
            {
                "custom_id": f"simplewiki-{str(i)}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": "text-embedding-3-small",
                    "input": cropped_text,
                },
            }
        )

    output.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse Simple Wikipedia articles and create embedding requests"
    )
    parser.add_argument(
        "input", type=argparse.FileType("r"), help="Path to the input file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        help="Path to the output file",
        default="embeddings.jsonl",
    )
    parser.add_argument(
        "-l",
        "--lead-only",
        action="store_true",
        help="Only use the lead section of each article",
    )
    args = parser.parse_args()

    main(args)
