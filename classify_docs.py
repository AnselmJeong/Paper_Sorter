from pathlib import Path

from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from argparse import ArgumentParser
from tqdm import tqdm


SYSTEM_MESSAGE = """The file name entered by the "user" includes the title and author of the academic book or articles.
Select only one of the following categories that best represent the subject or theme of this book or articles. 
The category which includes "in General" can only be selected when there is no other more specific category that applies.
If none of the provided categories can be applied, just select "Not Otherwise Specified."

IMPORTANT: NEVER use ad hoc categories that are not included in the below Categories.

CATEGORIES
=====================
{categories} 
"""


def get_chain(root_path: Path) -> Runnable:
    # Get existing directory names as categories
    categories = [d.name for d in root_path.iterdir() if d.is_dir()]

    # If no directories exist, create a default "Uncategorized" directory
    if not categories:
        categories = ["Uncategorized"]

    # Create the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_MESSAGE.format(categories="\n".join(categories)),
            ),
            ("user", "{file_name}"),
        ]
    )

    # Define the output schema
    class Category(BaseModel):
        """Category that best represents the given book or article's subject"""

        subject: Literal[tuple(categories)] = Field(
            ..., description="The most representative subject of the book or article"
        )

    # Create the chain by linking the prompt and the LLM.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", temperature=1
    ).with_structured_output(Category, strict=True)

    return prompt | llm


def get_category(files: list[Path], chain: Runnable) -> dict:
    # Get the categories of each of the files by invoking LLM.
    output = {}
    for file in tqdm(files, desc="Getting categories"):
        response = chain.invoke({"file_name": file.stem})
        output[file.name] = response
    return output


def move_to_subdirectory(file_path: Path, subdirectory: str):
    # Move one file to the subdirectory with the category name.
    subdirectory_path = Path(subdirectory)
    if not subdirectory_path.exists():
        subdirectory_path.mkdir(parents=True)
    file_path.rename(subdirectory_path / file_path.name)


def process_files(root_path: str | Path, chain: Runnable):
    # Process all the files in the root directory.
    if not isinstance(root_path, Path):
        root_path = Path(root_path)

    if not root_path.is_dir():
        raise ValueError(f"The path {root_path} is not a valid directory.")

    files = [
        file
        for file in root_path.iterdir()
        if file.suffix in [".pdf", ".epub", ".docx"]
    ]
    output = get_category(files, chain)

    for file in files:
        move_to_subdirectory(file, root_path / output[file.name].subject)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "root_path", type=str, help="Directory containing the files to be processed"
    )
    args = parser.parse_args()

    root_path = Path(args.root_path)
    chain = get_chain(root_path)
    process_files(root_path, chain)
