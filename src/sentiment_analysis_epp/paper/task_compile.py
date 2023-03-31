"""Tasks for compiling the paper of the sentiment analysis project."""
from pathlib import Path
import shutil
import pytask
from pytask_latex import compilation_steps as cs

from sentiment_analysis_epp.config import BLD
PAPER_DIR = Path(__file__).parent


document = "paper"

@pytask.mark.latex(
    script=PAPER_DIR / f"{document}.tex",
    document=BLD / "latex" / f"{document}.pdf",
    compilation_steps=cs.latexmk(
        options=("--pdf", "--interaction=nonstopmode", "--synctex=1", "--cd"),
    ),
)
@pytask.mark.task(id=document)
def task_compile_document():
    """Compile the paper specified in the latex decorator."""

kwargs = {
    "depends_on": BLD / "latex" / f"{document}.pdf",
    "produces": PAPER_DIR / f"{document}.pdf",
}

@pytask.mark.task(id=f"copy_{document}", kwargs=kwargs)
def task_copy_to_root(depends_on, produces):
    """Copy the compiled paper to the root directory."""
    shutil.copy(depends_on, produces)
