import shutil
from pathlib import Path
import pytask
from pytask_latex import compilation_steps as cs

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

DEPS = [
    BLD / "python" / "conf_matrix_logit.tex",
    BLD / "python" / "conf_matrix_nb.tex",
    BLD / "python" / "confusion_matrix_svm.tex",
    BLD / "plots" / "sentiment_hist.png",
    BLD / "plots" / "performance_plot_20230331_193625.png",
]

TEX_FILE = SRC / "paper.tex"
PDF_FILE = BLD / "latex" / "paper_py.pdf"
document = "paper_py"

@pytask.mark.latex(
    script=TEX_FILE,
    document=PDF_FILE,
    compilation_steps=cs.latexmk(
        options=("--pdf", "--interaction=nonstopmode", "--synctex=1", "--cd"),
    ),
)
def task_compile_paper():
    """Compile the paper using LaTeX."""
kwargs = { "depends_on": SRC / "paper.tex",
          "produces": BLD.parent.resolve() / "paper_py.pdf",
}       

@pytask.mark.task(id=document, kwargs = kwargs)
def task_copy_pdf_to_root(depends_on, produces):
    """Copy the PDF file to the root directory."""
    shutil.copy(depends_on, produces)
