import json
import math
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt, RGBColor


WORKSPACE = Path(__file__).resolve().parents[3]
BUILD = WORKSPACE / "runtime" / "feature_reference"
DATA = json.loads((BUILD / "model_features.json").read_text(encoding="utf-8"))
OUTPUT = BUILD / "Inference_Model_Feature_Reference.docx"

NAVY = "17365D"
PALE_BLUE = "F2F6FA"
LIGHT_BLUE = "D9EAF7"
LIGHT_GRAY = "D9D9D9"
MID_GRAY = "5B6573"
BLACK = RGBColor(0, 0, 0)


def set_font(run, name="Aptos", size=None, bold=None, color=None):
    run.font.name = name
    run_properties = run._element.get_or_add_rPr()
    run_properties.rFonts.set(qn("w:ascii"), name)
    run_properties.rFonts.set(qn("w:hAnsi"), name)
    if run_properties.find(qn("w:noProof")) is None:
        run_properties.append(OxmlElement("w:noProof"))
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=75, start=90, bottom=75, end=90):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color=LIGHT_GRAY, size="4"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_repeat_table_layout(table):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    repeat_header(table.rows[0])
    for row in table.rows:
        prevent_row_split(row)
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)


def format_table_header(row, size=7.5):
    for cell in row.cells:
        set_cell_shading(cell, NAVY)
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            for run in paragraph.runs:
                set_font(run, size=size, bold=True, color=RGBColor(255, 255, 255))


def format_body_row(row, index, font_size=7.4, centered_columns=()):
    if index % 2 == 0:
        for cell in row.cells:
            set_cell_shading(cell, PALE_BLUE)
    for col_index, cell in enumerate(row.cells):
        for paragraph in cell.paragraphs:
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if col_index in centered_columns else WD_ALIGN_PARAGRAPH.LEFT
            )
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            for run in paragraph.runs:
                set_font(run, size=font_size, color=BLACK)


def set_col_widths(table, widths):
    table_layout = table._tbl.tblPr.find(qn("w:tblLayout"))
    if table_layout is None:
        table_layout = OxmlElement("w:tblLayout")
        table._tbl.tblPr.append(table_layout)
    table_layout.set(qn("w:type"), "fixed")
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width)


def add_field(paragraph, instruction):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])
    set_font(run, size=8, color=RGBColor(90, 100, 112))


def add_footer(section):
    footer = section.footer
    table = footer.add_table(rows=1, cols=3, width=Inches(9.55))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = (4.0, 3.6, 1.95)
    set_col_widths(table, widths)
    left, center, right = table.rows[0].cells
    left.paragraphs[0].add_run("Inference Pipeline Model Feature Reference")
    center.paragraphs[0].add_run("27 October 2026")
    right.paragraphs[0].add_run("Page ")
    add_field(right.paragraphs[0], "PAGE")
    for i, cell in enumerate((left, center, right)):
        paragraph = cell.paragraphs[0]
        paragraph.alignment = (WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT)[i]
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            set_font(run, size=8, color=RGBColor(90, 100, 112))


def configure_section(section, columns=1):
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Mm(297)
    section.page_height = Mm(210)
    section.top_margin = Mm(14)
    section.bottom_margin = Mm(15)
    section.left_margin = Mm(14)
    section.right_margin = Mm(14)
    section.header_distance = Mm(6)
    section.footer_distance = Mm(6)
    columns_element = section._sectPr.find(qn("w:cols"))
    if columns_element is None:
        columns_element = OxmlElement("w:cols")
        section._sectPr.append(columns_element)
    columns_element.set(qn("w:num"), str(columns))
    columns_element.set(qn("w:space"), "300")
    columns_element.set(qn("w:equalWidth"), "1")


def add_heading(doc, text, level=1):
    paragraph = doc.add_heading(text, level=level)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(14 if level == 1 else 9)
    paragraph.paragraph_format.space_after = Pt(6)
    return paragraph


def add_body(doc, text, bold_lead=None):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(7)
    paragraph.paragraph_format.line_spacing = 1.08
    if bold_lead:
        run = paragraph.add_run(bold_lead)
        set_font(run, bold=True, size=9.5)
    run = paragraph.add_run(text)
    set_font(run, size=9.5)
    return paragraph


def add_inventory(doc):
    add_heading(doc, "Model Inventory", 1)
    add_body(
        doc,
        "The table lists every model artifact referenced by the active inference configuration. "
        "Artifacts that contain several component estimators are shown as one configured binding because they share one saved feature contract.",
    )
    headers = ["No", "Configuration key", "Algorithm", "Inference output", "Contract", "Count", "Preprocessing"]
    table = doc.add_table(rows=1, cols=len(headers))
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    for index, model in enumerate(DATA["models"], 1):
        values = [
            str(index), model["key"], model["algorithm"], model["output"], model["contract"],
            str(model["feature_count"]), model["preprocessing"],
        ]
        row = table.add_row()
        for cell, value in zip(row.cells, values):
            cell.text = value
        format_body_row(row, index, font_size=7.0, centered_columns=(0, 4, 5))
    set_repeat_table_layout(table)
    set_col_widths(table, (0.35, 1.70, 1.35, 1.95, 0.72, 0.42, 3.02))
    format_table_header(table.rows[0], size=7.1)


def add_contract_summary(doc):
    doc.add_page_break()
    add_heading(doc, "Feature Contract Summary", 1)
    introduction = add_body(
        doc,
        "Four exact ordered feature contracts cover the 17 configured model bindings. "
        "The short hash is calculated from the newline joined ordered feature names and can be used to detect schema drift.",
    )
    introduction.paragraph_format.keep_with_next = True
    headers = ["Contract", "Feature count", "Ordered feature hash", "Configured model bindings"]
    table = doc.add_table(rows=1, cols=4)
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    for index, contract in enumerate(DATA["contracts"], 1):
        row = table.add_row()
        values = [
            contract["id"], str(contract["feature_count"]), contract["hash"], ", ".join(contract["models"]),
        ]
        for cell, value in zip(row.cells, values):
            cell.text = value
        format_body_row(row, index, font_size=7.4, centered_columns=(0, 1, 2))
    set_repeat_table_layout(table)
    set_col_widths(table, (1.15, 0.85, 1.45, 6.06))
    format_table_header(table.rows[0], size=7.4)


def add_artifact_bindings(doc):
    add_heading(doc, "Artifact Bindings", 1)
    add_body(
        doc,
        "Paths are relative to the workspace root. The configuration file used for this reference is "
        f"{DATA['config']}.",
    )
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Configuration key"
    table.rows[0].cells[1].text = "Saved artifact"
    for index, model in enumerate(DATA["models"], 1):
        row = table.add_row()
        row.cells[0].text = model["key"]
        row.cells[1].text = model["artifact"]
        format_body_row(row, index, font_size=6.8)
    set_repeat_table_layout(table)
    for row in table.rows:
        for cell in row.cells:
            set_cell_margins(cell, top=35, bottom=35)
    set_col_widths(table, (2.5, 7.01))
    format_table_header(table.rows[0], size=7.0)


def add_feature_contract(doc, contract, *, last=False):
    add_heading(doc, f"{contract['id']} Feature Contract", 1)
    add_body(
        doc,
        f"This contract contains {contract['feature_count']} ordered fields. "
        f"Ordered feature hash {contract['hash']}. Models using it are {', '.join(contract['models'])}.",
    )
    if contract["id"] == "RV 168":
        add_body(
            doc,
            "The RV model consumes current option and market state, exact lags and changes through 60 minutes, "
            "spot return and realized variation measures, and additional engineered transforms. The saved prediction "
            "is inverted from log variance and multiplied by the artifact calibration value.",
        )
    elif contract["id"] == "Combined 151":
        add_body(
            doc,
            "All combined IV regressors and the convergence classifier use the same schema. The last two inputs are "
            "the predicted 120 minute RV and the current combined IV gap to that prediction.",
        )
    elif contract["id"] == "Leg 143":
        add_body(
            doc,
            "The call and put point and quantile models share this schema. Their raw target is the log ratio of future "
            "fixed strike IV to entry fixed strike IV. The history window stops at 60 minutes.",
        )
    elif contract["id"] == "Event 168":
        meta = DATA["event_metadata"]
        target = meta["target_definition"]
        add_body(
            doc,
            f"The event bundle uses a {meta['lookback_minutes']} minute maximum lookback and {len(meta['engineered_features'])} "
            f"engineered fields. Its fixed score threshold is {meta['threshold']}. The 120 minute event requires an RV "
            f"drop of at least {target['minimum_rv_drop']:.2f} and an RV decline at least "
            f"{target['minimum_rv_drop_beyond_iv']:.2f} greater than the fixed strike combined IV decline.",
        )

    feature_table = doc.add_table(rows=1, cols=3)
    feature_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    feature_table.autofit = False
    chunk_size = math.ceil(len(contract["features"]) / 3)
    for column_index, cell in enumerate(feature_table.rows[0].cells):
        set_cell_margins(cell, top=0, start=0, bottom=0, end=70)
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 0.9
        start = column_index * chunk_size
        stop = min(start + chunk_size, len(contract["features"]))
        for feature_index in range(start, stop):
            number = paragraph.add_run(f"{feature_index + 1:03d}  ")
            set_font(number, name="Aptos Mono", size=6.7, bold=True, color=RGBColor(23, 54, 93))
            name = paragraph.add_run(contract["features"][feature_index])
            set_font(name, name="Aptos Mono", size=6.7, color=BLACK)
            if feature_index + 1 != stop:
                name.add_break()
    set_col_widths(feature_table, (3.17, 3.17, 3.17))
    if not last:
        doc.add_page_break()


def build():
    doc = Document()
    section = doc.sections[0]
    configure_section(section, columns=1)
    add_footer(section)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = BLACK
    for style_name, size in (("Title", 28), ("Heading 1", 17), ("Heading 2", 12)):
        style = styles[style_name]
        style.font.name = "Aptos Display" if style_name != "Title" else "Aptos Display"
        style._element.rPr.rFonts.set(qn("w:ascii"), style.font.name)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), style.font.name)
        style.font.size = Pt(size)
        style.font.bold = style_name != "Title"
        style.font.color.rgb = BLACK
        if style_name == "Title":
            style_paragraph_properties = style._element.get_or_add_pPr()
            border = style_paragraph_properties.find(qn("w:pBdr"))
            if border is not None:
                style_paragraph_properties.remove(border)

    title = doc.add_paragraph(style="Title")
    title.paragraph_format.space_after = Pt(6)
    title.add_run("Inference Pipeline Model Feature Reference")
    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(20)
    run = subtitle.add_run("Active model bindings and exact ordered inference features")
    set_font(run, size=13, color=RGBColor(70, 80, 92))

    intro = doc.add_paragraph()
    intro.paragraph_format.space_after = Pt(14)
    intro.paragraph_format.line_spacing = 1.15
    run = intro.add_run(
        "This reference documents every model artifact used by the current fixed strike inference pipeline. "
        "It maps each configured model to its saved feature contract and records every feature in the exact order "
        "supplied during inference. Use it when reviewing model dependencies, preparing data, validating deployments, "
        "or investigating feature schema changes."
    )
    set_font(run, size=11)

    facts = [
        ("Configured model bindings", str(len(DATA["models"]))),
        ("Unique ordered feature contracts", str(len(DATA["contracts"]))),
        ("Contract feature entries", str(sum(c["feature_count"] for c in DATA["contracts"]))),
        ("Reference date", DATA["document_date"]),
    ]
    table = doc.add_table(rows=2, cols=4)
    for index, (label, value) in enumerate(facts):
        cell = table.cell(0, index)
        cell.text = label
        value_cell = table.cell(1, index)
        value_cell.text = value
        set_cell_shading(cell, LIGHT_BLUE)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        value_cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in cell.paragraphs[0].runs:
            set_font(run, size=8, bold=True, color=BLACK)
        for run in value_cell.paragraphs[0].runs:
            set_font(run, size=15, bold=True, color=RGBColor(23, 54, 93))
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        value_cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_margins(cell, top=110, bottom=110)
        set_cell_margins(value_cell, top=130, bottom=130)
    set_table_borders(table)
    set_col_widths(table, (2.38, 2.38, 2.38, 2.38))

    add_heading(doc, "Reading This Reference", 1)
    add_body(
        doc,
        "Feature order is part of the model contract. A dataset that contains the right names in the wrong order must "
        "be reindexed before prediction. Feature names are case sensitive, including fields such as Future_oi. Missing "
        "or infinite values are converted to missing values before the saved estimator or pipeline handles them.",
    )
    add_body(
        doc,
        "The inventory counts configured artifact bindings rather than individual estimators. The combined ensemble "
        "contains three component models, and the combined quantile artifact contains three quantile estimators, but "
        "each artifact uses one shared ordered contract.",
    )

    doc.add_page_break()
    add_inventory(doc)
    add_contract_summary(doc)
    add_artifact_bindings(doc)

    first_contract_section = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section(first_contract_section, columns=1)
    for index, contract in enumerate(DATA["contracts"]):
        add_feature_contract(doc, contract, last=index == len(DATA["contracts"]) - 1)

    doc.core_properties.title = "Inference Pipeline Model Feature Reference"
    doc.core_properties.subject = "Active model bindings and exact ordered inference features"
    doc.core_properties.keywords = "inference models features contracts IV RV BANKNIFTY"
    doc.core_properties.comments = "Generated from active saved model artifacts and configuration"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
