"""Insert the corrected-methodology note in the Persian thesis coherently.

Run only after `run_official_experiments.py`; values are read from generated
CSV files rather than typed into the document.
"""
from pathlib import Path
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "پایان_نامه.docx"
OFFICIAL = ROOT / "results_official"
MARKER = "یادداشت روش‌شناختی و نتایج اصلاح‌شده"


def add_before(document, anchor, text, style=None):
    # The thesis uses localized/custom Word styles, so English Heading names
    # are not guaranteed to exist.  Fall back to Normal while retaining a
    # readable bold heading.
    paragraph = document.add_paragraph(text)
    if style:
        try:
            paragraph.style = style
        except KeyError:
            for run in paragraph.runs:
                run.bold = True
    anchor._p.addprevious(paragraph._p)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    return paragraph


def metric(metrics, model):
    row = metrics.loc[metrics.model == model].iloc[0]
    return row


def main():
    document = Document(THESIS)
    if any(MARKER in p.text for p in document.paragraphs):
        raise SystemExit("Corrected-methodology section already exists; refusing to duplicate it.")
    anchors = [p for p in document.paragraphs if p.text.strip() == "فهرست منابع"]
    if not anchors:
        raise RuntimeError("Could not find the references heading in the thesis.")
    anchor = anchors[0]
    d1 = pd.read_csv(OFFICIAL / "d1_metrics.csv")
    base = metric(d1, "RF behavioral")
    extended = metric(d1, "RF extended behavioral")
    content = metric(d1, "RF content (matched text subset)")
    behavioural_matched = metric(d1, "RF behavioral (matched text subset)")
    fusion = metric(d1, "RF fusion (matched text subset)")
    add_before(document, anchor, MARKER, "Heading 1")
    add_before(document, anchor, "اصلاحات اعتبار روش و بازتولیدپذیری", "Heading 2")
    add_before(document, anchor,
        "پس از اجرای آزمایش‌های اولیه، مسیر ارزیابی بازبینی شد. خروجی‌های قدیمی در پوشه‌های results_custom و results_dataset2 حفظ شده‌اند، اما به‌دلیلِ برازشِ میانه‌گذاری روی کل داده پیش از اعتبارسنجی متقاطع، و انتخاب آستانه روی همان پیش‌بینی‌های OOF که برای گزارش عملکرد استفاده شده بودند، به‌عنوان نتایج اکتشافیِ پیش از اصلاح در نظر گرفته می‌شوند. حذف این آثار تاریخی به‌درستیِ علمی کمکی نمی‌کرد؛ بنابراین از نتایج رسمی جدید جدا نگه داشته شده‌اند.")
    add_before(document, anchor,
        "در مسیر رسمی جدید، در هر لایهٔ بیرونیِ Stratified CV، میانه‌گذاریِ مقادیر گم‌شده فقط با بخش آموزش برازش می‌شود و همان تبدیل بر بخش آزمون اعمال می‌گردد. مقیاس‌بندیِ مدل‌های خطی نیز درون Pipeline و فقط روی دادهٔ آموزش انجام می‌شود. برای معیارهای وابسته به آستانه، آستانه در CV داخلیِ دادهٔ آموزش انتخاب و فقط یک‌بار روی لایهٔ آزمون بیرونی ارزیابی می‌شود. بنابراین ROC-AUC از احتمال‌های OOF و F1/Precision/Recall از آستانه‌های مستقلِ هر لایه به‌دست می‌آیند.")
    add_before(document, anchor, "نتایج رسمیِ اصلاح‌شده برای دیتاست اول", "Heading 2")
    add_before(document, anchor,
        "جدول زیر خروجی واقعیِ اجرای مسیر اصلاح‌شده را گزارش می‌کند. برای مقایسهٔ رفتار/محتوا/فیوژن، هر سه مدل دقیقاً روی همان ۱۶۲ حسابِ دارای متن ارزیابی شده‌اند؛ ازاین‌رو اختلاف آن‌ها به تفاوت اندازهٔ نمونه نسبت داده نمی‌شود.")
    table = document.add_table(rows=1, cols=5)
    try:
        table.style = "Table Grid"
    except KeyError:
        pass
    headers = ["مدل", "n", "ROC-AUC (OOF)", "F1 میانگین بیرونی", "آستانهٔ میانه"]
    for cell, text in zip(table.rows[0].cells, headers): cell.text = text
    rows = [
        ("رفتاری پایه، ۲۴ فیچر", base), ("رفتاری گسترش‌یافته، ۴۵ فیچر", extended),
        ("رفتاری، زیرمجموعهٔ متن", behavioural_matched), ("محتوا، زیرمجموعهٔ متن", content),
        ("فیوژن، زیرمجموعهٔ متن", fusion),
    ]
    for label, row in rows:
        cells = table.add_row().cells
        values = [label, str(int(row.n)), f"{row.roc_auc_oof:.3f}",
                  f"{row.f1_outer_mean:.3f}", f"{row.selected_threshold_median:.2f}"]
        for cell, value in zip(cells, values): cell.text = value
    anchor._p.addprevious(table._tbl)
    add_before(document, anchor,
        f"روی کل دیتاست اول، ROC-AUC مدل پایه {base.roc_auc_oof:.3f} و مدل رفتاری گسترش‌یافته {extended.roc_auc_oof:.3f} است. در زیرمجموعهٔ متن، AUC رفتاری {behavioural_matched.roc_auc_oof:.3f}، محتوایی {content.roc_auc_oof:.3f} و فیوژن {fusion.roc_auc_oof:.3f} است. بنابراین فیچرهای فعالیتِ افزوده‌شده — نوآوری کوچک و قابل‌آزمون این پروژه — نسبت به پایهٔ ۲۴فیچری بهبود رتبه‌بندی ایجاد می‌کنند؛ اما فیوژن زبانی/محتوایی در این نمونهٔ کوچک برتریِ AUC نشان نداده است. این نتیجه نباید به‌عنوان اثر علّیِ فیچرها تفسیر شود.")
    add_before(document, anchor, "دامنهٔ نوآوری و محدودیت‌های تفسیر", "Heading 2")
    add_before(document, anchor,
        "روش اصلیِ Katyal شامل فیچرهای تاریخچهٔ حساب و Random Forest است. فیچرهای فعالیت و ترکیب نوع توییت، افزودهٔ این پروژه بر مبنای ستون‌های موجود در داده‌اند. ماژول hazm نیز انطباق فارسیِ ایدهٔ LFC مقالهٔ دوم است، نه بازتولیدِ مدل یا وظیفهٔ اصلی آن مقاله. ویژگی‌های هماهنگیِ مبتنی بر متن تکراری در تحلیل سراسری ماهیتی transductive دارند؛ برای ارزیابی inductive باید فقط با محتوای بخش آموزش ساخته شوند. تابع جداگانهٔ compute_inductive_coordination_features برای این منظور اضافه شده است.")
    add_before(document, anchor, "وضعیت دیتاست دوم و کار باقی‌مانده", "Heading 2")
    d2_path = OFFICIAL / "d2_metrics.csv"
    if d2_path.exists():
        d2 = pd.read_csv(d2_path)
        d2base = metric(d2, "RF behavioral")
        d2ext = metric(d2, "RF extended behavioral")
        add_before(document, anchor,
            f"مسیر رسمی دیتاست دوم نیز اجرا شد: AUC پایه {d2base.roc_auc_oof:.3f} و AUC گسترش‌یافته {d2ext.roc_auc_oof:.3f}. برچسب‌های این دیتاست همچنان proxy label هستند: prob1 با آستانهٔ ۰٫۵ و افزودنِ حساب تعلیق‌شده فقط در نبودِ برچسب؛ تعارض‌ها حفظ شده‌اند. گزارش تشخیصی یکتایی screen_name و پوشش اتصال متن در metadata.json ثبت می‌شود.")
    else:
        add_before(document, anchor,
            "اجرای رسمی دیتاست دوم در زمان به‌روزرسانی این سند در حال انجام/ثبت است؛ تا پیش از تولید فایل results_official/d2_metrics.csv نباید ارقام پیش از اصلاح به‌عنوان نتایج رسمی گزارش شوند. منشأ برچسب‌ها proxy label است: prob1 با آستانهٔ ۰٫۵ و افزودن حساب تعلیق‌شده فقط در نبود برچسب؛ تعارض‌ها باید حفظ و گزارش شوند.")
    add_before(document, anchor,
        "بازآفرینی‌پذیری", "Heading 2")
    add_before(document, anchor,
        "فایل requirements.txt، پیکربندی متمرکز seed و پارامترهای مدل، آزمون دودِ بدون داده، کنترل‌های schema، هش SHA-256 داده‌ها، نسخهٔ بسته‌ها، شاخص‌های هر fold و پیش‌بینی‌های OOF افزوده شده‌اند. دستور اجرای رسمی در README آمده است. آزمون مقاومت با بازنویسی heuristic، برچسب‌های proxy دیتاست دوم، و کمبود متن در دیتاست اول همچنان محدودیت‌های صریح پروژه‌اند.")
    document.save(THESIS)


if __name__ == "__main__":
    main()
