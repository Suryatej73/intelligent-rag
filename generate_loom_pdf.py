import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header line
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(40, letter[1] - 40, letter[0] - 40, letter[1] - 40)
        self.drawString(40, letter[1] - 35, "Intelligent RAG · Loom Video Script (Set A) · Surya tej Reddy (Rank #1)")
        
        # Footer line
        self.line(40, 45, letter[0] - 40, 45)
        self.drawString(40, 32, "NxtWave Academy · Gen AI Project · Zyro Dynamics HR Help Desk")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 40, 32, page_str)
        self.restoreState()

def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=55
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A202C"),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2B6CB0"),
        spaceAfter=10
    )

    badge_style = ParagraphStyle(
        'Badge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2C5282")
    )

    q_title_style = ParagraphStyle(
        'QuestionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=0,
        spaceAfter=4
    )

    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4A5568")
    )

    meta_val_style = ParagraphStyle(
        'MetaValStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748")
    )

    spoken_style = ParagraphStyle(
        'SpokenText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1A202C")
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=12,
        spaceAfter=3
    )

    checklist_style = ParagraphStyle(
        'ChecklistText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2D3748")
    )

    story = []

    # Title & Header
    story.append(Paragraph("IRP · ACADEMY · GEN AI PROJECT · LOOM VIDEO (SET A)", badge_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Intelligent RAG: HR Help Desk Chatbot", title_style))
    story.append(Paragraph("Official Loom Walkthrough Guide & 7-Question Spoken Script", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3182CE"), spaceBefore=2, spaceAfter=10))

    # Meta banner table
    summary_data = [
        [
            Paragraph("<b>Participant:</b> Surya tej Reddy (@suryatej73)", meta_val_style),
            Paragraph("<b>Target Duration:</b> 8 min max (Aim for 7:30)", meta_val_style)
        ],
        [
            Paragraph("<b>Leaderboard Standing:</b> 🥇 <b>Rank #1 (Score: 92.06000)</b>", meta_val_style),
            Paragraph("<b>Company / Corpus:</b> Zyro Dynamics (11 Policy PDFs)", meta_val_style)
        ]
    ]
    t_summary = Table(summary_data, colWidths=[270, 260])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 10))

    # Recording rules banner
    rules_text = (
        "<b>MANDATORY RECORDING RULES:</b><br/>"
        "1. Stop before 8:00 (aim for 7:30 buffer). • "
        "2. Answer all 7 questions in order. • "
        "3. <b>Say the question number out loud before each answer</b> (e.g. <i>'Question 1...'</i>). • "
        "4. Keep webcam + screen on throughout. • "
        "5. Show real notebook and app (no slides only). • "
        "6. Set Loom permission to <i>'Anyone with link can view'</i>."
    )
    t_rules = Table([[Paragraph(rules_text, checklist_style)]], colWidths=[530])
    t_rules.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EBF8FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#63B3ED")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_rules)
    story.append(Spacer(1, 14))

    # Questions Data
    questions = [
        {
            "num": "Question 1",
            "title": "Introduce your bot",
            "time": "0:00 – 0:45 (45 sec)",
            "screen": "Kaggle Leaderboard page showing your name (Surya tej Reddy) at Rank #1 with score 92.06000.",
            "spoken": (
                "\"Question 1: Introduce your bot.<br/><br/>"
                "This bot is built for the employees and HR operations team of Zyro Dynamics, a fast-growing 500+ "
                "employee SaaS company. It solves the critical bottleneck of HR staff being flooded with repetitive "
                "inquiries regarding leave policies, medical insurance coverages, payroll deadlines, and travel allowances. "
                "By using Retrieval-Augmented Generation across 11 internal policy PDFs, it provides instant, strictly grounded "
                "answers with exact page citations while preventing hallucinations.<br/><br/>"
                "My final score on the Kaggle public leaderboard is <b>92.06000</b>, currently holding <b>Rank #1</b>.\""
            )
        },
        {
            "num": "Question 2",
            "title": "Walk through your pipeline",
            "time": "0:45 – 2:00 (1 min 15 sec)",
            "screen": "Scroll through Cell 2 in Kaggle notebook showing PyPDFLoader, RecursiveCharacterTextSplitter, FAISS, and RAG chain.",
            "spoken": (
                "\"Question 2: Walk through your pipeline.<br/><br/>"
                "Here is the step-by-step engineering pipeline of how a raw PDF becomes a verified answer:<br/>"
                "• <b>PDF Loader</b>: I used LangChain's <code>PyPDFLoader</code> to ingest all 11 HR policy PDFs (39 pages total), extracting clean text and preserving document names and page numbers in metadata.<br/>"
                "• <b>Text Splitting</b>: Chunks are generated using <code>RecursiveCharacterTextSplitter</code> with a <b>chunk size of 800 characters</b> and an <b>overlap of 150 characters</b>.<br/>"
                "• <b>Embeddings</b>: Processed through <code>sentence-transformers/all-MiniLM-L6-v2</code> with L2 normalization for dense 384-dimensional vectors.<br/>"
                "• <b>Vector Store & Retrieval</b>: Indexed in a <b>FAISS CPU</b> store using <b>Maximum Marginal Relevance (MMR)</b> with k=5 and fetch_k=15.<br/>"
                "• <b>LLM & Generation</b>: Context is routed via LangChain LCEL into <b>Groq's llama-3.3-70b-versatile</b> (with llama-3.1-8b-instant fallback) using a strictly grounded HR system prompt.\""
            )
        },
        {
            "num": "Question 3",
            "title": "Your chunking choice",
            "time": "2:00 – 3:00 (1 min)",
            "screen": "Printed chunk from 02_Leave_Policy.pdf in notebook showing the Earned Leave accrual rule.",
            "spoken": (
                "\"Question 3: Your chunking choice.<br/><br/>"
                "I configured <b>chunk_size = 800</b> and <b>chunk_overlap = 150</b> based on empirical policy paragraph analysis. "
                "Zyro Dynamics' HR policy rules—such as Earned Leave accrual thresholds or maternity eligibility—average 400 to 650 characters. "
                "A smaller chunk size like 300 splits a policy rule from its mandatory exceptions, while larger sizes like 1500 pull unrelated sections, causing retrieval dilution.<br/><br/>"
                "<i>(Point to notebook output)</i>: Look at this printed chunk from <code>02_Leave_Policy.pdf</code>: it contains the complete Earned Leave rule—the 1-year service eligibility, the 240 days worked requirement, the 1.25 days per month accrual, and the 0.5 probation rate. Because of the 150-character overlap, the rule is kept 100% intact and never cut in half.\""
            )
        },
        {
            "num": "Question 4",
            "title": "Your retrieval settings",
            "time": "3:00 – 4:00 (1 min)",
            "screen": "Notebook code cell defining the MMR retriever (k=5, fetch_k=15, lambda_mult=0.7).",
            "spoken": (
                "\"Question 4: Your retrieval settings.<br/><br/>"
                "I specifically implemented <b>MMR (Maximum Marginal Relevance)</b> instead of plain similarity search. "
                "Plain cosine similarity often retrieves redundant chunks from the exact same page or table header. "
                "MMR optimizes for both query relevance and contextual diversity.<br/><br/>"
                "My final hyperparameters are: <b>k = 5</b>, <b>fetch_k = 15</b>, and <b>lambda_mult = 0.7</b>.<br/>"
                "• When I tested plain similarity or higher k (e.g. k=8), duplicate chunks crowded out important policy nuances and reduced answer conciseness.<br/>"
                "• When lambda_mult was too low (0.3), it forced diversity by pulling irrelevant documents.<br/>"
                "• A lambda of 0.7 provided the optimal sweet spot, retrieving the exact policy section along with its related payout and exception tables.\""
            )
        },
        {
            "num": "Question 5",
            "title": "Live demo on your deployed app",
            "time": "4:00 – 6:00 (2 min) — Core Demo",
            "screen": "Split screen or side-by-side tabs: Deployed Streamlit web app and the official policy PDFs.",
            "spoken": (
                "\"Question 5: Live demo on your deployed app.<br/><br/>"
                "<b>1. Leave Policy Query:</b><br/>"
                "I type: <i>'How much Earned Leave can I carry forward to next year and what happens to the excess?'</i><br/>"
                "Bot Answer: A maximum of 45 days of Earned Leave may be carried forward at the end of the financial year (31 March). Any excess is automatically encashed at the basic daily rate in April payroll.<br/>"
                "Citation: <code>02_Leave_Policy.pdf (Page 3)</code>. Switching to the PDF on Page 3, you see the exact sentence under Section 3.2.<br/><br/>"
                "<b>2. Travel & Expense Policy Query:</b><br/>"
                "I type: <i>'What is the hotel and daily allowance for an L4 employee on international travel?'</i><br/>"
                "Bot Answer: For L3 to L4 grades, the hotel allowance is up to USD 120 per night, and daily allowance is USD 60.<br/>"
                "Citation: <code>10_Travel_and_Expense_Policy.pdf (Page 2)</code>. Showing the table in the PDF confirms the USD 120 and USD 60 limits.<br/><br/>"
                "<b>3. Out-of-Scope Query:</b><br/>"
                "I type: <i>'Who won the last IPL?'</i> (or <i>'What was Zyro's revenue last year?'</i>)<br/>"
                "Bot Refusal: 'I'm sorry, but I can only answer HR-related questions based on Zyro Dynamics' internal policy documents. Your question appears to be outside my scope.' Verified 100% polite refusal.\""
            )
        },
        {
            "num": "Question 6",
            "title": "Out-of-scope detection",
            "time": "6:00 – 7:15 (1 min 15 sec)",
            "screen": "Guardrail classifier function/prompt in notebook and LangSmith trace dashboard.",
            "spoken": (
                "\"Question 6: Out-of-scope detection.<br/><br/>"
                "My bot uses a <b>hybrid guardrail architecture</b> combining rule-based boundary filters with an LLM intent classifier.<br/>"
                "• It classifies queries against 5 forbidden categories: external job applications/hiring, confidential individual compensation like personal ESOP vesting schedules, unlisted company financial metrics, competitor product comparisons (Salesforce/Zoho), and general trivia.<br/>"
                "• When a query triggers an out-of-scope intent, retrieval is bypassed entirely to save compute and eliminate hallucinations, instantly returning the official NxtWave polite refusal message.<br/><br/>"
                "<i>(Switch to LangSmith)</i>: In this LangSmith trace for 'Who won the last IPL?', you can see the execution tree tagged as <code>refusal_guardrail</code> with status <code>refused</code> and zero vector store calls, ensuring complete compliance with competition constraints.\""
            )
        },
        {
            "num": "Question 7",
            "title": "Reflect",
            "time": "7:15 – 8:00 (45 sec)",
            "screen": "Notebook showing Version 1 vs Version 2, or Question 13 (POSH complaint inquiry timeline).",
            "spoken": (
                "\"Question 7: Reflect.<br/><br/>"
                "In my initial Version 1 submission, which scored 82.89, the bot answered weakly on <b>Question 13</b> regarding the POSH complaint timeline. Because the word 'timeline' was heavily represented in the Annual Performance Review (APR) schedule, plain retrieval contaminated the context with appraisal deadlines instead of the Internal Complaints Committee (ICC) 60-day inquiry window.<br/><br/>"
                "To resolve this in Version 2, I added entity-specific intent routing for statutory compliance terms like POSH and ICC, and standardized the refusal formatting. This eliminated cross-policy contamination, boosted our score to <b>92.06000, and secured Rank #1 on the leaderboard</b>.<br/><br/>"
                "Thank you for watching!\""
            )
        }
    ]

    for q in questions:
        card_content = [
            [
                Paragraph(f"<b>{q['num']}: {q['title']}</b>", q_title_style),
                Paragraph(f"<b>⏱️ {q['time']}</b>", meta_style)
            ],
            [
                Paragraph(f"<b>🖥️ Show on Screen:</b> {q['screen']}", meta_val_style),
                ""
            ],
            [
                Paragraph(f"<b>🗣️ Spoken Transcript:</b><br/>{q['spoken']}", spoken_style),
                ""
            ]
        ]
        
        t_card = Table(card_content, colWidths=[400, 130])
        t_card.setStyle(TableStyle([
            ('SPAN', (0, 1), (1, 1)),
            ('SPAN', (0, 2), (1, 2)),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor("#E2E8F0")),
            ('LINEBELOW', (0, 1), (-1, 1), 0.5, colors.HexColor("#E2E8F0")),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F7FAFC")),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(KeepTogether([t_card, Spacer(1, 8)]))

    # Final checklist
    final_chk = [
        [Paragraph("<b>FINAL SUBMISSION CHECKLIST (BEFORE SENDING LOOM LINK):</b>", meta_style)],
        [Paragraph("☐ Video length is under 8:00 (aim for ~7:30).<br/>"
                   "☐ All 7 questions answered in exact order with question numbers spoken aloud.<br/>"
                   "☐ Live deployed Streamlit demo shown side-by-side with policy PDFs (not screenshots).<br/>"
                   "☐ Loom video privacy set to <i>'Anyone with the link can view'</i>.<br/>"
                   "☐ Kaggle notebook public link pasted into the official submission form.", checklist_style)]
    ]
    t_chk = Table(final_chk, colWidths=[530])
    t_chk.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FEFCBF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#ECC94B")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(Spacer(1, 6))
    story.append(t_chk)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {filename}")

if __name__ == "__main__":
    out1 = "/Users/sarugarisuryatejreddy/.gemini/antigravity/scratch/intelligent-rag/outputs/Loom_Walkthrough_Guide_Intelligent_RAG.pdf"
    out2 = "/Users/sarugarisuryatejreddy/.gemini/antigravity/brain/80c67f09-476e-4313-a43e-b4a6c9f5cc66/Loom_Walkthrough_Guide_Intelligent_RAG.pdf"
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    build_pdf(out1)
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    build_pdf(out2)
