# -*- coding: utf-8 -*-
import os
from fpdf import FPDF

DOCUMENTS = {
    "hr_policy.pdf": {
        "title": "Human Resources Policy Manual",
        "content": (
            "1. Remote Work Policy\n"
            "Employees may work remotely up to 3 days per week with manager approval. "
            "Full remote requires VP sign-off. Employees working remotely must ensure a secure network connection.\n\n"
            "2. Vacation Days\n"
            "New employees receive 15 days of paid vacation in their first year, increasing to 20 days after 3 years of service. "
            "Unused vacation days do not roll over to the next calendar year.\n\n"
            "3. Performance Review\n"
            "Performance reviews occur twice yearly: mid-year in June and year-end in December. "
            "Employees self-assess first, then meet with their manager to discuss ratings and set future goals.\n\n"
            "4. Overtime Calculation\n"
            "Overtime is defined as hours worked beyond 40 per week. Hourly employees receive 1.5x their regular rate. "
            "Salaried exempt employees are not eligible for overtime pay.\n\n"
            "5. Parental Leave\n"
            "Primary caregivers receive 16 weeks of paid parental leave. Secondary caregivers receive 4 weeks. "
            "Adoption is treated the same as birth for leave purposes."
        )
    },
    "security_policy.pdf": {
        "title": "Corporate Data Security Policy",
        "content": (
            "Incident Reporting Procedures\n"
            "Data security is the responsibility of all employees. Security incidents must be reported to "
            "security@company.com within 24 hours of detection. Include incident type, affected systems, "
            "and estimated impact in your report. Failure to report may result in disciplinary action."
        )
    },
    "travel_policy.pdf": {
        "title": "Business Travel and Expenses",
        "content": (
            "Per Diem Limits\n"
            "Employees traveling for business are entitled to a daily allowance for meals and incidentals. "
            "The standard per diem rate is $75 for meals and incidentals. Major cities may qualify for up to $110. "
            "Receipts are required for all expenses exceeding $25."
        )
    },
    "procurement_manual.pdf": {
        "title": "Procurement and Vendor Management Manual",
        "content": (
            "Vendor Onboarding Process\n"
            "All new vendors must go through a strict vetting process. Vendor onboarding requires: "
            "(1) Legal review of contracts, (2) Finance approval for budget, (3) IT security assessment, "
            "(4) Executive sign-off for contracts >$50,000. Vendors must also sign our standard NDA."
        )
    },
    "it_security_standards.pdf": {
        "title": "IT Security Standards and Requirements",
        "content": (
            "Password Complexity and Rotation\n"
            "To prevent unauthorized access, strong passwords are required for all corporate accounts. "
            "Passwords must be at least 12 characters and include uppercase, lowercase, numbers, and special characters. "
            "Passwords expire every 90 days and cannot be reused for 5 generations."
        )
    },
    "finance_policy.pdf": {
        "title": "Corporate Finance and Budgeting Policy",
        "content": (
            "Project Budget Approvals\n"
            "All new projects require appropriate financial sign-off before commencement. "
            "Projects under $10,000 require department head approval. Projects $10,000-$100,000 need Finance Director sign-off. "
            "Above $100,000 requires Board approval. Ensure all budget requests include a detailed ROI analysis."
        )
    },
    "it_policy.pdf": {
        "title": "Information Technology Acceptable Use Policy",
        "content": (
            "Approved Software\n"
            "Employees are strictly prohibited from installing unauthorized software on corporate devices. "
            "Only software on the approved software list (ASL) maintained by IT may be installed. "
            "Requests for new software must go through the IT helpdesk portal and require manager approval."
        )
    },
    "code_of_conduct.pdf": {
        "title": "Employee Code of Conduct",
        "content": (
            "Conflicts of Interest\n"
            "The company expects all employees to act with the highest level of integrity. "
            "Employees must disclose any personal financial interests in vendors, clients, or competitors to HR and their manager. "
            "Undisclosed conflicts are grounds for termination. Annual disclosure forms must be signed every January."
        )
    },
    "data_governance.pdf": {
        "title": "Data Governance and Retention Framework",
        "content": (
            "Email Retention and Archiving\n"
            "To comply with legal and regulatory standards, specific data must be retained for set periods. "
            "Customer emails must be retained for 7 years per regulatory requirements. "
            "After 7 years, they are automatically purged from the archive system. Do not manually delete customer communications."
        )
    },
    "benefits_guide.pdf": {
        "title": "Employee Benefits Guide",
        "content": (
            "Employee Stock Option Plan (ESOP)\n"
            "The company believes in sharing its success with its workforce. "
            "Full-time employees who have completed 1 year of service are eligible to participate in the ESOP. "
            "Part-time and contract workers are not eligible. Vesting schedules are outlined in your individual grant agreement."
        )
    },
    "customer_service_manual.pdf": {
        "title": "Customer Service Operations Manual",
        "content": (
            "Handling Customer Complaints\n"
            "We pride ourselves on excellent customer service. "
            "All complaints must be logged in the CRM within 24 hours. "
            "First response to the customer is required within 48 hours. "
            "Escalation to management occurs if not resolved within 5 business days."
        )
    }
}

def generate_pdfs(output_dir="dummy_docs"):
    os.makedirs(output_dir, exist_ok=True)
    for filename, data in DOCUMENTS.items():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, data["title"], ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        # Using built-in replace for simple character normalization since fpdf only supports latin-1 natively
        text = data["content"].replace('$', 'USD ').encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, text)
        
        output_path = os.path.join(output_dir, filename)
        pdf.output(output_path)
        print(f"Created: {output_path}")

if __name__ == "__main__":
    generate_pdfs()
